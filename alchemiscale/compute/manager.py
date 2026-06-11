"""
:mod:`alchemiscale.compute.manager` --- compute manager for creating compute services
=====================================================================================

"""

from abc import abstractmethod
from contextlib import contextmanager
import logging
import time

from ..storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerStatus,
)
from .client import (
    AlchemiscaleComputeManagerClient,
    AlchemiscaleComputeManagerClientError,
)
from .settings import ComputeManagerSettings, ComputeServiceSettings
from ..sleep import InterruptableSleep, SleepInterrupted


class ComputeManager:

    compute_manager_id: ComputeManagerID
    client: AlchemiscaleComputeManagerClient
    service_settings: ComputeServiceSettings
    settings: ComputeManagerSettings

    def __init__(
        self, settings: ComputeManagerSettings, service_settings: ComputeServiceSettings
    ):
        self.settings = settings
        self.compute_manager_id = ComputeManagerID.new_from_name(self.settings.name)
        self.service_settings = service_settings
        self.service_settings.compute_manager_id = self.compute_manager_id
        self.client = AlchemiscaleComputeManagerClient(
            api_url=self.service_settings.api_url,
            identifier=self.service_settings.identifier,
            key=self.service_settings.key,
            max_retries=self.settings.client_max_retries,
            retry_base_seconds=self.settings.client_retry_base_seconds,
            retry_max_seconds=self.settings.client_retry_max_seconds,
        )

        self._stop = False
        self.int_sleep = InterruptableSleep()

        logger = logging.getLogger("AlchemiscaleComputeManager")
        logger.setLevel(self.settings.loglevel)

        extra = {"compute_manager_id": self.compute_manager_id}
        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_manager_id)s] [%(levelname)s] %(message)s"
        )
        formatter.converter = time.gmtime  # use utc time for logging timestamps

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if self.settings.logfile is not None:
            fh = logging.FileHandler(self.settings.logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logging.LoggerAdapter(logger, extra)

    def _register(self, steal=False):
        try:
            self.client.register(self.compute_manager_id, steal=steal)
        except AlchemiscaleComputeManagerClientError as e:
            self.logger.error(f"Registration failed: '{e}'")
            raise e

    def _deregister(self):
        self.client.deregister(self.compute_manager_id)

    @contextmanager
    def _registered(self, steal=False):
        """Register this compute manager for the lifetime of the context.

        This guarantees that if anything interrupts startup after registration,
        including int_sleep.clear(), the manager is still deregistered.
        """
        registered = False

        try:
            self._register(steal=steal)
            registered = True

            self.logger.info(
                f"Registered compute manager '{self.compute_manager_id}'"
            )

            yield

        finally:
            if registered:
                self.logger.info(f"Deregistering '{self.compute_manager_id}'")

                self.stop()

                heartbeat_thread = getattr(self, "heartbeat_thread", None)
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=5)

                    if heartbeat_thread.is_alive():
                        self.logger.warning(
                            "Heartbeat thread did not stop within 5 seconds"
                        )

                self._deregister()
                self.logger.info("Deregistration successful")

    def start(self, max_cycles: int | None = None, steal=False):
        self.logger.info(f"Starting up compute manager '{self.settings.name}'")

        with self._registered(steal=steal):
            try:
                self._stop = False
                self.int_sleep.clear()

                count = 0
                self.logger.info("Starting main loop")

                while self.cycle():
                    count += 1

                    if max_cycles and count >= max_cycles:
                        self.logger.info("Reached maximum number of cycles")
                        break

                    self.logger.info(
                        f"Sleeping for {self.settings.sleep_interval} seconds"
                    )
                    self.int_sleep(self.settings.sleep_interval)

            except SleepInterrupted:
                self.logger.info("Compute manager stopping.")

            except KeyboardInterrupt:
                self.logger.info("Caught SIGINT/Keyboard interrupt.")

            except Exception as e:
                self.logger.error(f"Unknown exception raised: '{str(e)}'")
                self.logger.info("Updating manager status to 'ERROR'")

                self.client.update_status(
                    self.compute_manager_id,
                    ComputeManagerStatus.ERROR,
                    detail=repr(e),
                )

                raise

    @abstractmethod
    def create_compute_services(self, data: dict, target: int) -> int:
        """Submit up to ``target`` new compute services and return the actual
        number created.

        Called by :meth:`cycle` when scaling up is warranted. ``target`` is the
        per-cycle sizing decision computed by :meth:`_compute_jobs_to_create`,
        which already accounts for the number of waiting tasks, the per-cycle
        rate limit (``max_submit_per_cycle``), the remaining capacity
        (``max_compute_services - len(compute_service_ids)``), and the
        per-service ``claim_limit``.

        Subclasses still own backend-specific gating (e.g. checking whether
        any previously submitted jobs are still pending in the batch system,
        running health checks). ``target`` is an upper bound, not a guarantee.

        Parameters
        ----------
        data
            Payload from the OK ``ComputeManagerInstruction``. Includes
            ``compute_service_ids`` and ``num_tasks`` at minimum.
        target
            Upper bound on the number of services to create this cycle.
            Always ``>= 1`` when this method is called; the cycle short-
            circuits and never calls in if there is no work or no capacity.

        Returns
        -------
        int
            Number of compute services actually created.
        """
        raise NotImplementedError

    def _compute_jobs_to_create(self, num_tasks: int, num_active_services: int) -> int:
        """Decide how many new compute services to create this cycle.

        The formula:

            ``min(num_tasks, max_submit_per_cycle, remaining_capacity)
            // claim_limit``

        with a floor of one when the cycle has decided we should be scaling
        up (i.e. ``num_tasks > 0`` and ``remaining_capacity > 0``). The
        floor handles the case where the ``claim_limit`` divide collapses
        to zero (e.g. a single task with ``claim_limit=2``): rather than
        stalling, create one service and let the next cycle catch up.

        Subclasses should rarely override this. The default rule is the
        universal autoscaling sizing logic; backend-specific gating (e.g.
        "skip this cycle if any submitted jobs are still pending") belongs
        in :meth:`create_compute_services`, not here.

        Parameters
        ----------
        num_tasks
            Number of tasks waiting on the server.
        num_active_services
            Number of compute services currently registered with the server.

        Returns
        -------
        int
            Target number of new services to create. May be ``0`` if there is
            no work or no remaining capacity.
        """
        remaining_capacity = self.settings.max_compute_services - num_active_services
        if remaining_capacity <= 0 or num_tasks <= 0:
            return 0

        jobs = min(
            num_tasks,
            self.settings.max_submit_per_cycle,
            remaining_capacity,
        )

        # Each compute service claims up to claim_limit tasks at a time, so
        # we need fewer services than tasks to cover the queue.
        # ``claim_limit`` is validated as PositiveInt at config load.
        jobs //= self.service_settings.claim_limit

        # Floor at one: the parent has decided we should scale up, so create
        # at least one service even if the divide collapsed to zero.
        return jobs or 1

    def stop(self):
        self.int_sleep.interrupt()
        self._stop = True

    def cycle(self) -> bool:

        if self._stop:
            return False

        self.logger.info(f"Requesting instruction from '{self.client.api_url}'")
        instruction, data = self.client.get_instruction(
            self.service_settings.scopes or [],
            self.service_settings.protocols,
            self.compute_manager_id,
        )
        match instruction:
            case ComputeManagerInstruction.OK:
                total_services = len(data["compute_service_ids"])
                num_tasks = data["num_tasks"]
                self.logger.info(
                    f"Received instruction '{instruction}', {num_tasks} tasks available"
                )
                if (
                    total_services < self.settings.max_compute_services
                    and num_tasks > 0
                ):
                    target = self._compute_jobs_to_create(
                        num_tasks=num_tasks,
                        num_active_services=total_services,
                    )
                    self.logger.info(
                        f"Sizing: num_tasks={num_tasks}, "
                        f"max_submit_per_cycle={self.settings.max_submit_per_cycle}, "
                        f"remaining_capacity="
                        f"{self.settings.max_compute_services - total_services}, "
                        f"claim_limit={self.service_settings.claim_limit} "
                        f"-> target={target}"
                    )
                    if target > 0:
                        new_services = self.create_compute_services(data, target)
                        total_services += new_services
                        if new_services:
                            self.logger.info(
                                f"Created {new_services} new compute service(s)"
                            )
                        else:
                            self.logger.info("No new compute services created")
                    else:
                        self.logger.info(
                            "Sizing returned 0; no compute services created"
                        )
            case ComputeManagerInstruction.SKIP:
                total_services = len(data["compute_service_ids"])
                self.logger.info("Received skip instruction")
            case ComputeManagerInstruction.SHUTDOWN:
                shutdown_message = data["message"]
                self.logger.info(f'Received shutdown message: "{shutdown_message}"')
                return False
        self.logger.info(f"Updating manager status at '{self.client.api_url}'")
        self.client.update_status(
            self.compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=total_services / self.settings.max_compute_services,
        )
        return True

    def clear_error(self):
        try:
            self.logger.info(f"Clearing '{self.settings.name}' with the ERROR status")
            self.client.clear_error(self.settings.name)
        except ValueError:
            self.logger.info(
                f"Could not clear '{self.settings.name}', no such compute manager with the ERROR status"
            )
