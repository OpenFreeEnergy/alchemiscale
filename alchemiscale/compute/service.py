"""
:mod:`alchemiscale.compute.service` --- compute services for FEC execution
==========================================================================

"""

from contextlib import contextmanager
import gc
import time
import logging
from uuid import uuid4
import threading
from pathlib import Path
import shutil

from gufe import Transformation
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from .client import AlchemiscaleComputeClient
from .settings import ComputeServiceSettings
from ..storage.models import ComputeServiceID
from ..models import Scope, ScopedKey

# moved to alchemiscale.sleep; re-exported here for backwards compatibility
from ..sleep import InterruptableSleep, SleepInterrupted


class SynchronousComputeService:
    """Fully synchronous compute service.

    This service is intended for use as a reference implementation, and for
    testing/debugging protocols.

    """

    def __init__(self, settings: ComputeServiceSettings):
        """Create a `SynchronousComputeService` instance."""
        self.settings = settings

        self.api_url = self.settings.api_url
        self.name = self.settings.name
        self.compute_manager_id = self.settings.compute_manager_id
        self.sleep_interval = self.settings.sleep_interval
        self.deep_sleep_interval = self.settings.deep_sleep_interval
        self.heartbeat_interval = self.settings.heartbeat_interval
        self.claim_limit = self.settings.claim_limit
        self.max_tasks = self.settings.max_tasks
        self.max_time = self.settings.max_time

        self.client = AlchemiscaleComputeClient(
            api_url=self.settings.api_url,
            identifier=self.settings.identifier,
            key=self.settings.key,
            cache_directory=self.settings.client_cache_directory,
            cache_size_limit=self.settings.client_cache_size_limit,
            use_local_cache=self.settings.client_use_local_cache,
            max_retries=self.settings.client_max_retries,
            retry_base_seconds=self.settings.client_retry_base_seconds,
            retry_max_seconds=self.settings.client_retry_max_seconds,
            verify=self.settings.client_verify,
        )

        if self.settings.scopes is None:
            self.scopes = [Scope()]
        else:
            self.scopes = self.settings.scopes

        self.scopes_exclude = self.settings.scopes_exclude

        self.shared_basedir = Path(self.settings.shared_basedir).absolute()
        self.shared_basedir.mkdir(exist_ok=True)
        self.keep_shared = self.settings.keep_shared

        self.scratch_basedir = Path(self.settings.scratch_basedir).absolute()
        self.scratch_basedir.mkdir(exist_ok=True)
        self.keep_scratch = self.settings.keep_scratch

        self.compute_service_id = ComputeServiceID.new_from_name(self.name)

        # shared between the main loop and the heartbeat thread; both wake
        # on a single ``stop()`` (which calls ``int_sleep.interrupt()``).
        # If you split these into separate functors, be sure to interrupt
        # both from ``stop()`` --- otherwise the heartbeat thread stays
        # alive (and the worker stays multi-threaded) until its sleep
        # naturally expires.
        self.int_sleep = InterruptableSleep()

        self._stop = False

        # logging
        extra = {"compute_service_id": str(self.compute_service_id)}
        logger = logging.getLogger("AlchemiscaleSynchronousComputeService")
        logger.setLevel(self.settings.loglevel)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_service_id)s] [%(levelname)s] %(message)s"
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

    def _register(self):
        """Register this compute service with the compute API."""
        self.client.register(self.compute_service_id, self.settings.compute_manager_id)

    def _deregister(self):
        """Deregister this compute service with the compute API."""
        self.client.deregister(self.compute_service_id)

    def beat(self):
        """Deliver a heartbeat to the compute API, indicating this service is still alive."""
        self.client.heartbeat(self.compute_service_id)
        self.logger.debug("Updated heartbeat")

    def heartbeat(self):
        """Start up the heartbeat, sleeping for `self.heartbeat_interval`.

        A failing ``beat()`` is logged and the loop continues; the thread
        only exits on ``stop()`` (via ``SleepInterrupted``) or process
        teardown. The compute API exposes its own retry policy on the
        underlying client call, but if those retries are ever exhausted ---
        e.g. a sustained outage or auth-token expiry --- a raised exception
        here would otherwise silently kill the heartbeat thread while the
        main loop kept processing tasks. Swallow per-beat failures so a
        transient problem doesn't deregister the service from the API's
        point of view.
        """
        while not self._stop:
            try:
                self.beat()
            except Exception:
                self.logger.exception("Heartbeat failed; will retry next interval")
            try:
                self.int_sleep(self.heartbeat_interval)
            except SleepInterrupted:
                break

    def claim_tasks(self, count=1) -> list[ScopedKey | None]:
        """Get a Task to execute from compute API.

        Returns `None` if no Task was available matching service configuration.

        Parameters
        ----------
        count
            The maximum number of Tasks to claim.
        """

        tasks = self.client.claim_tasks(
            scopes=self.scopes,
            scopes_exclude=self.scopes_exclude,
            compute_service_id=self.compute_service_id,
            count=count,
            protocols=self.settings.protocols,
        )

        return tasks

    def task_to_protocoldag(
        self, task: ScopedKey
    ) -> tuple[ProtocolDAG, Transformation, ProtocolDAGResult | None]:
        """Given a Task, produce a corresponding ProtocolDAG that can be executed.

        Also gives the Transformation that this ProtocolDAG corresponds to.
        If the Task extends another Task, then the ProtocolDAGResult for that
        other Task is also given; otherwise `None` given.

        """

        (
            transformation,
            extends_protocoldagresult,
        ) = self.client.retrieve_task_transformation(task)

        protocoldag = transformation.create(
            extends=extends_protocoldagresult,
            name=str(task),
        )
        return protocoldag, transformation, extends_protocoldagresult

    def push_result(
        self, task: ScopedKey, protocoldagresult: ProtocolDAGResult
    ) -> ScopedKey:
        # TODO: this method should postprocess any paths,
        # leaf nodes in DAG for blob results that should go to object store

        # TODO: ship paths to object store

        # finally, push ProtocolDAGResult
        sk: ScopedKey = self.client.set_task_result(
            task, protocoldagresult, self.compute_service_id
        )

        return sk

    def execute(self, task: ScopedKey) -> ScopedKey:
        """Executes given Task.

        Returns ScopedKey of ProtocolDAGResultRef following push to database.

        """
        # obtain a ProtocolDAG from the task
        self.logger.info("Creating ProtocolDAG from '%s'...", task)
        protocoldag, transformation, extends = self.task_to_protocoldag(task)
        self.logger.info(
            "Created '%s' from '%s' performing '%s'",
            protocoldag,
            task,
            transformation.protocol,
        )

        # execute the task; this looks the same whether the ProtocolDAG is a
        # success or failure

        shared = self.shared_basedir / str(protocoldag.key)
        shared.mkdir()
        scratch = self.scratch_basedir / str(protocoldag.key)
        scratch.mkdir()

        self.logger.info("Executing '%s'...", protocoldag)
        try:
            protocoldagresult = execute_DAG(
                protocoldag,
                shared_basedir=shared,
                scratch_basedir=scratch,
                keep_scratch=self.keep_scratch,
                raise_error=False,
                n_retries=self.settings.n_retries,
            )
        finally:
            if not self.keep_shared:
                shutil.rmtree(shared)

            if not self.keep_scratch:
                shutil.rmtree(scratch)

        if protocoldagresult.ok():
            self.logger.info("'%s' -> '%s' : SUCCESS", protocoldag, protocoldagresult)
        else:
            for failure in protocoldagresult.protocol_unit_failures:
                self.logger.info(
                    "'%s' -> '%s' : FAILURE :: '%s' : %s",
                    protocoldag,
                    protocoldagresult,
                    failure,
                    failure.exception,
                )

        # push the result (or failure) back to the compute API
        result_sk = self.push_result(task, protocoldagresult)
        self.logger.info("Pushed result `%s'", protocoldagresult)

        return result_sk

    def _check_max_tasks(self, max_tasks):
        if max_tasks is not None:
            if self._tasks_counter >= max_tasks:
                self.logger.info(
                    "Performed %s tasks; at or beyond max tasks = %s",
                    self._tasks_counter,
                    max_tasks,
                )
                raise KeyboardInterrupt

    def _check_max_time(self, max_time):
        if max_time is not None:
            run_time = time.time() - self._start_time
            if run_time >= max_time:
                self.logger.info(
                    "Ran for %s seconds; at or beyond max time = %s seconds",
                    run_time,
                    max_time,
                )
                raise KeyboardInterrupt

    def cycle(self, max_tasks: int | None = None, max_time: int | None = None):
        self._check_max_tasks(max_tasks)
        self._check_max_time(max_time)

        # claim tasks from the compute API
        self.logger.info("Claiming tasks")
        tasks: list[ScopedKey] | None = self.claim_tasks(count=self.claim_limit)

        if tasks is None:
            self.logger.info("No tasks claimed. Compute API denied request.")
            self.int_sleep(self.deep_sleep_interval)
            return

        self.logger.info("Claimed %d tasks", len([t for t in tasks if t is not None]))

        # if no tasks claimed, sleep
        if all([task is None for task in tasks]):
            self.logger.info(
                "No tasks claimed; sleeping for %d seconds", self.sleep_interval
            )
            self.int_sleep(self.sleep_interval)
            return

        # otherwise, process tasks
        self.logger.info("Executing tasks...")
        for task in tasks:
            if task is None:
                continue

            # execute each task
            self.logger.info("Executing task '%s'...", task)
            self.execute(task)
            self.logger.info("Finished task '%s'", task)

            if max_tasks is not None:
                self._tasks_counter += 1

            # stop checks
            self._check_max_tasks(max_tasks)
            self._check_max_time(max_time)

        self._check_max_tasks(max_tasks)
        self._check_max_time(max_time)

    def _start_heartbeat(self):
        self.heartbeat_thread = threading.Thread(
            target=self.heartbeat,
            daemon=True,
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self, timeout=5):
        self.stop()

        heartbeat_thread = getattr(self, "heartbeat_thread", None)
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=timeout)

            if heartbeat_thread.is_alive():
                self.logger.warning(
                    "Heartbeat thread did not stop within %s seconds",
                    timeout,
                )

    @contextmanager
    def _running(self):
        """Register the service and run heartbeat for the lifetime of the context."""
        registered = False
        heartbeat_started = False

        try:
            self._register()
            registered = True

            self.logger.info(
                "Registered service with registration '%s'",
                str(self.compute_service_id),
            )

            self._stop = False
            self.int_sleep.clear()

            self._start_heartbeat()
            heartbeat_started = True

            yield

        finally:
            if heartbeat_started:
                self._stop_heartbeat(timeout=5)
            else:
                # If interrupted after registration but before heartbeat startup,
                # still request a clean stop.
                self.stop()

            if registered:
                self.logger.info(
                    "Deregistering service with registration '%s'",
                    str(self.compute_service_id),
                )
                self._deregister()
                self.logger.info("Deregistration successful")

    def start(self):
        """Start the service.

        The service runs until it is told to stop, or until one of the limits
        configured on its settings is reached. ``max_tasks`` caps the number of
        Tasks executed and ``max_time`` caps the number of seconds run; the
        first maximum to be hit triggers the service to exit. Either limit being
        ``None`` (the default) means no limit of that kind.

        """
        max_tasks = self.max_tasks
        max_time = self.max_time

        self.logger.info("Starting up service '%s'", self.name)

        with self._running():
            try:
                self._tasks_counter = 0
                self._start_time = time.time()

                self.logger.info("Starting main loop")

                while not self._stop:
                    self.cycle(max_tasks=max_tasks, max_time=max_time)
                    gc.collect()

            except SleepInterrupted:
                self.logger.info("Service stopping.")

            except KeyboardInterrupt:
                self.logger.info("Caught SIGINT/Keyboard interrupt.")

            except Exception:
                self.logger.exception("Unknown exception raised")
                raise

    def stop(self):
        self.int_sleep.interrupt()
        self._stop = True


class AsynchronousComputeService(SynchronousComputeService):
    """Asynchronous compute service.

    This service can be used in production cases, though it does not make use
    of Folding@Home.

    """

    def __init__(self, api_url):
        # self.loop = asyncio.get_event_loop()

        self._stop = False

    def get_new_tasks(self): ...

    def start(self):
        """Start the service; will keep going until told to stop."""
        self._stop = False

        while True:
            if self._stop:
                return

    def stop(self):
        self._stop = True
