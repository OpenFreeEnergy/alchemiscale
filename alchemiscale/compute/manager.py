"""
:mod:`alchemiscale.compute.manager` --- compute manager for creating compute services
=====================================================================================

"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import datetime
from importlib.metadata import version
import logging
import time

from packaging.version import Version

from ..storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerJobRecord,
    ComputeManagerState,
    ComputeManagerStateUpdate,
    ComputeManagerStatus,
    ComputeServiceID,
    JobRecordKind,
)
from .backend import (
    BackendError,
    ComputeBackend,
    FailureKind,
    Job,
    JobState,
    PermanentBackendError,
    TransientBackendError,
    make_job_name,
)
from .client import (
    AlchemiscaleComputeManagerClient,
    AlchemiscaleComputeManagerClientError,
    ComputeManagerNotRegisteredError,
)
from .settings import ComputeManagerSettings, ComputeServiceSettings
from ..sleep import InterruptableSleep, SleepInterrupted

#: The oldest alchemiscale server this compute manager can work with.
MIN_SERVER_VERSION = Version("0.9.0")


class IncompatibleServerError(RuntimeError):
    """Raised when the alchemiscale server is too old for this compute manager."""


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


@dataclass
class Reconciliation:
    """What one cycle learned by comparing the backend's jobs with the
    manager's shared state."""

    #: records to merge into the shared state
    records: list[ComputeManagerJobRecord] = field(default_factory=list)
    #: records to remove from the shared state
    forget: list[tuple[str, JobRecordKind]] = field(default_factory=list)
    #: time of the most recent sign of health found this cycle
    last_healthy_at: datetime.datetime | None = None
    #: jobs to remove once the records are stored
    to_remove: list[Job] = field(default_factory=list)
    #: names of jobs still pending or running
    active: set[str] = field(default_factory=set)
    #: whether any job is still pending
    pending: bool = False
    #: WORKLOAD failures found this cycle
    workload_failures: list[ComputeManagerJobRecord] = field(default_factory=list)

    def record(self, job_name: str, kind: JobRecordKind, now, **kwargs):
        self.records.append(
            ComputeManagerJobRecord(job_name=job_name, kind=kind, at=now, **kwargs)
        )

    def healthy(self, at: datetime.datetime):
        if self.last_healthy_at is None or at > self.last_healthy_at:
            self.last_healthy_at = at


class ComputeManager:
    """Autoscales compute services on a batch system, through a
    :class:`~alchemiscale.compute.backend.ComputeBackend`.

    Each cycle, whatever the instruction from the server, the manager lists
    the backend's jobs, handles each terminal job once, checks that running
    jobs' compute services have registered, and, if there is work and
    capacity, submits new jobs. Facts about jobs are kept on the
    registration, which all instances of the manager share, so that several
    instances under one name behave like one manager.

    """

    compute_manager_id: ComputeManagerID
    client: AlchemiscaleComputeManagerClient
    service_settings: ComputeServiceSettings
    settings: ComputeManagerSettings
    backend: ComputeBackend

    def __init__(
        self,
        settings: ComputeManagerSettings,
        service_settings: ComputeServiceSettings,
        backend: ComputeBackend,
    ):
        self.settings = settings
        self.backend = backend
        self.compute_manager_id = ComputeManagerID.new_from_name(self.settings.name)
        self.service_settings = service_settings
        self.service_settings.compute_manager_id = self.compute_manager_id
        self.backend.compute_manager_id = self.compute_manager_id
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

    def _check_server_version(self):
        """Raise :class:`IncompatibleServerError` if the server is too old.

        A server older than 0.9.0 doesn't report its version, and is
        treated as too old. A development build of this client also
        accepts a server at least as new as itself, so that unreleased
        clients and servers built from the same source work together.

        """
        client_version = Version(version("alchemiscale"))
        required = min(MIN_SERVER_VERSION, client_version)

        server_version = self.client.get_server_version()
        if server_version is None or server_version < required:
            reported = (
                server_version
                if server_version is not None
                else "older than 0.9.0, which doesn't report its version"
            )
            raise IncompatibleServerError(
                f"The alchemiscale server at '{self.client.api_url}' is version "
                f"{reported}, but this compute manager (alchemiscale "
                f"{client_version}) requires a server of version {required} or "
                "newer. Upgrade the server before the compute manager."
            )

        self.logger.info(f"Server version {server_version}")

    def _register(self, steal=False):
        """Register with the server, joining the registration under this
        manager's name if there is a live one, and adopt its ID."""
        try:
            compute_manager_id = self.client.register(
                ComputeManagerID.new_from_name(self.settings.name), steal=steal
            )
        except AlchemiscaleComputeManagerClientError as e:
            self.logger.error(f"Registration failed: '{e}'")
            raise e

        if compute_manager_id != self.compute_manager_id:
            self.logger.info(f"Using registration '{compute_manager_id}'")
        self.compute_manager_id = compute_manager_id
        self.service_settings.compute_manager_id = compute_manager_id
        self.backend.compute_manager_id = compute_manager_id
        if isinstance(self.logger, logging.LoggerAdapter):
            self.logger.extra["compute_manager_id"] = compute_manager_id

    def _deregister(self):
        try:
            self.client.deregister(self.compute_manager_id)
        except ComputeManagerNotRegisteredError:
            self.logger.warning(
                f"'{self.compute_manager_id}' was no longer registered with the server"
            )

    @contextmanager
    def _running(self, steal=False):
        """Register this compute manager for the lifetime of the context.

        This guarantees that if anything interrupts startup after registration,
        including int_sleep.clear(), the manager is still deregistered.
        """
        registered = False

        try:
            self._register(steal=steal)
            registered = True

            self.logger.info(f"Registered compute manager '{self.compute_manager_id}'")

            self._stop = False
            self.int_sleep.clear()

            yield

        finally:
            if registered:
                self.logger.info(f"Deregistering '{self.compute_manager_id}'")

                # kept here in case we add additional cleanup to stop later, such as other threads
                self.stop()

                self._deregister()
                self.logger.info("Deregistration successful")

    def start(self, max_cycles: int | None = None, steal=False):
        self.logger.info(f"Starting up compute manager '{self.settings.name}'")

        try:
            self._check_server_version()
        except IncompatibleServerError as e:
            self.logger.error(str(e))
            raise

        try:
            self.backend.validate()
        except BackendError as e:
            self.logger.error(f"Invalid backend configuration: {e}")
            raise

        with self._running(steal=steal):
            try:
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

                try:
                    self.client.update_status(
                        self.compute_manager_id,
                        ComputeManagerStatus.ERROR,
                        detail=repr(e),
                    )
                except ComputeManagerNotRegisteredError:
                    self.logger.warning(
                        f"Could not report the error: '{self.compute_manager_id}' "
                        "is no longer registered with the server"
                    )

                raise

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

        Parameters
        ----------
        num_tasks
            Number of tasks waiting on the server.
        num_active_services
            Number of distinct names across pending and running jobs and
            compute services registered with the server.

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
        """Run one cycle; return ``False`` if the manager should stop."""

        if self._stop:
            return False

        self.logger.info(f"Requesting instruction from '{self.client.api_url}'")
        try:
            instruction, data = self.client.get_instruction(
                self.service_settings.scopes or [],
                self.service_settings.protocols,
                self.compute_manager_id,
            )
        except ComputeManagerNotRegisteredError:
            # the registration expired, for example after a network partition;
            # join or recreate it, and carry on next cycle
            self.logger.warning(
                f"'{self.compute_manager_id}' is no longer registered with the "
                "server; registering again"
            )
            self._register()
            return True

        if instruction == ComputeManagerInstruction.SHUTDOWN:
            self.logger.info(f'Received shutdown message: "{data["message"]}"')
            return False

        state: ComputeManagerState = data["state"]
        num_tasks = data.get("num_tasks", 0)
        registered = {
            ComputeServiceID(csid).name for csid in data["compute_service_ids"]
        }
        if instruction == ComputeManagerInstruction.SKIP:
            self.logger.info("Received skip instruction")
        else:
            self.logger.info(
                f"Received instruction '{instruction}', {num_tasks} tasks available"
            )

        now = _utcnow()
        update = ComputeManagerStateUpdate(expected_version=state.registration.version)

        # the state was read before listing jobs, so a job another instance
        # recorded as submitted is already visible to the backend
        try:
            jobs = self.backend.list_jobs()
        except BackendError as e:
            self.logger.warning(f"Could not list jobs; skipping maintenance: {e}")
            jobs = None

        if jobs is not None:
            reconciliation = self._reconcile(jobs, state, registered, now)
            active = reconciliation.active
        else:
            reconciliation = Reconciliation()
            # without the backend's view, assume recorded jobs are still active
            active = set(state.records_of(JobRecordKind.ACTIVE))

        update.records += reconciliation.records
        update.forget += reconciliation.forget
        update.last_healthy_at = reconciliation.last_healthy_at
        update.prune_handled_before = self._prune_handled_before(now)

        in_use = active | registered

        if (
            instruction == ComputeManagerInstruction.OK
            and jobs is not None
            and not reconciliation.pending
        ):
            submitted = self._submit(
                self._compute_jobs_to_create(num_tasks, len(in_use)), now
            )
            in_use |= {record.job_name for record in submitted}
            update.records += submitted
        elif reconciliation.pending:
            self.logger.info("Jobs are still pending; not submitting")

        self._post_state(
            ComputeManagerStatus.OK, None, self._saturation(len(in_use)), update
        )

        # remove jobs only now that their outcomes are stored
        self._remove(reconciliation.to_remove)

        return True

    def _saturation(self, in_use: int) -> float:
        return min(1.0, in_use / self.settings.max_compute_services)

    def _post_state(
        self,
        status: ComputeManagerStatus,
        detail: str | None,
        saturation: float,
        update: ComputeManagerStateUpdate,
    ) -> tuple[bool, ComputeManagerState]:
        self.logger.info(f"Updating manager status at '{self.client.api_url}'")
        applied, state = self.client.update_state(
            self.compute_manager_id,
            status,
            detail=detail,
            saturation=saturation,
            state_update=update,
        )
        if not applied:
            self.logger.info(
                "Another instance changed the shared state this cycle; "
                "reconsidering next cycle"
            )
        return applied, state

    def _prune_handled_before(self, now) -> datetime.datetime | None:
        lookback = self.backend.finished_job_lookback
        return now - lookback if lookback is not None else None

    def _reconcile(
        self,
        jobs: list[Job],
        state: ComputeManagerState,
        registered: set[str],
        now: datetime.datetime,
    ) -> Reconciliation:
        """Compare the backend's jobs with the shared state.

        Each terminal job is handled once: its outcome is recorded, and it is
        removed. Running jobs whose compute service hasn't registered are
        checked against the registration grace period.

        """
        result = Reconciliation()
        handled = state.records_of(JobRecordKind.HANDLED)
        recorded_active = state.records_of(JobRecordKind.ACTIVE)
        seen_registered = state.records_of(JobRecordKind.REGISTERED)
        unregistered = state.records_of(JobRecordKind.UNREGISTERED)
        listed = {job.name for job in jobs}

        for job in jobs:
            if job.name in handled:
                # already handled; removing again is harmless, and finishes a
                # removal that failed or a job cancelled for not registering
                result.to_remove.append(job)
                continue

            if job.state.terminal:
                self._handle_terminal(
                    job,
                    job.name in seen_registered or job.name in registered,
                    result,
                    now,
                )
                result.to_remove.append(job)
                continue

            result.active.add(job.name)
            if job.state == JobState.PENDING:
                result.pending = True

            if job.name not in recorded_active:
                result.record(
                    job.name, JobRecordKind.ACTIVE, now, submitted_at=job.submitted_at
                )

            if job.name in registered:
                if job.name not in seen_registered:
                    result.record(job.name, JobRecordKind.REGISTERED, now)
                if job.name in unregistered:
                    result.forget.append((job.name, JobRecordKind.UNREGISTERED))
                continue

            # a job whose service has registered before may be ending: its
            # service has deregistered while the batch system still reports
            # it running
            if (
                job.state == JobState.RUNNING
                and job.started_at is not None
                and job.name not in seen_registered
            ):
                self._check_registration(job, unregistered.get(job.name), result, now)

        # jobs that disappeared without being seen in a terminal state
        for name in recorded_active:
            if name not in listed and name not in handled:
                self.logger.warning(
                    f"Job '{name}' disappeared without a final state; its outcome "
                    "is unknown, and it counts as neither a failure nor healthy"
                )
                result.record(name, JobRecordKind.HANDLED, now, ended_at=now)

        # without a lookback, a removed job is gone for good, and its handled
        # record is no longer needed
        if self.backend.finished_job_lookback is None:
            result.forget += [
                (name, JobRecordKind.HANDLED) for name in handled if name not in listed
            ]

        return result

    def _handle_terminal(
        self,
        job: Job,
        seen_registered: bool,
        result: Reconciliation,
        now: datetime.datetime,
    ):
        ended_at = job.ended_at or now
        reason = f": {job.reason}" if job.reason else ""

        match job.state:
            case JobState.SUCCEEDED if seen_registered:
                self.logger.info(f"Job '{job.name}' succeeded")
                result.healthy(ended_at)
            case JobState.SUCCEEDED:
                self.logger.warning(
                    f"Job '{job.name}' succeeded, but its compute service was never "
                    "seen registered; check that the job starts a compute service "
                    "registering under the job's name"
                )
            case JobState.FAILED:
                kind = job.failure or FailureKind.WORKLOAD
                # a failure after the service registered, such as a GPU lost
                # mid-run, is diagnosed differently from a broken template
                after = (
                    "after its compute service registered"
                    if seen_registered
                    else "before its compute service registered"
                )
                reason = f" {after}" + (f": {job.reason}" if job.reason else "")
                record = ComputeManagerJobRecord(
                    job_name=job.name,
                    kind=JobRecordKind.FAILURE,
                    at=now,
                    submitted_at=job.submitted_at,
                    ended_at=ended_at,
                    failure_kind=kind.value,
                    reason=f"{after}: {job.reason}" if job.reason else after,
                )
                result.records.append(record)
                if kind == FailureKind.WORKLOAD:
                    self.logger.warning(f"Job '{job.name}' failed{reason}")
                    result.workload_failures.append(record)
                else:
                    self.logger.info(
                        f"Job '{job.name}' failed{reason}; the failure is outside "
                        "its control, and isn't counted"
                    )

        result.record(job.name, JobRecordKind.HANDLED, now, ended_at=ended_at)

    def _check_registration(
        self,
        job: Job,
        unregistered: ComputeManagerJobRecord | None,
        result: Reconciliation,
        now: datetime.datetime,
    ):
        """Check that a running job's compute service has registered.

        A job is flagged only after two checks find it unregistered, at least
        ``registration_grace_period`` apart, counted from when it started
        running. A flagged job is recorded as a WORKLOAD failure and removed.

        """
        grace = datetime.timedelta(seconds=self.settings.registration_grace_period)
        if now - job.started_at < grace:
            return

        if unregistered is None:
            self.logger.info(
                f"Job '{job.name}' is running, but its compute service hasn't "
                "registered; checking again next cycle"
            )
            result.record(job.name, JobRecordKind.UNREGISTERED, now)
            return

        if now - unregistered.at < grace:
            return

        reason = (
            f"running since {job.started_at.isoformat()} without its compute "
            "service registering with the server"
        )
        self.logger.warning(f"Job '{job.name}' is {reason}; removing it")
        record = ComputeManagerJobRecord(
            job_name=job.name,
            kind=JobRecordKind.FAILURE,
            at=now,
            submitted_at=job.submitted_at,
            ended_at=now,
            failure_kind=FailureKind.WORKLOAD.value,
            reason=reason,
        )
        result.records.append(record)
        result.workload_failures.append(record)
        result.record(job.name, JobRecordKind.HANDLED, now, ended_at=now)
        result.active.discard(job.name)
        result.to_remove.append(job)

    def _submit(
        self, target: int, now: datetime.datetime
    ) -> list[ComputeManagerJobRecord]:
        """Submit up to ``target`` jobs, returning ACTIVE records for those
        the backend accepted."""
        submitted = []
        for _ in range(target):
            name = make_job_name(self.settings.name)
            try:
                self.backend.submit(name)
            except TransientBackendError as e:
                self.logger.info(f"Submission not accepted; trying next cycle: {e}")
                break
            except PermanentBackendError as e:
                self.logger.error(f"Submission failed: {e}")
                break

            self.logger.info(f"Submitted job '{name}'")
            submitted.append(
                ComputeManagerJobRecord(
                    job_name=name, kind=JobRecordKind.ACTIVE, at=now, submitted_at=now
                )
            )

        return submitted

    def _remove(self, jobs: list[Job]):
        for job in jobs:
            try:
                self.backend.remove(job)
            except BackendError as e:
                self.logger.warning(
                    f"Could not remove job '{job.name}'; retrying next cycle: {e}"
                )

    def clear_error(self):
        """Deprecated: an ERROR registration no longer prevents a compute
        manager from registering."""
        try:
            self.logger.info(f"Clearing '{self.settings.name}' with the ERROR status")
            self.client.clear_error(self.settings.name)
        except AlchemiscaleComputeManagerClientError as e:
            if e.status_code not in (404, 422):
                raise
            self.logger.info(
                f"Could not clear '{self.settings.name}', no such compute manager with the ERROR status"
            )
