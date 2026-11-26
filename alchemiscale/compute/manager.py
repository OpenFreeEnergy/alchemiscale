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

from packaging.version import InvalidVersion, Version

from ..storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerJobRecord,
    ComputeManagerRegistration,
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
from ..base.client import AlchemiscaleConnectionError
from .client import (
    AlchemiscaleComputeManagerClient,
    AlchemiscaleComputeManagerClientError,
    ComputeManagerNotRegisteredError,
)
from .settings import ComputeManagerSettings, ComputeServiceSettings
from ..sleep import InterruptableSleep, SleepInterrupted

#: The oldest alchemiscale server this compute manager can work with.
MIN_SERVER_VERSION = Version("0.9.0")
# reported by a build without version information, such as a source checkout
# without its git history
UNKNOWN_VERSION = Version("0.0.0")


class IncompatibleServerError(RuntimeError):
    """Raised when the alchemiscale server is too old for this compute manager."""


class AuthenticationFailedError(RuntimeError):
    """Raised when the compute manager can't authenticate with the server."""


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
    #: when the jobs that were signs of health this cycle were submitted
    healthy_submitted_at: list[datetime.datetime] = field(default_factory=list)
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

    def healthy(self, at: datetime.datetime, submitted_at: datetime.datetime):
        self.healthy_submitted_at.append(submitted_at)
        if self.last_healthy_at is None or at > self.last_healthy_at:
            self.last_healthy_at = at


def _later(a: datetime.datetime | None, b: datetime.datetime | None):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _server_side(error: AlchemiscaleComputeManagerClientError) -> bool:
    """Whether a failed request is the server's trouble, to wait out, rather
    than a request the server rejects."""
    return error.status_code is None or error.status_code >= 500


def _next_backoff(reg: ComputeManagerRegistration, settings) -> float:
    if reg.backoff_seconds:
        backoff = reg.backoff_seconds * 2
    else:
        backoff = settings.sleep_interval
    return min(backoff, settings.max_backoff)


class _Cycle:
    """One cycle's decisions: reconciling jobs, applying the failure policy,
    and pausing, probing and resuming.

    Every decision is made from the state read at the start of the cycle and
    the jobs listed during it, so that instances under one name behave like
    one manager. State transitions are sent with the version the state was
    read at, and lost if another instance changed it first.

    """

    def __init__(
        self,
        manager: "ComputeManager",
        state: ComputeManagerState,
        registered: set[str],
        has_work: bool,
        num_tasks: int,
    ):
        self.manager = manager
        self.settings = manager.settings
        self.backend = manager.backend
        self.logger = manager.logger
        self.state = state
        self.reg = state.registration
        self.registered = registered
        self.has_work = has_work
        self.num_tasks = num_tasks
        self.now = _utcnow()

        self.update = ComputeManagerStateUpdate(expected_version=self.reg.version)
        self.transition = self.update.transition
        self.reconciliation = Reconciliation()
        self.in_use: set[str] = set()
        self.jobs: list[Job] | None = None

        # a manager left in ERROR by an instance that exited is resumed
        if self.reg.status == ComputeManagerStatus.DEGRADED:
            self.status, self.detail = ComputeManagerStatus.DEGRADED, self.reg.detail
        else:
            self.status, self.detail = ComputeManagerStatus.OK, None

    @property
    def paused(self) -> bool:
        return self.status == ComputeManagerStatus.DEGRADED

    def run(self):
        list_error = None
        try:
            self.jobs = self.backend.list_jobs()
        except BackendError as e:
            self.logger.warning(f"Could not list jobs; skipping maintenance: {e}")
            list_error = e

        if self.jobs is not None:
            self.reconciliation = self.manager._reconcile(
                self.jobs, self.state, self.registered, self.now
            )
            active = self.reconciliation.active
            if self.reg.outage_started_at is not None:
                self.logger.info("Backend calls are succeeding again")
                self.update.outage = "clear"
        else:
            # without the backend's view, assume recorded jobs are still active
            active = set(self.state.records_of(JobRecordKind.ACTIVE))
            if isinstance(list_error, TransientBackendError):
                self.update.outage = "start"

        self.update.records += self.reconciliation.records
        self.update.forget += self.reconciliation.forget
        self.update.last_healthy_at = self.reconciliation.last_healthy_at
        self.in_use = active | self.registered

        if self.paused:
            self.while_paused()
        else:
            self.while_running(list_error)

    # facts

    @property
    def last_healthy_at(self) -> datetime.datetime | None:
        return _later(self.reg.last_healthy_at, self.reconciliation.last_healthy_at)

    def counted_failures(self) -> list[ComputeManagerJobRecord]:
        """WORKLOAD failures since the last sign of health, stored or found
        this cycle; derived from records, so nothing is lost or counted twice
        when instances race."""
        failures = self.state.records_of(JobRecordKind.FAILURE) | {
            r.job_name: r for r in self.reconciliation.workload_failures
        }
        healthy_at = self.last_healthy_at
        return sorted(
            (
                r
                for r in failures.values()
                if r.failure_kind == FailureKind.WORKLOAD.value
                and (healthy_at is None or (r.ended_at or r.at) > healthy_at)
            ),
            key=lambda r: r.ended_at or r.at,
        )

    def only_infrastructure_failures(self) -> bool:
        """Whether jobs ended this cycle, and all failed for INFRASTRUCTURE
        reasons, as when they keep landing on faulty nodes."""
        ended = [
            r for r in self.reconciliation.records if r.kind == JobRecordKind.HANDLED
        ]
        failures = [
            r
            for r in self.reconciliation.records
            if r.kind == JobRecordKind.FAILURE
            and r.failure_kind == FailureKind.INFRASTRUCTURE.value
        ]
        return bool(failures) and len(failures) == len(ended)

    def outage_duration(self) -> datetime.timedelta | None:
        if self.update.outage != "start":
            return None
        return self.now - (self.reg.outage_started_at or self.now)

    # transitions

    def next_backoff(self) -> float:
        return _next_backoff(self.reg, self.settings)

    def pause(self, cause: str, detail: str):
        backoff = self.next_backoff()
        self.logger.warning(
            f"Pausing: {detail}. Probing again in {backoff:.0f} seconds"
        )
        self.status, self.detail = ComputeManagerStatus.DEGRADED, detail
        self.transition.update(
            pause_cause=cause,
            paused_at=self.now,
            backoff_seconds=backoff,
            backoff_until=self.now + datetime.timedelta(seconds=backoff),
            probe_job=None,
            probation_started_at=None,
        )

    def resume(self, probation: bool):
        self.status, self.detail = ComputeManagerStatus.OK, None
        self.transition.update(pause_cause=None, backoff_until=None, probe_job=None)
        if probation:
            self.event("Probe passed; resuming on probation")
            self.transition.update(probation_started_at=self.now)
        else:
            self.event("A job submitted before the pause succeeded; resuming")
            self.transition.update(
                paused_at=None, backoff_seconds=None, probation_started_at=None
            )

    def probe_failed(self, reason: str):
        backoff = self.next_backoff()
        self.event(
            f"Probe failed: {reason}. Probing again in {backoff:.0f} seconds",
            logging.WARNING,
        )
        self.transition.update(
            probe_job=None,
            backoff_seconds=backoff,
            backoff_until=self.now + datetime.timedelta(seconds=backoff),
        )

    # policy

    def while_running(self, list_error: BackendError | None):
        if isinstance(list_error, PermanentBackendError):
            self.pause("permanent", f"Backend error: {list_error}")
            return

        remove_error = self.manager._remove_error
        if remove_error is not None:
            self.pause("permanent", f"Backend error on removal: {remove_error}")
            return

        outage = self.outage_duration()
        if outage is not None and outage.total_seconds() > (
            self.settings.max_backend_outage
        ):
            self.pause(
                "outage",
                f"Backend calls have failed for {outage.total_seconds():.0f} "
                f"seconds, longer than max_backend_outage="
                f"{self.settings.max_backend_outage}: {list_error}",
            )
            return

        failures = self.counted_failures()
        on_probation = self.reg.probation_started_at is not None
        if on_probation:
            # jobs left over from before the pause can't end probation
            since = self.reg.paused_at or self.reg.probation_started_at
            probation_failures = [
                r for r in failures if r.submitted_at and r.submitted_at >= since
            ]
            if probation_failures:
                last = probation_failures[-1]
                self.pause(
                    "workload",
                    f"Job '{last.job_name}' failed on probation"
                    + (f": {last.reason}" if last.reason else ""),
                )
                return

        # on probation, the failures that caused the pause are still on
        # record until a sign of health; only probation failures count
        limit = self.settings.max_failed_jobs
        if not on_probation and limit is not None and len(failures) > limit:
            last = failures[-1]
            self.pause(
                "workload",
                f"{len(failures)} jobs failed since the last healthy job, more "
                f"than max_failed_jobs={limit}; most recent: '{last.job_name}'"
                + (f": {last.reason}" if last.reason else ""),
            )
            return

        if on_probation and any(
            submitted_at >= since
            for submitted_at in self.reconciliation.healthy_submitted_at
        ):
            self.event("A job succeeded; probation over")
            self.transition.update(
                probation_started_at=None, paused_at=None, backoff_seconds=None
            )

        if self.jobs is None or not self.has_work:
            return
        if self.reconciliation.pending:
            self.logger.info("Jobs are still pending; not submitting")
            return

        target = self.manager._compute_jobs_to_create(self.num_tasks, len(self.in_use))
        if target > 1 and self.only_infrastructure_failures():
            full, target = target, target // 2
            self.logger.warning(
                "Every job that ended this cycle failed for reasons outside its "
                f"control; submitting {target} rather than {full}"
            )
        submitted, error = self.manager._submit(target, self.now)
        self.accept(submitted)
        if error is not None:
            self.pause("permanent", f"Backend error on submission: {error}")

    def while_paused(self):
        reg = self.reg

        # a sign of health from a job still running from before the pause
        # shows the configuration works
        healthy_at = self.last_healthy_at
        if healthy_at and reg.paused_at and healthy_at > reg.paused_at:
            self.resume(probation=False)
            return

        if reg.probe_job is not None:
            verdict = self.probe_verdict(reg.probe_job)
            if verdict is True:
                self.resume(probation=True)
            elif verdict is False:
                # the backoff stays as it is, and the next cycle probes again
                self.event(
                    f"Probe job '{reg.probe_job}' ended without showing whether "
                    "the configuration works; probing again"
                )
                self.transition.update(probe_job=None)
            elif verdict is not None:
                self.probe_failed(verdict)
            else:
                self.logger.info(f"Waiting on probe job '{reg.probe_job}'")
            return

        if reg.backoff_until is not None and self.now < reg.backoff_until:
            self.logger.info(
                f"Paused ({self.detail}); next probe after "
                f"{reg.backoff_until.isoformat()}"
            )
            return

        if reg.pause_cause == "error":
            # the probe is the cycle itself, which got this far
            self.resume(probation=True)
            return

        if reg.pause_cause == "outage":
            # the probe is the cycle's own backend calls
            if self.jobs is not None:
                self.resume(probation=True)
            else:
                self.probe_failed("backend calls still failing")
            return

        if self.jobs is None:
            self.logger.info("Paused; can't list jobs, so not probing")
            return
        # a probe job with nothing to claim proves nothing
        if not self.has_work:
            self.logger.info("Paused; waiting for work to probe with")
            return

        submitted, error = self.manager._submit(1, self.now)
        self.accept(submitted)
        if error is not None:
            self.probe_failed(f"submission failed: {error}")
        elif submitted:
            self.event(f"Probing with job '{submitted[0].job_name}'")
            self.transition.update(probe_job=submitted[0].job_name)

    def probe_verdict(self, name: str) -> bool | str | None:
        """Return ``True`` if the probe job passed, a reason if it failed,
        ``False`` if it ended without a verdict, as when it was preempted or
        disappeared, or ``None`` if there's no verdict yet."""
        failures = self.state.records_of(JobRecordKind.FAILURE) | {
            r.job_name: r
            for r in self.reconciliation.records
            if r.kind == JobRecordKind.FAILURE
        }
        if name in failures:
            if failures[name].failure_kind == FailureKind.INFRASTRUCTURE.value:
                return False
            return "the probe job failed"

        if self.jobs is None:
            return None

        if name in self.registered or name in self.state.records_of(
            JobRecordKind.REGISTERED
        ):
            return True

        listed = {job.name: job for job in self.jobs}
        if name not in listed:
            return False
        if listed[name].state == JobState.SUCCEEDED:
            return "the probe job ended without its compute service registering"

        return None

    def event(self, message: str, level: int = logging.INFO):
        """Log an event in the manager's pause and recovery, and record it in
        the error history with this cycle's transition."""
        self.logger.log(level, message)
        self.update.events.append(message)

    def accept(self, submitted: list[ComputeManagerJobRecord]):
        self.update.records += submitted
        self.in_use |= {record.job_name for record in submitted}


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
        self._cycle_errors = 0
        self._last_state: ComputeManagerState | None = None
        self._remove_error: PermanentBackendError | None = None
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
        treated as too old. A client older than 0.9.0, such as a development
        build before the release, also accepts a server at least as new as
        itself, so that unreleased clients and servers built from the same
        source work together. A server built without version information,
        which reports 0.0.0, is assumed to be compatible.

        """
        # local labels (+g<hash>) differ between builds of the same commit
        client_version = Version(Version(version("alchemiscale")).public)
        required = min(MIN_SERVER_VERSION, client_version)

        try:
            server_version = self.client.get_server_version()
        except InvalidVersion as e:
            raise IncompatibleServerError(
                f"The alchemiscale server at '{self.client.api_url}' reported "
                f"an invalid version: {e}"
            ) from e

        if server_version == UNKNOWN_VERSION:
            self.logger.warning(
                f"The alchemiscale server at '{self.client.api_url}' was built "
                "without version information; assuming it is compatible"
            )
            return

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

        self._adopt(compute_manager_id)

    def _register_again(self):
        # the registration expired, for example after a network partition;
        # join or recreate it, and carry on next cycle
        self.logger.warning(
            f"'{self.compute_manager_id}' is no longer registered with the "
            "server; registering again"
        )
        self._register()

    def _adopt(self, compute_manager_id: ComputeManagerID):
        """Use the ID of the registration this manager belongs to, which
        changes when another instance replaces an expired registration."""
        if compute_manager_id == self.compute_manager_id:
            return
        self.logger.info(f"Using registration '{compute_manager_id}'")
        self.compute_manager_id = compute_manager_id
        self.service_settings.compute_manager_id = compute_manager_id
        self.backend.compute_manager_id = compute_manager_id
        if isinstance(self.logger, logging.LoggerAdapter):
            self.logger.extra["compute_manager_id"] = compute_manager_id

    def _deregister(self):
        self.client.deregister(self.compute_manager_id)

    @contextmanager
    def _running(self, steal=False):
        """Register this compute manager for the lifetime of the context.

        This guarantees that if anything interrupts startup after registration,
        including int_sleep.clear(), the manager is still deregistered.
        """
        registered = False

        try:
            self._when_reachable(lambda: self._register(steal=steal))
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
            self._when_reachable(self._check_server_version)
        except IncompatibleServerError as e:
            self.logger.error(str(e))
            raise
        except SleepInterrupted:
            self.logger.info("Compute manager stopping.")
            return

        try:
            self.backend.validate()
        except BackendError as e:
            self.logger.error(f"Invalid backend configuration: {e}")
            raise

        with self._running(steal=steal):
            try:
                count = 0
                self.logger.info("Starting main loop")

                while self._cycle_through_errors():
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

            except AuthenticationFailedError as e:
                self.logger.error(str(e))
                raise

    def _when_reachable(self, call):
        """Call ``call``, retrying while the server is unreachable or failing."""
        while True:
            try:
                return call()
            except AlchemiscaleConnectionError as e:
                error = e
            except AlchemiscaleComputeManagerClientError as e:
                if not _server_side(e):
                    raise
                error = e
            self.logger.warning(
                f"Could not reach the server; retrying in "
                f"{self.settings.sleep_interval} seconds: {error}"
            )
            self.int_sleep(self.settings.sleep_interval)

    def _cycle_through_errors(self) -> bool:
        """Run a cycle, skipping the rest of it on an error.

        The manager keeps running through server outages, backend errors,
        which pause it, and unexpected exceptions, which pause it once
        ``max_cycle_errors`` cycles in a row raise one. It exits only if
        authentication fails.

        """
        try:
            return self.cycle()
        except AlchemiscaleComputeManagerClientError as e:
            if e.status_code == 401:
                raise AuthenticationFailedError(
                    f"Authentication against '{self.client.api_url}' failed: {e}"
                ) from e
            if not _server_side(e):
                self._unexpected_error(e)
                return True
            self.logger.warning(f"Request to the server failed; skipping cycle: {e}")
        except AlchemiscaleConnectionError as e:
            self.logger.warning(f"Could not reach the server; skipping cycle: {e}")
        except Exception as e:
            self._unexpected_error(e)
        return True

    def _unexpected_error(self, error: Exception):
        """Count a cycle that raised an unexpected exception, such as a bug or
        a request the server rejects, and pause once ``max_cycle_errors``
        cycles in a row have.

        The cycle posted nothing, so the registration is renewed here.

        """
        self._cycle_errors += 1
        self.logger.error(
            f"Unexpected error, {self._cycle_errors} cycle(s) in a row; "
            f"skipping cycle: {error!r}",
            exc_info=error,
        )

        state = self._last_state
        if state is None:
            # nothing read yet; the registration was just renewed by registering
            return

        reg = state.registration
        update = ComputeManagerStateUpdate(expected_version=reg.version)
        status, detail = reg.status, reg.detail
        if (
            self._cycle_errors >= self.settings.max_cycle_errors
            and reg.status == ComputeManagerStatus.OK
        ):
            now = _utcnow()
            backoff = _next_backoff(reg, self.settings)
            status = ComputeManagerStatus.DEGRADED
            detail = (
                f"{self._cycle_errors} cycles in a row failed with an unexpected "
                f"error: {error!r}"
            )
            self.logger.warning(
                f"Pausing: {detail}. Probing again in {backoff:.0f} seconds"
            )
            update.transition = {
                "pause_cause": "error",
                "paused_at": now,
                "backoff_seconds": backoff,
                "backoff_until": now + datetime.timedelta(seconds=backoff),
                "probe_job": None,
                "probation_started_at": None,
            }

        try:
            applied, _ = self.client.update_state(
                self.compute_manager_id,
                status,
                detail=detail,
                saturation=reg.saturation or 0.0,
                state_update=update,
            )
        except Exception as e:
            self.logger.warning(f"Could not report to the server: {e!r}")
            return
        if update.transition and not applied:
            self.logger.info(
                "Another instance changed the shared state; pausing next time "
                "if the errors continue"
            )

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
            self._register_again()
            return True

        if instruction == ComputeManagerInstruction.SHUTDOWN:
            self.logger.info(f'Received shutdown message: "{data["message"]}"')
            return False

        state: ComputeManagerState = data["state"]
        self._adopt(state.registration.to_compute_manager_id())
        self._last_state = state
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

        cycle = _Cycle(
            manager=self,
            state=state,
            registered=registered,
            has_work=instruction == ComputeManagerInstruction.OK and num_tasks > 0,
            num_tasks=num_tasks,
        )
        cycle.run()

        try:
            self._post_state(
                cycle.status,
                cycle.detail,
                self._saturation(len(cycle.in_use)),
                cycle.update,
            )
        except ComputeManagerNotRegisteredError:
            # the outcomes weren't stored, so the jobs stay to be handled again
            self._register_again()
            return True

        # remove jobs only now that their outcomes are stored
        self._remove(cycle.reconciliation.to_remove)

        # only a cycle that got all the way through ends a run of errors
        self._cycle_errors = 0
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
        failed = state.records_of(JobRecordKind.FAILURE)
        recorded_active = state.records_of(JobRecordKind.ACTIVE)
        seen_registered = state.records_of(JobRecordKind.REGISTERED)
        unregistered = state.records_of(JobRecordKind.UNREGISTERED)
        listed = {job.name for job in jobs}

        for job in jobs:
            if job.name in handled:
                if job.state.terminal or job.name in failed:
                    # removing again is harmless, and finishes a removal that
                    # failed or a job cancelled for not registering
                    result.to_remove.append(job)
                    continue
                # a live job missing from one listing, as when the batch
                # system's accounting lags, was taken to have disappeared
                self.logger.info(f"Job '{job.name}' is listed again; still active")
                result.forget.append((job.name, JobRecordKind.HANDLED))

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

        # a handled job's record is needed while the backend may still list
        # it: until it's gone, without a lookback, or for the lookback after
        # it ended
        lookback = self.backend.finished_job_lookback
        for name, record in handled.items():
            if (
                name not in listed
                if lookback is None
                else (record.ended_at or record.at) < now - lookback
            ):
                result.forget.append((name, JobRecordKind.HANDLED))

        return result

    def _handle_terminal(
        self,
        job: Job,
        seen_registered: bool,
        result: Reconciliation,
        now: datetime.datetime,
    ):
        ended_at = job.ended_at or now

        match job.state:
            case JobState.SUCCEEDED if seen_registered:
                self.logger.info(f"Job '{job.name}' succeeded")
                result.healthy(ended_at, job.submitted_at)
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
                reason = f"{after}: {job.reason}" if job.reason else after
                record = ComputeManagerJobRecord(
                    job_name=job.name,
                    kind=JobRecordKind.FAILURE,
                    at=now,
                    submitted_at=job.submitted_at,
                    ended_at=ended_at,
                    failure_kind=kind.value,
                    reason=reason,
                )
                result.records.append(record)
                if kind == FailureKind.WORKLOAD:
                    self.logger.warning(f"Job '{job.name}' failed {reason}")
                    result.workload_failures.append(record)
                else:
                    self.logger.info(
                        f"Job '{job.name}' failed {reason}; the failure is outside "
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
    ) -> tuple[list[ComputeManagerJobRecord], PermanentBackendError | None]:
        """Submit up to ``target`` jobs.

        Returns ACTIVE records for the jobs the backend accepted, and the
        :class:`PermanentBackendError` that stopped submission, if any.

        """
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
                return submitted, e

            self.logger.info(f"Submitted job '{name}'")
            submitted.append(
                ComputeManagerJobRecord(
                    job_name=name, kind=JobRecordKind.ACTIVE, at=now, submitted_at=now
                )
            )

        return submitted, None

    def _remove(self, jobs: list[Job]):
        """Remove jobs, keeping a :class:`PermanentBackendError` for the next
        cycle to pause on."""
        self._remove_error = None
        for job in jobs:
            try:
                self.backend.remove(job)
            except BackendError as e:
                self.logger.warning(
                    f"Could not remove job '{job.name}'; retrying next cycle: {e}"
                )
                if isinstance(e, PermanentBackendError):
                    self._remove_error = e

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
