"""ComputeManager's reconcile cycle, run on a fake backend against a real
state store."""

import contextlib
import datetime
from datetime import timedelta
import logging
import threading

import pytest

from alchemiscale.compute.backend import (
    FailureKind,
    JobState,
    PermanentBackendError,
    TransientBackendError,
)
from alchemiscale.compute.testing import FakeComputeBackend
from alchemiscale.storage.models import ComputeManagerInstruction, JobRecordKind

from .manager_utils import (
    deregister_services,
    get_state,
    make_manager,
    register_service,
)


class BarrierBackend(FakeComputeBackend):
    """Holds each ``list_jobs`` until all parties have listed."""

    def __init__(self, parties, **kwargs):
        super().__init__(**kwargs)
        self.barrier = threading.Barrier(parties)

    def list_jobs(self):
        jobs = super().list_jobs()
        if self.barrier is not None:
            self.barrier.wait(timeout=30)
        return jobs


def run_concurrently(functions):
    errors = []

    def run(f):
        try:
            f()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(f,)) for f in functions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    if errors:
        raise errors[0]


def ago(**kwargs):
    return datetime.datetime.now(tz=datetime.UTC) - timedelta(**kwargs)


@pytest.fixture
def backend():
    return FakeComputeBackend()


@pytest.fixture
def manager(n4js_fresh, backend):
    return make_manager(n4js_fresh, backend)


def run_job(n4js, backend, name):
    """Start a job and register its compute service."""
    backend.start(name)
    register_service(n4js, name)


def finish_job(n4js, backend, name, failed=False):
    deregister_services(n4js, name)
    if failed:
        backend.fail(name)
    else:
        backend.succeed(name)


class TestSubmission:
    def test_submits_up_to_capacity(self, n4js_fresh, manager, backend):
        manager.cycle()
        assert len(backend.names(JobState.PENDING)) == 2

        # nothing is submitted while jobs are pending
        manager.cycle()
        assert len(backend.jobs) == 2

        for name in backend.names():
            run_job(n4js_fresh, backend, name)
        manager.cycle()
        assert len(backend.jobs) == 4

        # at capacity: a running job and its registered service count once
        for name in backend.names(JobState.PENDING):
            run_job(n4js_fresh, backend, name)
        manager.cycle()
        assert len(backend.jobs) == 4
        assert get_state(manager).registration.saturation == 1

    def test_no_submission_without_work(self, manager, backend):
        manager.client.num_tasks = 0
        manager.cycle()
        assert not backend.jobs

        manager.client.num_tasks = 10
        manager.client.instruction = ComputeManagerInstruction.SKIP
        manager.cycle()
        assert not backend.jobs

    def test_job_names(self, manager, backend):
        manager.cycle()
        for name in backend.names():
            prefix, uuid_hex = name.rsplit(".", 1)
            assert prefix == "testmanager" and len(uuid_hex) == 32
        assert set(get_state(manager).records_of(JobRecordKind.ACTIVE)) == set(
            backend.names()
        )

    def test_full_queue(self, manager, backend, caplog):
        caplog.set_level(logging.INFO)
        backend.submit_error = TransientBackendError("QOSMaxSubmitJobPerUserLimit")
        assert manager.cycle()
        assert not backend.jobs
        assert "trying next cycle" in caplog.text

        backend.submit_error = None
        manager.cycle()
        assert len(backend.jobs) == 2


class TestMaintenance:
    """Maintenance runs every cycle, whatever the instruction."""

    @pytest.mark.parametrize("condition", ["no tasks", "skip", "no capacity"])
    def test_runs_every_cycle(self, n4js_fresh, manager, backend, condition):
        manager.cycle()
        names = backend.names()
        for name in names:
            run_job(n4js_fresh, backend, name)
        manager.cycle()

        match condition:
            case "no tasks":
                manager.client.num_tasks = 0
            case "skip":
                manager.client.instruction = ComputeManagerInstruction.SKIP
            case "no capacity":
                manager.settings.max_compute_services = 1

        finish_job(n4js_fresh, backend, names[0])
        finish_job(n4js_fresh, backend, names[1], failed=True)
        manager.cycle()

        state = get_state(manager)
        assert set(names) <= set(state.records_of(JobRecordKind.HANDLED))
        assert set(state.records_of(JobRecordKind.FAILURE)) == {names[1]}
        assert state.registration.last_healthy_at is not None
        # finished jobs were removed
        assert set(names) <= set(backend.removed)

    def test_terminal_job_handled_once(self, n4js_fresh, n4js, caplog):
        # a backend like SLURM keeps reporting finished jobs
        caplog.set_level(logging.INFO)
        backend = FakeComputeBackend(finished_job_lookback=timedelta(days=7))
        manager = make_manager(n4js_fresh, backend, max_submit_per_cycle=1)
        manager.client.num_tasks = 0

        manager.client.num_tasks = 10
        manager.cycle()
        manager.client.num_tasks = 0
        (ok,) = backend.names()
        run_job(n4js_fresh, backend, ok)
        manager.cycle()
        finish_job(n4js_fresh, backend, ok)

        manager.client.num_tasks = 10
        manager.cycle()
        manager.client.num_tasks = 0
        (failed,) = set(backend.names()) - {ok}
        run_job(n4js_fresh, backend, failed)
        manager.cycle()
        finish_job(n4js_fresh, backend, failed, failed=True)

        for _ in range(4):
            manager.cycle()

        # both are still listed, and each was handled once
        assert {ok, failed} <= set(backend.names())
        assert caplog.text.count(f"Job '{ok}' succeeded") == 1
        assert caplog.text.count(f"Job '{failed}' failed") == 1

        state = get_state(manager)
        assert list(state.records_of(JobRecordKind.FAILURE)) == [failed]
        healthy_at = state.registration.last_healthy_at
        assert healthy_at == backend.jobs[ok].ended_at

    def test_handled_records_pruned(self, n4js_fresh):
        backend = FakeComputeBackend(finished_job_lookback=timedelta(hours=1))
        manager = make_manager(n4js_fresh, backend, max_submit_per_cycle=1)
        manager.cycle()
        (name,) = backend.names()
        run_job(n4js_fresh, backend, name)
        manager.cycle()
        finish_job(n4js_fresh, backend, name)
        manager.client.num_tasks = 0
        manager.cycle()
        assert name in get_state(manager).records_of(JobRecordKind.HANDLED)

        # once the job ended longer ago than the lookback, the backend no
        # longer reports it, and its record is dropped
        n4js_fresh.execute_query(
            """
            MATCH (r:ComputeManagerJobRecord {kind: "handled"})
            SET r.ended_at = datetime($ended_at)
            """,
            ended_at=ago(hours=2).isoformat(),
        )
        backend.vanish(name)
        manager.cycle()
        assert not get_state(manager).records_of(JobRecordKind.HANDLED)

    def test_removed_job_record_dropped(self, n4js_fresh, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        run_job(n4js_fresh, backend, name)
        manager.cycle()
        finish_job(n4js_fresh, backend, name)
        manager.client.num_tasks = 0
        manager.cycle()
        assert name not in backend.jobs
        assert name in get_state(manager).records_of(JobRecordKind.HANDLED)

        manager.cycle()
        assert name not in get_state(manager).records_of(JobRecordKind.HANDLED)

    def test_vanished_job(self, n4js_fresh, manager, backend, caplog):
        manager.cycle()
        name = backend.names()[0]
        run_job(n4js_fresh, backend, name)
        manager.client.num_tasks = 0
        manager.cycle()
        before = get_state(manager)

        deregister_services(n4js_fresh, name)
        backend.vanish(name)
        manager.cycle()
        manager.cycle()

        after = get_state(manager)
        assert caplog.text.count(f"Job '{name}' disappeared") == 1
        assert not after.records_of(JobRecordKind.FAILURE)
        assert after.registration.last_healthy_at == before.registration.last_healthy_at

    def test_infrastructure_failure(self, n4js_fresh, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        backend.fail(name, FailureKind.INFRASTRUCTURE, "preempted")
        manager.cycle()
        (record,) = get_state(manager).records_of(JobRecordKind.FAILURE).values()
        assert record.failure_kind == "INFRASTRUCTURE"
        assert record.reason == "before its compute service registered: preempted"

    def test_failure_after_registering(self, n4js_fresh, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        run_job(n4js_fresh, backend, name)
        manager.cycle()
        backend.fail(name, reason="exit code 1")
        manager.cycle()
        (record,) = get_state(manager).records_of(JobRecordKind.FAILURE).values()
        assert record.reason == "after its compute service registered: exit code 1"

    def test_succeeded_without_registering(self, n4js_fresh, manager, backend, caplog):
        manager.cycle()
        name = backend.names()[0]
        backend.succeed(name)
        manager.cycle()
        assert get_state(manager).registration.last_healthy_at is None
        assert "never seen registered" in caplog.text

    def test_remove_retried(self, n4js_fresh, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        backend.fail(name)
        backend.remove_error = TransientBackendError("API timeout")
        manager.cycle()
        assert name in backend.jobs

        backend.remove_error = None
        manager.cycle()
        assert name not in backend.jobs
        assert len(get_state(manager).records_of(JobRecordKind.FAILURE)) == 1

    def test_list_jobs_fails(self, manager, backend, caplog):
        backend.list_error = TransientBackendError("slurmctld restarting")
        assert manager.cycle()
        assert not backend.jobs
        assert "skipping maintenance" in caplog.text


class TestRegistrationCheck:
    def test_pending_jobs_not_checked(self, manager, backend):
        manager.cycle()
        backend.jobs[backend.names()[0]].submitted_at = ago(hours=5)
        manager.cycle()
        assert not get_state(manager).records_of(JobRecordKind.UNREGISTERED)

    def test_grace_counted_from_start(self, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        # queued for hours, but only just started
        backend.jobs[name].submitted_at = ago(hours=5)
        backend.start(name)
        manager.cycle()
        assert not get_state(manager).records_of(JobRecordKind.UNREGISTERED)

    def test_flagged_after_two_checks(self, n4js_fresh, manager, backend):
        manager.settings.registration_grace_period = 60
        manager.cycle()
        name = backend.names()[0]
        backend.start(name, started_at=ago(minutes=5))

        # first check: recorded, checked again next cycle
        manager.cycle()
        state = get_state(manager)
        assert name in state.records_of(JobRecordKind.UNREGISTERED)
        assert name in backend.jobs

        # a second check within the grace period of the first does nothing
        manager.cycle()
        assert name in backend.jobs

        # once the checks span the grace period, the job is failed and removed
        n4js_fresh.execute_query(
            """
            MATCH (r:ComputeManagerJobRecord {job_name: $name, kind: "unregistered"})
            SET r.at = datetime($at)
            """,
            name=name,
            at=ago(minutes=2).isoformat(),
        )
        manager.client.num_tasks = 0
        manager.cycle()
        state = get_state(manager)
        assert name not in backend.jobs
        (record,) = state.records_of(JobRecordKind.FAILURE).values()
        assert record.failure_kind == "WORKLOAD"
        assert "without its compute service registering" in record.reason
        assert name in state.records_of(JobRecordKind.HANDLED)

    def test_registering_clears_check(self, n4js_fresh, manager, backend):
        manager.cycle()
        name = backend.names()[0]
        backend.start(name, started_at=ago(minutes=5))
        manager.cycle()
        assert name in get_state(manager).records_of(JobRecordKind.UNREGISTERED)

        register_service(n4js_fresh, name)
        manager.cycle()
        state = get_state(manager)
        assert name not in state.records_of(JobRecordKind.UNREGISTERED)
        assert name in state.records_of(JobRecordKind.REGISTERED)

    def test_end_of_life_not_flagged(self, n4js_fresh, manager, backend):
        # a job whose service registered and has deregistered, but which the
        # batch system still reports running, is ending, not failing
        manager.cycle()
        name = backend.names()[0]
        backend.start(name, started_at=ago(hours=1))
        register_service(n4js_fresh, name)
        manager.cycle()
        deregister_services(n4js_fresh, name)
        for _ in range(3):
            manager.cycle()
        assert not get_state(manager).records_of(JobRecordKind.UNREGISTERED)
        assert name in backend.jobs


class TestConcurrentInstances:
    def test_shared_capacity(self, n4js_fresh, backend):
        managers = [make_manager(n4js_fresh, backend) for _ in range(2)]
        assert managers[0].compute_manager_id == managers[1].compute_manager_id

        max_services = managers[0].settings.max_compute_services
        per_cycle = managers[0].settings.max_submit_per_cycle

        for _ in range(5):
            for manager in managers:
                manager.cycle()
                for name in backend.names(JobState.PENDING):
                    run_job(n4js_fresh, backend, name)
                # never more than one cycle's submissions over the limit
                assert len(backend.jobs) <= max_services + per_cycle

        assert len(backend.jobs) == max_services

    def test_simultaneous_reads(self, n4js_fresh):
        # both instances read the state and list jobs before either submits:
        # the overshoot is bounded by one cycle's submissions per extra
        # instance, and corrected as surplus services find no work
        backend = BarrierBackend(parties=2)
        managers = [make_manager(n4js_fresh, backend) for _ in range(2)]
        for manager in managers:
            manager.settings.max_compute_services = 2

        run_concurrently([manager.cycle for manager in managers])
        assert len(backend.jobs) == 4

        # once they see each other's jobs, neither submits more
        backend.barrier = None
        for name in backend.names():
            run_job(n4js_fresh, backend, name)
        for manager in managers:
            manager.cycle()
        assert len(backend.jobs) == 4

    def test_terminal_job_handled_once_across_instances(
        self, n4js_fresh, backend, caplog
    ):
        caplog.set_level(logging.INFO)
        backend.finished_job_lookback = timedelta(days=7)
        managers = [make_manager(n4js_fresh, backend) for _ in range(2)]
        managers[0].cycle()
        name = backend.names()[0]
        run_job(n4js_fresh, backend, name)
        managers[1].cycle()
        finish_job(n4js_fresh, backend, name, failed=True)

        for _ in range(3):
            for manager in managers:
                manager.client.num_tasks = 0
                manager.cycle()

        assert caplog.text.count(f"Job '{name}' failed") == 1
        assert len(get_state(managers[0]).records_of(JobRecordKind.FAILURE)) == 1


def test_validate_failure_exits(n4js_fresh):
    backend = FakeComputeBackend(
        validate_error=PermanentBackendError("partition 'gpu' doesn't exist")
    )
    manager = make_manager(n4js_fresh, backend)
    with pytest.raises(PermanentBackendError):
        manager.start(max_cycles=1)


def status_of(manager):
    return get_state(manager).registration


def hold_backoff(n4js):
    """Push the stored backoff an hour out, so no probe is due during a test."""
    n4js.execute_query(
        """
        MATCH (cmr:ComputeManagerRegistration)
        SET cmr.backoff_until = datetime($t)
        """,
        t=(datetime.datetime.now(tz=datetime.UTC) + timedelta(hours=1)).isoformat(),
    )


def expire_backoff(n4js, manager):
    """End the backoff, as an operator's probe request does."""
    n4js.request_compute_manager_probe(manager.settings.name)


def fail_jobs(n4js, manager, backend, count, kind=FailureKind.WORKLOAD):
    """Have the manager submit ``count`` jobs, which register and then fail."""
    names = []
    with one_submission_per_cycle(manager):
        while len(names) < count:
            name = next_job(manager, backend)
            run_job(n4js, backend, name)
            deregister_services(n4js, name)
            backend.fail(name, kind)
            names.append(name)
    return names


def succeed_job(n4js, manager, backend):
    with one_submission_per_cycle(manager):
        name = next_job(manager, backend)
        run_job(n4js, backend, name)
        manager.cycle()
        finish_job(n4js, backend, name)
    return name


def next_job(manager, backend):
    """Cycle, and return the job submitted, or one left pending by earlier
    cycles, which blocks submission."""
    before = set(backend.names())
    manager.cycle()
    new = set(backend.names()) - before
    if not new:
        new = set(backend.names(JobState.PENDING))
    assert new, "the manager submitted nothing"
    return sorted(new)[0]


@contextlib.contextmanager
def one_submission_per_cycle(manager):
    per_cycle = manager.settings.max_submit_per_cycle
    manager.settings.max_submit_per_cycle = 1
    try:
        yield
    finally:
        manager.settings.max_submit_per_cycle = per_cycle


class TestPausing:
    @pytest.fixture
    def manager(self, n4js_fresh, backend):
        return make_manager(n4js_fresh, backend, max_failed_jobs=2)

    def test_pauses_after_max_failed_jobs(self, n4js_fresh, manager, backend):
        fail_jobs(n4js_fresh, manager, backend, 2)
        manager.cycle()
        assert status_of(manager).status == "OK"

        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert reg.pause_cause == "workload"
        assert "3 jobs failed since the last healthy job" in reg.detail
        assert reg.backoff_seconds == manager.settings.sleep_interval

        # paused: no new jobs, but maintenance and status updates continue
        submitted = set(backend.names())
        for _ in range(3):
            manager.cycle()
        assert set(backend.names()) <= submitted
        assert status_of(manager).last_status_update > reg.last_status_update

        # the error is in the history
        errors = n4js_fresh.get_compute_manager_errors(manager.settings.name)
        assert errors[0].status == "DEGRADED"

    def test_max_failed_jobs_zero_and_none(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"

    def test_never_pauses_with_none(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=None)
        fail_jobs(n4js_fresh, manager, backend, 6)
        manager.cycle()
        assert status_of(manager).status == "OK"

    def test_infrastructure_failures_never_count(self, n4js_fresh, manager, backend):
        fail_jobs(n4js_fresh, manager, backend, 4, FailureKind.INFRASTRUCTURE)
        manager.cycle()
        assert status_of(manager).status == "OK"

    def test_sign_of_health_resets_count(self, n4js_fresh, manager, backend):
        fail_jobs(n4js_fresh, manager, backend, 2)
        succeed_job(n4js_fresh, manager, backend)
        fail_jobs(n4js_fresh, manager, backend, 2)
        manager.cycle()
        assert status_of(manager).status == "OK"

    def test_job_reported_many_cycles(self, n4js_fresh):
        # on SLURM, finished jobs keep being listed: one failed job counts
        # once, and one completed job resets the count once
        backend = FakeComputeBackend(finished_job_lookback=timedelta(days=7))
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=1)

        succeed_job(n4js_fresh, manager, backend)
        fail_jobs(n4js_fresh, manager, backend, 1)
        for _ in range(4):
            manager.cycle()
        assert status_of(manager).status == "OK"

        # the old completed job doesn't keep resetting the count
        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"

    def test_registered_service_is_not_health(self, n4js_fresh, manager, backend):
        # jobs whose services register and then fail (e.g. OOMKilled) still
        # count, however many register
        fail_jobs(n4js_fresh, manager, backend, 3)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"


class TestRecovery:
    @pytest.fixture
    def manager(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"
        return manager

    def test_no_probe_before_backoff(self, n4js_fresh, manager, backend):
        hold_backoff(n4js_fresh)
        jobs = set(backend.names())
        manager.cycle()
        assert set(backend.names()) == jobs

    def test_probe_waits_for_work(self, n4js_fresh, manager, backend):
        expire_backoff(n4js_fresh, manager)
        jobs = set(backend.names())

        manager.client.num_tasks = 0
        manager.cycle()
        manager.client.num_tasks = 10
        manager.client.instruction = ComputeManagerInstruction.SKIP
        manager.cycle()
        assert set(backend.names()) == jobs
        assert status_of(manager).probe_job is None

        manager.client.instruction = None
        manager.cycle()
        (probe,) = set(backend.names()) - jobs
        assert status_of(manager).probe_job == probe

    def test_probe_passes_then_probation(self, n4js_fresh, manager, backend):
        expire_backoff(n4js_fresh, manager)
        jobs = set(backend.names())
        manager.cycle()
        (probe,) = set(backend.names()) - jobs

        # one job only, ignoring the computed target
        manager.cycle()
        assert set(backend.names()) - jobs == {probe}

        run_job(n4js_fresh, backend, probe)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.probation_started_at is not None
        assert reg.probe_job is None

        # normal sizing resumes
        manager.cycle()
        assert len(set(backend.names()) - jobs) > 1

        # a sign of health ends probation, and resets the backoff
        deregister_services(n4js_fresh, probe)
        backend.succeed(probe)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.probation_started_at is None
        assert reg.backoff_seconds is None

    def test_failure_on_probation(self, n4js_fresh, manager, backend):
        backoff = status_of(manager).backoff_seconds
        expire_backoff(n4js_fresh, manager)
        jobs = set(backend.names())
        manager.cycle()
        (probe,) = set(backend.names()) - jobs
        run_job(n4js_fresh, backend, probe)
        manager.cycle()
        assert status_of(manager).probation_started_at is not None

        # the probe registered, then failed (e.g. OOMKilled)
        deregister_services(n4js_fresh, probe)
        backend.fail(probe)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert "failed on probation" in reg.detail
        assert reg.backoff_seconds == 2 * backoff

    def test_leftover_job_does_not_end_probation(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
        # a job submitted before the pause, still running
        manager.client.num_tasks = 1
        manager.cycle()
        (leftover,) = backend.names()
        run_job(n4js_fresh, backend, leftover)
        manager.client.num_tasks = 10

        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"

        expire_backoff(n4js_fresh, manager)
        jobs = set(backend.names())
        manager.cycle()
        (probe,) = set(backend.names()) - jobs
        run_job(n4js_fresh, backend, probe)
        manager.cycle()
        assert status_of(manager).probation_started_at is not None

        deregister_services(n4js_fresh, leftover)
        backend.fail(leftover)
        manager.cycle()
        assert status_of(manager).status == "OK"

    def test_probe_fails(self, n4js_fresh, manager, backend):
        backoff = status_of(manager).backoff_seconds
        expire_backoff(n4js_fresh, manager)
        jobs = set(backend.names())
        manager.cycle()
        (probe,) = set(backend.names()) - jobs

        backend.fail(probe)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert reg.probe_job is None
        assert reg.backoff_seconds == 2 * backoff

    def test_backoff_capped(self, n4js_fresh, manager, backend):
        manager.settings.max_backoff = 2
        for _ in range(3):
            expire_backoff(n4js_fresh, manager)
            jobs = set(backend.names())
            manager.cycle()
            (probe,) = set(backend.names()) - jobs
            backend.fail(probe)
            manager.cycle()
        assert status_of(manager).backoff_seconds == 2

    def test_health_from_leftover_job_ends_pause(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
        manager.client.num_tasks = 1
        manager.cycle()
        (leftover,) = backend.names()
        run_job(n4js_fresh, backend, leftover)
        manager.client.num_tasks = 10
        fail_jobs(n4js_fresh, manager, backend, 1)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"

        finish_job(n4js_fresh, backend, leftover)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.probation_started_at is None
        assert reg.backoff_seconds is None

    def test_restart_keeps_pause(self, n4js_fresh, manager, backend):
        # the pause lives on the registration, not in the manager
        restarted = make_manager(n4js_fresh, backend, max_failed_jobs=0)
        assert restarted.compute_manager_id == manager.compute_manager_id
        jobs = set(backend.names())
        hold_backoff(n4js_fresh)
        restarted.cycle()
        assert set(backend.names()) == jobs
        assert status_of(restarted).status == "DEGRADED"


class TestBackendErrors:
    def test_permanent_error_on_submit(self, n4js_fresh, manager, backend, caplog):
        backend.submit_error = PermanentBackendError("invalid partition 'gpu'")
        assert manager.cycle()
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert reg.pause_cause == "permanent"
        assert "invalid partition" in reg.detail

        # the probe is a submission; while it fails, the manager stays paused
        expire_backoff(n4js_fresh, manager)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"
        assert status_of(manager).backoff_seconds == 2 * reg.backoff_seconds

        # once fixed, the probe is accepted, and passes once it registers
        backend.submit_error = None
        expire_backoff(n4js_fresh, manager)
        manager.cycle()
        (probe,) = backend.names()
        run_job(n4js_fresh, backend, probe)
        manager.cycle()
        assert status_of(manager).status == "OK"

    def test_permanent_error_on_list(self, manager, backend):
        backend.list_error = PermanentBackendError("forbidden: cannot list jobs")
        assert manager.cycle()
        assert status_of(manager).pause_cause == "permanent"

    def test_runtime_permanent_error_does_not_exit(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend)
        backend.submit_error = PermanentBackendError("missing RBAC rule")
        manager.start(max_cycles=3)
        assert status_of(manager).status == "DEGRADED"

    def test_outage(self, n4js_fresh, manager, backend):
        manager.settings.max_backend_outage = 60
        backend.list_error = TransientBackendError("slurmctld not responding")
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.outage_started_at is not None

        # the outage is measured from its stored start
        n4js_fresh.execute_query(
            """
            MATCH (cmr:ComputeManagerRegistration)
            SET cmr.outage_started_at = datetime($t)
            """,
            t=ago(minutes=5).isoformat(),
        )
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert reg.pause_cause == "outage"

        # the probe is the cycle's own backend calls; no job is sent
        expire_backoff(n4js_fresh, manager)
        manager.cycle()
        assert status_of(manager).status == "DEGRADED"

        backend.list_error = None
        expire_backoff(n4js_fresh, manager)
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.outage_started_at is None
        assert not backend.jobs

    def test_brief_outage_cleared(self, n4js_fresh, manager, backend):
        backend.list_error = TransientBackendError("timeout")
        manager.cycle()
        backend.list_error = None
        manager.cycle()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.outage_started_at is None


class TestConcurrentPausing:
    def test_no_submission_while_other_paused(self, n4js_fresh, backend):
        a, b = (make_manager(n4js_fresh, backend, max_failed_jobs=0) for _ in "ab")
        fail_jobs(n4js_fresh, a, backend, 1)
        # a long backoff, so that b doesn't probe during the test
        a.settings.sleep_interval = 3600
        a.cycle()
        assert status_of(a).status == "DEGRADED"

        jobs = set(backend.names())
        for _ in range(3):
            b.cycle()
        assert set(backend.names()) == jobs
        assert status_of(b).status == "DEGRADED"

    def test_lost_compare_and_set(self, n4js_fresh):
        # both instances find failures and decide to pause from the same
        # version; one transition is rejected, and no failure is lost or
        # counted twice
        backend = BarrierBackend(parties=2)
        a, b = (make_manager(n4js_fresh, backend, max_failed_jobs=1) for _ in "ab")
        backend.barrier = None
        failed = fail_jobs(n4js_fresh, a, backend, 2)

        backend.barrier = threading.Barrier(2)
        run_concurrently([a.cycle, b.cycle])
        backend.barrier = None

        state = get_state(a)
        assert set(state.records_of(JobRecordKind.FAILURE)) == set(failed)
        assert state.registration.status == "DEGRADED"
        # one transition applied
        assert state.registration.backoff_seconds == a.settings.sleep_interval
        posted = [
            update for client in (a.client, b.client) for _, update in client.posted
        ]
        assert sum(1 for u in posted if u.transition.get("pause_cause")) == 2

    def test_counts_failures_from_both(self, n4js_fresh, backend):
        # failures recorded by different instances add up
        a, b = (make_manager(n4js_fresh, backend, max_failed_jobs=1) for _ in "ab")
        fail_jobs(n4js_fresh, a, backend, 1)
        a.cycle()
        fail_jobs(n4js_fresh, b, backend, 1)
        b.cycle()
        assert status_of(a).status == "DEGRADED"


def test_operator_probe_request(n4js_fresh, backend):
    from alchemiscale.storage.statestore import ComputeManagerStateError

    manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
    with pytest.raises(ComputeManagerStateError):
        n4js_fresh.request_compute_manager_probe("testmanager")

    fail_jobs(n4js_fresh, manager, backend, 1)
    manager.settings.sleep_interval = 3600
    manager.cycle()
    reg = status_of(manager)
    assert reg.backoff_until > ago(seconds=0)

    n4js_fresh.request_compute_manager_probe("testmanager")
    jobs = set(backend.names())
    manager.cycle()
    assert len(set(backend.names()) - jobs) == 1


class TestUnexpectedErrors:
    def test_pause_and_resume(self, n4js_fresh, backend):
        manager = make_manager(n4js_fresh, backend, max_cycle_errors=2)
        backend.list_error = RuntimeError("bug in the backend")
        manager.start(max_cycles=3)
        reg = status_of(manager)
        assert reg.status == "DEGRADED"
        assert reg.pause_cause == "error"
        assert "RuntimeError('bug in the backend')" in reg.detail
        # start() marks the manager stopped on returning
        manager._stop = False

        # while the error persists, the manager stays paused
        expire_backoff(n4js_fresh, manager)
        assert manager._cycle_through_errors()
        assert status_of(manager).status == "DEGRADED"

        # once a cycle completes after the backoff, it resumes on probation
        backend.list_error = None
        manager._cycle_through_errors()
        reg = status_of(manager)
        assert reg.status == "OK"
        assert reg.probation_started_at is not None


class TestInfrastructureFailures:
    def test_halves_submission(self, n4js_fresh, backend, caplog):
        manager = make_manager(
            n4js_fresh, backend, max_submit_per_cycle=4, max_compute_services=8
        )
        manager.cycle()
        assert len(backend.names()) == 4
        for name in backend.names():
            backend.fail(name, FailureKind.INFRASTRUCTURE, "node has a faulty GPU")
        manager.cycle()
        assert len(backend.names(JobState.PENDING)) == 2
        assert "submitting 2 rather than 4" in caplog.text
        assert status_of(manager).status == "OK"

    def test_not_halved_with_other_outcomes(self, n4js_fresh, backend):
        manager = make_manager(
            n4js_fresh, backend, max_submit_per_cycle=4, max_compute_services=8
        )
        manager.cycle()
        first, *rest = backend.names()
        backend.fail(first, FailureKind.INFRASTRUCTURE)
        for name in rest:
            backend.fail(name, FailureKind.WORKLOAD)
        manager.settings.max_failed_jobs = None
        manager.cycle()
        assert len(backend.names(JobState.PENDING)) == 4


def test_events_recorded(n4js_fresh, backend):
    manager = make_manager(n4js_fresh, backend, max_failed_jobs=0)
    fail_jobs(n4js_fresh, manager, backend, 1)
    manager.cycle()
    expire_backoff(n4js_fresh, manager)
    jobs = set(backend.names())
    manager.cycle()
    (probe,) = set(backend.names()) - jobs
    run_job(n4js_fresh, backend, probe)
    manager.cycle()

    history = [
        (e.status, e.detail)
        for e in n4js_fresh.get_compute_manager_errors(manager.settings.name)
    ]
    assert history[0] == ("OK", "Probe passed; resuming on probation")
    assert history[1] == ("DEGRADED", f"Probing with job '{probe}'")
    assert history[2][0] == "DEGRADED"
    assert "failed since the last healthy job" in history[2][1]
