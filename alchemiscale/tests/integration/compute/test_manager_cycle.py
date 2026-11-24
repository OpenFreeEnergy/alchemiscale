"""ComputeManager's reconcile cycle, run on a fake backend against a real
state store."""

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
