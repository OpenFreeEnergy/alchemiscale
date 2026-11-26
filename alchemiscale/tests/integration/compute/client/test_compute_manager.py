import datetime
from uuid import uuid4
import logging
import threading
import time
import sys
import signal

from multiprocessing import Process
from unittest.mock import MagicMock

from neo4j.time import DateTime

import pytest

from alchemiscale.models import Scope

from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerStatus,
    ComputeManagerInstruction,
)
from alchemiscale.compute.backend import ComputeBackend, Job, JobState
from alchemiscale.compute.manager import ComputeManager, ComputeManagerSettings
from alchemiscale.compute.settings import ComputeManagerSettings, ComputeServiceSettings
from alchemiscale.compute.client import AlchemiscaleComputeManagerClient


class LocalProcessBackend(ComputeBackend):
    """Runs each job as a local process running a compute service."""

    service_processes = {}
    exception = None
    service_max_time = None
    service_max_tasks = None

    def __init__(self, service_settings: ComputeServiceSettings):
        self.service_settings = service_settings
        self.submitted_at = {}

    def validate(self):
        pass

    def list_jobs(self):
        jobs = []
        for name, proc in LocalProcessBackend.service_processes.items():
            submitted_at = self.submitted_at[name]
            if proc.is_alive():
                state, ended_at = JobState.RUNNING, None
            else:
                state = JobState.SUCCEEDED if proc.exitcode == 0 else JobState.FAILED
                ended_at = datetime.datetime.now(tz=datetime.UTC)
            jobs.append(
                Job(
                    name=name,
                    state=state,
                    submitted_at=submitted_at,
                    started_at=submitted_at,
                    ended_at=ended_at,
                )
            )
        return jobs

    def submit(self, name):
        if exception := LocalProcessBackend.exception:
            raise exception

        # service lifetime limits live on the ComputeServiceSettings each
        # job's compute service runs with
        settings = self.service_settings.model_copy(
            update={
                "name": name,
                "compute_manager_id": self.compute_manager_id,
                "max_time": self.service_max_time or 10,
                "max_tasks": self.service_max_tasks or 2,
            }
        )
        proc = Process(
            target=LocalProcessBackend._create_compute_service, args=(settings,)
        )
        proc.start()
        LocalProcessBackend.service_processes[name] = proc
        self.submitted_at[name] = datetime.datetime.now(tz=datetime.UTC)
        # give the service time to register and claim tasks
        time.sleep(2)

    def remove(self, job):
        proc = LocalProcessBackend.service_processes.get(job.name)
        if proc is not None and proc.is_alive():
            proc.terminate()

    @staticmethod
    def _create_compute_service(service_settings):

        from alchemiscale.compute.service import SynchronousComputeService

        service = SynchronousComputeService(service_settings)

        # add signal handling
        for signame in {"SIGHUP", "SIGINT", "SIGTERM"}:

            def stop(*args, **kwargs):
                service.stop()
                raise KeyboardInterrupt()

            signal.signal(getattr(signal, signame), stop)

        try:
            service.start()
        except KeyboardInterrupt:
            pass


def spy_deregister(manager: ComputeManager) -> MagicMock:
    """Record calls to the manager client's ``deregister``.

    Registrations are shared and expire rather than being deleted, so
    whether the manager deregistered can't be seen in the state store.
    """
    deregister = MagicMock(wraps=manager.client.deregister)
    manager.client.deregister = deregister
    return deregister


@pytest.fixture()
def manager_settings():
    return ComputeManagerSettings(
        name="testmanager",
        logfile=None,
        max_compute_services=2,
        sleep_interval=3,
    )


@pytest.fixture()
def service_settings(
    compute_identity, single_scoped_credentialed_compute, compute_api_port
):
    return ComputeServiceSettings(
        name="testservice",
        api_url=f"http://127.0.0.1:{compute_api_port}/",
        identifier=single_scoped_credentialed_compute.identifier,
        key=compute_identity["key"],
        shared_basedir="./shared",
        scratch_basedir="./scratch",
        claim_limit=2,
    )


@pytest.fixture()
def manager(
    manager_settings,
    service_settings,
    tmpdir,
    uvicorn_server,
):
    return ComputeManager(
        manager_settings, service_settings, LocalProcessBackend(service_settings)
    )


class TestComputeManager:

    def setup_method(self):
        LocalProcessBackend.service_processes = {}
        LocalProcessBackend.exception = None
        LocalProcessBackend.service_max_time = None
        LocalProcessBackend.service_max_tasks = None

    def teardown_method(self):
        for proc in LocalProcessBackend.service_processes.values():
            proc.terminate()

    def test_manager_implementation(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        network_tyk2,
        scope_test,
    ):
        network_sk, taskhub_sk, _ = n4js_preloaded.assemble_network(
            network_tyk2, scope_test
        )
        get_num_unclaimed_tasks = lambda: len(
            n4js_preloaded.get_taskhub_unclaimed_tasks(taskhub_sk)
        )

        assert get_num_unclaimed_tasks() == 3

        # first cycle should pick up 2 tasks with one compute service
        manager.start(max_cycles=1)
        assert get_num_unclaimed_tasks() == 1
        assert len(LocalProcessBackend.service_processes) == 1

        # second cycle should pick up last task with an additional compute service
        manager.start(max_cycles=1)
        assert get_num_unclaimed_tasks() == 0
        assert len(LocalProcessBackend.service_processes) == 2

        # an additional cycle should not create another service
        manager.start(max_cycles=1)
        assert get_num_unclaimed_tasks() == 0
        assert len(LocalProcessBackend.service_processes) == 2

    def test_runtime_failure(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        network_tyk2,
        scope_test,
    ):
        # an unexpected exception skips the cycle; enough in a row pause the
        # manager, which keeps running
        LocalProcessBackend.exception = RuntimeError("Unexpected failure in manager")
        manager.settings.max_cycle_errors = 1

        manager.start(max_cycles=1)

        query = """MATCH (cmr:ComputeManagerRegistration) RETURN cmr"""
        record = n4js_preloaded.execute_query(query).records[0]["cmr"]
        assert record["status"] == ComputeManagerStatus.DEGRADED
        assert record["pause_cause"] == "error"
        assert "RuntimeError('Unexpected failure in manager')" in record["detail"]

    def test_clear_error(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        network_tyk2,
        scope_test,
    ):
        # an ERROR registration, as a 0.8 manager leaves behind
        manager._register()
        manager.client.update_status(
            manager.compute_manager_id,
            ComputeManagerStatus.ERROR,
            detail="RuntimeError('Unexpected failure in manager')",
        )

        query = """
        MATCH (cmr:ComputeManagerRegistration {status: "ERROR"})
        RETURN cmr
        """

        assert n4js_preloaded.execute_query(query).records
        with pytest.warns(DeprecationWarning):
            manager.clear_error()
        assert not n4js_preloaded.execute_query(query).records

    def test_manager_keyboard_interrupt(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        network_tyk2,
        scope_test,
        caplog,
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        LocalProcessBackend.exception = KeyboardInterrupt
        deregister = spy_deregister(manager)

        manager.start(max_cycles=1)
        deregister.assert_called_once_with(manager.compute_manager_id)

        assert "Caught SIGINT/Keyboard interrupt" in caplog.text

    def test_skip_instruction(self, n4js_preloaded, manager: ComputeManager, caplog):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        # ensure long lived service
        LocalProcessBackend.service_max_tasks = 10
        manager._register()
        manager.cycle()

        failure_times = [datetime.datetime.now(tz=datetime.UTC)] * 10

        query = """
        MATCH (csr:ComputeServiceRegistration)
        SET csr.failure_times = $failure_times
        RETURN csr
        """
        n4js_preloaded.execute_query(query, failure_times=failure_times)
        manager.cycle()

        assert "Received skip instruction" in caplog.text
        manager._deregister()

    def test_not_registered(self, n4js_preloaded, manager: ComputeManager, caplog):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        manager._register()

        # the registration is gone, as after it expired and was swept
        n4js_preloaded.execute_query(
            "MATCH (cmr:ComputeManagerRegistration) DETACH DELETE cmr"
        )

        # the manager registers again, and carries on
        assert manager.cycle() is True
        assert "registering again" in caplog.text
        state = n4js_preloaded.get_compute_manager_state(manager.compute_manager_id)
        assert state.registration.uuid == manager.compute_manager_id.uuid
        assert manager.service_settings.compute_manager_id == manager.compute_manager_id
        assert manager.cycle() is True

    def test_manager_interruptible_sleep(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        caplog,
        monkeypatch,
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)

        # suppress the subprocess spawn. LocalProcessBackend.submit
        # forks a multiprocessing.Process; combined with the background
        # thread below, that makes this test fork from a multi-threaded
        # parent. POSIX fork only carries the calling thread, so locks held
        # by other threads (logging, requests pools, ...) are inherited as
        # held-with-no-owner in the child, which then deadlocks. That hung
        # the xdist worker on 3.11/3.13 (3.12 happens to miss it). The test
        # is about stop() interrupting the sleep, not about spawning, so
        # we cut the spawn path here. See PR #503 discussion.
        monkeypatch.setattr(manager.backend, "submit", lambda name: None)

        # use a long sleep interval; if the sleep were *not* interruptible,
        # stop() would not take effect until this elapsed and this test would
        # time out waiting on the thread to join
        manager.settings.sleep_interval = 300
        deregister = spy_deregister(manager)

        thread = threading.Thread(target=manager.start)
        thread.start()

        try:
            # wait until the manager has entered its (interruptible) sleep
            deadline = time.monotonic() + 30
            while "Sleeping for" not in caplog.text:
                assert time.monotonic() < deadline, "manager never reached its sleep"
                time.sleep(0.05)

            # interrupting the sleep should let start() return promptly rather
            # than blocking for the full sleep_interval
            interrupt_time = time.monotonic()
            manager.stop()
            thread.join(timeout=30)

            assert not thread.is_alive()
            assert (time.monotonic() - interrupt_time) < 30
            assert "Compute manager stopping." in caplog.text
        finally:
            manager.stop()
            thread.join(timeout=30)

        # the manager should have deregistered itself on the way out
        deregister.assert_called_once_with(manager.compute_manager_id)

    def test_manager_start_deregisters_if_interrupted_during_startup_setup(
        self,
        n4js_preloaded,
        manager: ComputeManager,
        monkeypatch,
    ):
        """start() should deregister even if interrupted after registration.

        This simulates a signal/KeyboardInterrupt landing after the manager has
        registered, but before start() reaches the try/finally that normally
        performs deregistration.
        """
        interrupted = False

        def interrupt_after_registration():
            nonlocal interrupted
            interrupted = True
            raise KeyboardInterrupt

        monkeypatch.setattr(manager.int_sleep, "clear", interrupt_after_registration)
        deregister = spy_deregister(manager)

        try:
            manager.start(max_cycles=1)
        except KeyboardInterrupt:
            pass

        assert interrupted

        # This should fail without the fix: the manager registered, then the
        # simulated interrupt skipped the finally block, so the manager never
        # deregistered.
        deregister.assert_called_once_with(manager.compute_manager_id)
