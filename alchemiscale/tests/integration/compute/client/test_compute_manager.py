import datetime
from uuid import uuid4
import logging
import threading
import time
import sys
import signal

from multiprocessing import Process

from neo4j.time import DateTime

import pytest

from alchemiscale.models import Scope

from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerStatus,
    ComputeManagerInstruction,
)
from alchemiscale.compute.manager import ComputeManager, ComputeManagerSettings
from alchemiscale.compute.settings import ComputeManagerSettings, ComputeServiceSettings
from alchemiscale.compute.client import AlchemiscaleComputeManagerClient


class LocalTestingComputeManager(ComputeManager):

    service_processes = []
    exception = None
    service_max_time = None
    service_max_tasks = None

    def create_compute_services(self, data, target):
        if exception := LocalTestingComputeManager.exception:
            raise exception

        # service lifetime limits now live on the ComputeServiceSettings the
        # manager hands to each service, rather than being passed to start()
        self.service_settings.max_time = self.service_max_time or 10
        self.service_settings.max_tasks = self.service_max_tasks or 2

        # Honor ``target`` rather than always creating a single service.
        # ``_compute_jobs_to_create`` computes
        #     min(num_tasks, max_submit_per_cycle, remaining_capacity) // claim_limit
        # with a floor of 1 when scale-up is warranted. With the default
        # ``max_submit_per_cycle=1`` (and max_compute_services=2,
        # claim_limit=2), a 3-task fixture starting from zero services
        # yields target=1 on cycle 1 (one service claims 2 of 3 tasks)
        # and target=1 on cycle 2 (a second service claims the last
        # task), matching the test's expected ramp-up.
        for _ in range(target):
            proc = Process(
                target=LocalTestingComputeManager._create_compute_service,
                args=(self.service_settings,),
            )
            proc.start()
            LocalTestingComputeManager.service_processes.append(proc)
        time.sleep(2)
        return target

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
    return LocalTestingComputeManager(manager_settings, service_settings)


class TestComputeManager:

    def setup_method(self):
        LocalTestingComputeManager.service_processes = []
        LocalTestingComputeManager.exception = None
        LocalTestingComputeManager.service_max_time = None
        LocalTestingComputeManager.service_max_tasks = None

    def teardown_method(self):
        for proc in LocalTestingComputeManager.service_processes:
            proc.terminate()

    def test_manager_implementation(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
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
        assert len(LocalTestingComputeManager.service_processes) == 1

        # second cycle should pick up last task with an additional compute service
        manager.start(max_cycles=1)
        assert get_num_unclaimed_tasks() == 0
        assert len(LocalTestingComputeManager.service_processes) == 2

        # an additional cycle should not create another service
        manager.start(max_cycles=1)
        assert get_num_unclaimed_tasks() == 0
        assert len(LocalTestingComputeManager.service_processes) == 2

    def test_runtime_failure(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
        network_tyk2,
        scope_test,
    ):
        LocalTestingComputeManager.exception = RuntimeError(
            "Unexpected failure in manager"
        )

        with pytest.raises(RuntimeError):
            manager.start(max_cycles=1)

        query = """MATCH (cmr:ComputeManagerRegistration) RETURN cmr"""
        record = n4js_preloaded.execute_query(query).records[0]["cmr"]
        assert record["status"] == ComputeManagerStatus.ERROR
        assert record["detail"] == "RuntimeError('Unexpected failure in manager')"

    def test_clear_error(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
        network_tyk2,
        scope_test,
    ):
        LocalTestingComputeManager.exception = RuntimeError(
            "Unexpected failure in manager"
        )

        with pytest.raises(RuntimeError):
            manager.start(max_cycles=1)

        query = """
        MATCH (cmr:ComputeManagerRegistration {status: "ERROR"})
        RETURN cmr
        """

        assert n4js_preloaded.execute_query(query).records
        manager.clear_error()
        assert not n4js_preloaded.execute_query(query).records

    def test_manager_keyboard_interrupt(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
        network_tyk2,
        scope_test,
        caplog,
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        LocalTestingComputeManager.exception = KeyboardInterrupt

        manager.start(max_cycles=1)
        query = """MATCH (cmr:ComputeManagerRegistration) RETURN cmr"""
        assert not n4js_preloaded.execute_query(query).records

        assert "Caught SIGINT/Keyboard interrupt" in caplog.text

    def test_skip_instruction(
        self, n4js_preloaded, manager: LocalTestingComputeManager, caplog
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        # ensure long lived service
        LocalTestingComputeManager.service_max_tasks = 10
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

    def test_shutdown_instruction(
        self, n4js_preloaded, manager: LocalTestingComputeManager, caplog
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)
        manager._register()

        new_uuid = str(uuid4())

        query = """MATCH (cmr:ComputeManagerRegistration)
        SET cmr.uuid = $new_uuid
        """

        n4js_preloaded.execute_query(query, new_uuid=new_uuid)
        manager.cycle()

        assert "Received shutdown message" in caplog.text

    def test_manager_interruptible_sleep(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
        caplog,
        monkeypatch,
    ):
        caplog.set_level(logging.INFO, logger=manager.logger.name)

        # suppress the subprocess spawn. The default create_compute_services
        # forks a multiprocessing.Process; combined with the background
        # thread below, that makes this test fork from a multi-threaded
        # parent. POSIX fork only carries the calling thread, so locks held
        # by other threads (logging, requests pools, ...) are inherited as
        # held-with-no-owner in the child, which then deadlocks. That hung
        # the xdist worker on 3.11/3.13 (3.12 happens to miss it). The test
        # is about stop() interrupting the sleep, not about spawning, so
        # we cut the spawn path here. See PR #503 discussion.
        # signature is (data, target) since #502 moved autoscaling sizing
        # into ComputeManager; the no-op still returns 0 (no services created)
        monkeypatch.setattr(manager, "create_compute_services", lambda data, target: 0)

        # use a long sleep interval; if the sleep were *not* interruptible,
        # stop() would not take effect until this elapsed and this test would
        # time out waiting on the thread to join
        manager.settings.sleep_interval = 300

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
        query = """MATCH (cmr:ComputeManagerRegistration) RETURN cmr"""
        assert not n4js_preloaded.execute_query(query).records

    def test_manager_start_deregisters_if_interrupted_during_startup_setup(
        self,
        n4js_preloaded,
        manager: LocalTestingComputeManager,
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

        query = """MATCH (cmr:ComputeManagerRegistration) RETURN cmr"""

        try:
            try:
                manager.start(max_cycles=1)
            except KeyboardInterrupt:
                pass

            assert interrupted

            # This should fail without the fix: the manager registered, then the
            # simulated interrupt skipped the finally block, leaving an orphaned
            # ComputeManagerRegistration.
            assert not n4js_preloaded.execute_query(query).records

        finally:
            # Keep the failing test from poisoning later tests.
            if n4js_preloaded.execute_query(query).records:
                manager._deregister()
