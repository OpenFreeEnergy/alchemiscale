import datetime
from uuid import uuid4
import logging
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

    def create_compute_services(self, data):
        if exception := LocalTestingComputeManager.exception:
            raise exception

        params_start = {
            "max_time": self.service_max_time or 10,
            "max_tasks": self.service_max_tasks or 2,
        }

        proc = Process(
            target=LocalTestingComputeManager._create_compute_service,
            args=(self.service_settings, params_start),
        )
        proc.start()
        LocalTestingComputeManager.service_processes.append(proc)
        time.sleep(2)
        return 1

    @staticmethod
    def _create_compute_service(service_settings, params_start):

        from alchemiscale.compute.service import SynchronousComputeService

        service = SynchronousComputeService(service_settings)

        # add signal handling
        for signame in {"SIGHUP", "SIGINT", "SIGTERM"}:

            def stop(*args, **kwargs):
                service.stop()
                raise KeyboardInterrupt()

            signal.signal(getattr(signal, signame), stop)

        try:
            service.start(**params_start)
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
def service_settings(compute_identity, single_scoped_credentialed_compute):
    return ComputeServiceSettings(
        name="testservice",
        api_url="http://127.0.0.1:8000/",
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
