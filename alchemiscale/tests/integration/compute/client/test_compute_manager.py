from uuid import uuid4
from time import sleep
import sys
import signal

from multiprocessing import Process

import pytest

from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerStatus,
    ComputeManagerInstruction,
)
from alchemiscale.compute.manager import ComputeManager, ComputeManagerSettings
from alchemiscale.compute.client import AlchemiscaleComputeManagerClient


class LocalTestingComputeManager(ComputeManager):

    debug = {"services_started": 0, "service_processes": []}

    def __init__(self, settings: ComputeManagerSettings):
        self.settings = settings
        self.compute_manager_id = ComputeManagerID.new_from_manager_name(
            self.settings.name
        )

        self.service_settings_template = f"""
---
init:
  api_url: {self.settings.api_url}
  identifier: {self.settings.identifier}
  key: {self.settings.key}
  name: testservice
  compute_manager_id: {self.compute_manager_id}
  shared_basedir: "./shared"
  scratch_basedir: "./scratch"
  keep_shared: False
  keep_scratch: False
  n_retries: 1
  sleep_interval: 30
  heartbeat_interval: 300
  scopes:
    - '*-*-*'
  protocols: null
  claim_limit: 2
  loglevel: 'WARN'
  logfile: null
  client_cache_directory: null
  client_cache_size_limit: 1073741824
  client_use_local_cache: false
  client_max_retries: 0
  client_retry_base_seconds: 2.0
  client_retry_max_seconds: 60.0
  client_verify: true

start:
  max_tasks: 2
  max_time: 30
        """

        self.client = AlchemiscaleComputeManagerClient(
            api_url=self.settings.api_url,
            identifier=self.settings.identifier,
            key=self.settings.key,
        )
        self.sleep_interval = settings.sleep_interval

    def start(self):
        self._register()
        self.cycle()
        self._deregister()

    def cycle(self, run_n_cycles=None):

        num_cycles = 0

        try:
            while True:
                num_cycles += 1
                instruction, data = self.client.get_instruction(self.compute_manager_id)
                match instruction:
                    case ComputeManagerInstruction.OK:
                        total_services = len(data["compute_service_ids"])
                        num_tasks = data["num_tasks"]
                        if (
                            total_services < self.settings.max_compute_services
                            and num_tasks > 0
                        ):
                            proc = Process(
                                target=LocalTestingComputeManager.create_compute_service,
                                args=(self.service_settings_template,),
                            )
                            proc.start()
                            sleep(1)
                            self.debug["service_processes"].append(proc)
                            self.debug["services_started"] += 1
                            total_services += 1
                    case ComputeManagerInstruction.SKIP:
                        total_services = len(data["compute_service_ids"])
                    case ComputeManagerInstruction.SHUTDOWN:
                        shutdown_message = data["message"]
                        print(
                            f'Received shutdown message: "{shutdown_message}"',
                            file=sys.stderr,
                        )
                        break
                self.client.update_status(
                    self.compute_manager_id,
                    ComputeManagerStatus.OK,
                    saturation=total_services / self.settings.max_compute_services,
                )
                if run_n_cycles is not None and num_cycles == run_n_cycles:
                    break
                sleep(self.sleep_interval)
        except Exception as e:
            self.client.update_status(
                self.compute_manager_id, ComputeManagerStatus.ERRORED, repr(e)
            )
            raise e

    @staticmethod
    def create_compute_service(config: bytes):
        from alchemiscale.models import Scope
        from alchemiscale.compute.service import SynchronousComputeService
        from alchemiscale.compute.settings import ComputeServiceSettings
        import yaml

        params = yaml.safe_load(config)

        params_init = params.get("init", {})
        params_start = params.get("start", {})

        if "scopes" in params_init:
            params_init["scopes"] = [
                Scope.from_str(scope) for scope in params_init["scopes"]
            ]

        service = SynchronousComputeService(ComputeServiceSettings(**params_init))

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
def manager_settings(compute_manager_client):
    return ComputeManagerSettings(
        api_url=compute_manager_client.api_url,
        identifier=compute_manager_client.identifier,
        key=compute_manager_client.key,
        name="testmanager",
        logfile=None,
        max_compute_services=2,
        sleep_interval=3,
    )


@pytest.fixture()
def manager(
    manager_settings,
    tmpdir,
):
    return LocalTestingComputeManager(manager_settings)


class TestComputeManager:

    def setup_method(self):
        LocalTestingComputeManager.debug["service_processes"] = []

    def teardown_method(self):
        for proc in LocalTestingComputeManager.debug["service_processes"]:
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

        manager._register()

        # the first service will claim 2 of the tasks
        manager.cycle(run_n_cycles=1)
        assert manager.debug["services_started"] == 1
        sleep(1)
        assert get_num_unclaimed_tasks() == 1

        manager.cycle(run_n_cycles=1)
        assert manager.debug["services_started"] == 2
        sleep(1)
        assert get_num_unclaimed_tasks() == 0

        manager._deregister()
