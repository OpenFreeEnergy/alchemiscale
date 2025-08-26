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

    service_processes = []

    def create_compute_service(self):
        from alchemiscale.models import Scope
        from alchemiscale.compute.service import SynchronousComputeService
        from alchemiscale.compute.settings import ComputeServiceSettings
        import yaml

        config_template = f"""
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

        params = yaml.safe_load(config_template)

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
        LocalTestingComputeManager.service_processes = []

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

        manager.start(max_cycles=1)
