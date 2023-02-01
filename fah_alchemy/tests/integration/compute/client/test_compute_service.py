from typing import List, Optional

import pytest

from pathlib import Path

from fah_alchemy.models import ScopedKey
from fah_alchemy.storage.statestore import Neo4jStore
from fah_alchemy.compute.service import SynchronousComputeService


class TestSynchronousComputeService:
    ...

    @pytest.fixture
    def service(self, n4js_preloaded, compute_client, tmpdir):
        with tmpdir.as_cwd():
            return SynchronousComputeService(
                api_url=compute_client.api_url,
                identifier=compute_client.identifier,
                key=compute_client.key,
                name="test_compute_service",
                shared_path=Path(".").absolute(),
            )

    def test_get_tasks(self, n4js_preloaded, service):
        n4js: Neo4jStore = n4js_preloaded

        task_sks: List[Optional[ScopedKey]] = service.get_tasks(count=2)

        # should have 2 tasks
        assert len(task_sks) == 2

    def test_get_task_transformation(
        self, n4js_preloaded, service, network_tyk2, scope_test
    ):
        n4js: Neo4jStore = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskqueue(network_sk)

        task_sks = n4js.get_taskqueue_tasks(tq_sk)

        protocoldag, transformation, protocoldagresult = service.task_to_protocoldag(
            task_sks[0]
        )

        assert len(protocoldag.protocol_units) == 23

    def test_execute(
        self, n4js_preloaded, s3os_server_fresh, service, network_tyk2, scope_test
    ):

        n4js: Neo4jStore = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskqueue(network_sk)

        task_sks = n4js.get_taskqueue_tasks(tq_sk)

        protocoldagresultref_sk = service.execute(task_sks[0])

        # TODO: check that we can pull the result
