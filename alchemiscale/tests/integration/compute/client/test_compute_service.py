from typing import List, Optional
import threading
import time
import os

import pytest

from pathlib import Path

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.objectstore import S3ObjectStore
from alchemiscale.compute.service import SynchronousComputeService


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
                shared_basedir=Path("shared").absolute(),
                scratch_basedir=Path("scratch").absolute(),
                heartbeat_interval=1,
                sleep_interval=1,
            )

    def test_heartbeat(self, n4js_preloaded, service):
        n4js: Neo4jStore = n4js_preloaded

        # register service; normally happens on service start, but needed
        # for heartbeats
        service._register()

        # start up heartbeat thread
        heartbeat_thread = threading.Thread(target=service.heartbeat, daemon=True)
        heartbeat_thread.start()

        # give time for a heartbeat
        time.sleep(2)

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}})
        return csreg
        """
        csreg = n4js.graph.run(q).to_subgraph()
        assert csreg["registered"] < csreg["heartbeat"]

        # stop the service; should trigger heartbeat to stop
        service.stop()
        time.sleep(2)
        assert not heartbeat_thread.is_alive()

    def test_claim_tasks(self, n4js_preloaded, service):
        n4js: Neo4jStore = n4js_preloaded

        service._register()

        task_sks: List[Optional[ScopedKey]] = service.claim_tasks(count=2)

        # should have 2 tasks
        assert len(task_sks) == 2

        subgraph = n4js.graph.run(
            f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}}),
              (csreg)-[:CLAIMS]->(t:Task)
        return csreg, t
        """
        ).to_subgraph()

        assert len([node for node in subgraph.nodes if "Task" in node.labels]) == 2

    def test_task_to_protocoldag(
        self, n4js_preloaded, service, network_tyk2, scope_test
    ):
        n4js: Neo4jStore = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskhub(network_sk)

        task_sks = n4js.get_taskhub_tasks(tq_sk)

        protocoldag, transformation, protocoldagresult = service.task_to_protocoldag(
            task_sks[0]
        )

        assert len(protocoldag.protocol_units) == 23

    def test_push_result(self):
        # already tested with with client test for `set_task_result`
        # expand this test when we have result path handling added in
        ...

    def test_execute(
        self, n4js_preloaded, s3os_server_fresh, service, network_tyk2, scope_test
    ):
        n4js: Neo4jStore = n4js_preloaded
        s3os: S3ObjectStore = s3os_server_fresh
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskhub(network_sk)

        task_sks = n4js.get_taskhub_tasks(tq_sk)

        protocoldagresultref_sk = service.execute(task_sks[0])

        # examine object metadata
        protocoldagresultref = n4js.get_gufe(protocoldagresultref_sk)
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())
        assert len(objs) == 1
        assert objs[0].key == os.path.join(s3os.prefix, protocoldagresultref.location)

    def test_cycle(self, n4js_preloaded, s3os_server_fresh, service):
        n4js: Neo4jStore = n4js_preloaded
        s3os: S3ObjectStore = s3os_server_fresh

        service._register()

        q = """
        match (pdr:ProtocolDAGResultRef)
        return pdr
        """

        # preconditions
        protocoldagresultref = n4js.graph.run(q).to_subgraph()
        assert protocoldagresultref is None

        service.cycle()

        # postconditions
        protocoldagresultref = n4js.graph.run(q).to_subgraph()
        assert protocoldagresultref is not None
        assert protocoldagresultref["ok"] == True

        task = n4js.graph.run(
            """
        match (t:Task {status: 'complete'})
        return t
        """
        ).to_subgraph()

        assert task is not None

    def test_cycle_max_tasks(self): ...

    def test_cycle_max_time(self): ...

    def test_start(self, n4js_preloaded, s3os_server_fresh, service):
        n4js: Neo4jStore = n4js_preloaded
        s3os: S3ObjectStore = s3os_server_fresh

        # start up service in a thread; will register itself
        service_thread = threading.Thread(target=service.start, daemon=True)
        service_thread.start()

        # give time for execution
        time.sleep(2)

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}})
        return csreg
        """
        csreg = n4js.graph.run(q).to_subgraph()
        assert csreg["registered"] < csreg["heartbeat"]

        # stop the service
        service.stop()
        while True:
            if service_thread.is_alive():
                time.sleep(1)
            else:
                break

        task = n4js.graph.run(
            """
        match (t:Task {status: 'complete'})
        return t
        """
        ).to_subgraph()

        assert task is not None

        # service should now be deregistered
        csreg = n4js.graph.run(q).to_subgraph()
        assert csreg is None

    def test_stop(self):
        # tested as part of tests above to stop threaded components that
        # otherwise run forever
        ...

    # init kwarg tests

    def test_kwarg_keep_shared(self): ...

    def test_kwarg_keep_scratch(self): ...

    def test_kwarg_scopes(self):
        # TODO: add test here with alternative settings to `service` fixture
        scope = Scope("totally", "different", "scope")
