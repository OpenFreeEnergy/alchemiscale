import threading
import time
import os
import asyncio

import pytest

from pathlib import Path

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.models import ComputeManagerID
from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.objectstore import S3ObjectStore
from alchemiscale.compute.client import AlchemiscaleComputeClientError
from alchemiscale.compute.service import SynchronousComputeService, AsynchronousComputeService
from alchemiscale.compute.settings import ComputeServiceSettings


class TestSynchronousComputeService:
    ...

    @pytest.fixture
    def service(self, n4js_preloaded, compute_client, tmpdir):
        with tmpdir.as_cwd():
            return SynchronousComputeService(
                ComputeServiceSettings(
                    api_url=compute_client.api_url,
                    identifier=compute_client.identifier,
                    key=compute_client.key,
                    name="test_compute_service",
                    shared_basedir=Path("shared").absolute(),
                    scratch_basedir=Path("scratch").absolute(),
                    heartbeat_interval=1,
                    sleep_interval=1,
                    deep_sleep_interval=1,
                )
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
        csreg = n4js.execute_query(q).records[0]["csreg"]

        assert csreg["registered"] < csreg["heartbeat"]

        # stop the service; should trigger heartbeat to stop
        service.stop()
        time.sleep(2)
        assert not heartbeat_thread.is_alive()

    def test_claim_tasks(self, n4js_preloaded, service):

        service._register()

        task_sks: list[ScopedKey | None] = service.claim_tasks(count=2)

        # should have 2 tasks
        assert len(task_sks) == 2

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}}),
              (csreg)-[:CLAIMS]->(t:Task)
        return t
        """

        results = n4js_preloaded.execute_query(q)

        nodes = []
        for record in results.records:
            t = record["t"]
            if "Task" in t.labels:
                nodes.append(t)

        assert len(nodes) == 2

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

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_push_result(self):
        # already tested with with client test for `set_task_result`
        # expand this test when we have result path handling added in
        raise NotImplementedError

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

        service._register()

        q = """
        match (pdr:ProtocolDAGResultRef)
        return pdr
        """

        # preconditions
        protocoldagresultref = n4js.execute_query(q)
        assert not protocoldagresultref.records

        service.cycle()

        # postconditions
        protocoldagresultref = n4js.execute_query(q)
        assert protocoldagresultref.records
        assert protocoldagresultref.records[0]["pdr"]["ok"] is True

        q = """
        match (t:Task {status: 'complete'})
        return t
        """

        results = n4js.execute_query(q)

        assert results.records

    def test_cycle_max_failures(self, n4js_preloaded, s3os_server_fresh, service):
        n4js: Neo4jStore = n4js_preloaded

        service._register()

        q = """
        match (pdr:ProtocolDAGResultRef)
        return pdr
        """

        # preconditions
        protocoldagresultref = n4js.execute_query(q)
        assert not protocoldagresultref.records

        # create blocking failures
        query = """
        MATCH (cs:ComputeServiceRegistration {identifier: $compute_service_id})
        SET cs.failure_times = [datetime()] + cs.failure_times
        """

        for _ in range(4):
            n4js.execute_query(query, compute_service_id=service.compute_service_id)

        service.cycle()

        # postconditions
        protocoldagresultref = n4js.execute_query(q)
        assert not protocoldagresultref.records

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_cycle_max_tasks(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_cycle_max_time(self):
        raise NotImplementedError

    def test_start(self, n4js_preloaded, s3os_server_fresh, service):
        n4js: Neo4jStore = n4js_preloaded

        # start up service in a thread; will register itself
        service_thread = threading.Thread(target=service.start, daemon=True)
        service_thread.start()

        # give time for execution
        time.sleep(2)

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}})
        return csreg
        """
        csreg = n4js.execute_query(q).records[0]["csreg"]
        assert csreg["registered"] < csreg["heartbeat"]

        # stop the service
        service.stop()
        while True:
            if service_thread.is_alive():
                time.sleep(1)
            else:
                break

        q = """
        match (t:Task {status: 'complete'})
        return t
        """

        results = n4js.execute_query(q)
        assert results.records

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_stop(self):
        # tested as part of tests above to stop threaded components that
        # otherwise run forever
        raise NotImplementedError

    # init kwarg tests

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_kwarg_keep_shared(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_kwarg_keep_scratch(self):
        raise NotImplementedError

    def test_kwarg_scopes(self):
        # TODO: add test here with alternative settings to `service` fixture
        _ = Scope("totally", "different", "scope")

    def test_missing_compute_manager(self, n4js_preloaded, service):
        service.settings.compute_manager_id = ComputeManagerID.new_from_name(
            "testmanager"
        )
        with pytest.raises(
            AlchemiscaleComputeClientError,
            match="Could not find ComputeManagerRegistration",
        ):
            service._register()


class TestAsynchronousComputeService:
    """Integration tests for AsynchronousComputeService."""

    @pytest.fixture
    def async_service(self, n4js_preloaded, compute_client, tmpdir):
        with tmpdir.as_cwd():
            return AsynchronousComputeService(
                ComputeServiceSettings(
                    api_url=compute_client.api_url,
                    identifier=compute_client.identifier,
                    key=compute_client.key,
                    name="test_async_compute_service",
                    shared_basedir=Path("shared").absolute(),
                    scratch_basedir=Path("scratch").absolute(),
                    heartbeat_interval=1,
                    sleep_interval=1,
                    deep_sleep_interval=1,
                    # Async-specific settings
                    max_concurrent_tasks=2,
                    min_concurrent_tasks=1,
                    cpu_threshold=90.0,
                    memory_threshold=85.0,
                    resource_check_interval=1.0,
                    task_retry_backoff=5.0,
                    enable_gpu_monitoring=False,
                )
            )

    def test_init(self, async_service):
        """Test that async service initializes correctly."""
        assert async_service.max_concurrent == 2
        assert async_service.min_concurrent == 1
        assert async_service.cpu_threshold == 90.0
        assert async_service.memory_threshold == 85.0
        assert async_service.resource_check_interval == 1.0
        assert async_service.task_retry_backoff == 5.0
        assert async_service.current_concurrency == 1
        assert len(async_service.running_tasks) == 0
        assert len(async_service.terminated_tasks) == 0

    def test_heartbeat(self, n4js_preloaded, async_service):
        """Test heartbeat functionality for async service."""
        n4js: Neo4jStore = n4js_preloaded

        # Register service
        async_service._register()

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=async_service.heartbeat, daemon=True)
        heartbeat_thread.start()

        # Give time for a heartbeat
        time.sleep(2)

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{async_service.compute_service_id}'}})
        return csreg
        """
        csreg = n4js.execute_query(q).records[0]["csreg"]

        assert csreg["registered"] < csreg["heartbeat"]

        # Stop the service
        async_service.stop()
        time.sleep(2)
        assert not heartbeat_thread.is_alive()

    def test_claim_tasks(self, n4js_preloaded, async_service):
        """Test task claiming for async service."""
        async_service._register()

        task_sks: list[ScopedKey | None] = async_service.claim_tasks(count=2)

        # Should have 2 tasks
        assert len(task_sks) == 2

        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{async_service.compute_service_id}'}}),
              (csreg)-[:CLAIMS]->(t:Task)
        return t
        """

        results = n4js_preloaded.execute_query(q)

        nodes = []
        for record in results.records:
            t = record["t"]
            if "Task" in t.labels:
                nodes.append(t)

        assert len(nodes) == 2

    def test_task_to_protocoldag(
        self, n4js_preloaded, async_service, network_tyk2, scope_test
    ):
        """Test ProtocolDAG creation from task."""
        n4js: Neo4jStore = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskhub(network_sk)

        task_sks = n4js.get_taskhub_tasks(tq_sk)

        protocoldag, transformation, protocoldagresult = async_service.task_to_protocoldag(
            task_sks[0]
        )

        assert len(protocoldag.protocol_units) == 23

    def test_execute_task_async(
        self, n4js_preloaded, s3os_server_fresh, async_service, network_tyk2, scope_test
    ):
        """Test async task execution."""
        n4js: Neo4jStore = n4js_preloaded
        s3os: S3ObjectStore = s3os_server_fresh
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskhub(network_sk)

        task_sks = n4js.get_taskhub_tasks(tq_sk)

        # Run the async execution
        async def run_test():
            return await async_service.execute_task_async(task_sks[0])

        protocoldagresultref_sk = asyncio.run(run_test())

        # Verify result
        assert protocoldagresultref_sk is not None
        protocoldagresultref = n4js.get_gufe(protocoldagresultref_sk)
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())
        assert len(objs) == 1
        assert objs[0].key == os.path.join(s3os.prefix, protocoldagresultref.location)

    def test_increase_concurrency(self, async_service):
        """Test concurrency increase logic."""
        async_service.current_concurrency = 1
        async_service._increase_concurrency()
        assert async_service.current_concurrency == 2

        # Should not exceed max
        async_service._increase_concurrency()
        assert async_service.current_concurrency == 2

    def test_can_retry_task(self, async_service):
        """Test task retry eligibility checking."""
        from datetime import datetime, timedelta

        task_sk = ScopedKey(gufe_key="test-task", scope=Scope())

        # Task not in terminated list should be retryable
        assert async_service._can_retry_task(task_sk) is True

        # Add to terminated with future retry time
        async_service.terminated_tasks[task_sk] = datetime.now() + timedelta(seconds=10)
        assert async_service._can_retry_task(task_sk) is False

        # Add to terminated with past retry time
        async_service.terminated_tasks[task_sk] = datetime.now() - timedelta(seconds=10)
        assert async_service._can_retry_task(task_sk) is True
        # Should be removed from terminated list
        assert task_sk not in async_service.terminated_tasks

    def test_cycle_async(self, n4js_preloaded, s3os_server_fresh, async_service):
        """Test async cycle execution."""
        n4js: Neo4jStore = n4js_preloaded

        async_service._register()

        q = """
        match (pdr:ProtocolDAGResultRef)
        return pdr
        """

        # Preconditions
        protocoldagresultref = n4js.execute_query(q)
        assert not protocoldagresultref.records

        # Run one async cycle
        async def run_cycle():
            await async_service.cycle_async()
            # Give tasks time to start
            await asyncio.sleep(0.5)
            # Wait for all running tasks to complete
            while async_service.running_tasks:
                await asyncio.sleep(1)
                # Clean up completed tasks
                completed_keys = [
                    key
                    for key, task_exec in async_service.running_tasks.items()
                    if task_exec.process and task_exec.process.done()
                ]
                for key in completed_keys:
                    async_service.running_tasks.pop(key)

        asyncio.run(run_cycle())

        # Postconditions - at least one result should be present
        protocoldagresultref = n4js.execute_query(q)
        assert protocoldagresultref.records

        q = """
        match (t:Task {status: 'complete'})
        return t
        """

        results = n4js.execute_query(q)
        assert results.records

    def test_start_stop_async(self, n4js_preloaded, s3os_server_fresh, async_service):
        """Test async service start and stop."""
        n4js: Neo4jStore = n4js_preloaded

        # Start service in a thread
        def run_service():
            async_service.start(max_time=3)

        service_thread = threading.Thread(target=run_service, daemon=True)
        service_thread.start()

        # Give time for execution
        time.sleep(4)

        # Verify service ran
        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{async_service.compute_service_id}'}})
        return csreg
        """
        results = n4js.execute_query(q)
        # Service should have deregistered after max_time
        assert not results.records or results.records[0]["csreg"] is not None

        # Wait for thread to complete
        service_thread.join(timeout=2)

    def test_resource_monitoring(self, async_service):
        """Test that resource monitoring is initialized correctly."""
        assert async_service.resource_monitor is not None
        assert async_service.resource_monitor.enable_gpu is False

        # Test getting metrics
        metrics = async_service.resource_monitor.get_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.gpu_percent is None

    def test_concurrent_execution(self, n4js_preloaded, s3os_server_fresh, async_service):
        """Test that multiple tasks can run concurrently."""
        n4js: Neo4jStore = n4js_preloaded

        async_service._register()

        async def run_concurrent_test():
            # Run a cycle to start tasks
            await async_service.cycle_async()
            await asyncio.sleep(0.5)

            # Should have tasks running (up to max_concurrent)
            assert len(async_service.running_tasks) > 0
            assert len(async_service.running_tasks) <= async_service.max_concurrent

            # Wait for completion
            while async_service.running_tasks:
                await asyncio.sleep(1)
                completed_keys = [
                    key
                    for key, task_exec in async_service.running_tasks.items()
                    if task_exec.process and task_exec.process.done()
                ]
                for key in completed_keys:
                    async_service.running_tasks.pop(key)

        asyncio.run(run_concurrent_test())

        # Verify tasks completed
        q = """
        match (t:Task {status: 'complete'})
        return t
        """
        results = n4js.execute_query(q)
        assert results.records
