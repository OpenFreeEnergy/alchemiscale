import logging
import threading
import time
import os

import pytest

from pathlib import Path

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.models import ComputeManagerID
from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.objectstore import S3ObjectStore
from alchemiscale.compute.client import AlchemiscaleComputeClientError
from alchemiscale.compute.service import SynchronousComputeService
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

    def test_heartbeat_survives_beat_exception(self, service, monkeypatch, caplog):
        """A failing beat() must not kill the heartbeat thread.

        Without per-beat exception handling, a sustained outage or any
        exhausted-retry path in client.heartbeat would silently kill the
        thread while the main loop kept claiming tasks --- letting the API
        expire the service registration and orphan the claimed tasks.
        """
        caplog.set_level(logging.INFO, logger=service.logger.name)

        beats: list[bool] = []

        def flaky_beat():
            beats.append(True)
            if len(beats) == 1:
                raise RuntimeError("simulated transient heartbeat failure")

        monkeypatch.setattr(service, "beat", flaky_beat)

        heartbeat_thread = threading.Thread(target=service.heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            # heartbeat_interval=1 from the ``service`` fixture, so this wait
            # comfortably covers 3 beats: the first raises, the next two succeed
            time.sleep(3.5)

            assert heartbeat_thread.is_alive(), (
                "heartbeat thread died after the first failed beat; "
                "beat exception was not swallowed"
            )
            assert len(beats) >= 2, (
                f"only {len(beats)} beat(s) attempted; thread did not resume "
                "after the failure"
            )
            assert "Heartbeat failed" in caplog.text
        finally:
            service.stop()
            heartbeat_thread.join(timeout=5)

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

    def test_service_interruptible_sleep(
        self, n4js_preloaded, compute_client, tmpdir, caplog, monkeypatch
    ):
        n4js: Neo4jStore = n4js_preloaded

        # a service whose sleep interval is long enough that, were the sleep
        # *not* interruptible, stop() would block until it elapsed and this
        # test would time out joining the thread
        with tmpdir.as_cwd():
            service = SynchronousComputeService(
                ComputeServiceSettings(
                    api_url=compute_client.api_url,
                    identifier=compute_client.identifier,
                    key=compute_client.key,
                    name="test_compute_service_interruptible",
                    shared_basedir=Path("shared").absolute(),
                    scratch_basedir=Path("scratch").absolute(),
                    heartbeat_interval=300,
                    sleep_interval=300,
                    deep_sleep_interval=300,
                )
            )

        # force the "no tasks claimed; sleeping" branch every cycle so the
        # service parks itself in an interruptible sleep rather than executing
        monkeypatch.setattr(service, "claim_tasks", lambda count=1: [None])

        caplog.set_level(logging.INFO, logger=service.logger.name)

        service_thread = threading.Thread(target=service.start, daemon=True)
        service_thread.start()

        try:
            # wait until the service has actually entered its sleep
            deadline = time.monotonic() + 30
            while "No tasks claimed; sleeping" not in caplog.text:
                assert time.monotonic() < deadline, "service never reached its sleep"
                time.sleep(0.05)

            # interrupting the sleep should let start() return promptly rather
            # than blocking for the full sleep_interval
            interrupt_time = time.monotonic()
            service.stop()
            service_thread.join(timeout=30)

            assert not service_thread.is_alive()
            assert (time.monotonic() - interrupt_time) < 30
            assert "Service stopping." in caplog.text
        finally:
            service.stop()
            service_thread.join(timeout=30)

        # the service should have deregistered itself on the way out
        q = f"""
        match (csreg:ComputeServiceRegistration {{identifier: '{service.compute_service_id}'}})
        return csreg
        """
        assert not n4js.execute_query(q).records

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

    def test_start_non_stop_exit_stops_heartbeat_thread(
        self, compute_client, tmpdir, monkeypatch
    ):
        """start() should stop its heartbeat thread even when it exits without stop().

        This reproduces the max_tasks/max_time-style exit path: the main loop raises
        KeyboardInterrupt, start() catches it and returns, but service.stop() was
        never called by the test.
        """
        heartbeat_started = threading.Event()
        deregistered = threading.Event()

        with tmpdir.as_cwd():
            service = SynchronousComputeService(
                ComputeServiceSettings(
                    api_url=compute_client.api_url,
                    identifier=compute_client.identifier,
                    key=compute_client.key,
                    name="test_compute_service_heartbeat_lifecycle",
                    shared_basedir=Path("shared").absolute(),
                    scratch_basedir=Path("scratch").absolute(),
                    heartbeat_interval=300,
                    sleep_interval=300,
                    deep_sleep_interval=300,
                )
            )

        monkeypatch.setattr(service, "_register", lambda: None)
        monkeypatch.setattr(service, "_deregister", lambda: deregistered.set())

        # Ensure the heartbeat thread has really started and is about to park in
        # service.int_sleep(heartbeat_interval).
        monkeypatch.setattr(service, "beat", lambda: heartbeat_started.set())

        def raise_keyboard_interrupt_after_heartbeat(*args, **kwargs):
            assert heartbeat_started.wait(timeout=5), "heartbeat thread never started"
            raise KeyboardInterrupt

        # Simulate the same non-stop exit class as _check_max_tasks/_check_max_time.
        monkeypatch.setattr(service, "cycle", raise_keyboard_interrupt_after_heartbeat)

        service_thread = threading.Thread(target=service.start, daemon=True)
        service_thread.start()

        try:
            service_thread.join(timeout=5)

            assert not service_thread.is_alive()
            assert deregistered.is_set()

            # This is the assertion that should fail on the current code:
            # start() returned, but the heartbeat thread is still alive because
            # nobody called service.stop() / int_sleep.interrupt().
            service.heartbeat_thread.join(timeout=1)
            assert not service.heartbeat_thread.is_alive()
        finally:
            # Prevent the failing test from leaking a daemon heartbeat thread.
            service.stop()
            if hasattr(service, "heartbeat_thread"):
                service.heartbeat_thread.join(timeout=5)
