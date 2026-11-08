"""Tests for the shared event loop used by clients to run async requests."""

import asyncio
import multiprocessing
import sys
import threading
import time
import warnings

import pytest

from alchemiscale.base.client import (
    AlchemiscaleBaseClient,
    AlchemiscaleConnectionError,
    use_session,
)
import alchemiscale.interface.client
from alchemiscale.interface.client import AlchemiscaleClient


def make_client(cls=AlchemiscaleClient, **kwargs):
    return cls(
        api_url="http://localhost:1",
        identifier="test",
        key="test",
        use_local_cache=False,
        **kwargs,
    )


def _get_cached_result(client):
    return client._run_async(
        client._async_get_protocoldagresult("ref", "tf", "results", False)
    )


def _run_in_child(conn):
    client = make_client()

    async def value():
        return 42

    with warnings.catch_warnings():
        # the cache must not still be bound to the parent's loop
        warnings.simplefilter("error")
        conn.send((client._run_async(value()), _get_cached_result(client)))
    conn.close()


class TestSharedEventLoop:
    def test_concurrent_calls_use_own_session(self):
        """Calls from different threads each keep their own session."""
        client = make_client()
        both_started = threading.Barrier(2)
        errors = []

        @use_session
        async def request(self):
            session = self._session
            await asyncio.get_running_loop().run_in_executor(None, both_started.wait)
            await asyncio.sleep(0.1)
            assert self._session is session
            assert not session.is_closed

        def call():
            try:
                client._run_async(request(client))
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []

    def test_abandoned_call_does_not_close_new_session(self):
        """An interrupted call cleaning up late leaves later calls intact.

        Mimics a ``KeyboardInterrupt`` in Jupyter: the interrupted coroutine is
        still running when the user calls again, and finishes during that call.

        """
        client = make_client()
        client._run_async(asyncio.sleep(0))
        loop = AlchemiscaleBaseClient._shared_event_loop

        abandoned_closed = threading.Event()
        old_sessions = []

        @use_session
        async def abandoned(self):
            old_sessions.append(self._session)
            try:
                await asyncio.sleep(60)
            finally:
                # make the abandoned call finish while the new one is in flight
                loop.call_soon(abandoned_closed.set)

        @use_session
        async def new(self):
            session = self._session
            future.cancel()
            await asyncio.get_running_loop().run_in_executor(
                None, abandoned_closed.wait, 10
            )
            await asyncio.sleep(0.05)
            assert self._session is session
            assert not session.is_closed
            return "ok"

        future = asyncio.run_coroutine_threadsafe(abandoned(client), loop)
        while not old_sessions:
            time.sleep(0.01)

        assert client._run_async(new(client)) == "ok"
        assert old_sessions[0].is_closed

    def test_single_loop_for_all_clients(self):
        """Subclasses share one loop with the base client, whichever is used first."""

        class SubClient(AlchemiscaleClient):
            pass

        loops = []

        async def get_loop():
            return asyncio.get_running_loop()

        for cls in (SubClient, AlchemiscaleClient):
            loops.append(make_client(cls)._run_async(get_loop()))

        assert loops[0] is loops[1]
        assert "_shared_event_loop" not in vars(SubClient)
        assert "_shared_event_loop" not in vars(AlchemiscaleClient)

    @pytest.mark.timeout(30)
    def test_reentrant_call_raises(self):
        """Calling a sync method from the client's own loop raises, not deadlocks."""
        client = make_client()

        async def value():
            return 1

        async def reentrant():
            return client._run_async(value())

        with pytest.raises(RuntimeError, match="from within a coroutine"):
            client._run_async(reentrant())

    def test_retry_backoff_does_not_block_loop(self):
        """A request backing off doesn't stall other requests on the shared loop."""

        class FlakyClient(AlchemiscaleClient):
            attempts = 0

            @AlchemiscaleBaseClient._retry_async
            async def flaky(self):
                type(self).attempts += 1
                if type(self).attempts == 1:
                    raise AlchemiscaleConnectionError("unreachable")
                return "recovered"

        client = make_client(FlakyClient, retry_base_seconds=2.0, retry_max_seconds=2.0)
        result = []
        backing_off = threading.Thread(
            target=lambda: result.append(client._run_async(client.flaky()))
        )
        backing_off.start()
        while FlakyClient.attempts == 0:
            time.sleep(0.01)

        start = time.monotonic()
        client._run_async(asyncio.sleep(0))
        assert time.monotonic() - start < 1.0

        backing_off.join(timeout=10)
        assert result == ["recovered"]

    @pytest.mark.timeout(30)
    @pytest.mark.skipif(sys.platform == "win32", reason="requires fork")
    def test_forked_child_runs_own_loop(self, monkeypatch):
        """A forked child can make calls after the parent has started the loop."""

        async def get_resource(self, resource, compress=False):
            return [resource]

        monkeypatch.setattr(AlchemiscaleClient, "_get_resource_async", get_resource)
        monkeypatch.setattr(
            alchemiscale.interface.client, "pdr_from_bytes", lambda b: b.decode()
        )

        client = make_client()
        expected = "/transformations/tf/results/ref"
        assert _get_cached_result(client) == expected

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe()
        with warnings.catch_warnings():
            # forking a multi-threaded process warns on Python >= 3.12
            warnings.simplefilter("ignore", DeprecationWarning)
            child = ctx.Process(target=_run_in_child, args=(child_conn,))
            child.start()

        try:
            assert parent_conn.poll(10), "child hung waiting on the parent's loop"
            assert parent_conn.recv() == (42, expected)
        finally:
            child.join(timeout=10)
            if child.is_alive():
                child.kill()

        assert child.exitcode == 0
