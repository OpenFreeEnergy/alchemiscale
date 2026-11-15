"""Tests for the shared event loop used by clients to run async requests.

Clients run their async requests on a single event loop, running in a
background thread and shared by all clients (see
``AlchemiscaleBaseClient._run_async``). Each test here guards against a
specific way this can go wrong.

"""

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

# how long the retry test's request backs off before retrying
BACKOFF_SECONDS = 2.0

# how long a quick request may take while another is backing off; well under
# BACKOFF_SECONDS, so a backoff that blocks the loop is caught
MAX_STALL_SECONDS = 1.0

# upper bound on waiting for anything in these tests, so a regression fails
# instead of hanging
TIMEOUT_SECONDS = 10


def make_client(cls=AlchemiscaleClient, **kwargs):
    """A client for an API that is never contacted."""
    return cls(
        api_url="http://localhost:1",
        identifier="test",
        key="test",
        use_local_cache=False,
        **kwargs,
    )


async def wait_without_blocking_loop(waitable):
    """Wait on a ``threading.Event`` or ``threading.Barrier`` from a coroutine.

    Waits in a worker thread, so the event loop stays free to run other
    coroutines in the meantime.

    """
    await asyncio.get_running_loop().run_in_executor(
        None, waitable.wait, TIMEOUT_SECONDS
    )


class TestSharedEventLoop:
    def test_concurrent_calls_use_own_session(self):
        """Calls from different threads each keep their own session.

        Regression: the session was stored on the client, so a second call
        replaced the first call's session, and whichever call finished first
        closed the session the other was still using.

        """
        client = make_client()
        both_started = threading.Barrier(2)
        seen = []

        @use_session
        async def request(self):
            session_at_start = self._session
            # ensure both calls are in flight at the same time
            await wait_without_blocking_loop(both_started)
            await asyncio.sleep(0.1)
            return session_at_start, self._session, self._session.is_closed

        threads = [
            threading.Thread(
                target=lambda: seen.append(client._run_async(request(client)))
            )
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=TIMEOUT_SECONDS)

        assert len(seen) == 2
        for session_at_start, session_at_end, closed in seen:
            assert session_at_end is session_at_start
            assert not closed

        # and each call had a session of its own
        assert seen[0][0] is not seen[1][0]

    def test_abandoned_call_does_not_close_new_session(self):
        """An interrupted call cleaning up late leaves later calls intact.

        Regression: in Jupyter, pressing Ctrl-C and re-running a cell leaves
        the interrupted call still running on the loop. When it finished, it
        closed the *new* call's session, failing that call.

        """
        client = make_client()
        client._run_async(asyncio.sleep(0))  # start the shared loop
        loop = AlchemiscaleBaseClient._shared_event_loop

        abandoned_started = threading.Event()
        abandoned_cleaned_up = threading.Event()
        sessions = {}

        @use_session
        async def abandoned_call(self):
            sessions["abandoned"] = self._session
            abandoned_started.set()
            try:
                await asyncio.sleep(60)
            finally:
                # signal only once ``use_session`` has also closed this call's
                # session, which happens just after this ``finally``
                loop.call_soon(abandoned_cleaned_up.set)

        @use_session
        async def new_call(self):
            sessions["new"] = self._session
            abandoned.cancel()  # step 2
            await wait_without_blocking_loop(abandoned_cleaned_up)  # step 3
            await asyncio.sleep(0.05)  # leave room for any stray cleanup
            return self._session, self._session.is_closed

        # 1. start a call, then abandon it mid-flight (as Ctrl-C would)
        abandoned = asyncio.run_coroutine_threadsafe(abandoned_call(client), loop)
        assert abandoned_started.wait(TIMEOUT_SECONDS)

        # 2. start a new call, which cancels the abandoned one
        # 3. while the new call is in flight, the abandoned call cleans up
        session_at_end, closed_at_end = client._run_async(new_call(client))

        # 4. the new call kept its session open throughout; the abandoned
        # call's session was closed by its own cleanup
        assert session_at_end is sessions["new"]
        assert not closed_at_end
        assert sessions["abandoned"].is_closed

    def test_single_loop_for_all_clients(self):
        """Subclasses share one loop with the base client, whichever is used first.

        Regression: each client class got its own loop, but subclasses share
        their parent's ``alru_cache``, which is bound to one loop; switching
        loops cleared the cache, or raised on older ``async-lru`` versions.

        """

        class SubClient(AlchemiscaleClient):
            pass

        async def get_loop():
            return asyncio.get_running_loop()

        # use the subclass first, so it would get its own loop if it could
        subclass_loop = make_client(SubClient)._run_async(get_loop())
        base_loop = make_client(AlchemiscaleClient)._run_async(get_loop())

        assert subclass_loop is base_loop

    @pytest.mark.timeout(30)
    def test_reentrant_call_raises(self):
        """Calling a sync method from the client's own loop raises, not deadlocks.

        A coroutine on the loop that waits for ``_run_async`` would wait for
        itself forever; this must fail loudly instead.

        """
        client = make_client()

        async def value():
            return 1

        async def calls_run_async_from_loop():
            return client._run_async(value())

        with pytest.raises(RuntimeError, match="from within a coroutine"):
            client._run_async(calls_run_async_from_loop())

    def test_retry_backoff_does_not_block_loop(self):
        """A request backing off doesn't stall other requests on the shared loop.

        Regression: async retries slept with ``time.sleep``, which froze the
        shared loop, and with it every other in-flight request, for the whole
        backoff.

        """

        class FlakyClient(AlchemiscaleClient):
            """Fails its first request, then succeeds after one backoff."""

            @AlchemiscaleBaseClient._retry_async
            async def flaky_request(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise AlchemiscaleConnectionError("unreachable")
                return "recovered"

        client = make_client(
            FlakyClient,
            retry_base_seconds=BACKOFF_SECONDS,
            retry_max_seconds=BACKOFF_SECONDS,
        )
        client.attempts = 0

        result = []
        backing_off = threading.Thread(
            target=lambda: result.append(client._run_async(client.flaky_request()))
        )
        backing_off.start()
        while client.attempts == 0:  # wait for the first attempt to fail
            time.sleep(0.01)

        # a quick request finishes while the flaky one is backing off
        start = time.monotonic()
        client._run_async(asyncio.sleep(0))
        assert time.monotonic() - start < MAX_STALL_SECONDS

        backing_off.join(timeout=TIMEOUT_SECONDS)
        assert result == ["recovered"]

    @pytest.fixture
    def fake_server(self, monkeypatch):
        """Make cached ``ProtocolDAGResult`` fetches return their resource path.

        Avoids a real API server: the "result" for a fetch is just the path it
        requested.

        """

        async def get_resource(self, resource, compress=False):
            return [resource]

        monkeypatch.setattr(AlchemiscaleClient, "_get_resource_async", get_resource)
        monkeypatch.setattr(
            alchemiscale.interface.client, "pdr_from_bytes", lambda b: b.decode()
        )

    @staticmethod
    def get_cached_result(client):
        """Fetch a result through the ``alru_cache``-wrapped method."""
        return client._run_async(
            client._async_get_protocoldagresult("ref", "tf", "results", False)
        )

    @classmethod
    def run_in_child(cls, conn):
        """Make calls in a forked child, sending the results back over ``conn``."""
        client = make_client()

        async def value():
            return 42

        with warnings.catch_warnings():
            # turns a cache still bound to the parent's loop into an error
            warnings.simplefilter("error")
            conn.send((client._run_async(value()), cls.get_cached_result(client)))
        conn.close()

    @pytest.mark.timeout(30)
    @pytest.mark.skipif(sys.platform == "win32", reason="requires fork")
    def test_forked_child_runs_own_loop(self, fake_server):
        """A forked child can make calls after the parent has started the loop.

        Regression: the child inherited the parent's loop, but not the thread
        running it, so its first call waited forever. Its ``alru_cache`` was
        also still bound to the parent's loop.

        """
        # the parent starts the shared loop and fills the cache
        client = make_client()
        expected = "/transformations/tf/results/ref"
        assert self.get_cached_result(client) == expected

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe()
        with warnings.catch_warnings():
            # forking a multi-threaded process warns on Python >= 3.12
            warnings.simplefilter("ignore", DeprecationWarning)
            child = ctx.Process(target=self.run_in_child, args=(child_conn,))
            child.start()

        try:
            assert parent_conn.poll(
                TIMEOUT_SECONDS
            ), "child hung waiting on the parent's loop"
            assert parent_conn.recv() == (42, expected)
        finally:
            child.join(timeout=TIMEOUT_SECONDS)
            if child.is_alive():
                child.kill()

        assert child.exitcode == 0
