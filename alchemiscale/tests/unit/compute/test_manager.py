"""Unit tests for the ComputeManager base class.

Focuses on the pure-arithmetic sizing logic in ``_compute_jobs_to_create``,
which can be exercised without standing up Neo4j, S3, or a running compute
API. Integration coverage of the cycle loop lives in
``tests/integration/compute/client/test_compute_manager.py``.
"""

import datetime

from unittest.mock import MagicMock

import pytest

from alchemiscale.base.client import AlchemiscaleBaseClient
from alchemiscale.compute import manager as manager_module
from alchemiscale.compute.backend import is_own_job_name, make_job_name
from alchemiscale.base.client import AlchemiscaleConnectionError
from alchemiscale.compute.client import AlchemiscaleComputeManagerClientError
from alchemiscale.compute.manager import (
    AuthenticationFailedError,
    ComputeManager,
    IncompatibleServerError,
)
from alchemiscale.storage.models import (
    ComputeManagerRegistration,
    ComputeManagerState,
    ComputeManagerStatus,
    ComputeServiceID,
)
from alchemiscale.compute.settings import (
    ComputeManagerSettings,
    ComputeServiceSettings,
)


def _make_manager(
    *,
    max_compute_services: int = 10,
    max_submit_per_cycle: int = 1,
    claim_limit: int = 1,
) -> ComputeManager:
    """Construct a manager without going through ``__init__``.

    ``ComputeManager.__init__`` does network registration and logging setup
    that we don't need for arithmetic tests; bypass it and inject just the
    ``settings``/``service_settings`` the sizing method reads.
    """
    settings = ComputeManagerSettings(
        name="testmgr",
        logfile=None,
        max_compute_services=max_compute_services,
        max_submit_per_cycle=max_submit_per_cycle,
    )
    # Use MagicMock for service_settings so we can set claim_limit without
    # constructing the full ComputeServiceSettings (which requires several
    # real-looking fields).
    service_settings = MagicMock(spec=ComputeServiceSettings)
    service_settings.claim_limit = claim_limit

    mgr = ComputeManager.__new__(ComputeManager)
    mgr._cycle_errors = 0
    mgr._last_state = None
    mgr.settings = settings
    mgr.service_settings = service_settings
    return mgr


def test_sizing_capped_by_num_tasks():
    """Don't submit more services than there are waiting tasks."""
    mgr = _make_manager(max_compute_services=100, max_submit_per_cycle=100)
    assert mgr._compute_jobs_to_create(num_tasks=3, num_active_services=0) == 3


def test_sizing_capped_by_max_submit_per_cycle():
    """``max_submit_per_cycle`` rate-limits ramp-up."""
    mgr = _make_manager(max_compute_services=100, max_submit_per_cycle=2)
    assert mgr._compute_jobs_to_create(num_tasks=100, num_active_services=0) == 2


def test_sizing_capped_by_remaining_capacity():
    """``max_compute_services - active_services`` is the hard ceiling."""
    mgr = _make_manager(max_compute_services=5, max_submit_per_cycle=100)
    assert mgr._compute_jobs_to_create(num_tasks=100, num_active_services=4) == 1


def test_sizing_returns_zero_at_capacity():
    """At/over capacity, we don't submit anything (defensive — the cycle
    normally gates this case before calling us)."""
    mgr = _make_manager(max_compute_services=5)
    assert mgr._compute_jobs_to_create(num_tasks=10, num_active_services=5) == 0
    assert mgr._compute_jobs_to_create(num_tasks=10, num_active_services=6) == 0


def test_sizing_returns_zero_when_no_tasks():
    """Defensive: no tasks -> no jobs, even if there's slack capacity."""
    mgr = _make_manager()
    assert mgr._compute_jobs_to_create(num_tasks=0, num_active_services=0) == 0


def test_sizing_divides_by_claim_limit():
    """A service that claims N tasks at once means ~num_tasks/N services."""
    mgr = _make_manager(
        max_compute_services=100, max_submit_per_cycle=100, claim_limit=5
    )
    # 20 tasks, claim_limit 5 -> 4 services
    assert mgr._compute_jobs_to_create(num_tasks=20, num_active_services=0) == 4


def test_sizing_floors_to_one_when_divide_collapses():
    """If integer-divide by claim_limit gives 0, still create one service.

    Example: a single task with ``claim_limit=2`` would arithmetically yield
    ``1 // 2 == 0``, but the cycle has already decided we should be scaling
    up, so create one anyway. The next cycle catches up if needed.
    """
    mgr = _make_manager(max_compute_services=10, max_submit_per_cycle=10, claim_limit=5)
    # 2 tasks, claim_limit 5 -> 2 // 5 == 0 -> floor to 1
    assert mgr._compute_jobs_to_create(num_tasks=2, num_active_services=0) == 1


def test_sizing_combines_all_caps():
    """The min() takes effect when multiple caps would individually allow more."""
    mgr = _make_manager(
        max_compute_services=10,  # remaining capacity = 8 with 2 active
        max_submit_per_cycle=4,
        claim_limit=1,
    )
    # min(20 tasks, 4 rate, 8 capacity) = 4 -> divided by 1 = 4
    assert mgr._compute_jobs_to_create(num_tasks=20, num_active_services=2) == 4


def test_sizing_floor_does_not_override_capacity():
    """Floor-to-1 should not push us over capacity.

    This case can't happen given the cycle's gate, but the method should be
    safe in isolation.
    """
    mgr = _make_manager(max_compute_services=2)
    # Already at capacity -> 0, floor does NOT kick in
    assert mgr._compute_jobs_to_create(num_tasks=10, num_active_services=2) == 0


class TestServerVersionCheck:
    """``ComputeManager.start`` refuses a server older than it supports."""

    @staticmethod
    def _manager_with_server(info: dict, client_version: str = "0.9.0"):
        mgr = _make_manager()
        mgr.client = MagicMock()
        mgr.client.api_url = "http://server/"
        mgr.client.get_server_version.side_effect = lambda: (
            AlchemiscaleBaseClient.get_server_version(_FakeInfoClient(info))
        )
        mgr.logger = MagicMock()
        return mgr

    @pytest.fixture
    def client_version(self, monkeypatch):
        def set_version(v):
            monkeypatch.setattr(manager_module, "version", lambda _: v)

        set_version("0.9.0")
        return set_version

    def test_current_server(self, client_version):
        mgr = self._manager_with_server({"version": "0.9.0"})
        mgr._check_server_version()

        mgr = self._manager_with_server({"version": "0.10.1"})
        mgr._check_server_version()

    def test_older_server(self, client_version):
        mgr = self._manager_with_server({"version": "0.8.1"})
        with pytest.raises(IncompatibleServerError, match="0.8.1.*0.9.0"):
            mgr._check_server_version()

    def test_server_without_version(self, client_version):
        mgr = self._manager_with_server({"message": "nothing yet"})
        with pytest.raises(IncompatibleServerError, match="older than 0.9.0"):
            mgr._check_server_version()

    def test_development_builds(self, client_version):
        # a development client accepts a server at least as new as itself
        client_version("0.8.2.dev3+gabc1234")
        mgr = self._manager_with_server({"version": "0.8.2.dev3+gabc1234"})
        mgr._check_server_version()

        mgr = self._manager_with_server({"version": "0.8.2.dev1"})
        with pytest.raises(IncompatibleServerError):
            mgr._check_server_version()

    def test_start_exits_before_registering(self, client_version):
        mgr = self._manager_with_server({"version": "0.8.1"})
        mgr.settings = mgr.settings.model_copy(update={"name": "testmgr"})
        with pytest.raises(IncompatibleServerError):
            mgr.start()
        mgr.client.register.assert_not_called()


class _FakeInfoClient:
    def __init__(self, info):
        self._info = info

    def get_info(self):
        return self._info


class TestJobNames:
    def test_make_job_name(self):
        name = make_job_name("foo.bar")
        assert is_own_job_name("foo.bar", name)
        assert make_job_name("foo") != make_job_name("foo")

    def test_is_own_job_name_exact(self):
        hex32 = "0123456789abcdef" * 2
        assert is_own_job_name("foo", f"foo.{hex32}")
        # not a prefix match: other managers' jobs aren't claimed
        assert not is_own_job_name("foo", f"foo.bar.{hex32}")
        assert not is_own_job_name("foo", f"foobar.{hex32}")
        assert not is_own_job_name("foo.bar", f"foo.{hex32}")
        # the name is escaped, not a pattern
        assert not is_own_job_name("f.o", f"fxo.{hex32}")
        # exactly 32 lowercase hex characters
        assert not is_own_job_name("foo", f"foo.{hex32[:-1]}")
        assert not is_own_job_name("foo", f"foo.{hex32.upper()}")
        assert not is_own_job_name("foo", f"foo.{hex32}x")

    def test_job_name_is_a_compute_service_name(self):
        # the compute service a job starts registers under the job's name
        name = make_job_name("foo.bar")
        assert ComputeServiceID.new_from_name(name).name == name


class TestServerErrors:
    """The manager keeps cycling through server errors, but exits if it
    can't authenticate."""

    @staticmethod
    def _manager(error):
        mgr = _make_manager()
        mgr.client = MagicMock()
        mgr.client.api_url = "http://server/"
        mgr.logger = MagicMock()
        mgr.cycle = MagicMock(side_effect=error)
        return mgr

    @pytest.mark.parametrize(
        "error",
        [
            AlchemiscaleComputeManagerClientError("bad gateway", status_code=502),
            AlchemiscaleComputeManagerClientError("server error", status_code=500),
            AlchemiscaleConnectionError("connection refused"),
        ],
    )
    def test_skips_cycle(self, error):
        mgr = self._manager(error)
        assert mgr._cycle_through_errors() is True
        mgr.logger.warning.assert_called_once()

    def test_authentication_failure(self):
        mgr = self._manager(
            AlchemiscaleComputeManagerClientError("unauthorized", status_code=401)
        )
        with pytest.raises(AuthenticationFailedError):
            mgr._cycle_through_errors()


class TestUnexpectedErrors:
    """An unexpected exception skips the cycle; enough in a row pause the
    manager, which never exits for them."""

    @staticmethod
    def _manager(status="OK", max_cycle_errors=3):
        mgr = _make_manager()
        mgr.settings.max_cycle_errors = max_cycle_errors
        mgr.settings.sleep_interval = 10
        mgr.client = MagicMock()
        mgr.client.api_url = "http://server/"
        mgr.logger = MagicMock()
        mgr.compute_manager_id = "testmgr-" + "0" * 32
        now = datetime.datetime.now(tz=datetime.UTC)
        mgr._last_state = ComputeManagerState(
            registration=ComputeManagerRegistration(
                name="testmgr",
                uuid="0" * 32,
                registered=now,
                last_status_update=now,
                status=status,
                saturation=0.5,
                version=7,
            )
        )
        mgr.cycle = MagicMock(side_effect=RuntimeError("bug"))
        return mgr

    def test_pauses_after_max_cycle_errors(self):
        mgr = self._manager()
        for _ in range(2):
            assert mgr._cycle_through_errors() is True
        mgr.client.update_state.assert_not_called()

        assert mgr._cycle_through_errors() is True
        (call,) = mgr.client.update_state.call_args_list
        assert call.args[1] == ComputeManagerStatus.DEGRADED
        assert "3 cycles in a row" in call.kwargs["detail"]
        assert "RuntimeError('bug')" in call.kwargs["detail"]
        update = call.kwargs["state_update"]
        assert update.expected_version == 7
        assert update.transition["pause_cause"] == "error"
        assert update.transition["backoff_seconds"] == 10

        # further errors don't pause again
        assert mgr._cycle_through_errors() is True
        assert mgr.client.update_state.call_count == 1

    def test_success_resets_count(self):
        mgr = self._manager()
        mgr._cycle_through_errors()
        mgr._cycle_through_errors()
        mgr.cycle.side_effect = None
        mgr.cycle.return_value = True
        assert mgr._cycle_through_errors() is True
        mgr.cycle.side_effect = RuntimeError("bug")
        mgr._cycle_through_errors()
        mgr._cycle_through_errors()
        mgr.client.update_state.assert_not_called()

    def test_already_paused(self):
        mgr = self._manager(status="DEGRADED", max_cycle_errors=1)
        mgr._cycle_through_errors()
        mgr.client.update_state.assert_not_called()

    def test_report_fails(self):
        mgr = self._manager(max_cycle_errors=1)
        mgr.client.update_state.side_effect = ConnectionError("down")
        assert mgr._cycle_through_errors() is True
