"""Unit tests for the ComputeManager base class.

Focuses on the pure-arithmetic sizing logic in ``_compute_jobs_to_create``,
which can be exercised without standing up Neo4j, S3, or a running compute
API. Integration coverage of the cycle loop lives in
``tests/integration/compute/client/test_compute_manager.py``.
"""

from unittest.mock import MagicMock

from alchemiscale.compute.manager import ComputeManager
from alchemiscale.compute.settings import (
    ComputeManagerSettings,
    ComputeServiceSettings,
)


class _Manager(ComputeManager):
    """Concrete subclass for testing — implements the abstract method as a no-op."""

    def create_compute_services(self, data, target):
        return target


def _make_manager(
    *,
    max_compute_services: int = 10,
    max_submit_per_cycle: int = 1,
    claim_limit: int = 1,
) -> _Manager:
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

    mgr = _Manager.__new__(_Manager)
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
