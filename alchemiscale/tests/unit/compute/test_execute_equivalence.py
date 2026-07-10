"""Behavioral-equivalence tests for the alchemiscale-owned DAG executor.

This module is the guard run on every ``gufe`` upgrade. It runs *identical*
:class:`~gufe.protocols.ProtocolDAG`\\ s through both

* :func:`gufe.protocols.protocoldag.execute_DAG` (the upstream reference), and
* :func:`alchemiscale.compute.execute.execute_DAG` (the alchemiscale fork),

and asserts the resulting :class:`~gufe.protocols.ProtocolDAGResult`\\ s are
*behaviorally equivalent*.

Equivalence is compared **structurally**, never by result ``.key``: gufe units
tokenize with a ``uuid4`` per attempt, so two independent runs of the same DAG
produce results with different keys. We instead compare, over the two runs:

* overall ``.ok()``,
* the number of ``protocol_unit_results``,
* the multiset of result ``source_key``\\ s (which *is* stable --- it is the
  key of the originating :class:`ProtocolUnit`, shared by both runs of the
  same DAG object),
* the per-result ``.ok()`` sequence (grouped by ``source_key``, since raw
  ordering of same-level units is not guaranteed to match), and
* terminal / success / failure counts.

A separate, non-equivalence test exercises the alchemiscale-only
:class:`~alchemiscale.compute.execute.ExecutionHooks` seam directly.
"""

from collections import Counter

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gufe import ChemicalSystem, SmallMoleculeComponent
from gufe.protocols import Protocol, ProtocolUnit
from gufe.protocols import protocoldag as gufe_protocoldag
from gufe.protocols.errors import ExecutionInterrupt
from gufe.tests.test_protocol import BrokenProtocol, DummyProtocol

from alchemiscale.compute import execute as alchemiscale_execute
from alchemiscale.compute.execute import ExecutionHooks

# ---------------------------------------------------------------------------
# chemical-system + DAG fixtures
#
# The unit-test conftest only provides a heavyweight tyk2 network (module
# scope). For executor equivalence we want the smallest possible systems, so
# build minimal ones here (verified working per the task brief).
# ---------------------------------------------------------------------------


def _mol(smiles: str, name: str) -> SmallMoleculeComponent:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(m)
    return SmallMoleculeComponent.from_rdkit(m, name=name)


@pytest.fixture(scope="module")
def stateA() -> ChemicalSystem:
    return ChemicalSystem({"ligand": _mol("CCO", "ethanol")})


@pytest.fixture(scope="module")
def stateB() -> ChemicalSystem:
    return ChemicalSystem({"ligand": _mol("CCC", "propane")})


@pytest.fixture
def success_dag(stateA, stateB):
    """A DummyProtocol DAG: every unit succeeds (1 init + 21 sims + 1 finish)."""
    proto = DummyProtocol(settings=DummyProtocol.default_settings())
    return proto.create(stateA=stateA, stateB=stateB, mapping=None, name="success")


@pytest.fixture
def failure_dag(stateA, stateB):
    """A BrokenProtocol DAG: exactly one unit always fails, halting the DAG."""
    proto = BrokenProtocol(settings=BrokenProtocol.default_settings())
    return proto.create(stateA=stateA, stateB=stateB, mapping=None, name="failure")


# ---------------------------------------------------------------------------
# equivalence helpers
# ---------------------------------------------------------------------------


def _make_dirs(base, names):
    out = []
    for name in names:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        out.append(d)
    return out


def _ok_by_source(pdr):
    """Multiset of ``(source_key, ok)`` pairs across all unit results.

    Keyed on ``source_key`` (the stable originating-``ProtocolUnit`` key) rather
    than the per-attempt result ``.key`` (a fresh uuid4 each run). This captures
    "which units produced results, and with what ok-status" without depending on
    same-level ordering.
    """
    return Counter((str(r.source_key), r.ok()) for r in pdr.protocol_unit_results)


def assert_equivalent(gufe_pdr, alch_pdr):
    """Assert two ``ProtocolDAGResult``\\ s are behaviorally equivalent."""
    # overall success/failure
    assert gufe_pdr.ok() == alch_pdr.ok(), "overall .ok() differs"

    # same number of results
    assert len(gufe_pdr.protocol_unit_results) == len(
        alch_pdr.protocol_unit_results
    ), "number of protocol_unit_results differs"

    # same multiset of source_keys (which units produced results)
    gufe_sources = Counter(str(r.source_key) for r in gufe_pdr.protocol_unit_results)
    alch_sources = Counter(str(r.source_key) for r in alch_pdr.protocol_unit_results)
    assert gufe_sources == alch_sources, "multiset of source_keys differs"

    # same per-(source_key) .ok() multiset
    assert _ok_by_source(gufe_pdr) == _ok_by_source(
        alch_pdr
    ), "per-source .ok() sequence differs"

    # same success / failure / terminal counts
    assert len(gufe_pdr.protocol_unit_successes) == len(
        alch_pdr.protocol_unit_successes
    ), "success count differs"
    assert len(gufe_pdr.protocol_unit_failures) == len(
        alch_pdr.protocol_unit_failures
    ), "failure count differs"
    assert len(gufe_pdr.terminal_protocol_unit_results) == len(
        alch_pdr.terminal_protocol_unit_results
    ), "terminal result count differs"


def _run_both(dag, tmp_path, *, cache_basedir=None, **kwargs):
    """Run ``dag`` through both executors in separate temp dirs.

    Each executor gets its own ``shared``/``scratch`` (and optional ``cache``)
    trees so filesystem effects never cross-contaminate. Returns
    ``(gufe_pdr, alch_pdr)``.
    """
    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch = _make_dirs(gufe_dir, ["shared", "scratch"])
    a_shared, a_scratch = _make_dirs(alch_dir, ["shared", "scratch"])

    g_cache = a_cache = None
    if cache_basedir is not None:
        (g_cache,) = _make_dirs(gufe_dir, ["cache"])
        (a_cache,) = _make_dirs(alch_dir, ["cache"])

    gufe_pdr = gufe_protocoldag.execute_DAG(
        dag,
        shared_basedir=g_shared,
        scratch_basedir=g_scratch,
        cache_basedir=g_cache,
        **kwargs,
    )
    alch_pdr = alchemiscale_execute.execute_DAG(
        dag,
        shared_basedir=a_shared,
        scratch_basedir=a_scratch,
        cache_basedir=a_cache,
        **kwargs,
    )
    return gufe_pdr, alch_pdr


# ---------------------------------------------------------------------------
# 1. success DAG, n_retries=0
# ---------------------------------------------------------------------------


def test_equivalence_success(success_dag, tmp_path):
    gufe_pdr, alch_pdr = _run_both(success_dag, tmp_path, n_retries=0)

    assert gufe_pdr.ok() is True
    assert alch_pdr.ok() is True
    # all 23 units succeeded once
    assert len(alch_pdr.protocol_unit_results) == len(success_dag.protocol_units)
    assert_equivalent(gufe_pdr, alch_pdr)


# ---------------------------------------------------------------------------
# 2. failure DAG, n_retries=0  (one persistent failure halts the DAG)
# ---------------------------------------------------------------------------


def test_equivalence_failure_no_retry(failure_dag, tmp_path):
    gufe_pdr, alch_pdr = _run_both(
        failure_dag, tmp_path, n_retries=0, raise_error=False
    )

    assert gufe_pdr.ok() is False
    assert alch_pdr.ok() is False
    # both halt at the same point => same number of results
    assert_equivalent(gufe_pdr, alch_pdr)


# ---------------------------------------------------------------------------
# 3. failure DAG, n_retries=2  (retries then halts)
# ---------------------------------------------------------------------------


def test_equivalence_failure_with_retries(failure_dag, tmp_path):
    gufe_pdr, alch_pdr = _run_both(
        failure_dag, tmp_path, n_retries=2, raise_error=False
    )

    assert gufe_pdr.ok() is False
    assert alch_pdr.ok() is False

    # the single broken unit is attempted n_retries+1 == 3 times in both
    assert len(gufe_pdr.protocol_unit_failures) == len(alch_pdr.protocol_unit_failures)
    assert len(alch_pdr.protocol_unit_failures) == 3
    assert_equivalent(gufe_pdr, alch_pdr)


# ---------------------------------------------------------------------------
# 4. raise_error=True on the failure DAG  (both raise the same type)
# ---------------------------------------------------------------------------


def test_equivalence_raise_error(failure_dag, tmp_path):
    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch = _make_dirs(gufe_dir, ["shared", "scratch"])
    a_shared, a_scratch = _make_dirs(alch_dir, ["shared", "scratch"])

    with pytest.raises(Exception) as gufe_exc:
        gufe_protocoldag.execute_DAG(
            failure_dag,
            shared_basedir=g_shared,
            scratch_basedir=g_scratch,
            raise_error=True,
            n_retries=0,
        )

    with pytest.raises(Exception) as alch_exc:
        alchemiscale_execute.execute_DAG(
            failure_dag,
            shared_basedir=a_shared,
            scratch_basedir=a_scratch,
            raise_error=True,
            n_retries=0,
        )

    # both implementations raise the *same* exception type
    assert type(gufe_exc.value) is type(alch_exc.value)
    # and it originates from the broken unit's ValueError
    assert isinstance(alch_exc.value, ValueError)
    assert "I have failed my mission" in str(alch_exc.value)


# ---------------------------------------------------------------------------
# 5. cache-resume equivalence
#
# Fair comparison design: each implementation gets its OWN cache dir, but both
# are seeded IDENTICALLY by first running that same implementation once with
# keep_cache=True. We then run each implementation a SECOND time against its
# now-populated cache and assert the two second-runs are equivalent to each
# other (and that both skipped re-execution).
#
# "Skipped execution" is detected structurally: on a fully-cached resume, no
# shared_* directories are created (the per-unit execute() branch is never
# entered), so shared_basedir stays empty. We assert that for both.
# ---------------------------------------------------------------------------


def test_equivalence_cache_resume(success_dag, tmp_path):
    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"

    def _run(module, base, cache):
        shared, scratch = _make_dirs(base, ["shared", "scratch"])
        pdr = module.execute_DAG(
            success_dag,
            shared_basedir=shared,
            scratch_basedir=scratch,
            cache_basedir=cache,
            keep_cache=True,
            n_retries=0,
        )
        return pdr, shared

    (g_cache,) = _make_dirs(gufe_dir, ["cache"])
    (a_cache,) = _make_dirs(alch_dir, ["cache"])

    # first run: populate each cache
    gufe_first, _ = _run(gufe_protocoldag, gufe_dir / "run1", g_cache)
    alch_first, _ = _run(alchemiscale_execute, alch_dir / "run1", a_cache)
    assert gufe_first.ok() and alch_first.ok()

    # second run against the populated cache: should skip execution entirely
    gufe_second, gufe_shared2 = _run(gufe_protocoldag, gufe_dir / "run2", g_cache)
    alch_second, alch_shared2 = _run(alchemiscale_execute, alch_dir / "run2", a_cache)

    # both second runs succeeded and are equivalent to each other
    assert gufe_second.ok() and alch_second.ok()
    assert_equivalent(gufe_second, alch_second)

    # both second runs are also equivalent to their respective first runs
    assert_equivalent(gufe_first, gufe_second)
    assert_equivalent(alch_first, alch_second)

    # cache hit => no unit executed => no shared_* dirs created, in BOTH
    assert list(gufe_shared2.iterdir()) == [], "gufe re-executed despite cache"
    assert list(alch_shared2.iterdir()) == [], "alchemiscale re-executed despite cache"


# ---------------------------------------------------------------------------
# 6. keep_shared / keep_scratch parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keep_shared,keep_scratch",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_equivalence_keep_dirs(success_dag, tmp_path, keep_shared, keep_scratch):
    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch = _make_dirs(gufe_dir, ["shared", "scratch"])
    a_shared, a_scratch = _make_dirs(alch_dir, ["shared", "scratch"])

    gufe_pdr = gufe_protocoldag.execute_DAG(
        success_dag,
        shared_basedir=g_shared,
        scratch_basedir=g_scratch,
        keep_shared=keep_shared,
        keep_scratch=keep_scratch,
        n_retries=0,
    )
    alch_pdr = alchemiscale_execute.execute_DAG(
        success_dag,
        shared_basedir=a_shared,
        scratch_basedir=a_scratch,
        keep_shared=keep_shared,
        keep_scratch=keep_scratch,
        n_retries=0,
    )

    assert_equivalent(gufe_pdr, alch_pdr)

    # directory-retention parity: presence/absence of per-unit subdirs must match
    def _has_children(d):
        return any(d.iterdir())

    assert _has_children(g_shared) == _has_children(a_shared) == keep_shared
    assert _has_children(g_scratch) == _has_children(a_scratch) == keep_scratch


# ---------------------------------------------------------------------------
# focused test of the ExecutionHooks seam (alchemiscale-only, non-equivalence)
# ---------------------------------------------------------------------------


class RecordingHooks(ExecutionHooks):
    """Records every hook invocation for later assertion."""

    def __init__(self):
        self.dag_starts = []  # (units_total,)
        self.starts = []  # (source_key, attempt)
        self.ends = []  # (source_key, attempt, result_or_None)
        self.progress = []  # (units_completed, units_total)

    def on_dag_start(self, protocoldag, units_total):
        self.dag_starts.append(units_total)

    def on_unit_attempt_start(self, unit, attempt):
        self.starts.append((str(unit.key), attempt))

    def on_unit_attempt_end(self, unit, attempt, result):
        self.ends.append((str(unit.key), attempt, result))

    def on_progress(self, units_completed, units_total):
        self.progress.append((units_completed, units_total))


def test_hooks_success(success_dag, tmp_path):
    shared, scratch = _make_dirs(tmp_path, ["shared", "scratch"])
    hooks = RecordingHooks()

    pdr = alchemiscale_execute.execute_DAG(
        success_dag,
        shared_basedir=shared,
        scratch_basedir=scratch,
        n_retries=0,
        hooks=hooks,
    )
    assert pdr.ok()

    units_total = len(success_dag.protocol_units)

    # on_dag_start fired once with the correct total
    assert hooks.dag_starts == [units_total]

    # on_progress: first call is (0, N), last is (N, N)
    assert hooks.progress[0] == (0, units_total)
    assert hooks.progress[-1] == (units_total, units_total)
    # progress is monotonically non-decreasing and ends at completion
    completed = [c for c, _ in hooks.progress]
    assert completed == sorted(completed)
    assert completed[-1] == units_total

    # every start has exactly one matching end (balanced)
    assert len(hooks.starts) == len(hooks.ends)
    start_keys = Counter((k, a) for k, a in hooks.starts)
    end_keys = Counter((k, a) for k, a, _ in hooks.ends)
    assert start_keys == end_keys

    # every end on a success DAG carries an ok result
    assert all(r is not None and r.ok() for _, _, r in hooks.ends)


def test_hooks_failure(failure_dag, tmp_path):
    shared, scratch = _make_dirs(tmp_path, ["shared", "scratch"])
    hooks = RecordingHooks()

    pdr = alchemiscale_execute.execute_DAG(
        failure_dag,
        shared_basedir=shared,
        scratch_basedir=scratch,
        n_retries=0,
        raise_error=False,
        hooks=hooks,
    )
    assert not pdr.ok()

    units_total = len(failure_dag.protocol_units)
    assert hooks.dag_starts == [units_total]

    # starts and ends stay balanced even through the failure
    assert len(hooks.starts) == len(hooks.ends)
    start_keys = Counter((k, a) for k, a in hooks.starts)
    end_keys = Counter((k, a) for k, a, _ in hooks.ends)
    assert start_keys == end_keys

    # exactly one end carries a non-ok result (the broken unit)
    non_ok_ends = [(k, a, r) for k, a, r in hooks.ends if r is not None and not r.ok()]
    assert len(non_ok_ends) == 1

    # progress starts at (0, N) and freezes at the pre-failure completed count:
    # the DAG halts before all units finish, so the final completed count is
    # strictly less than the total, and equals the number of ok ends.
    assert hooks.progress[0] == (0, units_total)
    n_ok_ends = sum(1 for _, _, r in hooks.ends if r is not None and r.ok())
    assert hooks.progress[-1] == (n_ok_ends, units_total)
    assert hooks.progress[-1][0] < units_total


# ---------------------------------------------------------------------------
# custom protocols for the introspection-executor behaviors
#
# Each is a minimal 2-unit DAG (an upstream unit feeding a downstream unit) so
# that "downstream executed" is observable. They subclass DummyProtocol purely
# to inherit its settings/gather machinery and override `_create`.
# ---------------------------------------------------------------------------


class _PassUnit(ProtocolUnit):
    """A trivially-succeeding unit; used as the downstream in these DAGs."""

    @staticmethod
    def _execute(ctx, **inputs):
        return {"ok": True}


class _InterruptUnit(ProtocolUnit):
    """A unit whose ``_execute`` raises a chosen ``BaseException`` subclass.

    ``gufe``'s ``ProtocolUnit.execute`` lets ``KeyboardInterrupt`` and
    ``ExecutionInterrupt`` propagate rather than converting them into a
    ``ProtocolUnitFailure``; this unit lets us exercise that path.
    """

    @staticmethod
    def _execute(ctx, *, exc_name, **inputs):
        if exc_name == "ExecutionInterrupt":
            raise ExecutionInterrupt("unrecoverable")
        elif exc_name == "KeyboardInterrupt":
            raise KeyboardInterrupt("ctrl-c")
        raise AssertionError(f"unknown exc_name {exc_name!r}")  # pragma: no cover


class InterruptProtocol(DummyProtocol):
    """A DAG whose first unit raises an interrupt; a downstream unit follows."""

    exc_name = "ExecutionInterrupt"

    def _create(self, stateA, stateB, mapping=None, extends=None):
        head = _InterruptUnit(
            settings=self.settings, name="interrupter", exc_name=self.exc_name
        )
        tail = _PassUnit(settings=self.settings, name="downstream", upstream=head)
        return [head, tail]


class KeyboardInterruptProtocol(InterruptProtocol):
    exc_name = "KeyboardInterrupt"


# module-level attempt counter for the retry-then-success unit, keyed by the
# source unit's gufe key. Units are stateless and re-executed per attempt, so
# state must live outside the unit; the test RESETS this before each executor
# invocation so both runs see identical behavior.
_RETRY_ATTEMPTS: dict[str, int] = {}
# number of leading failures before success
_RETRY_FAIL_COUNT = 2


class _FlakyUnit(ProtocolUnit):
    """Fails its first ``_RETRY_FAIL_COUNT`` attempts, then succeeds.

    Attempt bookkeeping is external (``_RETRY_ATTEMPTS``, keyed by the unit's
    ``counter_key`` input) since the unit is re-instantiated/re-executed per
    attempt. The two executor runs share the same DAG object --- hence the same
    ``counter_key`` --- so, after the test resets the counter before each run,
    both observe identical flaky behavior.
    """

    @staticmethod
    def _execute(ctx, *, counter_key, **inputs):
        seen = _RETRY_ATTEMPTS.get(counter_key, 0)
        _RETRY_ATTEMPTS[counter_key] = seen + 1
        if seen < _RETRY_FAIL_COUNT:
            raise ValueError(f"flaky failure #{seen}")
        return {"ok": True, "attempts": seen + 1}


class RetryThenSucceedProtocol(DummyProtocol):
    """A DAG whose head unit is flaky (N failures then success); tail follows."""

    def _create(self, stateA, stateB, mapping=None, extends=None):
        head = _FlakyUnit(settings=self.settings, name="flaky", counter_key="flaky")
        tail = _PassUnit(settings=self.settings, name="downstream", upstream=head)
        return [head, tail]


class _StreamUnit(ProtocolUnit):
    """Writes one file into ``ctx.stdout`` and one into ``ctx.stderr``."""

    @staticmethod
    def _execute(ctx, **inputs):
        (ctx.stdout / "out.txt").write_text("hello stdout\n")
        (ctx.stderr / "err.txt").write_text("hello stderr\n")
        return {"ok": True}


class StreamProtocol(DummyProtocol):
    """A DAG whose head unit writes stream files; tail follows."""

    def _create(self, stateA, stateB, mapping=None, extends=None):
        head = _StreamUnit(settings=self.settings, name="streamer")
        tail = _PassUnit(settings=self.settings, name="downstream", upstream=head)
        return [head, tail]


class _BalanceHooks(ExecutionHooks):
    """Records start/end pairs (and the end ``result``) for balance checks."""

    def __init__(self):
        self.starts = []  # (source_key, attempt)
        self.ends = []  # (source_key, attempt, result_or_None)

    def on_unit_attempt_start(self, unit, attempt):
        self.starts.append((str(unit.key), attempt))

    def on_unit_attempt_end(self, unit, attempt, result):
        self.ends.append((str(unit.key), attempt, result))


# ---------------------------------------------------------------------------
# TASK A.1 --- interrupt propagation (ExecutionInterrupt / KeyboardInterrupt)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proto_cls,exc_type",
    [
        (InterruptProtocol, ExecutionInterrupt),
        (KeyboardInterruptProtocol, KeyboardInterrupt),
    ],
)
def test_equivalence_interrupt_propagates(
    proto_cls, exc_type, stateA, stateB, tmp_path
):
    proto = proto_cls(settings=proto_cls.default_settings())
    dag = proto.create(stateA=stateA, stateB=stateB, mapping=None, name="interrupt")

    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch = _make_dirs(gufe_dir, ["shared", "scratch"])
    a_shared, a_scratch = _make_dirs(alch_dir, ["shared", "scratch"])

    # BOTH executors let the interrupt propagate (NOT converted to a failure),
    # even with raise_error=False --- interrupts derive from BaseException.
    with pytest.raises(exc_type):
        gufe_protocoldag.execute_DAG(
            dag,
            shared_basedir=g_shared,
            scratch_basedir=g_scratch,
            raise_error=False,
            n_retries=0,
        )

    hooks = _BalanceHooks()
    with pytest.raises(exc_type):
        alchemiscale_execute.execute_DAG(
            dag,
            shared_basedir=a_shared,
            scratch_basedir=a_scratch,
            raise_error=False,
            n_retries=0,
            hooks=hooks,
        )

    # the end hook still fires for the interrupted attempt (so log capture is
    # always closed): starts and ends stay balanced...
    assert len(hooks.starts) == len(hooks.ends) == 1
    assert Counter(hooks.starts) == Counter((k, a) for k, a, _ in hooks.ends)
    # ...and the end hook received result=None for the interrupted attempt
    assert hooks.ends[0][2] is None


# ---------------------------------------------------------------------------
# TASK A.2 --- retry-then-success (the retry loop's success arm)
# ---------------------------------------------------------------------------


def _flaky_source_key(dag):
    """The stable source key of the flaky head unit in a RetryThenSucceed DAG."""
    for unit in dag.protocol_units:
        if unit.name == "flaky":
            return str(unit.key)
    raise AssertionError("no flaky unit found")  # pragma: no cover


def test_equivalence_retry_then_success(stateA, stateB, tmp_path):
    proto = RetryThenSucceedProtocol(
        settings=RetryThenSucceedProtocol.default_settings()
    )
    dag = proto.create(stateA=stateA, stateB=stateB, mapping=None, name="retry")

    flaky_key = _flaky_source_key(dag)
    n_retries = _RETRY_FAIL_COUNT + 1  # comfortably >= N

    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch = _make_dirs(gufe_dir, ["shared", "scratch"])
    a_shared, a_scratch = _make_dirs(alch_dir, ["shared", "scratch"])

    # RESET the counter immediately before EACH executor invocation so both
    # runs of the same DAG observe identical flaky behavior.
    _RETRY_ATTEMPTS.clear()
    gufe_pdr = gufe_protocoldag.execute_DAG(
        dag,
        shared_basedir=g_shared,
        scratch_basedir=g_scratch,
        raise_error=False,
        n_retries=n_retries,
    )

    _RETRY_ATTEMPTS.clear()
    alch_pdr = alchemiscale_execute.execute_DAG(
        dag,
        shared_basedir=a_shared,
        scratch_basedir=a_scratch,
        raise_error=False,
        n_retries=n_retries,
    )

    # both succeed overall (the flaky unit eventually passes; downstream runs)
    assert gufe_pdr.ok() is True
    assert alch_pdr.ok() is True

    def _for_source(pdr, source_key):
        return [r for r in pdr.protocol_unit_results if str(r.source_key) == source_key]

    for pdr in (gufe_pdr, alch_pdr):
        flaky_results = _for_source(pdr, flaky_key)
        failures = [r for r in flaky_results if not r.ok()]
        successes = [r for r in flaky_results if r.ok()]
        # exactly N failures + 1 success for the flaky source unit
        assert len(failures) == _RETRY_FAIL_COUNT
        assert len(successes) == 1

        # downstream unit executed and produced a (successful) result
        downstream = [
            r for r in pdr.protocol_unit_results if str(r.source_key) != flaky_key
        ]
        assert len(downstream) == 1
        assert downstream[0].ok()

    assert_equivalent(gufe_pdr, alch_pdr)


# ---------------------------------------------------------------------------
# TASK A.3 --- stream-dir parity (embedded stdout/stderr; per-attempt cleanup)
# ---------------------------------------------------------------------------


def test_equivalence_stream_dirs(stateA, stateB, tmp_path):
    proto = StreamProtocol(settings=StreamProtocol.default_settings())
    dag = proto.create(stateA=stateA, stateB=stateB, mapping=None, name="stream")

    gufe_dir = tmp_path / "gufe"
    alch_dir = tmp_path / "alch"
    g_shared, g_scratch, g_stdout, g_stderr = _make_dirs(
        gufe_dir, ["shared", "scratch", "stdout", "stderr"]
    )
    a_shared, a_scratch, a_stdout, a_stderr = _make_dirs(
        alch_dir, ["shared", "scratch", "stdout", "stderr"]
    )

    gufe_pdr = gufe_protocoldag.execute_DAG(
        dag,
        shared_basedir=g_shared,
        scratch_basedir=g_scratch,
        stdout_basedir=g_stdout,
        stderr_basedir=g_stderr,
        n_retries=0,
    )
    alch_pdr = alchemiscale_execute.execute_DAG(
        dag,
        shared_basedir=a_shared,
        scratch_basedir=a_scratch,
        stdout_basedir=a_stdout,
        stderr_basedir=a_stderr,
        n_retries=0,
    )

    assert gufe_pdr.ok() and alch_pdr.ok()
    assert_equivalent(gufe_pdr, alch_pdr)

    def _streamer_result(pdr):
        (r,) = [r for r in pdr.protocol_unit_results if r.name == "streamer"]
        return r

    g_res = _streamer_result(gufe_pdr)
    a_res = _streamer_result(alch_pdr)

    # the streamer unit's embedded stdout/stderr (filename -> bytes) match
    assert g_res.stdout == a_res.stdout
    assert g_res.stderr == a_res.stderr
    # and carry the expected content
    assert a_res.stdout == {"out.txt": b"hello stdout\n"}
    assert a_res.stderr == {"err.txt": b"hello stderr\n"}

    # per-attempt stream subdirs are rmtree'd (gufe/alchemiscale clean them up)
    # while the base dirs remain, in BOTH executors.
    for base in (g_stdout, g_stderr, a_stdout, a_stderr):
        assert base.exists()
        assert list(base.iterdir()) == []
