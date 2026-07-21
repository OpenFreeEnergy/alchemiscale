"""
:mod:`alchemiscale.compute.execute` --- alchemiscale-owned DAG executor
=======================================================================

An alchemiscale-maintained executor for :class:`~gufe.protocols.ProtocolDAG`\\ s,
used by :class:`~alchemiscale.compute.service.SynchronousComputeService`.

It mirrors the execution semantics of :func:`gufe.protocols.protocoldag.execute_DAG`
--- topological iteration, per-attempt :class:`~gufe.protocols.protocolunit.Context`
construction (including stream directories), retry behavior, all-attempts result
accumulation, halt on persistent failure, ``KeyboardInterrupt``/``ExecutionInterrupt``
pass-through, and ``gufe``-compatible unit-result caching --- while adding
**unit-attempt start/end hooks**. Those hooks are the single seam that exact
progress reporting and per-attempt log capture plug into.

This is a deliberate direction choice: ``SynchronousComputeService`` is the
reference implementation *for alchemiscale compute services*, and it should
support every behavior we want alchemiscale services to have. The fork risk ---
``gufe``'s executor keeps evolving --- is mitigated by a behavioral-equivalence
test suite (``tests/.../test_execute_equivalence.py``) that runs identical DAGs
through both implementations and asserts equivalent ``ProtocolDAGResult``\\ s,
exercised on every ``gufe`` upgrade.

The ``gufe`` version this mirrors is pinned in ``pyproject.toml``. ``gufe``'s
private input-mapping and cache-validation helpers (``_pu_to_pur``,
``_get_valid_unit_results``) are reused directly so that resume/cache semantics
cannot silently diverge; if a future ``gufe`` removes or changes them, the
equivalence suite fails loudly.
"""

import shutil
import warnings
from json import JSONDecodeError
from pathlib import Path

from gufe.protocols.protocoldag import (
    ProtocolDAG,
    ProtocolDAGResult,
    _get_valid_unit_results,
    _pu_to_pur,
)
from gufe.protocols.protocolunit import (
    Context,
    ProtocolUnit,
    ProtocolUnitResult,
)
from gufe.tokenization import GufeKey


class ExecutionHooks:
    """Hooks invoked by :func:`execute_DAG` at DAG and unit-attempt boundaries.

    Subclass and override the methods of interest; every default is a no-op, so
    a bare ``ExecutionHooks()`` reproduces plain ``gufe`` execution semantics.

    The hooks fire in the execution thread and must be cheap and non-blocking:
    progress pushes are fire-and-forget, and log-capture open/close must not
    stall DAG execution.
    """

    def on_dag_start(self, protocoldag: ProtocolDAG, units_total: int) -> None:
        """Called once, before any unit executes."""

    def on_unit_attempt_start(self, unit: ProtocolUnit, attempt: int) -> None:
        """Called immediately before a single unit-attempt executes.

        The log-capture handler for this attempt should be opened here, so that
        records emitted during the attempt are attributed to its result.
        """

    def on_unit_attempt_end(
        self,
        unit: ProtocolUnit,
        attempt: int,
        result: ProtocolUnitResult | None,
    ) -> None:
        """Called after a single unit-attempt executes (or is interrupted).

        Always called exactly once per ``on_unit_attempt_start``, even if the
        attempt raised (in which case ``result`` is ``None``) --- so the
        log-capture handler is guaranteed to be closed. When ``result`` is not
        ``None``, captured logs should be associated with ``result.key`` for
        later upload.
        """

    def on_progress(self, units_completed: int, units_total: int) -> None:
        """Called with the running count of distinct successfully completed
        units against the DAG's total unit count.

        Fired once at DAG start with ``(0, units_total)`` --- so the denominator
        is visible immediately --- and again after each unit successfully
        completes.
        """


def execute_DAG(
    protocoldag: ProtocolDAG,
    *,
    shared_basedir: Path,
    scratch_basedir: Path,
    cache_basedir: Path | None = None,
    stderr_basedir: Path | None = None,
    stdout_basedir: Path | None = None,
    keep_shared: bool = False,
    keep_scratch: bool = False,
    keep_cache: bool = False,
    raise_error: bool = True,
    n_retries: int = 0,
    hooks: ExecutionHooks | None = None,
) -> ProtocolDAGResult:
    """Locally execute a full :class:`ProtocolDAG` in serial and in-process.

    A behavioral mirror of :func:`gufe.protocols.protocoldag.execute_DAG` with
    added unit-attempt hooks. All keyword parameters have identical meaning to
    the ``gufe`` function; see its docstring. The only addition is ``hooks``.

    Parameters
    ----------
    hooks
        :class:`ExecutionHooks` invoked at DAG start and at each unit-attempt
        start/end, plus progress updates. Defaults to a no-op set, in which
        case execution is semantically identical to ``gufe``'s.

    Raises
    ------
    ProtocolDAGExecutionError
        If the ``ProtocolDAG`` cannot be executed due to an invalid cache state.
    """
    if n_retries < 0:
        raise ValueError("Must give positive number of retries")

    if hooks is None:
        hooks = ExecutionHooks()

    # `protocol_units` is in DAG-dependency order
    units_total = len(protocoldag.protocol_units)
    hooks.on_dag_start(protocoldag, units_total)
    # publish the denominator immediately (0 of N complete)
    hooks.on_progress(0, units_total)

    # load any cached unit results (disabled by SynchronousComputeService, but
    # implemented for gufe equivalence)
    all_cached_results: list[ProtocolUnitResult] = []
    if cache_basedir is not None:
        dag_unitresults_cache = cache_basedir / f"{str(protocoldag.key)}-results_cache"
        dag_unitresults_cache.mkdir(exist_ok=True, parents=True)

        for file in dag_unitresults_cache.rglob("*.json"):
            try:
                unit_result = ProtocolUnitResult.from_json(file)
            except JSONDecodeError as e:
                warnings.warn(f"Unable to read file, skipping {file}: {e}")
            else:
                all_cached_results.append(unit_result)

    # handle results & optionally caching
    results: dict[GufeKey, ProtocolUnitResult] = _get_valid_unit_results(
        protocoldag, all_cached_results
    )
    all_results = []  # successes AND failures
    shared_paths = []
    for unit in protocoldag.protocol_units:
        # If we already have results (from cache), skip execution
        if unit.key in results:
            all_results.append(results[unit.key])
            continue

        # translate each `ProtocolUnit` in input into corresponding `ProtocolUnitResult`
        inputs = _pu_to_pur(unit.inputs, results)

        attempt = 0
        while attempt <= n_retries:
            shared = shared_basedir / f"shared_{str(unit.key)}_attempt_{attempt}"
            shared_paths.append(shared)
            shared.mkdir(exist_ok=True)

            scratch = scratch_basedir / f"scratch_{str(unit.key)}_attempt_{attempt}"
            scratch.mkdir(exist_ok=True)

            stderr = None
            if stderr_basedir:
                stderr = stderr_basedir / f"stderr_{str(unit.key)}_attempt_{attempt}"
                stderr.mkdir(exist_ok=True)

            stdout = None
            if stdout_basedir:
                stdout = stdout_basedir / f"stdout_{str(unit.key)}_attempt_{attempt}"
                stdout.mkdir(exist_ok=True)

            context = Context(
                shared=shared, scratch=scratch, stderr=stderr, stdout=stdout
            )

            # execute this unit-attempt, guaranteeing the end hook fires exactly
            # once (so log capture is always closed). KeyboardInterrupt and
            # gufe ExecutionInterrupt derive from BaseException and propagate.
            # Contract: hook methods must not raise --- a hook exception in this
            # `finally` would replace an in-flight interrupt/error being
            # propagated. The supplied `ExecutionHooks` implementations honor
            # this (they only mutate in-memory state and logging handlers).
            hooks.on_unit_attempt_start(unit, attempt)
            result = None
            try:
                result = unit.execute(
                    context=context, raise_error=raise_error, **inputs
                )
                all_results.append(result)
            finally:
                hooks.on_unit_attempt_end(unit, attempt, result)

            # clean up outputs
            if stderr:
                shutil.rmtree(stderr)
            if stdout:
                shutil.rmtree(stdout)

            if not keep_scratch:
                shutil.rmtree(scratch)

            if result.ok():
                # attach result to this `ProtocolUnit`
                results[unit.key] = result

                # Serialize results if requested
                if cache_basedir is not None:
                    result.to_json(
                        dag_unitresults_cache / f"{str(unit.key)}_unitresults.json"
                    )

                # progress: one more distinct unit successfully completed.
                # NOTE: `results` includes any cache-resumed entries, so under
                # caching (disabled by SynchronousComputeService) the count
                # would include pre-completed units. Harmless while caching is
                # off; revisit when #180-style resume lands.
                hooks.on_progress(len(results), units_total)
                break
            attempt += 1

        if not result.ok():
            # persistent failure halts DAG execution; downstream units yield no
            # results at all, freezing progress at the point of failure
            break

    if not keep_shared:
        for shared_path in shared_paths:
            shutil.rmtree(shared_path)

    if not keep_cache and cache_basedir is not None:
        shutil.rmtree(dag_unitresults_cache)

    return ProtocolDAGResult(
        name=protocoldag.name,
        protocol_units=protocoldag.protocol_units,
        protocol_unit_results=all_results,
        transformation_key=protocoldag.transformation_key,
        extends_key=protocoldag.extends_key,
    )
