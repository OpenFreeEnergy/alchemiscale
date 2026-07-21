"""
:mod:`alchemiscale.compute.capture` --- per-unit log capture and progress hooks
================================================================================

Execution hooks for :class:`~alchemiscale.compute.service.SynchronousComputeService`
that plug into the alchemiscale executor (:mod:`alchemiscale.compute.execute`) to:

- capture log records emitted through ``gufe``'s sanctioned logging channel
  (:attr:`gufe.tokenization.GufeTokenizable.logger`, the ``gufekey.{module}.{qualname}``
  namespace), scoped to a single unit attempt, so attribution is exact per
  `ProtocolUnitResult`/`ProtocolUnitFailure`;
- push live progress counts, fire-and-forget, at unit boundaries.

We deliberately capture *only* the ``gufekey`` namespace --- what protocols emit
through ``ProtocolUnit.logger`` --- not third-party library loggers (OpenMM,
openff-toolkit, RDKit, ...), whose volume is unbounded and uncurated. A protocol
wanting library logs kept routes them through a ``Context.stdout``/``stderr``
`FileHandler` instead (the stream channel).
"""

import logging
import time

from gufe.protocols.protocolunit import ProtocolUnit, ProtocolUnitResult

from ..models import ScopedKey
from ..storage.models import ComputeServiceID
from .execute import ExecutionHooks

# the logger namespace every GufeTokenizable.logger writes to; hierarchy
# propagation delivers all descendants to a handler attached here
GUFEKEY_LOGGER_NAME = "gufekey"


class GufeKeyLogHandler(logging.Handler):
    """A `logging.Handler` that accumulates formatted, timestamped log lines.

    Attached to the ``gufekey`` logger for the duration of a single unit
    attempt. Each record carries ``record.gufekey`` (the emitting unit's gufe
    key), which the formatter includes for labeling and sanity-checking.
    """

    def __init__(self, level: int | str = logging.NOTSET):
        super().__init__(level=level)
        # `defaults` covers records that reach the `gufekey` logger without the
        # `GufeTokenizable.logger` adapter's `record.gufekey` stamp (the
        # sanctioned channel always sets it, but we must not raise on a record
        # that doesn't).
        formatter = logging.Formatter(
            "[%(asctime)s] [%(gufekey)s] [%(levelname)s] %(message)s",
            defaults={"gufekey": "-"},
        )
        formatter.converter = time.gmtime  # UTC timestamps
        self.setFormatter(formatter)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # pragma: no cover - defensive, mirrors logging
            self.handleError(record)

    def text(self) -> str:
        return "\n".join(self.lines)


class SynchronousExecutionHooks(ExecutionHooks):
    """Execution hooks binding the DAG execution of one `Task` to log capture and
    progress reporting.

    Parameters
    ----------
    task
        The `Task` whose DAG is being executed (for the progress payload).
    compute_service_id
        The claiming compute service (for the progress payload).
    progress_callback
        A fire-and-forget callable ``(task, units_completed, units_total)`` that
        pushes progress; it must not raise or block (the service wraps its own
        transport in a swallow-and-log).
    gufekey_loglevel
        The level to which the ``gufekey`` logger is set so protocol ``INFO``
        logs reach the capture handler (the logger otherwise inherits the root
        ``WARNING``). Applied once, when the hooks are constructed.
    log_cap_bytes
        Per-unit-result cap on captured log text; the tail (where errors live)
        is kept.
    """

    def __init__(
        self,
        *,
        task: ScopedKey,
        compute_service_id: ComputeServiceID,
        progress_callback,
        capture_logs: bool = True,
        gufekey_loglevel: int | str = logging.INFO,
        log_cap_bytes: int = 1_048_576,
    ):
        self.task = task
        self.compute_service_id = compute_service_id
        self.progress_callback = progress_callback
        self.capture_logs = capture_logs
        self.log_cap_bytes = log_cap_bytes

        # captured log text per unit-result gufe key, ready for upload
        self.unit_logs: dict[str, str] = {}

        self._logger = logging.getLogger(GUFEKEY_LOGGER_NAME)
        # ensure protocol logs at gufekey_loglevel reach the handler; only mutate
        # the (process-global) logger level when capture is actually enabled
        if self.capture_logs:
            self._logger.setLevel(gufekey_loglevel)
        self._handler: GufeKeyLogHandler | None = None

    def on_unit_attempt_start(self, unit: ProtocolUnit, attempt: int) -> None:
        if not self.capture_logs:
            return
        # open a fresh capture handler scoped to this single unit attempt
        self._handler = GufeKeyLogHandler()
        self._logger.addHandler(self._handler)

    def on_unit_attempt_end(
        self,
        unit: ProtocolUnit,
        attempt: int,
        result: ProtocolUnitResult | None,
    ) -> None:
        if self._handler is None:
            return
        self._logger.removeHandler(self._handler)
        if result is not None:
            text = self._handler.text()
            if text:
                data = text.encode("utf-8")
                if len(data) > self.log_cap_bytes:
                    # keep the tail
                    data = data[-self.log_cap_bytes :]
                    text = data.decode("utf-8", errors="replace")
                self.unit_logs[str(result.key)] = text
        self._handler = None

    def on_progress(self, units_completed: int, units_total: int) -> None:
        self.progress_callback(self.task, units_completed, units_total)
