"""Unit tests for :mod:`alchemiscale.compute.capture`.

These exercise the log-capture handler and the ``SynchronousExecutionHooks``
seam in isolation --- no Neo4j, no network, no real ``ProtocolUnit`` execution.
We drive the hooks by hand (calling ``on_unit_attempt_start`` /
``on_unit_attempt_end`` / ``on_progress`` directly) and inspect the resulting
``unit_logs`` and the ``gufekey`` logger's handler set.
"""

import logging

import pytest

from gufe.tokenization import GufeKey

from alchemiscale.compute.capture import (
    GUFEKEY_LOGGER_NAME,
    GufeKeyLogHandler,
    SynchronousExecutionHooks,
)
from alchemiscale.models import ScopedKey
from alchemiscale.storage.models import ComputeServiceID

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


class FakeProgressCallback:
    """Records every ``(task, units_completed, units_total)`` call."""

    def __init__(self):
        self.calls = []

    def __call__(self, task, units_completed, units_total):
        self.calls.append((task, units_completed, units_total))


class DummyResult:
    """Minimal stand-in for a ``ProtocolUnitResult`` --- only exposes ``.key``."""

    def __init__(self, key):
        self.key = key


@pytest.fixture
def task():
    return ScopedKey.from_str("Task-x-o-c-p")


@pytest.fixture
def compute_service_id():
    return ComputeServiceID("svc-" + "0" * 32)


@pytest.fixture
def progress_callback():
    return FakeProgressCallback()


@pytest.fixture
def gufekey_logger():
    return logging.getLogger(GUFEKEY_LOGGER_NAME)


@pytest.fixture(autouse=True)
def _restore_gufekey_logger(gufekey_logger):
    """Snapshot/restore the process-global ``gufekey`` logger around each test.

    The hooks mutate this logger's level and handler set; keep tests hermetic.
    """
    saved_level = gufekey_logger.level
    saved_handlers = list(gufekey_logger.handlers)
    try:
        yield
    finally:
        gufekey_logger.setLevel(saved_level)
        gufekey_logger.handlers[:] = saved_handlers


def _make_hooks(task, compute_service_id, progress_callback, **kwargs):
    return SynchronousExecutionHooks(
        task=task,
        compute_service_id=compute_service_id,
        progress_callback=progress_callback,
        **kwargs,
    )


def _emit(logger_name, msg, *, gufekey=None):
    """Emit one record on ``logger_name``.

    When ``gufekey`` is given, stamp it onto the record via ``extra`` exactly as
    ``gufe``'s ``_GufeLoggerAdapter`` would; otherwise emit a bare record with no
    ``gufekey`` attribute.
    """
    logger = logging.getLogger(logger_name)
    extra = {"gufekey": gufekey} if gufekey is not None else None
    logger.info(msg, extra=extra)


# a syntactically valid gufe key for labeling captured records
_GUFE_KEY = GufeKey("FakeUnit-" + "a" * 32)


# ---------------------------------------------------------------------------
# GufeKeyLogHandler
# ---------------------------------------------------------------------------


def test_handler_accumulates_lines():
    handler = GufeKeyLogHandler()
    rec = logging.LogRecord(
        name="gufekey.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rec.gufekey = str(_GUFE_KEY)
    handler.emit(rec)
    handler.emit(rec)

    assert len(handler.lines) == 2
    # each formatted line carries the gufe key and message
    for line in handler.lines:
        assert str(_GUFE_KEY) in line
        assert "hello" in line


def test_handler_missing_gufekey_renders_dash_no_raise():
    """A record with no ``gufekey`` attribute must not raise and renders ``[-]``."""
    handler = GufeKeyLogHandler()
    # a bare record, as if ``logging.getLogger("gufekey.x").info(...)`` were
    # called without the GufeTokenizable.logger adapter's stamp
    rec = logging.LogRecord(
        name="gufekey.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="unstamped",
        args=(),
        exc_info=None,
    )
    assert not hasattr(rec, "gufekey")

    handler.emit(rec)  # must not raise

    assert len(handler.lines) == 1
    assert "[-]" in handler.lines[0]
    assert "unstamped" in handler.lines[0]


def test_handler_text_joins_lines():
    handler = GufeKeyLogHandler()
    handler.lines = ["a", "b", "c"]
    assert handler.text() == "a\nb\nc"


def test_handler_text_empty():
    handler = GufeKeyLogHandler()
    assert handler.text() == ""


# ---------------------------------------------------------------------------
# SynchronousExecutionHooks --- handler attach/detach + capture
# ---------------------------------------------------------------------------


def test_start_attaches_end_removes_handler(
    task, compute_service_id, progress_callback, gufekey_logger
):
    baseline = len(gufekey_logger.handlers)
    hooks = _make_hooks(task, compute_service_id, progress_callback)

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    assert len(gufekey_logger.handlers) == baseline + 1
    assert isinstance(hooks._handler, GufeKeyLogHandler)

    hooks.on_unit_attempt_end(unit=None, attempt=0, result=DummyResult(_GUFE_KEY))
    # handler count returns to baseline
    assert len(gufekey_logger.handlers) == baseline
    assert hooks._handler is None


def test_record_between_start_and_end_is_captured(
    task, compute_service_id, progress_callback
):
    hooks = _make_hooks(task, compute_service_id, progress_callback)
    result = DummyResult(_GUFE_KEY)

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    # a record on a gufekey descendant logger, stamped with a real gufe key
    _emit("gufekey.some.Unit", "captured message", gufekey=str(_GUFE_KEY))
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=result)

    stored = hooks.unit_logs[str(result.key)]
    assert "captured message" in stored
    assert str(_GUFE_KEY) in stored


def test_records_before_start_and_after_end_not_captured(
    task, compute_service_id, progress_callback
):
    hooks = _make_hooks(task, compute_service_id, progress_callback)
    result = DummyResult(_GUFE_KEY)

    # before the handler is attached
    _emit("gufekey.some.Unit", "before start", gufekey=str(_GUFE_KEY))

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    _emit("gufekey.some.Unit", "during", gufekey=str(_GUFE_KEY))
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=result)

    # after the handler is removed
    _emit("gufekey.some.Unit", "after end", gufekey=str(_GUFE_KEY))

    stored = hooks.unit_logs[str(result.key)]
    assert "during" in stored
    assert "before start" not in stored
    assert "after end" not in stored


def test_truncation_keeps_tail(task, compute_service_id, progress_callback):
    cap = 200
    hooks = _make_hooks(task, compute_service_id, progress_callback, log_cap_bytes=cap)
    result = DummyResult(_GUFE_KEY)

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    # emit far more than `cap` bytes, with a distinctive marker at the very end
    for i in range(200):
        _emit("gufekey.some.Unit", f"line-{i:05d}-padding", gufekey=str(_GUFE_KEY))
    _emit("gufekey.some.Unit", "THE_VERY_LAST_LINE", gufekey=str(_GUFE_KEY))
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=result)

    stored = hooks.unit_logs[str(result.key)]
    # kept text is the tail and within the cap
    assert len(stored.encode("utf-8")) <= cap
    assert "THE_VERY_LAST_LINE" in stored
    # an early line should have been dropped from the head
    assert "line-00000-padding" not in stored


def test_end_with_none_result_closes_handler_stores_nothing(
    task, compute_service_id, progress_callback, gufekey_logger
):
    baseline = len(gufekey_logger.handlers)
    hooks = _make_hooks(task, compute_service_id, progress_callback)

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    _emit("gufekey.some.Unit", "should be dropped", gufekey=str(_GUFE_KEY))
    # interrupted attempt: result is None
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=None)

    # handler closed (count back to baseline), nothing stored
    assert len(gufekey_logger.handlers) == baseline
    assert hooks._handler is None
    assert hooks.unit_logs == {}


def test_empty_capture_not_stored(task, compute_service_id, progress_callback):
    """A result with no captured text stores no entry (guarded by ``if text``)."""
    hooks = _make_hooks(task, compute_service_id, progress_callback)
    result = DummyResult(_GUFE_KEY)

    hooks.on_unit_attempt_start(unit=None, attempt=0)
    # emit nothing
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=result)

    assert str(result.key) not in hooks.unit_logs


# ---------------------------------------------------------------------------
# SynchronousExecutionHooks --- capture_logs=False disables capture entirely
# ---------------------------------------------------------------------------


def test_capture_disabled_no_handler_no_logs_no_level_mutation(
    task, compute_service_id, progress_callback, gufekey_logger
):
    baseline_handlers = len(gufekey_logger.handlers)
    baseline_level = gufekey_logger.level

    hooks = _make_hooks(task, compute_service_id, progress_callback, capture_logs=False)

    # constructing the hooks must NOT mutate the (process-global) logger level
    assert gufekey_logger.level == baseline_level

    result = DummyResult(_GUFE_KEY)
    hooks.on_unit_attempt_start(unit=None, attempt=0)
    # no handler attached
    assert len(gufekey_logger.handlers) == baseline_handlers
    assert hooks._handler is None

    _emit("gufekey.some.Unit", "ignored", gufekey=str(_GUFE_KEY))
    hooks.on_unit_attempt_end(unit=None, attempt=0, result=result)

    # nothing captured, level still untouched
    assert hooks.unit_logs == {}
    assert gufekey_logger.level == baseline_level


def test_capture_enabled_sets_logger_level(
    task, compute_service_id, progress_callback, gufekey_logger
):
    """With capture on, the ``gufekey`` logger level is set at construction."""
    hooks = _make_hooks(
        task, compute_service_id, progress_callback, gufekey_loglevel=logging.INFO
    )
    assert gufekey_logger.level == logging.INFO


# ---------------------------------------------------------------------------
# SynchronousExecutionHooks --- progress forwarding
# ---------------------------------------------------------------------------


def test_on_progress_forwards_to_callback_with_task(
    task, compute_service_id, progress_callback
):
    hooks = _make_hooks(task, compute_service_id, progress_callback)

    hooks.on_progress(0, 5)
    hooks.on_progress(3, 5)

    assert progress_callback.calls == [
        (task, 0, 5),
        (task, 3, 5),
    ]
