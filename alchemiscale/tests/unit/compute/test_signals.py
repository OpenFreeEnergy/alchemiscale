import signal

import pytest

from alchemiscale.compute.signals import (
    DEFAULT_STOP_SIGNALS,
    install_stop_handlers,
)


class StopRecorder:
    """Minimal stoppable that records how many times ``stop()`` was called."""

    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1


@pytest.fixture
def available_signals():
    return [name for name in DEFAULT_STOP_SIGNALS if hasattr(signal, name)]


@pytest.fixture
def restore_handlers(available_signals):
    """Save and restore process signal handlers around a test."""
    originals = {
        name: signal.getsignal(getattr(signal, name)) for name in available_signals
    }
    try:
        yield
    finally:
        for name, original in originals.items():
            signal.signal(getattr(signal, name), original)


def test_install_stop_handlers_registers_each_signal(
    available_signals, restore_handlers
):
    recorder = StopRecorder()

    install_stop_handlers(recorder)

    for name in available_signals:
        handler = signal.getsignal(getattr(signal, name))
        assert callable(handler)


def test_handler_calls_stop_and_raises_keyboard_interrupt(
    available_signals, restore_handlers
):
    recorder = StopRecorder()

    install_stop_handlers(recorder)

    # invoke each registered handler directly (rather than actually raising the
    # signal, which would interfere with the test process)
    for name in available_signals:
        handler = signal.getsignal(getattr(signal, name))
        with pytest.raises(KeyboardInterrupt):
            handler(getattr(signal, name), None)

    # stop() should have been called exactly once per handled signal
    assert recorder.stop_count == len(available_signals)


def test_install_stop_handlers_skips_unknown_signals(restore_handlers):
    recorder = StopRecorder()

    # a bogus signal name must not raise; it is simply skipped
    install_stop_handlers(recorder, signal_names=("SIGINT", "NOT_A_REAL_SIGNAL"))

    handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert recorder.stop_count == 1


def test_handler_without_keyboard_interrupt_calls_stop_only(
    available_signals, restore_handlers
):
    recorder = StopRecorder()

    install_stop_handlers(recorder, raise_keyboard_interrupt=False)

    # invoking each handler should call stop() but NOT raise KeyboardInterrupt
    for name in available_signals:
        handler = signal.getsignal(getattr(signal, name))
        # no pytest.raises: handler must return normally
        handler(getattr(signal, name), None)

    assert recorder.stop_count == len(available_signals)
