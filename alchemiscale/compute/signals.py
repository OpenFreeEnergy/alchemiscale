"""
:mod:`alchemiscale.compute.signals` --- graceful shutdown signal handling
==========================================================================

Helpers for wiring OS signals to the ``stop()`` method of a long-running
service or compute manager.

Signal disposition is process-global and :func:`signal.signal` may only be
called from the main thread, so installing handlers is the responsibility of
the *entry point* (a CLI command) that owns the process --- not of the library
classes themselves. This module provides that wiring as a single reusable call
so each entry point does not reimplement (and risk forgetting) it.

"""

import signal
from typing import Protocol


class Stoppable(Protocol):
    """Anything exposing an idempotent ``stop()`` method."""

    def stop(self) -> None: ...


#: Signals that, by default, trigger a graceful shutdown.
DEFAULT_STOP_SIGNALS: tuple[str, ...] = ("SIGHUP", "SIGINT", "SIGTERM")


def install_stop_handlers(
    stoppable: Stoppable,
    signal_names: tuple[str, ...] = DEFAULT_STOP_SIGNALS,
    *,
    raise_keyboard_interrupt: bool = True,
) -> None:
    """Install signal handlers that gracefully stop ``stoppable``.

    For each name in ``signal_names``, registers a handler that calls
    ``stoppable.stop()`` and (by default) then raises
    :class:`KeyboardInterrupt`. The ``stop()`` call tells the object's main
    loop to wind down (waking any interruptible sleep), while the raised
    :class:`KeyboardInterrupt` unwinds the main thread so the loop's
    shutdown/deregistration logic runs promptly.

    Must be called from the main thread; :func:`signal.signal` raises
    otherwise. Signal names not present on the current platform are skipped.

    Parameters
    ----------
    stoppable
        An object exposing a ``stop()`` method, e.g. a
        :class:`~alchemiscale.compute.service.SynchronousComputeService` or a
        :class:`~alchemiscale.compute.manager.ComputeManager`.
    signal_names
        Names of the signals to handle; defaults to
        :data:`DEFAULT_STOP_SIGNALS`.
    raise_keyboard_interrupt
        If ``True`` (default), the handler also raises
        :class:`KeyboardInterrupt` after calling ``stop()``. Set ``False``
        when the caller relies on cooperative shutdown only and a
        :class:`KeyboardInterrupt` would interfere with teardown --- e.g. a
        service whose ``stop()`` triggers a
        :class:`~concurrent.futures.ProcessPoolExecutor` shutdown that must
        not be interrupted partway through.
    """

    def handler(signum, frame):
        stoppable.stop()
        if raise_keyboard_interrupt:
            raise KeyboardInterrupt()

    for name in signal_names:
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, handler)
