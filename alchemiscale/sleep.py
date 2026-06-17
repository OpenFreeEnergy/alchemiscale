"""
:mod:`alchemiscale.sleep` --- interruptible sleep primitive
===========================================================

A small, dependency-free sleep primitive shared by the long-running services
(compute service, compute manager, strategist) so that a blocking sleep between
work cycles can be woken early from another thread (e.g. a signal handler
calling ``stop()``).

"""

import threading


class SleepInterrupted(BaseException):
    """
    Exception class used to signal that an InterruptableSleep was interrupted

    This (like KeyboardInterrupt) derives from BaseException to prevent
    it from being handled with "except Exception".
    """

    pass


class InterruptableSleep:
    """
    A class for sleeping, but interruptable

    This class uses threading Events to wake up from a sleep before the entire sleep
    duration has run. If the sleep is interrupted, then an SleepInterrupted exception is raised.

    This class is a functor: call an instance with a delay in seconds to sleep for that
    duration (e.g. ``int_sleep(30)``), and call :meth:`interrupt` from another thread to
    wake it early.
    """

    def __init__(self):
        self._event = threading.Event()

    def __call__(self, delay: float):
        interrupted = self._event.wait(delay)
        if interrupted:
            raise SleepInterrupted()

    def interrupt(self):
        self._event.set()

    def clear(self):
        self._event.clear()
