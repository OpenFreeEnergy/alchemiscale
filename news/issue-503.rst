**Added:**

* New :mod:`alchemiscale.sleep` module providing the ``InterruptableSleep`` primitive (and ``SleepInterrupted`` exception), shared by the compute service, compute manager, and strategist service.
* ``alchemiscale.compute.signals.install_stop_handlers`` helper for wiring ``SIGHUP``/``SIGINT``/``SIGTERM`` to a service's ``stop()`` from an entry point.

**Changed:**

* ``InterruptableSleep`` and ``SleepInterrupted`` now live in :mod:`alchemiscale.sleep`; they remain importable from :mod:`alchemiscale.compute.service` for backwards compatibility.

**Deprecated:**

* <news item>

**Removed:**

* Removed the unused ``sched.scheduler`` instances from ``SynchronousComputeService`` and ``AsynchronousComputeService``.

**Fixed:**

* ``ComputeManager``, ``SynchronousComputeService``, and ``StrategistService`` now use an interruptible sleep between cycles, so ``stop()`` (and termination signals) takes effect promptly instead of waiting for the full sleep interval to elapse. ``SynchronousComputeService`` previously constructed an ``InterruptableSleep`` but never used it for sleeping, leaving termination unresponsive during a sleep.

**Security:**

* <news item>
