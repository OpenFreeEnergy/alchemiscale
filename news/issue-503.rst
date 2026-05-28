**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* Removed the unused ``sched.scheduler`` instances from ``SynchronousComputeService`` and ``AsynchronousComputeService``.

**Fixed:**

* ``ComputeManager`` and ``SynchronousComputeService`` now use an interruptible sleep between cycles, so ``stop()`` (and SIGINT) takes effect immediately instead of waiting for the full sleep interval to elapse.

**Security:**

* <news item>
