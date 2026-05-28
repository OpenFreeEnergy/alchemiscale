**Added:**

* ``ComputeServiceSettings`` gained ``max_tasks`` and ``max_time`` fields (both ``int | None``, default ``None`` = no limit), folding service lifetime limits into the single settings object.

**Changed:**

* ``SynchronousComputeService.start()`` no longer takes ``max_tasks`` or ``max_time`` parameters; both are read from the service's settings.
* The ``alchemiscale compute synchronous`` CLI now loads a flat YAML config — all fields live at the top level rather than under ``init:`` / ``start:`` keys.

**Deprecated:**

* <news item>

**Removed:**

* The ``init:`` / ``start:`` nested YAML wrapper for compute service configs. Legacy nested configs now fail validation with a clear error (hard cutover).

**Fixed:**

* <news item>

**Security:**

* <news item>

