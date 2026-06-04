**Added:**

* Compute services now accept a ``scopes_exclude`` setting that filters out
  ``TaskHub``\s in matching ``Scope``\s when claiming tasks; applied as a
  filter after ``scopes``.

**Changed:**

* The compute API ``/claim`` endpoint now consistently pads its response with
  ``None`` up to the requested ``count`` when no eligible ``TaskHub``\s remain
  after scope filtering, matching the behavior of the normal claim path.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
