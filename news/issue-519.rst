**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* ``alchemiscale api`` and ``alchemiscale compute api`` now sign JSON web tokens with the ``JWT_SECRET_KEY`` (or ``--jwt-secret``) they are given, instead of always discarding it and generating a random per-process key. This broke multi-replica deployments and invalidated all tokens on restart (`#519 <https://github.com/OpenFreeEnergy/alchemiscale/issues/519>`_).
* A random key is now only generated when no secret is supplied, and a warning is printed to stderr when this happens.

**Security:**

* <news item>
