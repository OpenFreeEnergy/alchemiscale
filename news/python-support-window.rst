**Added:**

* Support for Python 3.14.

**Changed:**

* Environment pins bumped to ``gufe`` 1.13.1, ``openfe`` 1.13.0, and ``openmm`` 8.4.0, the earliest versions installable on Python 3.14.

**Deprecated:**

* <news item>

**Removed:**

* Support for Python 3.11; ``alchemiscale`` now requires Python 3.12 or later.
* The ``libsqlite<3.49`` pin from environment files; ``diskcache`` no longer fails with newer ``libsqlite``.

**Fixed:**

* ``AlchemiscaleClient`` no longer uses ``nest_asyncio``, which is incompatible with Python 3.14; async requests now run on a shared event loop in a background thread, which also works where an event loop is already running (e.g. Jupyter).

**Security:**

* <news item>
