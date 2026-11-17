===============
CHANGELOG
===============

.. current developments

v0.8.1
====================

**Added:**

* Support for Python 3.14.

**Changed:**

* Environment pins bumped to ``gufe`` 1.13.1, ``openfe`` 1.13.0, and ``openmm`` 8.4.0, the earliest versions installable on Python 3.14.

**Removed:**

* Support for Python 3.11; ``alchemiscale`` now requires Python 3.12 or later.
* The ``libsqlite<3.49`` pin from environment files; ``diskcache`` no longer fails with newer ``libsqlite``.

**Fixed:**

* Registering a compute manager with ``steal=True`` no longer fails with a server error when no manager with that name is already registered.
* ``AlchemiscaleClient`` no longer uses ``nest_asyncio``, which is incompatible with Python 3.14; async requests now run on a shared event loop in a background thread, which also works where an event loop is already running (e.g. Jupyter).
* ``AlchemiscaleClient`` calls overlapping on one client (from several threads, or re-running after an interrupted call) no longer close each other's HTTP session.
* ``AlchemiscaleClient`` calls in a forked child process no longer hang.
* Retry backoff in async ``AlchemiscaleClient`` requests no longer blocks other in-flight requests.


v0.8.0
====================

**Added:**

* ``alchemiscale.base.api.JSONHandlerResponse``, for returning content that is already JSON-ready but needs ``gufe``'s ``JSON_HANDLER`` codecs.
* Compute services now accept a ``scopes_exclude`` setting that filters out
  ``TaskHub``\s in matching ``Scope``\s when claiming tasks; applied as a
  filter after ``scopes``.
* ``AlchemiscaleClient.get_network_archive`` and ``AlchemiscaleClient.get_network_archives`` produce archival-quality ``AlchemicalArchive`` extracts, bundling an ``AlchemicalNetwork`` with all successful ``ProtocolDAGResult``\s for its ``Transformation``\s, with optional user-supplied metadata. Networks that are not found yield ``None``.
* ``AlchemiscaleClient.get_network_results`` and ``AlchemiscaleClient.get_transformation_results`` now take a ``return_as`` keyword argument for getting back a single aggregated ``ProtocolResult``, a list of ``ProtocolResult``\s, or a list of ``ProtocolDAGResult``\s.
* New :mod:`alchemiscale.sleep` module providing the ``InterruptableSleep`` primitive (and ``SleepInterrupted`` exception), shared by the compute service, compute manager, and strategist service.
* ``alchemiscale.compute.signals.install_stop_handlers`` helper for wiring ``SIGHUP``/``SIGINT``/``SIGTERM`` to a service's ``stop()`` from an entry point.
* ``ComputeServiceSettings`` gained ``max_tasks`` and ``max_time`` fields (both ``int | None``, default ``None`` = no limit), folding service lifetime limits into the single settings object.

**Changed:**

* The compute API's ``/tasks/{task_scoped_key}/transformation/gufe`` endpoint now returns the ``Transformation`` in keyed chain form, served directly from the state store. Previously it built the ``Transformation`` and re-serialized it with ``to_dict``, which emits every referenced object once per reference --- both ``ChemicalSystem``\s of a ``Transformation`` in full, including a shared ``ProteinComponent``.
* ``AlchemiscaleComputeClient`` now requests and sends compressed payloads when retrieving ``Transformation``\s and submitting results.

  Compute services must be upgraded alongside the compute API; the two no longer interoperate across this change.
* Docs are now built on Read the Docs with micromamba instead of conda; conda's environment solve exceeded the build's memory limit.
* The compute API ``/claim`` endpoint now consistently pads its response with
  ``None`` up to the requested ``count`` when no eligible ``TaskHub``\s remain
  after scope filtering, matching the behavior of the normal claim path.
* ``InterruptableSleep`` and ``SleepInterrupted`` now live in :mod:`alchemiscale.sleep`; they remain importable from :mod:`alchemiscale.compute.service` for backwards compatibility.
* ``SynchronousComputeService.start()`` no longer takes ``max_tasks`` or ``max_time`` parameters; both are read from the service's settings.
* The ``alchemiscale compute synchronous`` CLI now loads a flat YAML config — all fields live at the top level rather than under ``init:`` / ``start:`` keys.
* ``ComputeServiceID`` now enforces a structured ``{name}-{uuid}`` form (parity with ``ComputeManagerID``). Names must match ``[a-zA-Z][a-zA-Z0-9_.:]*`` — i.e. start with a letter and contain only alphanumerics, underscores, periods, or colons; the ``-`` is reserved as the name/uuid delimiter. UUIDs must be hex form.
* ``ComputeServiceID`` and ``ComputeManagerID`` now share validation via a new ``ComputeIDBase`` base class. Use ``ComputeServiceID.new_from_name(name)`` / ``ComputeManagerID.new_from_name(name)`` to construct fresh IDs.

**Deprecated:**

* The ``return_protocoldagresults`` keyword argument in ``AlchemiscaleClient.get_network_results`` and ``AlchemiscaleClient.get_transformation_results`` is now deprecated.

**Removed:**

* Removed the unused ``sched.scheduler`` instances from ``SynchronousComputeService`` and ``AsynchronousComputeService``.
* The ``init:`` / ``start:`` nested YAML wrapper for compute service configs. Legacy nested configs now fail validation with a clear error (hard cutover).

**Fixed:**

* Greatly reduced compute API memory use and processing time for ``AlchemicalNetwork``\s built on large systems. For a ``ChemicalSystem`` with a ~150k-atom ``ProteinComponent``, serving a ``Transformation`` took roughly seven minutes of CPU and ~380 MB of memory per ``Task`` claim, and sent a 60 MB response; it now does no ``GufeTokenizable`` construction at all and sends ~4 MB.
* Result submission to the compute API no longer runs on the event loop, so a large ``ProtocolDAGResult`` upload no longer blocks every other request served by that worker.
* ``ComputeManager``, ``SynchronousComputeService``, and ``StrategistService`` now use an interruptible sleep between cycles, so ``stop()`` (and termination signals) takes effect promptly instead of waiting for the full sleep interval to elapse. ``SynchronousComputeService`` previously constructed an ``InterruptableSleep`` but never used it for sleeping, leaving termination unresponsive during a sleep.
* ``alchemiscale api`` and ``alchemiscale compute api`` now sign JSON web tokens with the ``JWT_SECRET_KEY`` (or ``--jwt-secret``) they are given, instead of always discarding it and generating a random per-process key. This broke multi-replica deployments and invalidated all tokens on restart (`#519 <https://github.com/OpenFreeEnergy/alchemiscale/issues/519>`_).
* A random key is now only generated when no secret is supplied, and a warning is printed to stderr when this happens.
* ``AlchemicalNetwork``\s featuring components with attributes that Neo4j cannot store as node properties — such as the ``pint.Quantity`` ``box_vectors`` of a ``SolvatedPDBComponent`` or ``ProteinMembraneComponent`` — can now be created. Previously these failed with ``AlchemiscaleClientError: Status Code 422 : Unprocessable Entity : Values of type <class 'pint.Quantity'> are not supported``; such attributes are now JSON-serialized on write and deserialized on read, as already done for ``dict``, ``list``, and ``tuple`` attributes.


v0.7.2
====================

**Added:**

* Added ``Strategist`` service to included ``docker-compose`` deployment
* ``ComputeServiceID`` now includes ``new_from_name`` method for UUID generation

**Changed:**

* Disabled UNRECOGNIZED warnings handling by ``neo4j`` driver, which appear often in normal operation due to the form of our Cypher queries
* ``ComputeManagerID`` now allows underscores (``_``), periods (``.``), and colons (``:``)
* Now using ``traefik`` v2.11 in ``docker-compose`` deployment

**Fixed:**

* Protocol names now validated upon ``Task`` claiming
* Compute managers now respect ``Protocol`` filtering from compute service settings


v0.7.1
====================

**Added:**

* Added commandline-level override of compute service name (#450)
* Added ``Scope``\s validation for ``ComputeServiceSettings``

**Changed:**

* Enabled parallel execution of test suite

**Fixed:**

* Fixed startup issue in ``StrategistService``


v0.7.0
====================

**Added:**

* Compute services that regularly fail will be denied when they request to claim ``Task``s
  * This behavior is controlled through the ``ALCHEMISCALE_COMPUTE_API_FORGIVE_TIME_SECONDS`` and the ``ALCHEMISCALE_COMPUTE_API_MAX_FAILURES`` ``ComputeAPISettings`` attributes. The former determines how long it takes for a failure to be considered forgiven while the latter dictates the number of allowed failures before a denial occurs.
* Support for horizontal autoscaling in ``alchemiscale`` through the new compute manager implementation
  * Compute API endpoints for managing compute managers
  * ``ComputeManager`` base class to facilitate platform specific manager implementation
* Optimizations for submitting ``AlchemicalNetwork`` objects to the Neo4j database
* Optimizations for retrieving ``GufeTokenizable`` objects from the Neo4j database
* Added ``StrategistService`` as a deployable service for automating the performance of user-submitted ``Strategy``\s
* Added ``Strategy`` submission and handling to ``AlchemiscaleClient``

**Changed:**

* Updated ``AlchemiscaleClient.get_transformation_chemicalsystems`` method:
  * Now returns the ``ScopedKeys`` of a given ``Transformation`` ordered by the states they represent; the first ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateA`` and the second ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateB``.
  * If provided with a ``NonTransformation`` ``ScopedKey``, the method returns a list containing only the single key representing the ``system``.
* User Guide now broken up into sub-pages to accommodate length and expansion of user-facing features

**Removed:**

* Removed module for ``strategies`` in ``alchemiscale``


