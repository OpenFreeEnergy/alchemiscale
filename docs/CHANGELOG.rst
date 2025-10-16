===============
CHANGELOG
===============

.. current developments

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
  * Now returns the ``ScopedKeys`` of a given ``Transformation`` ordered by the states they represent:
    * The first ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateA``.
    * The second ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateB``.
  * If provided with a ``NonTransformation`` ``ScopedKey``, the method returns a list containing only the single key representing the ``system``.
* User Guide now broken up into sub-pages to accommodate length and expansion of user-facing features

**Removed:**

* Removed module for ``strategies`` in ``alchemiscale``


