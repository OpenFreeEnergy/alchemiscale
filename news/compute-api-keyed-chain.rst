**Added:**

* ``alchemiscale.base.api.JSONHandlerResponse``, for returning content that is already JSON-ready but needs ``gufe``'s ``JSON_HANDLER`` codecs.

**Changed:**

* The compute API's ``/tasks/{task_scoped_key}/transformation/gufe`` endpoint now returns the ``Transformation`` in keyed chain form, served directly from the state store. Previously it built the ``Transformation`` and re-serialized it with ``to_dict``, which emits every referenced object once per reference --- both ``ChemicalSystem``\s of a ``Transformation`` in full, including a shared ``ProteinComponent``.
* ``AlchemiscaleComputeClient`` now requests and sends compressed payloads when retrieving ``Transformation``\s and submitting results.

  Compute services must be upgraded alongside the compute API; the two no longer interoperate across this change.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Greatly reduced compute API memory use and processing time for ``AlchemicalNetwork``\s built on large systems. For a ``ChemicalSystem`` with a ~150k-atom ``ProteinComponent``, serving a ``Transformation`` took roughly seven minutes of CPU and ~380 MB of memory per ``Task`` claim, and sent a 60 MB response; it now does no ``GufeTokenizable`` construction at all and sends ~4 MB.
* Result submission to the compute API no longer runs on the event loop, so a large ``ProtocolDAGResult`` upload no longer blocks every other request served by that worker.

**Security:**

* <news item>
