**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* ``AlchemicalNetwork``\s featuring components with attributes that Neo4j cannot store as node properties — such as the ``pint.Quantity`` ``box_vectors`` of a ``SolvatedPDBComponent`` or ``ProteinMembraneComponent`` — can now be created. Previously these failed with ``AlchemiscaleClientError: Status Code 422 : Unprocessable Entity : Values of type <class 'pint.Quantity'> are not supported``; such attributes are now JSON-serialized on write and deserialized on read, as already done for ``dict``, ``list``, and ``tuple`` attributes.

**Security:**

* <news item>
