**Changed:**

* Compute manager endpoints return a 404 when the manager isn't registered, instead of a 500 or, for the instruction endpoint, a ``SHUTDOWN`` instruction. They return a 422 when the request conflicts with the registration's state.
* ``AlchemiscaleComputeManagerClient`` raises ``ComputeManagerNotRegisteredError`` on a 404, and no longer retries 404 responses.
* ``Neo4jStore.get_compute_manager_id`` and ``get_compute_manager_status`` raise ``ComputeManagerNotFoundError`` instead of returning ``None``.

**Fixed:**

* ``clear_error`` and ``deregister`` for a compute manager that isn't registered, or ``clear_error`` for one without the ``ERROR`` status, no longer fail with a server error.
