**Fixed:**

* A ``ProtocolDAG`` creation failure no longer kills the compute service. The affected ``Task`` is now set to ``error`` with an explanatory ``reason``, and the service continues running.
