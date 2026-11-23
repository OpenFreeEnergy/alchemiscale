**Added:**

* Clients send their alchemiscale version in an ``X-Alchemiscale-Version`` header with every authenticated request.
* ``AlchemiscaleBaseClient.get_server_version``, which returns the alchemiscale version reported by an API server, or ``None`` for a server too old to report one.
* ``ComputeManager.start`` checks the server version before registering, and raises ``IncompatibleServerError`` naming both versions if the server is older than 0.9.0.

**Changed:**

* ``GET /info`` on the user and compute APIs returns the server's alchemiscale version, as ``{"version": "<version>"}``.
