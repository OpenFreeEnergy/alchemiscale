**Added:**

* ``AlchemiscaleClient.merge_networks`` for combining multiple existing ``AlchemicalNetwork``\s into a new ``AlchemicalNetwork`` in a target ``Scope``. ``Task``\s in ``complete`` or ``error`` state on the source networks are cloned into the new network along with their ``ProtocolDAGResultRef``\s so previously-computed results do not need to be re-run; ``Task``\s in other statuses (``waiting``, ``running``, ``invalid``, ``deleted``) are not carried over. Backed by ``Neo4jStore.merge_networks`` and a new ``POST /networks/merge`` endpoint on the user API.
* ``AlchemiscaleClient.copy_network`` for copying a single ``AlchemicalNetwork`` to a new ``Scope``, with all of its ``Task``\s (regardless of status) and their ``ProtocolDAGResultRef``\s carried over. An optional ``name`` argument creates the copy under a new name (and therefore a new gufe key); when ``name`` is omitted, the source network's name and gufe key are preserved. Backed by ``Neo4jStore.copy_network`` and a new ``POST /networks/{network_scoped_key}/copy`` endpoint.
* ``AlchemiscaleClient.merge_scopes`` for copying every ``AlchemicalNetwork`` in a set of source ``Scope``\s into a single target ``Scope``, via per-network ``copy_network`` calls.
* For all three methods, cloned ``Task``\s are wired to their ``Transformation``\s via ``PERFORMS`` so the standard ``(:AlchemicalNetwork)-[:DEPENDS_ON]->(:Transformation)<-[:PERFORMS]-(:Task)`` traversal sees them on the new network. Cloned ``Task``\s are intentionally **not** actioned to the new network's ``TaskHub``; callers wanting cloned ``Task``\s picked up by compute services should call ``action_tasks`` against the new network's ``TaskHub`` after the copy completes.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>

