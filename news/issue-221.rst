**Added:**

* ``AlchemiscaleClient.merge_networks`` for combining multiple existing ``AlchemicalNetwork``\s into a new ``AlchemicalNetwork`` in a target ``Scope``. Backed by ``Neo4jStore.merge_networks`` and a new ``POST /networks/merge`` endpoint on the user API.
* ``AlchemiscaleClient.copy_network`` for copying a single ``AlchemicalNetwork`` to a new ``Scope``. An optional ``name`` argument creates the copy under a new name (and therefore a new gufe key); when ``name`` is omitted, the source network's name and gufe key are preserved. Backed by ``Neo4jStore.copy_network`` and a new ``POST /networks/{network_scoped_key}/copy`` endpoint.
* ``AlchemiscaleClient.merge_scopes`` for copying every ``active`` ``AlchemicalNetwork`` in a set of source ``Scope``\s into a single target ``Scope``, via per-network ``copy_network`` calls. Networks in ``inactive``, ``invalid``, or ``deleted`` state on the source scopes are skipped entirely, so consolidation never silently reactivates soft-deleted or invalidated networks. It is not transactional, but is safe to rerun after a partial failure since ``copy_network`` is idempotent.
* All three methods operate only on ``AlchemicalNetwork``\s in ``active`` state. ``merge_networks`` and ``copy_network`` raise ``ValueError`` if any source network is in a non-``active`` state; ``merge_scopes`` filters them out.
* For all three methods, only ``Task``\s in ``complete`` state on the source networks are cloned into the new network -- along with their ``ProtocolDAGResultRef``\s -- so previously-computed results do not need to be re-run. ``Task``\s in any other status are not carried over. Cloned ``Task``\s are wired to their ``Transformation``\s via ``PERFORMS`` so the standard ``(:AlchemicalNetwork)-[:DEPENDS_ON]->(:Transformation)<-[:PERFORMS]-(:Task)`` traversal sees them on the new network. Cloned ``Task``\s have their serialized ``extends`` property rewritten to reference the target-scope parent's ``ScopedKey`` (or ``None`` if the parent was not carried), so ``get_transformation_tasks(..., return_as="graph")`` on a copied ``Transformation`` returns a graph entirely in the target scope.
* Any ``Strategy`` (``AlchemiscaleClient.set_network_strategy``) or ``TaskRestartPattern``\s (``AlchemiscaleClient.add_task_restart_patterns``) on the source networks are not carried over by any of the three new methods. Set them explicitly on the new network afterward if needed.

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

