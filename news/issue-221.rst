**Added:**

* ``AlchemiscaleClient.merge_networks`` for combining multiple existing ``AlchemicalNetwork``\s into a new ``AlchemicalNetwork`` in a target ``Scope``. Backed by ``Neo4jStore.merge_networks`` and a new ``POST /networks/merge`` endpoint on the user API.
* ``AlchemiscaleClient.copy_network`` for copying a single ``AlchemicalNetwork`` to a new ``Scope``. An optional ``name`` argument creates the copy under a new name (and therefore a new gufe key); when ``name`` is omitted, the source network's name and gufe key are preserved. Backed by ``Neo4jStore.copy_network`` and a new ``POST /networks/{network_scoped_key}/copy`` endpoint.
* ``AlchemiscaleClient.merge_scopes`` for copying every ``AlchemicalNetwork`` in a set of source ``Scope``\s into a single target ``Scope``, via per-network ``copy_network`` calls. Each copy preserves its source network's *state* (``active``, ``inactive``, ``invalid``, or ``deleted``), so this method is a faithful move rather than a silent reactivation of soft-deleted or invalidated networks. It is not transactional, but is safe to rerun after a partial failure since ``copy_network`` is idempotent.
* For all three methods, only ``Task``\s in ``complete`` state on the source networks are cloned into the new network -- along with their ``ProtocolDAGResultRef``\s -- so previously-computed results do not need to be re-run. ``Task``\s in any other status are not carried over. Cloned ``Task``\s are wired to their ``Transformation``\s via ``PERFORMS`` so the standard ``(:AlchemicalNetwork)-[:DEPENDS_ON]->(:Transformation)<-[:PERFORMS]-(:Task)`` traversal sees them on the new network. ``EXTENDS`` edges from a carried ``Task`` are preserved only when the extended parent is itself ``complete`` (and therefore also carried); a carried retry whose parent is ``error``, ``waiting``, ``running``, ``invalid``, or ``deleted`` lands in the target scope without its parent and without a dangling ``EXTENDS`` edge.
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

