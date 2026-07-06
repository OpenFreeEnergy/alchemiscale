**Added:**

* ``AlchemiscaleClient.merge_networks`` for combining multiple existing ``AlchemicalNetwork``\s into a new ``AlchemicalNetwork`` in a target ``Scope``. Backed by ``Neo4jStore.merge_networks`` and a new ``POST /networks/merge`` endpoint on the user API.
* ``AlchemiscaleClient.copy_network`` for copying a single ``AlchemicalNetwork`` to a new ``Scope``. An optional ``name`` argument creates the copy under a new name (and therefore a new gufe key); when ``name`` is omitted, the source network's name and gufe key are preserved. Backed by ``Neo4jStore.copy_network`` and a new ``POST /networks/{network_scoped_key}/copy`` endpoint.
* ``AlchemiscaleClient.merge_scopes`` for copying every ``AlchemicalNetwork`` in a set of source ``Scope``\s into a single target ``Scope``, via per-network ``copy_network`` calls.
* For all three methods, only ``Task``\s in ``complete`` state on the source networks are cloned into the new network -- along with their ``ProtocolDAGResultRef``\s -- so previously-computed results do not need to be re-run. ``Task``\s in any other status (``waiting``, ``running``, ``error``, ``invalid``, ``deleted``) are not carried over. Cloned ``Task``\s are wired to their ``Transformation``\s via ``PERFORMS`` so the standard ``(:AlchemicalNetwork)-[:DEPENDS_ON]->(:Transformation)<-[:PERFORMS]-(:Task)`` traversal sees them on the new network. Cloned ``Task``\s are intentionally **not** actioned to the new network's ``TaskHub``; callers wanting them picked up by compute services should call ``action_tasks`` against the new network's ``TaskHub`` after the copy completes.
* The source networks' execution-orchestration state -- any attached ``Strategy`` (``AlchemiscaleClient.set_network_strategy``) and any ``TaskRestartPattern``\s on the ``TaskHub`` (``AlchemiscaleClient.add_task_restart_patterns``) -- is intentionally **not** carried over by any of the three new methods, since these govern how ``Task``\s run rather than the results themselves. Set them explicitly on the new network afterward if needed.

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

