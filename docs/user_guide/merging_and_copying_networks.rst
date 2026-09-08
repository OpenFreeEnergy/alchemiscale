.. _merging-and-copying-networks:

######################################
Merging and copying AlchemicalNetworks
######################################

Once you have :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s submitted with :py:class:`~alchemiscale.storage.models.Task`\s and results attached, you may want to combine several of them into a single :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, or move an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` (or a whole :py:class:`~alchemiscale.models.Scope` of them) into a different :py:class:`~alchemiscale.models.Scope`.
Three :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` methods support these workflows:

* :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_networks` — combine several :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s into a single new one.
* :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` — copy one :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` into a different :py:class:`~alchemiscale.models.Scope`.
* :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_scopes` — copy every :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` from one or more source :py:class:`~alchemiscale.models.Scope`\s into a single target :py:class:`~alchemiscale.models.Scope`.

The destination :py:class:`~alchemiscale.models.Scope` for each of these must be a *specific* :py:class:`~alchemiscale.models.Scope` (no wildcards), and your user must have permissions on the source and destination :py:class:`~alchemiscale.models.Scope`\s.

All three methods operate only on :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s in ``active`` state.
:py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_networks` and :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` reject a non-``active`` source with an error; :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_scopes` silently skips :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s in ``inactive``, ``invalid``, or ``deleted`` state.
This keeps consolidation intentional: soft-deleted or invalidated networks are not silently reactivated on the target :py:class:`~alchemiscale.models.Scope`.

All three methods also share the same Task-retention policy: **only** :py:class:`~alchemiscale.storage.models.Task`\s in ``complete`` status carry over to the new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, together with their :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRef`\s.
:py:class:`~alchemiscale.storage.models.Task`\s in any other status (``waiting``, ``running``, ``error``, ``invalid``, ``deleted``) are not carried over.
This keeps the semantics simple: **the results of computed work are preserved; nothing else is**.

.. note::

   Any ``Strategy`` (via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.set_network_strategy`) or ``TaskRestartPattern``\s (via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.add_task_restart_patterns`) on the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s are not copied.
   Set these on the new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` yourself after the merge or copy if you need them.


***********************************
Merging multiple AlchemicalNetworks
***********************************

Use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_networks` to combine multiple :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s into a single new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` in the target :py:class:`~alchemiscale.models.Scope`::

    >>> merged_sk = asc.merge_networks(
    ...     networks=[an_sk_a, an_sk_b, an_sk_c],
    ...     name='combined_campaign',
    ...     scope=Scope('my_org', 'my_campaign', 'combined_project'),
    ... )
    >>> merged_sk
    <ScopedKey('AlchemicalNetwork-8f...-my_org-my_campaign-combined_project')>

The new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` contains the union of the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s and :external+gufe:py:class:`~gufe.transformations.transformation.NonTransformation`\s from every source network.
The source networks themselves are unchanged.

The ``complete`` :py:class:`~alchemiscale.storage.models.Task`\s attached to those source :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s are cloned onto the merged network's :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s along with their :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRef`\s, so previously-computed results do not need to be re-run.

If you want to add fresh :py:class:`~alchemiscale.storage.models.Task`\s to the merged network -- either to retry a :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` that previously errored, or to gather more repeats -- create new ones with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.create_tasks` on the relevant :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s and then :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.action_tasks` them on the merged network.


**********************************
Copying a single AlchemicalNetwork
**********************************

Use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` to copy one :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` into a different :py:class:`~alchemiscale.models.Scope`::

    >>> copy_sk = asc.copy_network(
    ...     network=an_sk,
    ...     scope=Scope('my_org', 'my_campaign', 'shared_project'),
    ... )

Like :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_networks`, only the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\'s ``complete`` :py:class:`~alchemiscale.storage.models.Task`\s are carried over, together with their :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRef`\s.

If ``name`` is not given, the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\'s name is preserved.
Pass ``name`` to rename the copy::

    >>> renamed_copy_sk = asc.copy_network(
    ...     network=an_sk,
    ...     scope=Scope('my_org', 'my_campaign', 'shared_project'),
    ...     name='shared_snapshot',
    ... )


*********************************************************
Copying every AlchemicalNetwork from one Scope to another
*********************************************************

Use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_scopes` to copy every :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` in one or more source :py:class:`~alchemiscale.models.Scope`\s into a single target :py:class:`~alchemiscale.models.Scope`::

    >>> new_sks = asc.merge_scopes(
    ...     scopes=[Scope('my_org', 'my_campaign', 'project_a'),
    ...             Scope('my_org', 'my_campaign', 'project_b')],
    ...     target_scope=Scope('my_org', 'my_campaign', 'consolidated'),
    ... )
    >>> len(new_sks)
    17

This is a convenience method: each ``active`` :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` in a source :py:class:`~alchemiscale.models.Scope` is copied via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` in turn, with its name preserved.
Networks in ``inactive``, ``invalid``, or ``deleted`` state on a source :py:class:`~alchemiscale.models.Scope` are skipped entirely.
It is useful for consolidating the *live* :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s from multiple :py:class:`~alchemiscale.models.Scope`\s into a shared workspace, or for relocating a :py:class:`~alchemiscale.models.Scope`\'s ``active`` :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.

Note that :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_scopes` is not transactional: if a per-network copy fails partway through, the copies that already succeeded remain in the target :py:class:`~alchemiscale.models.Scope`.
Because :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` is idempotent -- re-copying an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` that already exists in the target :py:class:`~alchemiscale.models.Scope` dedups onto the existing node -- rerunning after a failure is safe.
