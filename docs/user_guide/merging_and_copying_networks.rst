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

.. note::

   For all three methods, the following execution-orchestration state is intentionally **not** carried over from the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s to the new one, since these govern *how* :py:class:`~alchemiscale.storage.models.Task`\s run rather than the results themselves:

   * Any :py:class:`~alchemiscale.storage.models.Task`\s that were previously *actioned* on a source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` are not automatically actioned on the new one; call :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.action_tasks` on the new network afterward for any :py:class:`~alchemiscale.storage.models.Task`\s you want compute services to pick up.
   * Any ``Strategy`` set via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.set_network_strategy` is not copied.
   * Any ``TaskRestartPattern``\s added via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.add_task_restart_patterns` are not copied.

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

The new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` contains the union of the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s and ``NonTransformation``\s from every source network.
The source networks themselves are unchanged.

Existing :py:class:`~alchemiscale.storage.models.Task`\s attached to those source :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s that are in ``complete`` or ``error`` state are cloned onto the merged network's :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s along with their :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRef`\s, so previously-computed results do not need to be re-run.
:py:class:`~alchemiscale.storage.models.Task`\s in other states (``waiting``, ``running``, ``invalid``, ``deleted``) are **not** carried over.

To retry the cloned ``error`` :py:class:`~alchemiscale.storage.models.Task`\s on the merged network, first set their status back to ``waiting`` with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.set_tasks_status`, then :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.action_tasks` them::

    >>> errored = asc.get_network_tasks(merged_sk, status='error')
    >>> asc.set_tasks_status(errored, 'waiting')
    >>> asc.action_tasks(errored, merged_sk)


**********************************
Copying a single AlchemicalNetwork
**********************************

Use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` to copy one :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` into a different :py:class:`~alchemiscale.models.Scope`::

    >>> copy_sk = asc.copy_network(
    ...     network=an_sk,
    ...     scope=Scope('my_org', 'my_campaign', 'shared_project'),
    ... )

Unlike :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.merge_networks`, :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` carries over **every** existing :py:class:`~alchemiscale.storage.models.Task` for the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\'s :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s, regardless of status, together with their :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRef`\s.

If ``name`` is not given, the source :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\'s name is preserved and the copy has the same ``gufe.key`` as the source (so its :py:class:`~alchemiscale.models.ScopedKey` differs from the source's only in :py:class:`~alchemiscale.models.Scope`).
Pass ``name`` to rename the copy; this yields a fresh ``gufe.key`` derived from the renamed content::

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

This is a convenience method: each :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` is copied via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.copy_network` in turn, with names preserved.
It is useful for consolidating :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s from multiple :py:class:`~alchemiscale.models.Scope`\s into a shared workspace, or for relocating a whole :py:class:`~alchemiscale.models.Scope`\'s worth of :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.
