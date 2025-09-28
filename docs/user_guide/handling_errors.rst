.. _handling-errors:

###############
Handling Errors
###############

If you observe many errored :py:class:`~alchemiscale.storage.models.Task`\s from running :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_transformation_status`, you can introspect the traceback raised by the :py:class:`~alchemiscale.storage.models.Task` on execution.
For a given :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, you can pull down all failed results and print their exceptions and tracebacks with::

    >>> # failed_protocol_dag_results : List[ProtocolDAGResult]
    >>> failed_protocol_dag_results = asc.get_transformation_failures(tf_sk)
    >>> 
    >>> for failure in failed_protocol_dag_results:
    >>>     for failed_unit in failure.protocol_unit_failures:
    >>>         print(failed_unit.exception)
    >>>         print(failed_unit.traceback)

This may give you clues as to what is going wrong with your :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s.
A failure may be a symptom of the environments the compute services are running with; it could also indicate some fundamental problems with the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s you are attempting to execute, and in this case trying to reproduce the error locally and experimenting with possible solutions is appropriate.
You may want to try different :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` settings, different ``Mapping``\s, or try to adjust the components in your :external+gufe:py:class:`~gufe.chemicalsystem.ChemicalSystem`\s.

For a given :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, you can execute it locally with::

    >>> from gufe.protocols import execute_DAG
    >>> from pathlib import Path
    >>> 
    >>> transformation = asc.get_transformation(tf_sk)
    >>> protocol_dag = transformation.create()
    >>> 
    >>> testdir = Path('transformation-test/')
    >>> testdir.mkdir(exist_ok=True)
    >>> 
    >>> protocol_dag_result = execute_DAG(protocol_dag, 
    >>>                                   shared_basedir=testdir,
    >>>                                   scratch_basedir=testdir)
    >>>                                   
    >>> protocol_result = transformation.gather([protocol_dag_result])
    >>> protocol_result.get_estimate()
    >>> protocol_result.get_uncertainty()

Note that for some :external+gufe:py:class:`~gufe.protocols.protocol.Protocol`\s, your local machine may need to meet certain requirements:

* :py:class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol`: NVIDIA GPU if ``settings.platform == 'CUDA'``
* :py:class:`~feflow.protocols.nonequilibrium_cycling.NonEquilibriumCyclingProtocol`: OpenEye Toolkit license, NVIDIA GPU if ``settings.platform == 'CUDA'``

************************
Re-running errored Tasks
************************

If you believe an errored :py:class:`~alchemiscale.storage.models.Task` is due to a random failure (such as landing on a flaky compute host, or due to inherent stochasticity in the :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` itself), or due to a systematic failure that has been resolved (such as a misconfigured compute environment, now remediated), you can choose to set that :py:class:`~alchemiscale.storage.models.Task`\'s status back to ``'waiting'``.
This will make it eligible for being claimed and executed again, perhaps succesfully.

Given a set of :py:class:`~alchemiscale.storage.models.Task`\s you wish to set back to ``'waiting'``, you can do::

    >>> asc.set_tasks_status(tasks, 'waiting')

Only :py:class:`~alchemiscale.storage.models.Task`\s with status ``'error'`` or ``'running'`` can be set back to ``'waiting'``; it is not possible to set :py:class:`~alchemiscale.storage.models.Task`\s with status ``'complete'``, ``'invalid'``, or ``'deleted'`` back to ``'waiting'``.

If you’re feeling confident, you could set all errored :py:class:`~alchemiscale.storage.models.Task`\s on a given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` with::

    >>> # first, get all tasks associated with network with status 'error'
    >>> tasks = asc.get_network_tasks(an_sk, status='error')
    >>> 
    >>> # set all these tasks to status 'waiting'
    >>> asc.set_tasks_status(tasks, 'waiting')
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]

***************************************************
Re-running Errored Tasks with Task Restart Patterns
***************************************************

Re-running errored :py:class:`~alchemiscale.storage.models.Task`\s manually for known failure modes (such as those described in the previous section) quickly becomes tedious, especially for large networks.
Alternatively, you can add `regular expression <https://en.wikipedia.org/wiki/Regular_expression>`_ strings as :py:class:`~alchemiscale.storage.models.Task` restart patterns to an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.
:py:class:`~alchemiscale.storage.models.Task`\s actioned on that :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` will be automatically restarted if the :py:class:`~alchemiscale.storage.models.Task` fails during any part of its execution, provided that an enforcing pattern matches a traceback within the :py:class:`~alchemiscale.storage.models.Task`\'s failed :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`.
The number of restarts is controlled by the ``num_allowed_restarts`` parameter of the :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.add_task_restart_patterns` method.
If a :py:class:`~alchemiscale.storage.models.Task` is restarted more than ``num_allowed_restarts`` times, the :py:class:`~alchemiscale.storage.models.Task` is canceled on that :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` and left in an ``error`` status.

As an example, if you wanted to rerun any :py:class:`~alchemiscale.storage.models.Task` that failed with a ``RuntimeError`` or a ``MemoryError`` and attempt it at most 5 times, you could add the following patterns::

  >>> asc.add_task_restart_patterns(an_sk, [r"RuntimeError: .+", r"MemoryError: Unable to allocate \d+ GiB"], 5)

Providing too general a pattern, such as the example above, you may consume compute resources on failures that are unavoidable.
On the other hand, an overly strict pattern (such as specifying explicit ``gufe`` keys) will likely do nothing.
Therefore, it is best to find a balance in your patterns that matches your use case.

Restart patterns enforcing an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` can be retrieved with::

  >>> asc.get_task_restart_patterns(an_sk)
  {"RuntimeError: .+": 5, "MemoryError: Unable to allocate \d+ GiB": 5}

The number of allowed restarts can also be modified::

  >>> asc.set_task_restart_patterns_allowed_restarts(an_sk, ["RuntimeError: .+"], 3)
  >>> asc.set_task_restart_patterns_allowed_restarts(an_sk, ["MemoryError: Unable to allocate \d+ GiB"], 2)
  >>> asc.get_task_restart_patterns(an_sk)
  {"RuntimeError: .+": 3, "MemoryError: Unable to allocate \d+ GiB": 2}

Patterns can be removed by specifying the patterns in a list::

  >>> asc.remove_task_restart_patterns(an_sk, ["MemoryError: Unable to allocate \d+ GiB"])
  >>> asc.get_task_restart_patterns(an_sk)
  {"RuntimeError: .+": 3}

Or by clearing all enforcing patterns::

  >>> asc.clear_task_restart_patterns(an_sk)
  >>> asc.get_task_restart_patterns(an_sk)
  {}


***********************************
Marking Tasks as deleted or invalid
***********************************

If you created many :py:class:`~alchemiscale.storage.models.Task`\s that are problematic, perhaps because they will always fail, would give scientifically dubious results, or are otherwise unwanted, you can choose to set their status to either ``invalid`` or ``deleted``.
Although technically equivalent, ``invalid`` :py:class:`~alchemiscale.storage.models.Task`\s are ones that have a known problem that you wish to mark as such, while ``deleted`` :py:class:`~alchemiscale.storage.models.Task`\s are marked as fair game for removal by the administrator at a future time.
Setting a :py:class:`~alchemiscale.storage.models.Task` to either of these statuses will automatically cancel them from any and all :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s they are actioned on, so choosing one of these statuses is the easiest way to ensure no compute is wasted on a :py:class:`~alchemiscale.storage.models.Task` you no longer want results for.

You can set any :py:class:`~alchemiscale.storage.models.Task` you create to either ``invalid`` or ``deleted``, although once a :py:class:`~alchemiscale.storage.models.Task` is set to either of these statuses, it cannot be changed to another.
To set a number of :py:class:`~alchemiscale.storage.models.Task`\s to ``invalid``::

    >>> asc.set_tasks_status(tasks, 'invalid')
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]


Or instead to ``deleted``::

    >>> asc.set_tasks_status(tasks, 'deleted')
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]


***********************************************************
Marking AlchemicalNetworks as inactive, deleted, or invalid
***********************************************************

Over time, you may find that the number of :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s in the :py:class:`~alchemiscale.models.Scope`\s you have access to is becoming difficult to manage, with many no longer relevant to the work you are currently doing.
By default, new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s are set to an ``active`` state, but you can change this to any one of ``inactive``, ``deleted``, or ``invalid``, similar to statuses for :py:class:`~alchemiscale.storage.models.Task`\s detailed previously.

Unlike :py:class:`~alchemiscale.storage.models.Task` statuses, all :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` states are reversible, and currently only serve as a way for users to disable default visibility in :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.query_networks` and :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_scope_status`.
Semantically, ``inactive`` is for networks that are no longer of interest, ``deleted`` is for networks that are marked as fair game for deletion by an administrator, and ``invalid`` is for networks that have a known problem and are not expected to give reasonable results.

To get the current state of an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, you can use :meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_network_state`::

    >>> asc.get_network_state(an_sk)
    'active'

We can likewise set its state to e.g. ``inactive`` with::

    >>> asc.set_network_state(an_sk, 'inactive')
    <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>

Subsequent use of :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.query_networks` shows only ``active`` networks by default, but you can show all networks regardless of state by setting ``state=None``::

    >>> asc.query_networks(state=None)
    [<ScopedKey('AlchemicalNetwork-4617c8d8d6599124af3b4561b8d910a0-my_org-my_campaign-my_project')>,
     <ScopedKey('AlchemicalNetwork-d90bd97079cd965b887b373307ea7bab-my_org-my_campaign-my_project')>,
     <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>
     ...]

Likewise, :py:class:`~alchemiscale.storage.models.Task` status counts over whole :py:class:`~alchemiscale.models.Scope`\s obtained from :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_scope_status` by default counts only :py:class:`~alchemiscale.storage.models.Task`\s that are associated with at least one ``active`` network, but we can disregard network state by setting ``network_state=None``::

    >>> asc.get_scope_status(Scope('my_org', 'my_campaign'), network_state=None)
    {'complete': 324,
     'error': 37,
     'invalid': 6,
     'deleted': 13,
     'waiting': 372,
     'running': 66}

Both of the above methods can take any valid network state (``active``, ``inactive``, ``deleted``, or ``invalid``) to filter down to only networks with the matching state.
They can also take regular expressions (regexes), allowing you to filter for multiple states at once with e.g. ``inactive|active``.
