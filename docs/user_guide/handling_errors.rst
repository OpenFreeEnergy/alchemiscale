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


****************************
Getting tracebacks in bulk
****************************

Pulling down full :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_transformation_failures` (as above) transfers every failed result object, which can be slow when you only want to read the exceptions.
When your goal is fast failure triage for a single :py:class:`~alchemiscale.storage.models.Task`, use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_tracebacks` instead.
It returns only the traceback text, one :py:class:`~alchemiscale.storage.models.TaskTracebacks` record per failed :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult` of that :py:class:`~alchemiscale.storage.models.Task`, most recent first::

    >>> task: ScopedKey
    >>> for attempt in asc.get_task_tracebacks(task):
    >>>     print(attempt.protocoldagresultref, attempt.datetime_created)
    >>>     for unit in attempt.tracebacks:
    >>>         print(unit.source_key)
    >>>         print(unit.traceback)

Each :py:class:`~alchemiscale.storage.models.TaskTracebacks` carries the :py:class:`~alchemiscale.models.ScopedKey` of the failed result (``protocoldagresultref``) and a list of per-:external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnitFailure` tracebacks; each of those carries the ``source_key`` of the failing :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit` and its ``traceback`` string.

If a :py:class:`~alchemiscale.storage.models.Task` has been attempted many times, you can limit the number of failed results returned to just the most recent ones with the ``limit`` keyword argument::

    >>> # only the tracebacks from the most recent failed result
    >>> asc.get_task_tracebacks(task, limit=1)


*******************************
Drilling into per-unit logs
*******************************

Tracebacks tell you *where* a :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit` failed, but often the surrounding logs and captured stdout/stderr are what tell you *why*.
These artifacts are captured per :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit` at execution time (see :ref:`compute` for the compute-side settings that control capture), and you can drill into them without transferring the full result objects.

Start from a :py:class:`~alchemiscale.storage.models.Task` and list records describing its :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_result_recs`.
Pass ``ok=False`` to look only at failures, ``ok=True`` for successes, or leave it unset for all::

    >>> task: ScopedKey
    >>> pdrrs = asc.get_task_result_recs(task, ok=False)
    >>> pdrrs
    [<ProtocolDAGResultRec scoped_key=... ok=False>, ...]

Each :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRec` carries the :py:class:`~alchemiscale.models.ScopedKey` of the underlying result as its ``scoped_key`` attribute.
For a given record, list the per-unit records with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_recs`, then pull the captured artifacts for any unit of interest with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_logs`, :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_stdout`, and :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_stderr`::

    >>> for pdrr in pdrrs:
    >>>     for purr in asc.get_result_unit_recs(pdrr):
    >>>         # skip units with nothing captured
    >>>         if purr.has_logs:
    >>>             print(asc.get_result_unit_logs(purr))
    >>>         if purr.has_stdout:
    >>>             print(asc.get_result_unit_stdout(purr))
    >>>         if purr.has_stderr:
    >>>             print(asc.get_result_unit_stderr(purr))

Each :py:class:`~alchemiscale.storage.models.ProtocolUnitResultRec` exposes ``has_logs``, ``has_stdout``, and ``has_stderr`` flags so you can skip units that captured nothing.
:py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_logs` returns the captured log text as a single string (or ``None``), while :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_stdout` and :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_unit_stderr` return a mapping of filename to captured text (or ``None``).

.. note::
   The drill-down methods accept either a :py:class:`~alchemiscale.models.ScopedKey` or the corresponding record object (a :py:class:`~alchemiscale.storage.models.ProtocolDAGResultRec` for ``pdrr`` arguments, a :py:class:`~alchemiscale.storage.models.ProtocolUnitResultRec` for ``purr`` arguments), so the chain above composes naturally.
   The same :py:class:`~alchemiscale.models.ScopedKey`\s can be copied between Python sessions.

When you don't need per-unit granularity, three convenience methods render everything for you.
:py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_result_logs` returns a single human-readable rendering of all unit logs for one :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`; the default ``order='unit'`` groups each unit's logs under a header, while ``order='time'`` interleaves all units' log lines by timestamp::

    >>> pdrr = asc.get_task_result_recs(task, ok=False)[0]
    >>> print(asc.get_result_logs(pdrr))
    >>> # or interleave across units by timestamp
    >>> print(asc.get_result_logs(pdrr, order='time'))

:py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_stdout` and :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_stderr` go one level higher, concatenating the captured stdout/stderr across *all* of a :py:class:`~alchemiscale.storage.models.Task`\'s :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s (most recent first), with section headers identifying each result, unit, and filename::

    >>> print(asc.get_task_stdout(task))
    >>> print(asc.get_task_stderr(task))

Each returns ``""`` when nothing was captured.

.. note::
   Logs and stream capture are opt-in on the compute side and depend on what each :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` chooses to emit and archive.
   If a :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` writes nothing to its per-unit stdout/stderr, or logs nothing through :external+gufe:py:attr:`~gufe.protocols.protocolunit.ProtocolUnit.logger`, these methods will return empty results even for a :py:class:`~alchemiscale.storage.models.Task` that failed.
   See :ref:`compute` for details on the capture mechanism and the settings that govern it.


********************************************
The reason field and DAG-creation failures
********************************************

Not every failure produces a :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult` with tracebacks to inspect.
Before any :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit` runs, the compute service must first build the :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAG` for a :py:class:`~alchemiscale.storage.models.Task` from its :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`.
If that construction itself raises, there is no result to store; instead, the :py:class:`~alchemiscale.storage.models.Task` is set directly to ``error``, and the traceback is recorded on the ``reason`` field of the :py:class:`~alchemiscale.storage.models.Task`.

You can read this ``reason`` back through :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_details` (bulk) or :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_history` (per-attempt); both are covered in :ref:`introspection`::

    >>> (detail,) = asc.get_tasks_details([task])
    >>> print(detail.status)   # 'error'
    >>> print(detail.reason)   # the DAG-creation traceback

.. warning::
   A :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAG`-creation failure is treated as a *systematic* problem with the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, not a transient one.
   Such :py:class:`~alchemiscale.storage.models.Task`\s are **not** eligible for automatic retry via :py:class:`~alchemiscale.storage.models.Task` restart patterns (see below), since re-running them would fail again in exactly the same way.
   Resolve the underlying problem with the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` before setting these :py:class:`~alchemiscale.storage.models.Task`\s back to ``waiting``.

Finally, when you mark :py:class:`~alchemiscale.storage.models.Task`\s ``invalid`` or ``deleted`` (see below), you can attach your own ``reason`` for the record::

    >>> asc.set_tasks_status(tasks, 'invalid', reason='superseded by re-parameterized transformation')

The ``reason`` is recorded only for ``invalid`` and ``deleted`` transitions, and surfaces through :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_details`.


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
