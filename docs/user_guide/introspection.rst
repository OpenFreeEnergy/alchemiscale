.. _introspection:

############################
Understanding Task execution
############################

The status counts described in :ref:`getting-started` tell you *how many* of your :py:class:`~alchemiscale.storage.models.Task`\s are ``waiting``, ``running``, ``error``, or ``complete``, but they don't tell you *what happened* to any one :py:class:`~alchemiscale.storage.models.Task`, *where* it ran, or *how far* a currently-running one has gotten.
This document covers the introspection methods on the :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` that answer those questions: per-:py:class:`~alchemiscale.storage.models.Task` execution history, bulk indicators across many :py:class:`~alchemiscale.storage.models.Task`\s, live progress for running :py:class:`~alchemiscale.storage.models.Task`\s, and your compute share within a :py:class:`~alchemiscale.models.Scope`.

For failure triage specifically — tracebacks, per-unit logs, and captured stdout/stderr — see :ref:`handling-errors`.


***********************************
Task history and execution attempts
***********************************

A single :py:class:`~alchemiscale.storage.models.Task` may be executed several times over its lifetime: it can land on a flaky host, be released when you change its status mid-run, or be restarted by a :py:class:`~alchemiscale.storage.models.Task` restart pattern.
Each of these is a distinct *attempt*.
To retrieve the full attempt history of a :py:class:`~alchemiscale.storage.models.Task`, use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_history`, which returns a list of :py:class:`~alchemiscale.storage.models.TaskAttempt` records, most recent first::

    >>> task: ScopedKey
    >>> for attempt in asc.get_task_history(task):
    >>>     print(attempt.compute_service_id,
    >>>           attempt.hostname,
    >>>           attempt.datetime_claimed,
    >>>           attempt.datetime_end,
    >>>           attempt.outcome)

Each :py:class:`~alchemiscale.storage.models.TaskAttempt` records:

* ``compute_service_id`` and ``hostname`` — which compute service claimed the attempt, and the host it ran on
* ``manager_name`` — the compute manager responsible for the service, if any
* ``datetime_claimed`` and ``datetime_end`` — when the attempt was claimed and when it ended (``datetime_end`` is ``None`` while the attempt is still in flight)
* ``outcome`` — one of ``complete``, ``error``, ``expired`` (the compute service lost its registration before producing a result), or ``released`` (you forced the :py:class:`~alchemiscale.storage.models.Task` to another status before it finished)
* ``units_completed`` and ``units_total`` — how far the attempt progressed through its :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit`\s
* ``protocoldagresultref`` — the :py:class:`~alchemiscale.models.ScopedKey` of the :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult` the attempt produced, where one exists (``expired`` and ``released`` attempts have none)

You can limit the history to the most recent attempts with the ``limit`` keyword argument::

    >>> # just the most recent attempt
    >>> asc.get_task_history(task, limit=1)

Recall from :ref:`handling-errors` that a :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAG`-creation failure sets the :py:class:`~alchemiscale.storage.models.Task` to ``error`` and records the traceback on the :py:class:`~alchemiscale.storage.models.Task`\'s ``reason`` (surfaced via :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_details`, below) rather than producing a result to inspect.


****************************
Bulk indicators for Tasks
****************************

When you want a compact status summary across many :py:class:`~alchemiscale.storage.models.Task`\s at once — for a dashboard, a triage sweep, or a quick sanity check — use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_details`.
It returns a list of :py:class:`~alchemiscale.storage.models.TaskDetails`, one per input :py:class:`~alchemiscale.storage.models.Task` and in the same order (with ``None`` in place of any :py:class:`~alchemiscale.storage.models.Task` that doesn't exist)::

    >>> tasks = asc.get_network_tasks(an_sk)
    >>> for detail in asc.get_tasks_details(tasks):
    >>>     if detail is None:
    >>>         continue
    >>>     print(detail.task,
    >>>           detail.status,
    >>>           detail.datetime_status_changed,
    >>>           detail.num_claims)

Each :py:class:`~alchemiscale.storage.models.TaskDetails` bundles:

* ``task`` — the :py:class:`~alchemiscale.models.ScopedKey` of the :py:class:`~alchemiscale.storage.models.Task`
* ``status`` and ``datetime_status_changed`` — the current status and when it last changed
* ``reason`` — the human-readable reason for the current status, where one was recorded; this is where a :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAG`-creation traceback appears for an errored :py:class:`~alchemiscale.storage.models.Task`, and where a ``reason`` you passed to :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.set_tasks_status` appears for ``invalid``/``deleted`` :py:class:`~alchemiscale.storage.models.Task`\s
* ``num_claims`` — how many times the :py:class:`~alchemiscale.storage.models.Task` has been claimed for execution
* ``current_claim`` — the live claim on a ``running`` :py:class:`~alchemiscale.storage.models.Task` (compute service, host, claim time, and progress), or ``None``
* ``most_recent_attempt`` — the most recent :py:class:`~alchemiscale.storage.models.TaskAttempt`, matching the first element of :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_task_history`

For example, to find all errored :py:class:`~alchemiscale.storage.models.Task`\s on a network and print why each stopped::

    >>> tasks = asc.get_network_tasks(an_sk, status='error')
    >>> for detail in asc.get_tasks_details(tasks):
    >>>     print(detail.task, detail.reason)


******************************
Live progress of running Tasks
******************************

For :py:class:`~alchemiscale.storage.models.Task`\s that are currently ``running``, you can watch how far each has progressed through its :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit`\s with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_progress`.
It returns a list in the same order as the input, where each element is a ``(units_completed, units_total)`` tuple for a ``running`` :py:class:`~alchemiscale.storage.models.Task` that is reporting progress, or ``None`` otherwise (for example, a :py:class:`~alchemiscale.storage.models.Task` that isn't running, or a running one that hasn't yet reported)::

    >>> tasks = asc.get_network_tasks(an_sk, status='running')
    >>> for task, progress in zip(tasks, asc.get_tasks_progress(tasks)):
    >>>     if progress is None:
    >>>         print(task, 'no progress reported')
    >>>     else:
    >>>         completed, total = progress
    >>>         print(task, f'{completed}/{total} units')

Progress is best-effort telemetry pushed by the compute service between :external+gufe:py:class:`~gufe.protocols.protocolunit.ProtocolUnit`\s, so a ``None`` result does not imply anything is wrong — only that no progress datapoint is currently available.


****************************
Compute share within a Scope
****************************

When compute is contended, it's useful to know what fraction of it your work is currently receiving.
:py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_scope_compute_share` returns, as a ``float``, the fraction of currently-running :py:class:`~alchemiscale.storage.models.Task`\s in a given :py:class:`~alchemiscale.models.Scope` relative to its sibling :py:class:`~alchemiscale.models.Scope`\s at the same level::

    >>> asc.get_scope_compute_share(Scope('my_org', 'my_campaign', 'my_project'))
    0.42

The share is computed server-side as this :py:class:`~alchemiscale.models.Scope`\'s aggregate fraction of running :py:class:`~alchemiscale.storage.models.Task`\s relative to its siblings; only the aggregate fraction is returned.
Your identity must hold the given :py:class:`~alchemiscale.models.Scope`.
A value near ``1.0`` means nearly all currently-running :py:class:`~alchemiscale.storage.models.Task`\s among the sibling :py:class:`~alchemiscale.models.Scope`\s belong to this one; a value near ``0.0`` means the :py:class:`~alchemiscale.models.Scope` is getting little of the available compute right now.
Because it reflects only the *instantaneous* running population, the value fluctuates as :py:class:`~alchemiscale.storage.models.Task`\s are claimed and completed.
