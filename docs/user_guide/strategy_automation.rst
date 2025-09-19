.. _strategy-automation:

####################################
Automating Execution with a Strategy
####################################

After submitting an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` and creating some initial :py:class:`~alchemiscale.storage.models.Task`\s as described in :ref:`getting-started`, you can automate the ongoing management of your computation with a ``Strategy``.
A ``Strategy`` intelligently analyzes your network's current results and automatically prioritizes which transformations should receive computational resources next.

*******************
What is a Strategy?
*******************

A ``Strategy`` is an algorithm that continuously monitors your :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` and:

- Analyzes the current state of your network and existing results
- Assigns priority weights to transformations (0.0 to 1.0 scale)
- Automatically creates and actions (or cancels) :py:class:`~alchemiscale.storage.models.Task`\s based on these priorities
- Adapts its decisions as new results become available

This automation saves you from manually monitoring progress and creating new :py:class:`~alchemiscale.storage.models.Task`\s, while ensuring computational resources are allocated where they will be most beneficial.

********************
Basic Strategy Usage
********************

Setting up a Strategy for your Network
======================================

Once you have submitted an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` (see :ref:`user-guide-submit-network`), you can attach a ``Strategy`` to it::

    >>> from stratocaster.strategies import ConnectivityStrategy
    >>> from alchemiscale import AlchemiscaleClient, Scope
    
    >>> # Assuming you already have a submitted network
    >>> asc = AlchemiscaleClient()
    >>> scope = Scope('my_org', 'my_campaign', 'my_project')
    >>> 
    >>> # Create a connectivity-based strategy
    >>> strategy = ConnectivityStrategy(ConnectivityStrategy.default_settings())
    >>> 
    >>> # Attach it to your network
    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     max_tasks_per_transformation=5
    ... )

This connectivity strategy will prioritize transformations critical for overall connectivity between ``ChemicalSystem``\s within your network, helping to improve overall convergence.

Monitoring Strategy Progress
============================

You can check on your ``Strategy``\'s current status::

    >>> # Check strategy state
    >>> state = asc.get_network_strategy_state(network_sk)
    >>> print(f"Status: {state.status}")
    >>> print(f"Iterations: {state.iterations}")
    >>> print(f"Last run: {state.last_iteration}")

The ``Strategy`` will automatically run periodically (every 1 hour by default) to reassess your network and adjust :py:class:`~alchemiscale.storage.models.Task` allocation accordingly.

****************************
Understanding Strategy Modes
****************************

Strategies operate in different modes that control how aggressively they manage your :py:class:`~alchemiscale.storage.models.Task`\s:

Partial Mode (Default)
======================

In ``partial`` mode, the ``Strategy`` takes a conservative approach::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     mode="partial"
    ... )

- Creates new :py:class:`~alchemiscale.storage.models.Task`\s when transformation priorities increase
- Never cancels existing :py:class:`~alchemiscale.storage.models.Task`\s
- Accumulates computational work over time
- Safe choice when you want to avoid wasting any computation

Full Mode
=========

In ``full`` mode, the ``Strategy`` actively reallocates resources::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     mode="full"
    ... )

- Creates new :py:class:`~alchemiscale.storage.models.Task`\s when priorities increase
- Cancels existing :py:class:`~alchemiscale.storage.models.Task`\s when priorities decrease
- Aggressively optimizes resource allocation
- May cancel running :py:class:`~alchemiscale.storage.models.Task`\s if they become lower priority

.. warning::
   Use ``full`` mode carefully, as it may cancel in-progress work if transformation priorities change significantly.

*********************************
Task Scaling and Resource Control
*********************************

You can control how many :py:class:`~alchemiscale.storage.models.Task`\s are created based on transformation priorities:

Linear Scaling
==============

With linear scaling, the number of tasks increases proportionally with priority::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     task_scaling="linear",
    ...     max_tasks_per_transformation=6
    ... )

The number of tasks proposed corresponds to::

    tasks = math.floor(1 + (weight × max_tasks_per_transformation))

So a transformation with weight 0.5 would get ``1 + (0.5 × 6) = 4`` tasks.

Linear scaling gives the following qualitative relationship between weight and `Task` counts, assuming `max_tasks_per_transformation = 6`::

    tasks        1           2           3          4           5          6
            |----------|-----------|----------|-----------|----------|-----------|
    weight  0                                                                    1


Exponential Scaling (Default)
=============================

With exponential scaling, high-priority transformations receive disproportionately more resources::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     task_scaling="exponential",
    ...     max_tasks_per_transformation=6
    ... )

The number of tasks proposed corresponds to::

    tasks = math.floor((1 + max_tasks_per_transformation) ^ weight)

So a transformation with weight 0.5 would get ``(1 + 6)^0.5 ≈ 2.6`` (rounded to 2) tasks.

This gives high-priority transformations much more computational power while still allocating some resources to lower-priority ones.
Exponential scaling gives the following qualitative relationship between weight and :py:class:`.Task` counts, assuming `max_tasks_per_transformation = 6`::


    tasks                   1                         2            3      4   5 6
            |--------------------------------|----------------|--------|----|--|-|
    weight  0                                                                    1


*************************
Strategy Lifecycle States
*************************

Your ``Strategy`` also features a ``status``, similar to :py:class:`.Task` ``status``:

Awake Status
============

When ``'awake'``, the ``Strategy`` is actively working::

    >>> status = asc.get_network_strategy_status(network_sk)
    >>> print(status)
    'awake'

The ``Strategy`` analyzes your network, assigns weights to transformations, and creates and actions (or cancels) :py:class:`.Task`\s according to its ``mode``.

Dormant Status
==============

A ``Strategy`` goes ``'dormant'`` when it determines no further work is needed::

    >>> status = asc.get_network_strategy_status(network_sk)
    >>> print(status)
    'dormant'

This happens when all transformation weights are ``None``, indicating the ``Strategy`` has reached its stop condition.

A ``Strategy`` will automatically go from ``'dormant'`` to ``'awake'`` if new results have appeared since it went ``'dormant'``,
giving it a chance to evaluate whether to allocate additional effort given the new information.
You can also manually wake up a ``'dormant'`` ``Strategy`` with::

    >>> asc.set_network_strategy_awake(network_sk)

Error Status
============

If the ``Strategy`` encounters an error during execution, it will enter the ``'error'`` ``status``::

    >>> status = asc.get_network_strategy_status(network_sk)
    >>> print(status)
    'error'

You can introspect the problem using::

    >>> state = asc.get_network_strategy_state(network_sk)
    >>> if state.exception:
    ...     print(f"Error: {state.exception}")
    ...     print(f"Traceback: {state.traceback}")

A ``Strategy`` in the ``'error'`` ``status`` will no longer be performed.
You should address the issue indicated by the traceback, and then set the ``Strategy`` back to the ``'awake'`` ``status`` to continue::

    >>> asc.set_network_strategy_awake(network_sk)


*************************************
Managing Strategy Execution Frequency
*************************************

You can control how often your ``Strategy`` runs::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     sleep_interval=600  # 10 minutes between runs
    ... )

Shorter intervals mean more responsive automation but higher computational overhead for the strategist service.
Longer intervals reduce overhead but may be slower to respond to changing conditions.

The ``alchemiscale`` ``Strategist`` service will be configured with a minimum sleep interval,
so setting this too low will have no effect if it is lower than that interval.

********************
Disabling a Strategy
********************

If you need to pause strategy execution temporarily::

    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     mode="disabled"
    ... )

This completely stops the ``Strategy`` from creating or canceling any :py:class:`.Task`\s.
You can re-enable it later by changing the mode back to ``partial`` or ``full``.

**************
Best Practices
**************

Start Conservative
==================

When first using a ``Strategy``:

- Begin with ``partial`` mode to avoid unexpected cancellations
- Use lower ``max_tasks_per_transformation`` values initially
- Monitor strategy behavior before scaling up

Resource Planning
=================

- Use ``linear`` scaling for predictable resource usage
- Use ``exponential`` scaling when you want to heavily prioritize important transformations
- Adjust ``max_tasks_per_transformation`` based on your available compute resources

Monitoring
==========

Regular monitoring helps ensure your ``Strategy`` is working as expected::

    >>> # Check strategy state periodically
    >>> state = asc.get_network_strategy_state(network_sk)
    >>> print(f"Status: {state.status}, Iterations: {state.iterations}")
    >>> 
    >>> # Monitor overall network progress
    >>> status_counts = asc.get_network_status(network_sk)
    >>> print(status_counts)

***************
Troubleshooting
***************

Strategy Not Running
====================

If your ``Strategy`` isn't executing:

- Verify with the ``alchemiscale`` server administrator that the Strategist service is running and accessible
- Check that your network is in a :py:class:`~alchemiscale.models.Scope` visible to the Strategist service
- Ensure sufficient time has passed since the last iteration (respecting ``sleep_interval``)

Unexpected Task Behavior
========================

If :py:class:`.Task`\s are being created or canceled unexpectedly:

- Review your ``Strategy`` mode (``partial`` vs ``full``)
- Check ``max_tasks_per_transformation`` and ``task_scaling`` settings

Poor Performance
================

If ``Strategy`` execution is slow:

- Increase the strategist service cache size if running your own service
- Consider reducing ``max_workers`` if the host is overloaded, or increasing if underutilized
- Evaluate whether your chosen ``Strategy`` algorithm is efficient for large networks

****************
Example Workflow
****************

Here's a complete example showing typical ``Strategy`` usage::

    >>> from alchemiscale import AlchemiscaleClient, Scope, ScopedKey
    >>> from stratocaster.strategies import ConnectivityStrategy
    >>> 
    >>> # Set up client and submit network (assuming this is done)
    >>> asc = AlchemiscaleClient()
    >>> network_sk = ScopedKey.from_str("<your-network-scoped-key>")
    >>> 
    >>> # Create and attach a conservative connectivity strategy
    >>> strategy = ConnectivityStrategy(ConnectivityStrategy.default_settings())
    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     mode="partial",
    ...     task_scaling="linear",
    ...     max_tasks_per_transformation=3,
    ...     sleep_interval=1800
    ... )
    >>> 
    >>> # Monitor progress
    >>> state = asc.get_network_strategy_state(network_sk)
    >>> print(f"Strategy status: {state.status}")
    >>> 
    >>> # Later, if you want more aggressive optimization
    >>> asc.set_network_strategy(
    ...     network=network_sk,
    ...     strategy=strategy,
    ...     mode="full",
    ...     task_scaling="exponential",
    ...     max_tasks_per_transformation=10
    ... )

This workflow starts conservatively and becomes more aggressive as you gain confidence in the ``Strategy``'s behavior.
