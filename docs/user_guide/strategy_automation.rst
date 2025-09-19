.. _strategy-automation:

####################################
Automating Execution with a Strategy
####################################

Overview
========

Strategies in **alchemiscale** provide intelligent, adaptive task management for :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.
A Strategy analyzes your network's current results and automatically prioritizes which transformations should receive computational resources, optimizing the overall free energy calculation workflow.

Key Concepts
============

What is a Strategy?
-------------------

A **Strategy** is an algorithm that:

- Analyzes the current state of your AlchemicalNetwork and existing results
- Assigns priority weights to transformations (0.0 to 1.0)
- Dynamically adjusts task allocation based on evolving results
- Can go dormant when no further work is needed

Strategy Modes
--------------

**Partial Mode** (``partial``) - *Default*
  - Only creates new tasks when weights increase
  - Never cancels existing tasks
  - Conservative approach - accumulates computational work

**Full Mode** (``full``)
  - Creates tasks when weights increase
  - Cancels tasks when weights decrease
  - Aggressive optimization - actively reallocates resources

**Disabled Mode** (``disabled``)
  - Strategy is completely inactive
  - No task creation or cancellation
  - Used to pause strategy execution

Task Scaling
------------

**Linear Scaling** (``linear``)::

    tasks = 1 + (weight × max_tasks_per_transformation)

**Exponential Scaling** (``exponential``) - *Default*::

    tasks = (1 + max_tasks_per_transformation) ^ weight


Which gives the following qualitative relationship between weight and `Task` counts, assuming `max_tasks_per_transformation = 6`::

    max_tasks_per_transformation = 6
    
    # linear
    
    tasks        1           2           3          4           5          6
            |----------|-----------|----------|-----------|----------|-----------|
    weight  0                                                                    1
    
    
    # exponential
    
    tasks                   1                         2            3      4   5 6
            |--------------------------------|----------------|--------|----|--|-|
    weight  0                                                                    1


Basic Usage
===========

1. Submit an AlchemicalNetwork with Strategy
---------------------------------------------

.. code-block:: python

    from alchemiscale import AlchemiscaleClient
    from stratocaster.strategies import ConnectivityStrategy

    client = AlchemiscaleClient('https://your-server.com', 'your-token')

    # Create your AlchemicalNetwork
    network = create_your_network()  # Your network creation logic
    scope = Scope('my-org', 'my-campaign', 'my-project')

    # Submit your network
    network_sk = client.create_network(network, scope)

2. Configure Strategy Settings
------------------------------

.. code-block:: python

    from alchemiscale.storage.models import (
        StrategyState, 
        StrategyModeEnum, 
        StrategyTaskScalingEnum
    )

    # Get the current strategy to avoid removing it
    current_strategy = client.get_network_strategy(network_sk)
    
    # Configure strategy behavior using set_network_strategy
    client.set_network_strategy(
        network=network_sk,
        strategy=current_strategy,  # Keep existing strategy
        mode="full",  # or "partial" (default) or "disabled"
        task_scaling="linear",  # or "exponential" (default)
        max_tasks_per_transformation=5,
        sleep_interval=300  # 5 minutes between strategy runs
    )

3. Monitor Strategy Progress
----------------------------

.. code-block:: python

    # Check current strategy state
    state = client.get_network_strategy_state(network_sk)
    print(f"Status: {state.status}")
    print(f"Iterations: {state.iterations}")
    print(f"Last run: {state.last_iteration}")
    if state.exception:
        print(f"Last error: {state.exception}")
    
    # Check if strategy needs to be woken up (if dormant or errored)
    status = client.get_network_strategy_status(network_sk)
    if status in ["dormant", "error"]:
        client.set_network_strategy_awake(network_sk)

Strategy Lifecycle
==================

**Awake State**
   - Strategy actively analyzes network and results
   - Assigns weights to transformations
   - Creates/cancels tasks based on weights and mode

**Dormant State**
   - All transformation weights are ``None``
   - Strategy pauses execution until new results appear
   - In ``full`` mode: cancels all remaining tasks
   - In ``partial`` mode: leaves existing tasks running

**Error State**
   - Strategy execution failed (e.g., code error, missing dependencies)
   - Check ``strategy_state.exception`` and ``strategy_state.traceback``
   - Strategy will retry on next service cycle

Advanced Configuration
======================

Strategy Service Settings
--------------------------

If you're running your own strategist service:

.. code-block:: yaml

    # strategist-settings.yaml
    sleep_interval: 300  # Check for ready strategies every 5 minutes
    max_workers: 4       # Parallel strategy execution
    cache_directory: "/opt/cache/strategist"
    cache_size_limit: 1073741824  # 1 GiB
    use_local_cache: true
    scopes:
      - org: "my-org"
        campaign: "my-campaign"


Best Practices
==============

Strategy Selection
------------------

- **Use simple strategies first** (e.g., connectivity-based)
- **Test with** ``partial`` **mode** before using ``full`` mode
- **Choose appropriate** ``max_tasks_per_transformation`` **based on your compute resources**

Resource Management
-------------------

- **Use** ``linear`` **scaling** for predictable resource usage
- **Monitor strategy iterations** to ensure reasonable execution frequency

Error Handling
--------------

- **Check strategy state regularly** for error conditions
- **Validate strategy dependencies** before deployment
- **Test strategies on small networks** before large-scale usage

Troubleshooting
===============

Strategy Not Running
---------------------

- Check that strategist service is running
- Verify network is in correct scope for service
- Ensure ``min_iteration_interval`` has elapsed

Unexpected Task Behavior
------------------------

- Review strategy mode (``partial`` vs ``full``)
- Check ``max_tasks_per_transformation`` and scaling settings
- Examine strategy weights and transformation status

Performance Issues
------------------

- Increase strategist service ``cache_size_limit``
- Reduce ``max_workers`` if system is overloaded
- Optimize strategy algorithm efficiency

Examples
========

Connectivity Strategy
---------------------

.. code-block:: python

   my_network: AlchemicalNetwork
   my_scope: Scope

    # Prioritize poorly connected transformations
    network_sk = client.create_network(my_network, my_scope)

    strategy = ConnectivityStrategy(ConnectivityStrategy.default_settings())

    # Use conservative settings
    client.set_network_strategy(
        network_sk,
        strategy,
        mode="partial",  # doesn't ever cancel tasks
        max_tasks_per_transformation=3,
    )

    # Or use more aggressive resource reallocation
    client.set_network_strategy(
        network_sk,
        strategy,
        mode="full",  # allows for task cancellation
        task_scaling="linear",
        max_tasks_per_transformation=10
    )
