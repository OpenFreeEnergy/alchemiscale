.. _strategy-guide:

##################################
Using Strategies with alchemiscale
##################################

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
    from stratocaster.strategies import UncertaintyStrategy

    client = AlchemiscaleClient('https://your-server.com', 'your-token')

    # Create your AlchemicalNetwork
    network = create_your_network()  # Your network creation logic

    # Submit with a strategy
    network_sk = client.create_network(
        network=network, 
        strategy=UncertaintyStrategy(target_uncertainty=0.5)  # Target 0.5 kcal/mol uncertainty
    )

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

Custom Strategy Development
---------------------------

.. code-block:: python

    from stratocaster.base import Strategy, StrategyResult
    from stratocaster.base.models import StrategySettings
    from gufe import AlchemicalNetwork, ProtocolResult
    from gufe.tokenization import GufeKey
    from pydantic import Field

    class MyCustomStrategySettings(StrategySettings):
        uncertainty_threshold: float = Field(
            default=0.5, 
            description="Uncertainty threshold in kcal/mol"
        )

    class MyCustomStrategy(Strategy):
        _settings_cls = MyCustomStrategySettings
        
        def _propose(
            self, 
            network: AlchemicalNetwork,
            protocol_results: dict[GufeKey, ProtocolResult]
        ) -> StrategyResult:
            # Your strategy logic here
            settings = self.settings
            weights = {}
            
            for state_a, state_b in network.graph.edges():
                # Get the transformation key from the edge
                transformation_key = network.graph.get_edge_data(state_a, state_b)[0]["object"].key
                
                # Analyze results for this transformation
                result = protocol_results.get(transformation_key)
                
                if result is None:
                    # No results yet - high priority
                    weights[transformation_key] = 1.0
                elif result.uncertainty > settings.uncertainty_threshold:
                    # Needs more work - medium priority  
                    weights[transformation_key] = 0.5
                else:
                    # Sufficient results - no priority
                    weights[transformation_key] = None
                    
            return StrategyResult(weights)
        
        @classmethod
        def _default_settings(cls) -> StrategySettings:
            return MyCustomStrategySettings()

Best Practices
==============

Strategy Selection
------------------

- **Use simple strategies first** (e.g., connectivity-based)
- **Test with** ``additive`` **mode** before using ``full`` mode
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

- Review strategy mode (``additive`` vs ``full``)
- Check ``max_tasks_per_transformation`` and scaling settings
- Examine strategy weights and transformation status

Performance Issues
------------------

- Increase strategist service ``cache_size_limit``
- Reduce ``max_workers`` if system is overloaded
- Optimize strategy algorithm efficiency

Examples
========

Simple Connectivity Strategy
-----------------------------

.. code-block:: python

    # Prioritize poorly connected transformations
    network_sk = client.create_network(
        network=my_network,
        strategy=ConnectivityStrategy()
    )

    # Use conservative settings
    client.set_network_strategy(
        network=network_sk,
        strategy=ConnectivityStrategy(),
        mode="partial",  # Conservative mode
        max_tasks_per_transformation=3,
    )

Uncertainty-Based Strategy
---------------------------

.. code-block:: python

    from stratocaster.strategies import UncertaintyStrategy
    from openff.units import unit

    # Create strategy that targets transformations with uncertainty > 0.3 kcal/mol
    uncertainty_strategy = UncertaintyStrategy(
        target_uncertainty=0.3 * unit.kilocalorie_per_mole,  # Target 0.3 kcal/mol uncertainty
        min_samples=5,                                       # Require at least 5 samples before considering uncertainty
        max_uncertainty_cap=3.0 * unit.kilocalorie_per_mole, # Ignore transformations with uncertainty > 3.0 (likely problematic)
        max_samples=15                                       # Hard limit: stop after 15 samples regardless of uncertainty
    )

    # Submit network with uncertainty-based prioritization
    network_sk = client.create_network(
        network=my_network,
        strategy=uncertainty_strategy
    )

    # Use with aggressive resource reallocation
    client.set_network_strategy(
        network=network_sk,
        strategy=uncertainty_strategy,
        mode="full",  # Allow task cancellation
        task_scaling="exponential",
        max_tasks_per_transformation=10
    )
