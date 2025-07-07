"""
:mod:`alchemiscale.strategist.service` --- strategist service
=============================================================

"""

import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np

from gufe import AlchemicalNetwork, ProtocolResult
from gufe.protocols import ProtocolDAGResult

from ..models import Scope, ScopedKey
from ..storage.statestore import get_n4js
from ..storage.objectstore import get_s3os
from ..storage.models import (
    StrategyState,
    StrategyStatusEnum,
    StrategyModeEnum,
    StrategyTaskScalingEnum,
    TaskStatusEnum,
)
from ..settings import Neo4jStoreSettings, S3ObjectStoreSettings
from .settings import StrategistSettings


logger = logging.getLogger(__name__)



class StrategistService:
    """Service for executing strategies on AlchemicalNetworks.
    
    This service runs in an infinite loop, periodically:
    1. Querying for strategies ready for execution
    2. Executing strategies in parallel using ProcessPoolExecutor
    3. Converting strategy weights to task counts
    4. Creating/cancelling tasks as needed
    5. Updating strategy state
    """
    
    def __init__(self, settings: StrategistSettings):
        self.settings = settings
        self.sleep_interval = settings.sleep_interval
        self.max_workers = settings.max_workers
        self.scopes = settings.scopes or [Scope()]
        
        # Initialize storage components
        self.n4js = get_n4js(settings.neo4j_settings or Neo4jStoreSettings())
        self.s3os = get_s3os(settings.s3_settings or S3ObjectStoreSettings())
        
        # LRU cache for ProtocolDAGResults to reduce object store hits
        self._result_cache = {}
        self._cache_max_size = settings.cache_size
        
        self._stop = False
        
    @lru_cache(maxsize=1000)
    def _get_protocol_results(self, network_sk: ScopedKey) -> dict[ScopedKey, list[ProtocolResult]]:
        """Get ProtocolResults for all transformations in a network.
        
        This method is cached to avoid repeated expensive lookups.
        """
        results = {}
        transformations = self.n4js.get_network_transformations(network_sk)
        
        for transformation_sk in transformations:
            # Get successful ProtocolDAGResults for this transformation
            result_refs = self.n4js.get_transformation_results(
                transformation_sk, 
                status=TaskStatusEnum.complete
            )
            
            protocol_results = []
            for result_ref in result_refs:
                if result_ref.ok:
                    # Check cache first
                    cache_key = str(result_ref.obj_key)
                    if cache_key in self._result_cache:
                        pdr = self._result_cache[cache_key]
                    else:
                        # Load from object store
                        pdr = self.s3os.get_protocoldagresult(result_ref)
                        
                        # Add to cache with LRU eviction
                        if len(self._result_cache) >= self._cache_max_size:
                            # Remove oldest item (simple FIFO for now)
                            oldest_key = next(iter(self._result_cache))
                            del self._result_cache[oldest_key]
                        self._result_cache[cache_key] = pdr
                    
                    # Extract ProtocolResults from ProtocolDAGResult
                    if hasattr(pdr, 'protocol_results'):
                        protocol_results.extend(pdr.protocol_results)
            
            results[transformation_sk] = protocol_results
            
        return results
    
    def _weights_to_task_counts(
        self, 
        weights: dict[ScopedKey, float | None],
        max_tasks_per_transformation: int,
        max_tasks_per_network: int | None,
        task_scaling: StrategyTaskScalingEnum,
    ) -> dict[ScopedKey, int]:
        """Convert strategy weights to discrete task counts."""
        task_counts = {}
        
        for transformation_sk, weight in weights.items():
            if weight is None or weight == 0:
                task_counts[transformation_sk] = 0
            elif weight == 1:
                task_counts[transformation_sk] = max_tasks_per_transformation
            else:
                if task_scaling == StrategyTaskScalingEnum.linear:
                    tasks = int(1 + weight * max_tasks_per_transformation)
                elif task_scaling == StrategyTaskScalingEnum.exponential:
                    tasks = int((1 + max_tasks_per_transformation) ** weight)
                else:
                    raise ValueError(f"Unknown task scaling: {task_scaling}")
                
                task_counts[transformation_sk] = min(tasks, max_tasks_per_transformation)
        
        # Scale down if total exceeds max_tasks_per_network
        total_tasks = sum(task_counts.values())
        if max_tasks_per_network is not None and total_tasks > max_tasks_per_network:
            scale_factor = max_tasks_per_network / total_tasks
            task_counts = {
                sk: int(count * scale_factor) 
                for sk, count in task_counts.items()
            }
        
        return task_counts
    
    def _execute_strategy(
        self, 
        network_sk: ScopedKey, 
        strategy_sk: ScopedKey, 
        strategy_state: StrategyState
    ) -> StrategyState:
        """Execute a single strategy and return updated state."""
        try:
            # Check if strategy went dormant - count successful results
            current_result_count = len([
                ref for transformation_sk in self.n4js.get_network_transformations(network_sk)
                for ref in self.n4js.get_transformation_results(transformation_sk, TaskStatusEnum.complete)
                if ref.ok
            ])
            
            # If dormant, check if new results appeared
            if strategy_state.status == StrategyStatusEnum.dormant:
                if current_result_count == strategy_state.last_iteration_result_count:
                    # Still dormant, no new results
                    return strategy_state
                else:
                    # New results appeared, wake up
                    strategy_state.status = StrategyStatusEnum.awake
            
            # Get network and results
            network = self.n4js.get_gufe(network_sk)
            protocol_results = self._get_protocol_results(network_sk)
            
            # Get strategy object
            strategy = self.n4js.get_network_strategy(network_sk)
            if strategy is None:
                raise ValueError(f"Strategy not found for network {network_sk}")
            
            # Execute strategy to get weights
            # Note: This requires the strategy to have a propose() method
            # that takes AlchemicalNetwork and dict of ProtocolResults
            strategy_result = strategy.propose(network, protocol_results)
            weights = strategy_result.resolve()  # Get normalized weights
            
            # Check if all weights are None (stop condition)
            if all(w is None for w in weights.values()):
                strategy_state.status = StrategyStatusEnum.dormant
                
                # If in full mode, cancel all actioned tasks
                if strategy_state.mode == StrategyModeEnum.full:
                    network_tasks = self.n4js.get_network_tasks(network_sk)
                    for task_sk in network_tasks:
                        task = self.n4js.get_task(task_sk)
                        if task.status in [TaskStatusEnum.waiting, TaskStatusEnum.running]:
                            self.n4js.cancel_task(task_sk)
                
                strategy_state.last_iteration_result_count = current_result_count
                strategy_state.last_iteration = datetime.utcnow()
                strategy_state.iterations += 1
                return strategy_state
            
            # Set weights to None for transformations with errored tasks
            transformations = self.n4js.get_network_transformations(network_sk)
            for transformation_sk in transformations:
                tasks = self.n4js.get_transformation_tasks(transformation_sk)
                for task_sk in tasks:
                    task = self.n4js.get_task(task_sk)
                    if task.status == TaskStatusEnum.error:
                        weights[transformation_sk] = None
                        break
            
            # Convert weights to task counts
            task_counts = self._weights_to_task_counts(
                weights,
                strategy_state.max_tasks_per_transformation,
                strategy_state.max_tasks_per_network,
                strategy_state.task_scaling,
            )
            
            # Set task counts for each transformation
            for transformation_sk, target_count in task_counts.items():
                current_tasks = self.n4js.get_transformation_tasks(transformation_sk)
                actioned_tasks = [
                    task_sk for task_sk in current_tasks
                    if self.n4js.get_task(task_sk).status in [TaskStatusEnum.waiting, TaskStatusEnum.running]
                ]
                current_count = len(actioned_tasks)
                
                if target_count > current_count:
                    # Create new tasks
                    for _ in range(target_count - current_count):
                        self.n4js.create_task(transformation_sk)
                        
                elif target_count < current_count and strategy_state.mode == StrategyModeEnum.full:
                    # Cancel excess tasks (prioritize unclaimed ones)
                    excess = current_count - target_count
                    tasks_to_cancel = []
                    
                    # First cancel unclaimed tasks
                    for task_sk in actioned_tasks:
                        task = self.n4js.get_task(task_sk)
                        if task.status == TaskStatusEnum.waiting and task.claim is None:
                            tasks_to_cancel.append(task_sk)
                            if len(tasks_to_cancel) >= excess:
                                break
                    
                    # Then cancel claimed but not running tasks if needed
                    if len(tasks_to_cancel) < excess:
                        for task_sk in actioned_tasks:
                            if task_sk not in tasks_to_cancel:
                                task = self.n4js.get_task(task_sk)
                                if task.status == TaskStatusEnum.waiting:
                                    tasks_to_cancel.append(task_sk)
                                    if len(tasks_to_cancel) >= excess:
                                        break
                    
                    # Cancel the selected tasks
                    for task_sk in tasks_to_cancel:
                        self.n4js.cancel_task(task_sk)
            
            # Update strategy state
            strategy_state.last_iteration = datetime.utcnow()
            strategy_state.last_iteration_result_count = current_result_count
            strategy_state.iterations += 1
            strategy_state.exception = None
            strategy_state.traceback = None
            
            return strategy_state
            
        except Exception as e:
            # Strategy execution failed
            logger.exception(f"Strategy execution failed for network {network_sk}")
            
            strategy_state.status = StrategyStatusEnum.error
            strategy_state.exception = (type(e).__name__, str(e))
            strategy_state.traceback = traceback.format_exc()
            strategy_state.last_iteration = datetime.utcnow()
            strategy_state.iterations += 1
            
            return strategy_state
    
    def _iteration(self):
        """Perform one iteration of strategy execution."""
        # Get strategies ready for execution
        ready_strategies = self.n4js.get_strategies_for_execution(
            scopes=self.scopes,
            min_sleep_interval=self.sleep_interval
        )
        
        if not ready_strategies:
            logger.debug("No strategies ready for execution")
            return
        
        logger.info(f"Executing {len(ready_strategies)} strategies")
        
        # Execute strategies in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all strategy executions
            future_to_network = {
                executor.submit(self._execute_strategy, network_sk, strategy_sk, strategy_state): network_sk
                for network_sk, strategy_sk, strategy_state in ready_strategies
            }
            
            # Collect results and update database
            for future in as_completed(future_to_network):
                network_sk = future_to_network[future]
                try:
                    updated_state = future.result()
                    
                    # Update strategy state in database
                    success = self.n4js.update_strategy_state(network_sk, updated_state)
                    if not success:
                        logger.error(f"Failed to update strategy state for network {network_sk}")
                    else:
                        logger.debug(f"Updated strategy state for network {network_sk}")
                        
                except Exception as e:
                    logger.exception(f"Strategy execution future failed for network {network_sk}: {e}")
    
    def start(self):
        """Start the Strategist service."""

        logger.info("Starting Strategist service")
        self._stop = True
        
        try:
            while self._stop:
                start_time = time.time()
                
                try:
                    self._iteration()
                except Exception as e:
                    logger.exception(f"Iteration failed: {e}")
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                remaining_sleep = max(0, self.sleep_interval - elapsed)
                
                if remaining_sleep > 0:
                    logger.debug(f"Sleeping for {remaining_sleep:.1f} seconds")
                    time.sleep(remaining_sleep)
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self._stop = False
            logger.info("Strategist service stopped")
    
    def stop(self):
        """Stop the strategist service."""
        self._stop = False
