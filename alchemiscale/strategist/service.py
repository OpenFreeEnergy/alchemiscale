"""
:mod:`alchemiscale.strategist.service` --- strategist service
=============================================================

"""

import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings
import numpy as np

from diskcache import Cache
import zstandard as zstd

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
from ..compression import compress_keyed_chain_zstd, decompress_gufe_zstd, json_to_gufe
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
    
    @staticmethod
    def _determine_cache_dir(cache_directory: Path | str | None = None) -> Path:
        """Determine the cache directory path following XDG conventions."""
        if cache_directory is not None:
            return Path(cache_directory)
        
        # Use XDG_CACHE_HOME if set, otherwise default to ~/.cache
        if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
            cache_base = Path(xdg_cache_home)
        else:
            cache_base = Path.home() / ".cache"
        
        return cache_base / "alchemiscale-strategist"
    
    def __init__(self, settings: StrategistSettings):
        self.settings = settings
        self.sleep_interval = settings.sleep_interval
        self.max_workers = settings.max_workers
        self.scopes = settings.scopes or [Scope()]
        
        # Initialize storage components
        self.n4js = get_n4js(settings.neo4j_settings or Neo4jStoreSettings())
        self.s3os = get_s3os(settings.s3_settings or S3ObjectStoreSettings())
        
        # Initialize disk-based cache
        self._cache_enabled = settings.use_local_cache
        if self._cache_enabled:
            self._cache_directory = self._determine_cache_dir(settings.cache_directory)
            self._cache_directory.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(
                str(self._cache_directory),
                size_limit=settings.cache_size_limit,
                eviction_policy="least-recently-used",
            )
        else:
            self._cache = None
        
        self._stop = False
        
    def _get_protocoldagresult_cached(self, result_ref, transformation_sk: ScopedKey) -> ProtocolDAGResult:
        """Get ProtocolDAGResult with disk caching."""
        cache_key = str(result_ref.obj_key)
        
        if self._cache_enabled:
            try:
                if cached_content := self._cache.get(cache_key, None):
                    return decompress_gufe_zstd(cached_content)
            except zstd.ZstdError:
                # Handle decompression errors by removing corrupted cache entry
                warnings.warn(f"Error decompressing cached ProtocolDAGResult {cache_key}, "
                             "deleting entry and retrieving new content.")
                self._cache.delete(cache_key)
        
        # Retrieve from object store using s3os.pull_protocoldagresult
        try:
            pdr_bytes = self.s3os.pull_protocoldagresult(
                ScopedKey(gufe_key=result_ref.obj_key, **result_ref.scope.dict()),
                transformation_sk,
                ok=result_ref.ok
            )
        except Exception:
            # Fallback to location-based retrieval
            pdr_bytes = self.s3os.pull_protocoldagresult(
                location=result_ref.location,
                ok=result_ref.ok
            )
        
        # Decompress the raw bytes to get ProtocolDAGResult
        try:
            pdr = decompress_gufe_zstd(pdr_bytes)
        except zstd.ZstdError:
            # Fallback to JSON deserialization for uncompressed data
            pdr = json_to_gufe(pdr_bytes.decode("utf-8"))
        
        if self._cache_enabled:
            try:
                # Cache the already compressed bytes directly 
                self._cache.set(cache_key, pdr_bytes)
            except Exception as e:
                warnings.warn(f"Failed to cache ProtocolDAGResult {cache_key}: {e}")
        
        return pdr
    
    def _get_protocol_results(self, network_sk: ScopedKey) -> dict[ScopedKey, list[ProtocolResult]]:
        """Get ProtocolResults for all transformations in a network.
        
        This method uses disk-based caching to avoid repeated expensive lookups.
        """
        results = {}
        transformations = self.n4js.get_network_transformations(network_sk)
        
        for transformation_sk in transformations:
            # Get the transformation object
            transformation = self.n4js.get_gufe(transformation_sk)
            
            # Get successful ProtocolDAGResults for this transformation
            result_refs = self.n4js.get_transformation_results(transformation_sk)
            
            # Collect all ProtocolDAGResults for this transformation
            pdrs = []
            for result_ref in result_refs:
                if result_ref.ok:
                    # Get ProtocolDAGResult with caching
                    pdr = self._get_protocoldagresult_cached(result_ref, transformation_sk)
                    pdrs.append(pdr)
            
            # Use transformation.gather() to get ProtocolResult from ProtocolDAGResults
            if len(pdrs) != 0:
                protocol_result = transformation.gather(pdrs)
                results[transformation_sk] = protocol_result
            else:
                results[transformation_sk] = None
            
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
                for ref in self.n4js.get_transformation_results(transformation_sk)
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
            protocol_results: list[ProtocolResult|None] = self._get_protocol_results(network_sk)
            
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
                    taskhub_sk = self.n4js.get_taskhub(network_sk)
                    with self.n4js.transaction() as tx:
                        task_sks = self.n4js.get_taskhub_tasks(taskhub_sk, tx=tx)
                        self.n4js.cancel_tasks(task_sks, taskhub_sk, tx=tx)
                
                strategy_state.last_iteration_result_count = current_result_count
                strategy_state.last_iteration = datetime.utcnow()
                strategy_state.iterations += 1
                return strategy_state
            
            # Set weights to None for transformations with errored tasks
            transformations = self.n4js.get_network_transformations(network_sk)
            for transformation_sk in transformations:
                counts = self.n4js.get_transformation_status(transformation_sk)
                if counts.get(TaskStatusEnum.error.value):
                    weights[transformation_sk] = None
            
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
    
    def cycle(self):
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
            network_to_future = {value: key for key, value in future_to_network.items()}

            while future_to_network and not self._stop:
                # Collect results and update database
                try:
                    for future in as_completed(list(future_to_network.keys()), timeout=self.sleep_interval):
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
                            logger.exception(f"Strategy execution failed for network {network_sk}: {e}")

                        future_to_network.pop(future)
                        network_to_future.pop(network_sk)

                except TimeoutError:
                    # Check if we should stop before adding new strategies
                    if self._stop:
                        break
                        
                    # if we ran out of time waiting for strategies to finish, check for new ones
                    # and continue
                    ready_strategies = self.n4js.get_strategies_for_execution(
                        scopes=self.scopes,
                        min_sleep_interval=self.sleep_interval
                    )

                    for network_sk, strategy_sk, strategy_state in ready_strategies:
                        # Only submit strategies that aren't already running
                        if network_sk not in network_to_future:
                            # Submit new strategy execution
                            future = executor.submit(self._execute_strategy, network_sk, strategy_sk, strategy_state)
                            future_to_network[future] = network_sk
                            network_to_future[network_sk] = future
                            logger.debug(f"Added new strategy execution for network {network_sk}")

    def start(self):
        """Start the Strategist service."""

        logger.info("Starting Strategist service")
        self._stop = True
        
        try:
            while not self._stop:
                start_time = time.time()
                
                try:
                    self.cycle()
                except Exception as e:
                    logger.exception(f"Iteration failed: {e}")
                
                # Check if we should stop before sleeping
                if self._stop:
                    break
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                remaining_sleep = max(0, self.sleep_interval - elapsed)
                
                if remaining_sleep > 0:
                    logger.debug(f"Sleeping for {remaining_sleep:.1f} seconds")
                    time.sleep(remaining_sleep)
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self._stop = True
            logger.info("Strategist service stopped")
    
    def stop(self):
        """Stop the strategist service."""
        self._stop = True
