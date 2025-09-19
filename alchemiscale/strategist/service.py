"""
:mod:`alchemiscale.strategist.service` --- strategist service
=============================================================

"""

import logging
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
from pathlib import Path
from typing import Any
import warnings
import numpy as np

from diskcache import Cache
import zstandard as zstd

from gufe import AlchemicalNetwork, ProtocolResult, Transformation
from gufe.tokenization import GufeKey
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


def execute_strategy_worker(
    network_sk: ScopedKey,
    strategy_state: StrategyState,
    settings: StrategistSettings,
) -> "StrategyState":
    """Standalone worker function for executing strategies in ProcessPoolExecutor.

    This function creates its own StrategistService instance to avoid serialization
    issues with bound methods and maintains cache functionality per worker process.

    Parameters
    ----------
    network_sk : ScopedKey
        The scoped key of the network to execute strategy for
    strategy_state : StrategyState
        The current state of the strategy
    settings : StrategistSettings
        Settings to instantiate the StrategistService

    Returns
    -------
    StrategyState
        Updated strategy state after execution
    """
    # Create a fresh StrategistService instance for this worker
    service = StrategistService(settings)

    # Execute the strategy using the service's method
    results = service._execute_strategy(network_sk, strategy_state)

    # Close the service's database connection
    service.close()

    return results


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

    def _get_protocoldagresult_cached(
        self, result_ref, transformation_sk: ScopedKey
    ) -> ProtocolDAGResult:
        """Get ProtocolDAGResult with disk caching."""
        cache_key = str(result_ref.obj_key)

        if self._cache_enabled:
            try:
                if cached_content := self._cache.get(cache_key, None):
                    return decompress_gufe_zstd(cached_content)
            except zstd.ZstdError:
                # Handle decompression errors by removing corrupted cache entry
                warnings.warn(
                    f"Error decompressing cached ProtocolDAGResult {cache_key}, "
                    "deleting entry and retrieving new content."
                )
                self._cache.delete(cache_key)

        # Retrieve from object store using s3os.pull_protocoldagresult
        try:
            pdr_bytes = self.s3os.pull_protocoldagresult(
                ScopedKey(gufe_key=result_ref.obj_key, **result_ref.scope.dict()),
                transformation_sk,
                ok=result_ref.ok,
            )
        except Exception:
            # Fallback to location-based retrieval
            pdr_bytes = self.s3os.pull_protocoldagresult(
                location=result_ref.location, ok=result_ref.ok
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

    def _get_protocol_results(
        self, network_sk: ScopedKey
    ) -> tuple[dict[ScopedKey, ProtocolResult | None], dict[ScopedKey, Transformation]]:
        """Get ProtocolResults for all transformations in a network.

        This method uses disk-based caching to avoid repeated expensive lookups.
        """
        results = {}
        transformations = self.n4js.get_network_transformations(network_sk)

        # Build mapping between transformation objects and their scoped keys
        sk_to_tf: dict[ScopedKey, Transformation] = {}

        for transformation_sk in transformations:
            # Get the transformation object
            # TODO: wrap in on-disk caching
            transformation = self.n4js.get_gufe(transformation_sk)

            sk_to_tf[transformation_sk] = transformation

            # Get successful ProtocolDAGResults for this transformation
            result_ref_sks = self.n4js.get_transformation_results(transformation_sk)

            # Collect all ProtocolDAGResults for this transformation
            pdrs = []
            for result_ref_sk in result_ref_sks:
                # Get the actual ProtocolDAGResultRef object from the ScopedKey
                result_ref = self.n4js.get_gufe(result_ref_sk)
                if result_ref.ok:
                    # Get ProtocolDAGResult with caching
                    pdr = self._get_protocoldagresult_cached(
                        result_ref, transformation_sk
                    )
                    pdrs.append(pdr)

            # Use transformation.gather() to get ProtocolResult from ProtocolDAGResults
            if len(pdrs) != 0:
                protocol_result = transformation.gather(pdrs)
                results[transformation_sk] = protocol_result
            else:
                results[transformation_sk] = None

        return results, sk_to_tf

    def _weights_to_task_counts(
        self,
        weights: dict[GufeKey, float | None],
        max_tasks_per_transformation: int,
        task_scaling: StrategyTaskScalingEnum,
    ) -> dict[GufeKey, int]:
        """Convert strategy weights to discrete task counts."""
        task_counts = {}

        for transformation_key, weight in weights.items():
            if weight is None or weight == 0:
                task_counts[transformation_key] = 0
            elif weight == 1:
                task_counts[transformation_key] = max_tasks_per_transformation
            else:
                if task_scaling == StrategyTaskScalingEnum.linear:
                    tasks = int(1 + weight * max_tasks_per_transformation)
                elif task_scaling == StrategyTaskScalingEnum.exponential:
                    tasks = int((1 + max_tasks_per_transformation) ** weight)
                else:
                    raise ValueError(f"Unknown task scaling: {task_scaling}")

                task_counts[transformation_key] = min(
                    tasks, max_tasks_per_transformation
                )

        return task_counts

    def _count_successful_results(self, network_sk: ScopedKey) -> int:
        """Count successful ProtocolDAGResults for all transformations in network."""
        return len(
            [
                ref_sk
                for transformation_sk in self.n4js.get_network_transformations(
                    network_sk
                )
                for ref_sk in self.n4js.get_transformation_results(transformation_sk)
                if self.n4js.get_gufe(ref_sk).ok
            ]
        )

    def _check_dormant_status(
        self, strategy_state: StrategyState, current_result_count: int
    ) -> tuple[bool, StrategyState]:
        """Check if strategy is dormant and handle accordingly.

        Returns:
            (should_continue, updated_strategy_state)
        """
        if strategy_state.status == StrategyStatusEnum.dormant:
            if current_result_count == strategy_state.last_iteration_result_count:
                # Still dormant, no new results
                return False, strategy_state
            else:
                # New results appeared, wake up
                strategy_state.status = StrategyStatusEnum.awake
                return True, strategy_state
        return True, strategy_state

    def _execute_strategy_logic(
        self,
        network_sk: ScopedKey,
        protocol_results: dict[ScopedKey, ProtocolResult | None],
        sk_to_tf: dict[ScopedKey, Transformation],
    ) -> dict[GufeKey, float]:
        """Execute strategy logic to get transformation weights."""
        # Get network
        # TODO: wrap in on-disk caching
        network = self.n4js.get_gufe(network_sk)

        # Convert protocol_results to format expected by strategy
        transformation_results = {
            sk_to_tf[key].key: value for key, value in protocol_results.items()
        }

        # Get strategy object
        strategy = self.n4js.get_network_strategy(network_sk)
        if strategy is None:
            raise ValueError(f"Strategy not found for network {network_sk}")

        # Execute strategy to get weights
        strategy_result = strategy.propose(network, transformation_results)
        return strategy_result.resolve()  # Get normalized weights

    def _handle_dormant_strategy(
        self,
        strategy_state: StrategyState,
        network_sk: ScopedKey,
        current_result_count: int,
    ) -> StrategyState:
        """Handle strategy going dormant (all weights are None)."""
        strategy_state.status = StrategyStatusEnum.dormant

        # If in full mode, cancel all actioned tasks
        if strategy_state.mode == StrategyModeEnum.full:
            taskhub_sk = self.n4js.get_taskhub(network_sk)
            with self.n4js.transaction() as tx:
                task_sks = self.n4js.get_taskhub_tasks(taskhub_sk, tx=tx)
                self.n4js.cancel_tasks(task_sks, taskhub_sk, tx=tx)

        strategy_state.last_iteration_result_count = current_result_count
        strategy_state.last_iteration = datetime.datetime.now(tz=datetime.UTC)
        strategy_state.iterations += 1
        return strategy_state

    def _filter_errored_transformations(
        self, weights: dict[GufeKey, float], sk_to_tf: dict[ScopedKey, Transformation]
    ) -> dict[GufeKey, float]:
        """Set weights to None for transformations with errored tasks."""
        for transformation_sk in sk_to_tf:
            counts = self.n4js.get_transformation_status(transformation_sk)
            if counts.get(TaskStatusEnum.error.value):
                weights[sk_to_tf[transformation_sk].key] = None
        return weights

    def _get_actionable_tasks(
        self, transformation_sk: ScopedKey, actioned_tasks: list[ScopedKey]
    ) -> dict[ScopedKey, TaskStatusEnum]:
        """Get actionable tasks for a transformation, excluding already actioned ones."""
        transformation_tasks = self.n4js.get_transformation_tasks(transformation_sk)
        return {
            task_sk: status
            for task_sk, status in zip(
                transformation_tasks,
                self.n4js.get_task_status(transformation_tasks),
            )
            if status in [TaskStatusEnum.waiting, TaskStatusEnum.running]
            and task_sk not in actioned_tasks
        }

    def _select_tasks_to_action(
        self, actionable_tasks_status: dict[ScopedKey, TaskStatusEnum], required: int
    ) -> list[ScopedKey]:
        """Select tasks to action, prioritizing running tasks first."""
        tasks_to_action = []

        # Add actionable tasks not already actioned, starting with those that are already running
        for task_sk, status in actionable_tasks_status.items():
            if status == TaskStatusEnum.running:
                tasks_to_action.append(task_sk)
                if len(tasks_to_action) >= required:
                    break

        # If we still need more, add actionable tasks that are waiting
        if len(tasks_to_action) < required:
            for task_sk, status in actionable_tasks_status.items():
                if status == TaskStatusEnum.waiting:
                    tasks_to_action.append(task_sk)
                    if len(tasks_to_action) >= required:
                        break

        return tasks_to_action

    def _action_additional_tasks(
        self,
        transformation_sk: ScopedKey,
        taskhub_sk: ScopedKey,
        target_count: int,
        actioned_count: int,
        actioned_tasks: list[ScopedKey],
    ) -> None:
        """Action additional tasks for a transformation, creating new ones if needed."""
        required = target_count - actioned_count

        # Get existing actionable tasks for the transformation
        actionable_tasks_status = self._get_actionable_tasks(
            transformation_sk, actioned_tasks
        )

        # Select tasks to action, prioritizing running tasks
        tasks_to_action = self._select_tasks_to_action(
            actionable_tasks_status, required
        )

        # Create new tasks if needed
        if len(tasks_to_action) < required:
            new_tasks = self.n4js.create_tasks(
                [transformation_sk] * (required - len(tasks_to_action))
            )
            tasks_to_action.extend(new_tasks)

        self.n4js.action_tasks(tasks_to_action, taskhub_sk)

    def _select_tasks_to_cancel(
        self, actioned_tasks: list[ScopedKey], excess: int
    ) -> list[ScopedKey]:
        """Select tasks to cancel, prioritizing waiting tasks first."""
        tasks_to_cancel = []
        actioned_status = self.n4js.get_task_status(actioned_tasks)

        # First cancel waiting tasks
        for task_sk, status in zip(actioned_tasks, actioned_status):
            if status == TaskStatusEnum.waiting:
                tasks_to_cancel.append(task_sk)
                if len(tasks_to_cancel) >= excess:
                    break

        # Then cancel running tasks if needed
        if len(tasks_to_cancel) < excess:
            for task_sk, status in zip(actioned_tasks, actioned_status):
                if status == TaskStatusEnum.running:
                    tasks_to_cancel.append(task_sk)
                    if len(tasks_to_cancel) >= excess:
                        break

        return tasks_to_cancel

    def _cancel_excess_tasks(
        self,
        transformation_sk: ScopedKey,
        taskhub_sk: ScopedKey,
        target_count: int,
        actioned_count: int,
        actioned_tasks: list[ScopedKey],
    ) -> None:
        """Cancel excess tasks for a transformation in full mode."""
        excess = actioned_count - target_count

        tasks_to_cancel = self._select_tasks_to_cancel(actioned_tasks, excess)
        self.n4js.cancel_tasks(tasks_to_cancel, taskhub_sk)

    def _apply_task_counts(
        self,
        task_counts: dict[GufeKey, int],
        sk_to_tf: dict[ScopedKey, Transformation],
        network_sk: ScopedKey,
        strategy_state: StrategyState,
    ) -> None:
        """Apply task count targets by creating/cancelling tasks as needed."""
        taskhub_sk = self.n4js.get_taskhub(network_sk)

        # Create reverse mapping from transformation gufe key to ScopedKey
        tf_key_to_sk = {value.key: key for key, value in sk_to_tf.items()}

        # Set task counts for each transformation
        for transformation_key, target_count in task_counts.items():
            transformation_sk = tf_key_to_sk[transformation_key]

            # Get actioned tasks once per transformation
            actioned_tasks = self.n4js.get_transformation_actioned_tasks(
                transformation_sk, taskhub_sk
            )
            actioned_count = len(actioned_tasks)

            if target_count > actioned_count:
                # Action additional tasks, creating them as necessary
                self._action_additional_tasks(
                    transformation_sk,
                    taskhub_sk,
                    target_count,
                    actioned_count,
                    actioned_tasks,
                )

            elif (
                target_count < actioned_count
                and strategy_state.mode == StrategyModeEnum.full
            ):
                # Cancel excess tasks
                self._cancel_excess_tasks(
                    transformation_sk,
                    taskhub_sk,
                    target_count,
                    actioned_count,
                    actioned_tasks,
                )

    def _update_strategy_state_success(
        self, strategy_state: StrategyState, current_result_count: int
    ) -> StrategyState:
        """Update strategy state after successful execution."""
        strategy_state.last_iteration = datetime.datetime.now(tz=datetime.UTC)
        strategy_state.last_iteration_result_count = current_result_count
        strategy_state.iterations += 1
        strategy_state.exception = None
        strategy_state.traceback = None
        return strategy_state

    def _update_strategy_state_error(
        self,
        strategy_state: StrategyState,
        current_result_count: int,
        exception: Exception,
    ) -> StrategyState:
        """Update strategy state after execution error."""
        strategy_state.last_iteration = datetime.datetime.now(tz=datetime.UTC)
        strategy_state.last_iteration_result_count = current_result_count
        strategy_state.status = StrategyStatusEnum.error
        strategy_state.exception = (exception.__class__.__qualname__, str(exception))
        strategy_state.traceback = traceback.format_exc()
        strategy_state.iterations += 1
        return strategy_state

    def _execute_strategy(
        self, network_sk: ScopedKey, strategy_state: StrategyState
    ) -> StrategyState:
        """Execute a single strategy and return updated state."""
        current_result_count = 0

        try:
            # Check if strategy went dormant - count successful results
            current_result_count = self._count_successful_results(network_sk)

            # If dormant, check if new results appeared
            should_continue, strategy_state = self._check_dormant_status(
                strategy_state, current_result_count
            )
            if not should_continue:
                return strategy_state

            # Get protocol results and transformation mappings (expensive operation - do once)
            protocol_results, sk_to_tf = self._get_protocol_results(network_sk)

            # Execute strategy logic to get weights
            weights = self._execute_strategy_logic(
                network_sk, protocol_results, sk_to_tf
            )

            # Check if all weights are None (stop condition)
            if all(w is None for w in weights.values()):
                return self._handle_dormant_strategy(
                    strategy_state, network_sk, current_result_count
                )

            # Set weights to None for transformations with errored tasks
            weights = self._filter_errored_transformations(weights, sk_to_tf)

            # Convert weights to task counts
            task_counts = self._weights_to_task_counts(
                weights,
                strategy_state.max_tasks_per_transformation,
                strategy_state.task_scaling,
            )

            # Apply task count targets
            self._apply_task_counts(task_counts, sk_to_tf, network_sk, strategy_state)

            # Update strategy state for successful execution
            return self._update_strategy_state_success(
                strategy_state, current_result_count
            )

        except Exception as e:
            # Strategy execution failed
            logger.exception(f"Strategy execution failed for network {network_sk}")
            return self._update_strategy_state_error(
                strategy_state, current_result_count, e
            )

    def cycle(self):
        """Perform one iteration of strategy execution."""

        # avoid issues with forking processes, such as with pytest
        ctx = mp.get_context("spawn")

        # Get strategies ready for execution
        ready_strategies = self.n4js.get_strategies_for_execution(
            scopes=self.scopes, min_sleep_interval=self.sleep_interval
        )

        if not ready_strategies:
            logger.debug("No strategies ready for execution")
            return

        logger.info(f"Executing {len(ready_strategies)} strategies")

        # Use ProcessPoolExecutor for parallel strategy execution
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx
        ) as executor:
            # Submit all strategy executions using standalone worker function
            future_to_network = {
                executor.submit(
                    execute_strategy_worker, network_sk, strategy_state, self.settings
                ): network_sk
                for network_sk, strategy_sk, strategy_state in ready_strategies
            }
            network_to_future = {value: key for key, value in future_to_network.items()}

            while future_to_network and not self._stop:
                # Collect results and update database
                try:
                    for future in as_completed(
                        list(future_to_network.keys()), timeout=self.sleep_interval
                    ):
                        network_sk = future_to_network[future]
                        try:
                            updated_state = future.result()

                            # Update strategy state in database
                            success = self.n4js.update_strategy_state(
                                network_sk, updated_state
                            )
                            if not success:
                                logger.error(
                                    f"Failed to update strategy state for network {network_sk}"
                                )
                            else:
                                logger.debug(
                                    f"Updated strategy state for network {network_sk}"
                                )

                        except Exception as e:
                            logger.exception(
                                f"Strategy execution failed for network {network_sk}: {e}"
                            )

                        future_to_network.pop(future)
                        network_to_future.pop(network_sk)

                except TimeoutError:
                    # Check if we should stop before adding new strategies
                    if self._stop:
                        break

                    # if we ran out of time waiting for strategies to finish, check for new ones
                    # and continue
                    ready_strategies = self.n4js.get_strategies_for_execution(
                        scopes=self.scopes, min_sleep_interval=self.sleep_interval
                    )

                    for network_sk, strategy_sk, strategy_state in ready_strategies:
                        # Only submit strategies that aren't already running AND haven't been recently executed
                        if network_sk not in network_to_future and (
                            strategy_state.last_iteration is None
                            or (
                                datetime.datetime.now(tz=datetime.UTC)
                                - strategy_state.last_iteration
                            ).total_seconds()
                            >= self.sleep_interval
                        ):
                            # Submit new strategy execution using standalone worker function
                            future = executor.submit(
                                execute_strategy_worker,
                                network_sk,
                                strategy_state,
                                self.settings,
                            )
                            future_to_network[future] = network_sk
                            network_to_future[network_sk] = future
                            logger.debug(
                                f"Added new strategy execution for network {network_sk}"
                            )

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

    def close(self):
        """Close the Neo4j driver for this service instance."""
        self.n4js.close()
