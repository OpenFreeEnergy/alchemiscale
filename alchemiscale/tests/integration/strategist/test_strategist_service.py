"""Integration tests for StrategistService."""

import pytest
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.objectstore import S3ObjectStore
from alchemiscale.storage.models import (
    StrategyState,
    StrategyStatusEnum,
    StrategyModeEnum,
    StrategyTaskScalingEnum,
    TaskStatusEnum,
)
from alchemiscale.strategist.service import StrategistService
from alchemiscale.strategist.settings import StrategistSettings
from alchemiscale.settings import Neo4jStoreSettings, S3ObjectStoreSettings
from alchemiscale.compression import compress_gufe_zstd
from ..conftest import DummyStrategy


class DormantStrategy(DummyStrategy):
    """Strategy that returns None weights to trigger dormant state."""

    def _propose(self, alchemical_network, protocol_results):
        from stratocaster.base import StrategyResult

        # Return None weights to trigger dormant state
        weights = {tf.key: None for tf in alchemical_network.edges}
        return StrategyResult(weights)


class BrokenStrategy(DummyStrategy):
    """Strategy that raises an exception for error handling tests."""

    def _propose(self, alchemical_network, protocol_results):
        raise ValueError("Intentional test error")


class TestStrategistService:
    """Test the StrategistService integration with Neo4j and S3."""

    @pytest.fixture
    def strategist_settings(self, tmpdir, s3objectstore_settings_endpoint, uri):
        """Settings for StrategistService tests."""
        return StrategistSettings(
            sleep_interval=1,
            max_workers=2,
            cache_directory=Path(tmpdir) / "strategist_cache",
            cache_size_limit=10485760,  # 10 MB
            use_local_cache=True,
            neo4j_settings=Neo4jStoreSettings(
                NEO4J_URL=uri, NEO4J_USER="neo4j", NEO4J_PASS="password"
            ),
            s3_settings=s3objectstore_settings_endpoint,
        )

    @pytest.fixture
    def strategist_service(self, strategist_settings, n4js_fresh, s3os_server_fresh):
        """StrategistService instance for testing."""
        return StrategistService(strategist_settings)

    @pytest.fixture
    def preloaded_network_with_strategy(
        self, n4js_fresh, s3os_server_fresh, network_tyk2, scope_test
    ):
        """Network with strategy and some completed results."""
        n4js: Neo4jStore = n4js_fresh
        s3os: S3ObjectStore = s3os_server_fresh

        # Create network
        network_sk, taskhub_sk, _ = n4js.assemble_network(network_tyk2, scope_test)

        # Set a strategy
        strategy = DummyStrategy()
        n4js.set_network_strategy(
            network_sk,
            strategy,
            StrategyState(
                status=StrategyStatusEnum.awake,
                mode=StrategyModeEnum.partial,
                max_tasks_per_transformation=2,
                task_scaling=StrategyTaskScalingEnum.linear,
                last_iteration=None,
                last_iteration_result_count=0,
                iterations=0,
            ),
        )

        # Create some initial tasks
        transformation_sks = n4js.get_network_transformations(network_sk)[
            :2
        ]  # Just first 2
        task_sks = n4js.create_tasks(
            transformation_sks * 3
        )  # 3 tasks per transformation

        return {
            "network_sk": network_sk,
            "taskhub_sk": taskhub_sk,
            "transformation_sks": transformation_sks,
            "task_sks": task_sks,
            "n4js": n4js,
            "s3os": s3os,
        }

    def test_strategist_service_initialization(self, strategist_service):
        """Test that StrategistService initializes correctly."""
        assert strategist_service.sleep_interval == 1
        assert strategist_service.max_workers == 2
        assert strategist_service._cache_enabled is True
        assert strategist_service._cache is not None
        assert strategist_service.n4js is not None
        assert strategist_service.s3os is not None

    def test_cache_directory_creation(self, tmpdir, strategist_settings):
        """Test that cache directory is created correctly."""
        cache_dir = Path(tmpdir) / "test_cache"
        assert not cache_dir.exists()

        settings = StrategistSettings(
            cache_directory=cache_dir,
            use_local_cache=True,
            neo4j_settings=strategist_settings.neo4j_settings,
            s3_settings=strategist_settings.s3_settings,
        )
        service = StrategistService(settings)

        assert cache_dir.exists()
        assert service._cache_directory == cache_dir

    def test_execute_strategy_no_ready_strategies(self, strategist_service, n4js_fresh):
        """Test cycle method when no strategies are ready."""
        # Should complete without error
        strategist_service.cycle()

        # Verify no strategies were found (just checking it doesn't crash)
        assert True

    def test_execute_strategy_basic(
        self, strategist_service, preloaded_network_with_strategy
    ):
        """Test basic strategy execution."""
        data = preloaded_network_with_strategy
        network_sk = data["network_sk"]
        n4js = data["n4js"]

        # Get the current strategy state
        strategy_state = n4js.get_network_strategy_state(network_sk)
        assert strategy_state.status == StrategyStatusEnum.awake
        assert strategy_state.iterations == 0

        # Execute one cycle
        strategist_service.cycle()

        # Check that strategy was executed
        updated_state = n4js.get_network_strategy_state(network_sk)
        assert updated_state.iterations == 1
        assert updated_state.last_iteration is not None

    def test_strategy_with_dormant_state(
        self, strategist_service, preloaded_network_with_strategy
    ):
        """Test strategy execution when strategy goes dormant."""
        data = preloaded_network_with_strategy
        network_sk = data["network_sk"]
        n4js = data["n4js"]

        # Use the module-level DormantStrategy class
        dormant_strategy = DormantStrategy()
        n4js.set_network_strategy(
            network_sk,
            dormant_strategy,
            StrategyState(
                status=StrategyStatusEnum.awake,
                mode=StrategyModeEnum.partial,
                max_tasks_per_transformation=2,
                task_scaling=StrategyTaskScalingEnum.linear,
                last_iteration=None,
                last_iteration_result_count=0,
                iterations=0,
            ),
        )

        # Execute one cycle
        strategist_service.cycle()

        # Strategy should now be dormant
        updated_state = n4js.get_network_strategy_state(network_sk)
        assert updated_state.status == StrategyStatusEnum.dormant
        assert updated_state.iterations == 1

    def test_strategy_task_creation_and_actioning(
        self, strategist_service, preloaded_network_with_strategy
    ):
        """Test that strategy execution creates and actions tasks correctly."""
        data = preloaded_network_with_strategy
        network_sk = data["network_sk"]
        taskhub_sk = data["taskhub_sk"]
        transformations = data["transformation_sks"]
        n4js = data["n4js"]

        # Initially should have no actioned tasks
        for tf_sk in transformations:
            actioned = n4js.get_transformation_actioned_tasks(tf_sk, taskhub_sk)
            assert len(actioned) == 0

        # Execute strategy
        strategist_service.cycle()

        # Now should have actioned tasks based on strategy weights
        total_actioned = 0
        for tf_sk in transformations:
            actioned = n4js.get_transformation_actioned_tasks(tf_sk, taskhub_sk)
            total_actioned += len(actioned)

            # With DummyStrategy returning 0.5 weight and max_tasks_per_transformation=2,
            # linear scaling should give 1 + 0.5 * 2 = 2 tasks per transformation
            assert len(actioned) <= 2  # Should not exceed max

        assert total_actioned > 0

    def test_strategy_error_handling(
        self, strategist_service, preloaded_network_with_strategy
    ):
        """Test strategy execution error handling."""
        data = preloaded_network_with_strategy
        network_sk = data["network_sk"]
        n4js = data["n4js"]

        # Use the module-level BrokenStrategy class
        broken_strategy = BrokenStrategy()
        n4js.set_network_strategy(
            network_sk,
            broken_strategy,
            StrategyState(
                status=StrategyStatusEnum.awake,
                mode=StrategyModeEnum.partial,
                max_tasks_per_transformation=2,
                task_scaling=StrategyTaskScalingEnum.linear,
                last_iteration=None,
                last_iteration_result_count=0,
                iterations=0,
            ),
        )

        # Execute one cycle - should not crash
        strategist_service.cycle()

        # Strategy should be in error state
        updated_state = n4js.get_network_strategy_state(network_sk)
        assert updated_state.status == StrategyStatusEnum.error
        assert updated_state.exception is not None
        assert updated_state.traceback is not None
        assert "Intentional test error" in updated_state.exception[1]

    def test_cache_functionality(
        self,
        strategist_service,
        preloaded_network_with_strategy,
        protocoldagresults,
        transformation,
    ):
        """Test that caching works correctly."""
        data = preloaded_network_with_strategy

        network_sk = data["network_sk"]
        transformations = data["transformation_sks"]
        task_sks = data["task_sks"]
        n4js = data["n4js"]
        s3os = data["s3os"]

        strategist_service._cache.clear()
        strategist_service._cache.stats(reset=True)

        # Store some results in S3 using the transformation fixture that matches protocoldagresults
        tf_sk = n4js.get_scoped_key(transformation, network_sk.scope)

        protocoldagresult = protocoldagresults[0]

        result_ref = s3os.push_protocoldagresult(
            compress_gufe_zstd(protocoldagresult),
            protocoldagresult.ok(),
            protocoldagresult.key,
            transformation=tf_sk,
        )

        # Create a task for this transformation since it may not have any
        task_sk = n4js.create_task(tf_sk)
        n4js.set_task_result(task_sk, result_ref)

        # Get protocol results (should cache them)
        results1, sk_to_tf1 = strategist_service._get_protocol_results(network_sk)

        # We expect cache misses since the requested results weren't cached on
        # the call above yet
        assert (
            strategist_service._cache.stats() == (0, 1)
            and len(strategist_service._cache) == 1
        )

        # Get them again (should use cache)
        results2, sk_to_tf2 = strategist_service._get_protocol_results(network_sk)

        assert (
            strategist_service._cache.stats() == (1, 1)
            and len(strategist_service._cache) == 1
        )

        # Should be the same
        assert results1 == results2

    def test_weights_to_task_counts_linear(self, strategist_service):
        """Test weight to task count conversion with linear scaling."""
        from alchemiscale.models import ScopedKey

        weights = {
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-abc123"
            ): 0.5,
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-def456"
            ): 1.0,
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-ghi789"
            ): 0.0,
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-jkl012"
            ): None,
        }

        task_counts = strategist_service._weights_to_task_counts(
            weights,
            max_tasks_per_transformation=4,
            task_scaling=StrategyTaskScalingEnum.linear,
        )

        # Linear scaling: 1 + weight * max_tasks_per_transformation
        # 0.5 -> 1 + 0.5 * 4 = 3
        # 1.0 -> 4 (max)
        # 0.0 -> 0
        # None -> 0

        assert task_counts[list(weights.keys())[0]] == 3  # 0.5 weight
        assert task_counts[list(weights.keys())[1]] == 4  # 1.0 weight
        assert task_counts[list(weights.keys())[2]] == 0  # 0.0 weight
        assert task_counts[list(weights.keys())[3]] == 0  # None weight

    def test_weights_to_task_counts_exponential(self, strategist_service):
        """Test weight to task count conversion with exponential scaling."""
        from alchemiscale.models import ScopedKey

        weights = {
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-abc123"
            ): 0.5,
            ScopedKey.from_str(
                "Transformation-test_org-test_campaign-test_project-def456"
            ): 1.0,
        }

        task_counts = strategist_service._weights_to_task_counts(
            weights,
            max_tasks_per_transformation=4,
            task_scaling=StrategyTaskScalingEnum.exponential,
        )

        # Exponential scaling: (1 + max_tasks_per_transformation) ** weight
        # 0.5 -> (1 + 4) ** 0.5 = 5 ** 0.5 ≈ 2.236 -> 2
        # 1.0 -> (1 + 4) ** 1.0 = 5 -> 4 (capped at max)

        assert task_counts[list(weights.keys())[0]] == 2  # 0.5 weight
        assert task_counts[list(weights.keys())[1]] == 4  # 1.0 weight (capped)

    def test_service_stop_flag(self, strategist_service):
        """Test that service stop flag works correctly."""
        assert strategist_service._stop is False

        strategist_service.stop()
        assert strategist_service._stop is True
