import pytest
from datetime import datetime
from unittest.mock import MagicMock

from uuid import uuid4

from alchemiscale.storage.models import (
    NetworkStateEnum,
    NetworkMark,
    TaskRestartPattern,
    Tracebacks,
    StrategyState,
    StrategyModeEnum,
    StrategyStatusEnum,
    StrategyTaskScalingEnum,
    ComputeManagerID,
)
from alchemiscale import ScopedKey


class TestNetworkState:

    network_sk = ScopedKey.from_str(
        "AlchemicalNetwork-foo123-FakeOrg-FakeCampaign-FakeProject"
    )

    @pytest.mark.parametrize(
        ("state", "should_fail"),
        [
            ("active", False),
            ("inactive", False),
            ("invalid", False),
            ("deleted", False),
            ("Active", True),
            ("NotAState", True),
        ],
    )
    def test_enum_values(self, state: str, should_fail: bool):

        if should_fail:
            with pytest.raises(ValueError, match="`state` = "):
                NetworkMark(self.network_sk, state)
        else:
            NetworkMark(self.network_sk, state)

    def test_suggested_states_message(self):
        try:
            NetworkMark(self.network_sk, "NotAState")
        except ValueError as e:
            emessage = str(e)
            suggested_states = emessage.split(":")[1].strip().split(", ")
            assert len(suggested_states) == len(NetworkStateEnum)
            for state in suggested_states:
                NetworkStateEnum(state)


class TestTaskRestartPattern:

    pattern_value_error = "`pattern` must be a non-empty string"
    max_retries_value_error = "`max_retries` must have a positive integer value."

    def test_empty_pattern(self):
        with pytest.raises(ValueError, match=self.pattern_value_error):
            _ = TaskRestartPattern(
                "", 3, "FakeScopedKey-1234-fake_org-fake_campaign-fake_project"
            )

    def test_non_string_pattern(self):
        with pytest.raises(ValueError, match=self.pattern_value_error):
            _ = TaskRestartPattern(
                None, 3, "FakeScopedKey-1234-fake_org-fake_campaign-fake_project"
            )

        with pytest.raises(ValueError, match=self.pattern_value_error):
            _ = TaskRestartPattern(
                [], 3, "FakeScopedKey-1234-fake_org-fake_campaign-fake_project"
            )

    def test_non_positive_max_retries(self):

        with pytest.raises(ValueError, match=self.max_retries_value_error):
            TaskRestartPattern(
                "Example pattern",
                0,
                "FakeScopedKey-1234-fake_org-fake_campaign-fake_project",
            )

        with pytest.raises(ValueError, match=self.max_retries_value_error):
            TaskRestartPattern(
                "Example pattern",
                -1,
                "FakeScopedKey-1234-fake_org-fake_campaign-fake_project",
            )

    def test_non_int_max_retries(self):
        with pytest.raises(ValueError, match=self.max_retries_value_error):
            TaskRestartPattern(
                "Example pattern",
                4.0,
                "FakeScopedKey-1234-fake_org-fake_campaign-fake_project",
            )

    def test_to_dict(self):
        expected = {
            "taskhub_scoped_key": "FakeScopedKey-1234-fake_org-fake_campaign-fake_project",
            "max_retries": 3,
            "pattern": "Example pattern",
        }

        trp = TaskRestartPattern(
            expected["pattern"],
            expected["max_retries"],
            expected["taskhub_scoped_key"],
        )
        dict_trp = trp.to_dict()

        for key, value in expected.items():
            assert dict_trp[key] == value

    def test_from_dict(self):

        original_pattern = "Example pattern"
        original_max_retries = 3
        original_taskhub_scoped_key = (
            "FakeScopedKey-1234-fake_org-fake_campaign-fake_project"
        )

        trp_orig = TaskRestartPattern(
            original_pattern, original_max_retries, original_taskhub_scoped_key
        )
        trp_dict = trp_orig.to_dict()
        trp_reconstructed: TaskRestartPattern = TaskRestartPattern.from_dict(trp_dict)

        assert trp_reconstructed.pattern == original_pattern
        assert trp_reconstructed.max_retries == original_max_retries
        assert trp_reconstructed.taskhub_scoped_key == original_taskhub_scoped_key

        assert trp_orig is trp_reconstructed


class TestTracebacks:

    valid_entry = ["traceback1", "traceback2", "traceback3"]
    source_keys = ["ProtocolUnit-ABC123", "ProtocolUnit-DEF456", "ProtocolUnit-GHI789"]
    failure_keys = [
        "ProtocolUnitFailure-ABC123",
        "ProtocolUnitFailure-DEF456",
        "ProtocolUnitFailure-GHI789",
    ]
    tracebacks_value_error = (
        "`tracebacks` must be a non-empty list of non-empty string values"
    )

    def test_empty_string_element(self):
        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks(self.valid_entry + [""], self.source_keys, self.failure_keys)

    def test_non_list_parameter(self):
        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks(None, self.source_keys, self.failure_keys)

        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks(100, self.source_keys, self.failure_keys)

        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks(
                "not a list, but still an iterable that yields strings",
                self.source_keys,
                self.failure_keys,
            )

    def test_list_non_string_elements(self):
        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks(self.valid_entry + [None], self.source_keys, self.failure_keys)

    def test_empty_list(self):
        with pytest.raises(ValueError, match=self.tracebacks_value_error):
            Tracebacks([], self.source_keys, self.failure_keys)

    def test_to_dict(self):

        expected = {
            "tracebacks": self.valid_entry,
            "source_keys": self.source_keys,
            "failure_keys": self.failure_keys,
        }

        tb = Tracebacks(
            expected["tracebacks"], expected["source_keys"], expected["failure_keys"]
        )
        tb_dict = tb.to_dict()

        for key, value in expected.items():
            assert tb_dict[key] == value

    def test_from_dict(self):
        tb_orig = Tracebacks(self.valid_entry, self.source_keys, self.failure_keys)
        tb_dict = tb_orig.to_dict()
        tb_reconstructed: TaskRestartPattern = TaskRestartPattern.from_dict(tb_dict)

        assert tb_reconstructed.tracebacks == self.valid_entry
        tb_orig is tb_reconstructed


class TestStrategyState:

    def test_default_values(self):
        """Test that StrategyState has correct default values."""
        state = StrategyState()

        assert state.mode == StrategyModeEnum.partial
        assert state.status == StrategyStatusEnum.awake
        assert state.iterations == 0
        assert state.sleep_interval == 3600
        assert state.last_iteration is None
        assert state.last_iteration_result_count == 0
        assert state.max_tasks_per_transformation == 3
        assert state.task_scaling == StrategyTaskScalingEnum.exponential
        assert state.exception is None
        assert state.traceback is None

    def test_enum_validation(self):
        """Test that enum fields validate correctly."""
        # Valid enum values should work
        state = StrategyState(
            mode=StrategyModeEnum.full,
            status=StrategyStatusEnum.dormant,
            task_scaling=StrategyTaskScalingEnum.linear,
        )
        assert state.mode == StrategyModeEnum.full
        assert state.status == StrategyStatusEnum.dormant
        assert state.task_scaling == StrategyTaskScalingEnum.linear

        # String enum values should also work
        state = StrategyState(
            mode="disabled",
            status="error",
            task_scaling="exponential",
        )
        assert state.mode == StrategyModeEnum.disabled
        assert state.status == StrategyStatusEnum.error
        assert state.task_scaling == StrategyTaskScalingEnum.exponential

    def test_positive_int_validation(self):
        """Test that PositiveInt fields validate correctly."""
        # Valid positive integers should work
        state = StrategyState(
            sleep_interval=1800,
            max_tasks_per_transformation=5,
        )
        assert state.sleep_interval == 1800
        assert state.max_tasks_per_transformation == 5

        # Zero and negative values should fail
        with pytest.raises(ValueError):
            StrategyState(sleep_interval=0)

        with pytest.raises(ValueError):
            StrategyState(sleep_interval=-1)

        with pytest.raises(ValueError):
            StrategyState(max_tasks_per_transformation=0)

        with pytest.raises(ValueError):
            StrategyState(max_tasks_per_transformation=-1)

    def test_neo4j_datetime_conversion(self):
        """Test that Neo4j DateTime objects are converted to Python datetime."""
        # Mock a Neo4j DateTime object
        mock_neo4j_datetime = MagicMock()
        mock_neo4j_datetime.to_native.return_value = datetime(2023, 12, 25, 10, 30, 0)

        state = StrategyState(last_iteration=mock_neo4j_datetime)

        # Should be converted to Python datetime
        assert isinstance(state.last_iteration, datetime)
        assert state.last_iteration == datetime(2023, 12, 25, 10, 30, 0)
        mock_neo4j_datetime.to_native.assert_called_once()

    def test_datetime_passthrough(self):
        """Test that regular datetime objects are passed through unchanged."""
        test_datetime = datetime(2023, 12, 25, 10, 30, 0)
        state = StrategyState(last_iteration=test_datetime)

        assert state.last_iteration == test_datetime
        assert isinstance(state.last_iteration, datetime)

    def test_none_datetime(self):
        """Test that None datetime values are handled correctly."""
        state = StrategyState(last_iteration=None)
        assert state.last_iteration is None

    def test_exception_tuple(self):
        """Test exception tuple handling."""
        exception_info = ("ValueError", "Test error message")
        state = StrategyState(exception=exception_info)

        assert state.exception == exception_info
        assert state.exception[0] == "ValueError"
        assert state.exception[1] == "Test error message"

    def test_to_dict(self):
        """Test conversion to dictionary with enum values as strings."""
        test_datetime = datetime(2023, 12, 25, 10, 30, 0)
        exception_info = ("ValueError", "Test error")

        state = StrategyState(
            mode=StrategyModeEnum.full,
            status=StrategyStatusEnum.dormant,
            iterations=5,
            sleep_interval=1800,
            last_iteration=test_datetime,
            last_iteration_result_count=10,
            max_tasks_per_transformation=5,
            task_scaling=StrategyTaskScalingEnum.linear,
            exception=exception_info,
            traceback="Test traceback",
        )

        result = state.to_dict()

        assert result["mode"] == "full"
        assert result["status"] == "dormant"
        assert result["task_scaling"] == "linear"
        assert result["iterations"] == 5
        assert result["sleep_interval"] == 1800
        assert result["last_iteration"] == test_datetime
        assert result["last_iteration_result_count"] == 10
        assert result["max_tasks_per_transformation"] == 5
        assert result["exception"] == exception_info
        assert result["traceback"] == "Test traceback"

    def test_from_dict(self):
        """Test creation from dictionary."""
        test_datetime = datetime(2023, 12, 25, 10, 30, 0)
        exception_info = ("ValueError", "Test error")

        data = {
            "mode": "full",
            "status": "dormant",
            "iterations": 5,
            "sleep_interval": 1800,
            "last_iteration": test_datetime,
            "last_iteration_result_count": 10,
            "max_tasks_per_transformation": 5,
            "task_scaling": "linear",
            "exception": exception_info,
            "traceback": "Test traceback",
        }

        state = StrategyState.from_dict(data)

        assert state.mode == StrategyModeEnum.full
        assert state.status == StrategyStatusEnum.dormant
        assert state.task_scaling == StrategyTaskScalingEnum.linear
        assert state.iterations == 5
        assert state.sleep_interval == 1800
        assert state.last_iteration == test_datetime
        assert state.last_iteration_result_count == 10
        assert state.max_tasks_per_transformation == 5
        assert state.exception == exception_info
        assert state.traceback == "Test traceback"

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict and from_dict are consistent."""
        original = StrategyState(
            mode=StrategyModeEnum.disabled,
            status=StrategyStatusEnum.error,
            iterations=3,
            sleep_interval=900,
            last_iteration=datetime(2023, 12, 25, 10, 30, 0),
            last_iteration_result_count=7,
            max_tasks_per_transformation=2,
            task_scaling=StrategyTaskScalingEnum.exponential,
            exception=("RuntimeError", "Test runtime error"),
            traceback="Full traceback here",
        )

        # Convert to dict and back
        data = original.to_dict()
        reconstructed = StrategyState.from_dict(data)

        assert original == reconstructed

    def test_validate_assignment_config(self):
        """Test that validate_assignment=True works correctly."""
        state = StrategyState()

        # Should allow valid enum assignment
        state.mode = StrategyModeEnum.full
        state.status = "dormant"  # String should be converted to enum

        assert state.mode == StrategyModeEnum.full
        assert state.status == StrategyStatusEnum.dormant

        # Should validate positive integers on assignment
        state.sleep_interval = 1200
        assert state.sleep_interval == 1200

        with pytest.raises(ValueError):
            state.sleep_interval = -1

    def test_all_enum_values(self):
        """Test that all enum values work correctly."""
        # Test all StrategyModeEnum values
        for mode in StrategyModeEnum:
            state = StrategyState(mode=mode)
            assert state.mode == mode

        # Test all StrategyStatusEnum values
        for status in StrategyStatusEnum:
            state = StrategyState(status=status)
            assert state.status == status

        # Test all StrategyTaskScalingEnum values
        for scaling in StrategyTaskScalingEnum:
            state = StrategyState(task_scaling=scaling)
            assert state.task_scaling == scaling


class TestComputeManagerID:

    name = "testmanager"

    def test_to_from_dict(self):
        manager_id = ComputeManagerID.new_from_name(self.name)
        dct_form = manager_id.to_dict()
        assert ComputeManagerID.from_dict(dct_form) == manager_id

    def test_invalid_uuid_form(self):
        # try using the int form of the uuid
        manager_uuid = str(int(uuid4()))

        with pytest.raises(ValueError, match="ComputeManagerID must have the form"):
            manager_id = ComputeManagerID(self.name + "-" + manager_uuid)

    def test_broken_uuid(self):
        original = "676b919a-a206-4f24-9134-3cb326ad127b"
        manager_uuid = "Z" + original

        with pytest.raises(ValueError, match="Could not interpret the provided UUID"):
            manager_id = ComputeManagerID(self.name + "-" + manager_uuid)

    def test_bad_name(self):
        name = "test_manager"

        with pytest.raises(
            ValueError, match="ComputeManagerID only allows alpha-numeric names"
        ):
            ComputeManagerID.new_from_name(name)
