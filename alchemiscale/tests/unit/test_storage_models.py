import pytest

from alchemiscale.storage.models import (
    NetworkStateEnum,
    NetworkMark,
    TaskRestartPattern,
    Tracebacks,
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

    def test_to_dict(self): ...

    def test_from_dict(self): ...
