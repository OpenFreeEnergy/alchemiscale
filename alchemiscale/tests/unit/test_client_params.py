"""Tests for AlchemiscaleBaseClientParam and environment variable handling."""

from contextlib import nullcontext
from enum import Enum, auto
import pytest
import warnings

from alchemiscale.base.client import AlchemiscaleBaseClientParam


class RecordType(Enum):
    WARN = auto()
    RAISE = auto()
    NONE = auto()


class TestAlchemiscaleBaseClientParam:
    def test_get_value_from_explicit(self):
        """Test that explicit parameter values are used correctly."""
        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=True,
        )

        value = param.get_value("explicit_value")
        assert value == "explicit_value"

    def test_get_value_from_env(self, monkeypatch):
        """Test that environment variables are used when no explicit value is provided."""
        monkeypatch.setenv("TEST_VAR", "env_value")

        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=True,
        )

        value = param.get_value(None)
        assert value == "env_value"

    def test_get_value_explicit_overrides_env(self, monkeypatch):
        """Test that explicit values override environment variables with warning."""
        monkeypatch.setenv("TEST_VAR", "env_value")

        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=True,
        )

        with pytest.warns(UserWarning) as record:
            value = param.get_value("explicit_value")

        assert value == "explicit_value"
        assert len(record) == 1
        assert "Environment variable TEST_VAR is set to 'env_value'" in str(
            record[0].message
        )
        assert "but an explicit test parameter 'explicit_value' is provided" in str(
            record[0].message
        )
        assert "Using the explicit test parameter" in str(record[0].message)

    def test_get_value_no_value_raises(self):
        """Test that ValueError is raised when no value is available."""
        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR_NOT_SET",
            human_name="test parameter",
            render_value=True,
        )

        with pytest.raises(ValueError) as exc:
            param.get_value(None)

        assert (
            "No test parameter provided and TEST_VAR_NOT_SET environment variable not set"
            in str(exc.value)
        )

    def test_render_value_false(self, monkeypatch):
        """Test that sensitive values are not rendered in warnings."""
        monkeypatch.setenv("TEST_VAR", "sensitive_value")

        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=False,
        )

        with pytest.warns(UserWarning) as record:
            value = param.get_value("explicit_sensitive")

        assert value == "explicit_sensitive"
        assert len(record) == 1
        assert "Environment variable TEST_VAR is set" in str(record[0].message)
        assert "sensitive_value" not in str(record[0].message)
        assert "explicit_sensitive" not in str(record[0].message)

    # Test for all 5 cases (both same, both different, env only, param only, both not set)
    @pytest.mark.parametrize(
        "parameter,env_var,expected_context",
        [
            (
                "same_value",
                "same_value",
                {"context": nullcontext(), "msg": None, "type": RecordType.NONE},
            ),
            (
                "value",
                "different_value",
                {
                    "context": pytest.warns(UserWarning),
                    "msg": (
                        "Environment variable TEST_VAR is set to 'different_value'"
                        ", but an explicit test parameter 'value' is provided."
                        " Using the explicit test parameter."
                    ),
                    "type": RecordType.WARN,
                },
            ),
            (
                None,
                "value",
                {
                    "context": warnings.catch_warnings(action="error"),
                    "msg": None,
                    "type": RecordType.NONE,
                },
            ),
            (
                "value",
                None,
                {
                    "context": warnings.catch_warnings(action="error"),
                    "msg": None,
                    "type": RecordType.NONE,
                },
            ),
            (
                None,
                None,
                {
                    "context": pytest.raises(ValueError),
                    "msg": "No test parameter provided and TEST_VAR environment variable not set",
                    "type": RecordType.RAISE,
                },
            ),
        ],
    )
    def test_warn_override(self, monkeypatch, parameter, env_var, expected_context):
        """Test that no warning is issued when explicit value matches environment variable."""
        if env_var is not None:
            monkeypatch.setenv("TEST_VAR", env_var)

        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=True,
        )

        with expected_context["context"] as record:
            value = param.get_value(parameter)

        match expected_context["type"]:
            case RecordType.WARN:
                assert len(record) == 1
                assert record[0].message.args[0] == expected_context["msg"]
            case RecordType.RAISE:
                assert str(record.value) == expected_context["msg"]
                return
            case RecordType.NONE:
                pass

        assert value == parameter if parameter is not None else env_var
