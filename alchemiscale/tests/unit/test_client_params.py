"""Tests for AlchemiscaleBaseClientParam and environment variable handling."""

import os
import pytest
import warnings

from alchemiscale.base.client import AlchemiscaleBaseClientParam


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

    def test_same_value_no_warning(self, monkeypatch):
        """Test that no warning is issued when explicit value matches environment variable."""
        monkeypatch.setenv("TEST_VAR", "same_value")

        param = AlchemiscaleBaseClientParam(
            param_name="test",
            env_var_name="TEST_VAR",
            human_name="test parameter",
            render_value=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            value = param.get_value("same_value")

        assert value == "same_value"
