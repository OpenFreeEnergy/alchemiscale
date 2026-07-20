"""Unit tests for best-effort compute-environment capture
(:mod:`alchemiscale.compute.environment`)."""

import json
import subprocess

import pytest

from alchemiscale.compute import environment as envmod
from alchemiscale.compute.environment import capture_environment, _parse_packages


def _fake_run_factory(outputs: dict[str, tuple[int, str]]):
    """Build a fake ``subprocess.run`` returning per-tool ``(returncode, stdout)``."""

    def _fake_run(argv, **kwargs):
        tool = argv[0]
        returncode, stdout = outputs.get(tool, (1, ""))
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout=stdout, stderr=""
        )

    return _fake_run


def _install(monkeypatch, present: set[str], outputs: dict[str, tuple[int, str]]):
    monkeypatch.setattr(
        envmod.shutil, "which", lambda name: name if name in present else None
    )
    monkeypatch.setattr(envmod.subprocess, "run", _fake_run_factory(outputs))


CONDA_JSON = json.dumps(
    [
        {"name": "gufe", "version": "1.10.0", "channel": "conda-forge"},
        {"name": "python", "version": "3.11.9", "channel": "conda-forge"},
    ]
)
PIP_JSON = json.dumps(
    [{"name": "gufe", "version": "1.10.0"}, {"name": "pip", "version": "24.0"}]
)


class TestParsePackages:
    def test_conda_shape(self):
        assert _parse_packages(CONDA_JSON) == {"gufe": "1.10.0", "python": "3.11.9"}

    def test_pip_shape(self):
        assert _parse_packages(PIP_JSON) == {"gufe": "1.10.0", "pip": "24.0"}

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            _parse_packages("[]")

    def test_non_list_raises(self):
        with pytest.raises(ValueError):
            _parse_packages('{"not": "a list"}')

    def test_bad_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_packages("not json")


class TestCaptureEnvironment:
    def test_first_tool_wins(self, monkeypatch):
        # micromamba present and successful -> used, later tools untried
        _install(
            monkeypatch,
            present={"micromamba", "conda", "pip"},
            outputs={"micromamba": (0, CONDA_JSON)},
        )
        env = capture_environment()
        assert env["tool"] == "micromamba"
        assert env["packages"] == {"gufe": "1.10.0", "python": "3.11.9"}
        assert env["captured_at"]

    def test_falls_through_missing_tools(self, monkeypatch):
        # micromamba/mamba/conda absent -> pip used
        _install(monkeypatch, present={"pip"}, outputs={"pip": (0, PIP_JSON)})
        env = capture_environment()
        assert env["tool"] == "pip"
        assert env["packages"] == {"gufe": "1.10.0", "pip": "24.0"}

    def test_falls_through_nonzero_exit(self, monkeypatch):
        # conda present but errors -> falls through to pip
        _install(
            monkeypatch,
            present={"conda", "pip"},
            outputs={"conda": (1, ""), "pip": (0, PIP_JSON)},
        )
        assert capture_environment()["tool"] == "pip"

    def test_falls_through_unparseable_output(self, monkeypatch):
        # conda present, returns garbage -> falls through to pip
        _install(
            monkeypatch,
            present={"conda", "pip"},
            outputs={"conda": (0, "not json"), "pip": (0, PIP_JSON)},
        )
        assert capture_environment()["tool"] == "pip"

    def test_falls_through_empty_listing(self, monkeypatch):
        # conda present, returns empty list (no packages) -> falls through
        _install(
            monkeypatch,
            present={"conda", "pip"},
            outputs={"conda": (0, "[]"), "pip": (0, PIP_JSON)},
        )
        assert capture_environment()["tool"] == "pip"

    def test_no_tools_returns_none(self, monkeypatch):
        _install(monkeypatch, present=set(), outputs={})
        assert capture_environment() is None

    def test_all_tools_fail_returns_none(self, monkeypatch):
        _install(
            monkeypatch,
            present={"micromamba", "mamba", "conda", "pip"},
            outputs={t: (1, "") for t in ("micromamba", "mamba", "conda", "pip")},
        )
        assert capture_environment() is None

    def test_subprocess_error_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(envmod.shutil, "which", lambda name: name)

        def _boom(argv, **kwargs):
            if argv[0] == "pip":
                return subprocess.CompletedProcess(argv, 0, PIP_JSON, "")
            raise subprocess.TimeoutExpired(argv, 1)

        monkeypatch.setattr(envmod.subprocess, "run", _boom)
        # the conda-family tools time out; pip succeeds -> never raises
        assert capture_environment()["tool"] == "pip"
