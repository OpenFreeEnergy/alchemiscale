import datetime
import json

from click.testing import CliRunner

from alchemiscale.cli import cli
from alchemiscale.compute.client import AlchemiscaleComputeManagerClient
from alchemiscale.storage.models import (
    ComputeManagerError,
    ComputeManagerRegistration,
    ComputeManagerState,
)

NOW = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.UTC)

ARGS = ["--api-url", "http://server/", "--identifier", "me", "--key", "secret"]


def _status(registered: bool):
    state = None
    if registered:
        state = ComputeManagerState(
            registration=ComputeManagerRegistration(
                name="testmanager",
                uuid="a" * 32,
                registered=NOW,
                last_status_update=NOW,
                status="DEGRADED",
                detail="6 failed jobs since the last healthy one",
                saturation=0.5,
                pause_cause="workload",
                backoff_until=NOW,
                identities=["compute-a", "compute-b"],
            )
        )
    return {
        "name": "testmanager",
        "state": state,
        "expired": False,
        "failure_count": 6 if registered else None,
        "errors": [
            ComputeManagerError(
                name="testmanager",
                time=NOW,
                status="DEGRADED",
                detail="6 failed jobs since the last healthy one",
            )
        ],
    }


def test_status(monkeypatch):
    monkeypatch.setattr(
        AlchemiscaleComputeManagerClient, "get_status", lambda self, name: _status(True)
    )
    result = CliRunner().invoke(
        cli, ["compute", "manager", "status", "testmanager", *ARGS]
    )
    assert result.exit_code == 0, result.output
    assert "status:             DEGRADED" in result.output
    assert "pause_cause:        workload" in result.output
    assert "identities:         compute-a, compute-b" in result.output
    assert "failed jobs:        6" in result.output
    assert "DEGRADED: 6 failed jobs since the last healthy one" in result.output


def test_status_not_registered(monkeypatch):
    monkeypatch.setattr(
        AlchemiscaleComputeManagerClient,
        "get_status",
        lambda self, name: _status(False),
    )
    result = CliRunner().invoke(
        cli, ["compute", "manager", "status", "testmanager", *ARGS]
    )
    assert result.exit_code == 0, result.output
    assert "not registered" in result.output
    assert "Recent errors" in result.output


def test_status_json(monkeypatch):
    monkeypatch.setattr(
        AlchemiscaleComputeManagerClient, "get_status", lambda self, name: _status(True)
    )
    result = CliRunner().invoke(
        cli, ["compute", "manager", "status", "testmanager", "--json", *ARGS]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"]["registration"]["status"] == "DEGRADED"
    assert data["errors"][0]["detail"] == "6 failed jobs since the last healthy one"


def test_probe(monkeypatch):
    requested = []
    monkeypatch.setattr(
        AlchemiscaleComputeManagerClient,
        "request_probe",
        lambda self, name: requested.append(name),
    )
    result = CliRunner().invoke(
        cli, ["compute", "manager", "probe", "testmanager", *ARGS]
    )
    assert result.exit_code == 0, result.output
    assert requested == ["testmanager"]
    assert "will probe on its next cycle" in result.output


def test_probe_refused(monkeypatch):
    from alchemiscale.compute.client import AlchemiscaleComputeManagerClientError

    def refuse(self, name):
        raise AlchemiscaleComputeManagerClientError(
            "Compute manager 'testmanager' isn't paused", status_code=422
        )

    monkeypatch.setattr(AlchemiscaleComputeManagerClient, "request_probe", refuse)
    result = CliRunner().invoke(
        cli, ["compute", "manager", "probe", "testmanager", *ARGS]
    )
    assert result.exit_code == 1
    assert "isn't paused" in result.output
    assert "Traceback" not in result.output
