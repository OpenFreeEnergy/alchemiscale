"""Robustness tests for the `AlchemiscaleClient` introspection ``visualize``
renderings (``get_task_history``/``get_tasks_details``/``get_tasks_progress``/
``get_task_tracebacks``).

These render via ``rich``; the concern is that they never raise on edge cases
(open/running attempts, not-found Tasks, zero-total progress, empty input) and
never mis-interpret bracketed content (e.g. ``[Errno 2]``) as ``rich`` markup.
The visualizers only use class-level state, so an un-``__init__``-ed client
instance suffices --- no server needed. ``rich`` strips ANSI when stdout is not
a TTY (as under ``capsys``), so plain-text assertions hold.
"""

import datetime

import pytest
from gufe.tokenization import GufeKey

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.models import (
    TaskAttempt,
    TaskClaim,
    TaskDetails,
    TaskOutcomeEnum,
    TaskStatusEnum,
    TaskTracebacks,
    TaskUnitTraceback,
)
from alchemiscale.interface.client import AlchemiscaleClient

NOW = datetime.datetime(2026, 7, 10, 12, 0, 0, tzinfo=datetime.UTC)
LATER = datetime.datetime(2026, 7, 10, 13, 30, 0, tzinfo=datetime.UTC)
CSID = "compute-a.svc-" + "0" * 32
T = ScopedKey.from_str("Task-aaa111-org-camp-proj")
T2 = ScopedKey.from_str("Task-bbb222-org-camp-proj")
PDRR = ScopedKey.from_str("ProtocolDAGResultRef-abc123-org-camp-proj")


@pytest.fixture
def client():
    # bypass __init__ (no server): the visualizers only touch class-level state
    return AlchemiscaleClient.__new__(AlchemiscaleClient)


class TestVisualizeIntrospection:
    def test_task_history(self, client, capsys):
        attempts = [
            TaskAttempt(
                compute_service_id=CSID,
                hostname="node-7",
                datetime_claimed=NOW,
                datetime_end=LATER,
                outcome=TaskOutcomeEnum.complete,
                units_completed=5,
                units_total=5,
                protocoldagresultref=PDRR,
            ),
            # an open/running attempt: no end, no outcome, no result
            TaskAttempt(
                compute_service_id=CSID,
                hostname="node-9",
                datetime_claimed=NOW,
                outcome=None,
                units_completed=1,
                units_total=5,
            ),
        ]
        client._visualize_task_history(T, attempts)
        out = capsys.readouterr().out
        assert "complete" in out
        assert "running" in out  # open attempt shown as running
        assert "1h30m" in out  # duration rendering

    def test_tasks_details_handles_missing(self, client, capsys):
        details = TaskDetails(
            task=T,
            status=TaskStatusEnum.error,
            datetime_status_changed=NOW,
            reason="something went wrong",
            num_claims=1,
            current_claim=TaskClaim(compute_service_id=CSID, hostname="node-9"),
        )
        client._visualize_tasks_details([T, T2], [details, None])
        out = capsys.readouterr().out
        assert "error" in out
        assert "not found" in out  # the None entry is rendered, not skipped

    def test_tasks_progress_edges(self, client, capsys):
        # a reporting Task, a non-reporting (None) Task, and a zero-total Task
        client._visualize_tasks_progress([T, T2, T], [(3, 10), None, (0, 0)])
        out = capsys.readouterr().out
        assert "3/10" in out
        assert "reporting" in out  # "— not reporting —" for the None entry

    def test_tracebacks_markup_safe(self, client, capsys):
        # bracketed content must render literally, never as rich markup
        tut = TaskUnitTraceback(
            failure_key=GufeKey("ProtocolUnitFailure-f1"),
            source_key=GufeKey("ProtocolUnit-u1"),
            traceback='raise ValueError("[boom]")  # [Errno 2]',
        )
        tt = TaskTracebacks(
            protocoldagresultref=PDRR,
            datetime_created=NOW,
            creator=CSID,
            tracebacks=[tut],
        )
        client._visualize_task_tracebacks(T, [tt])
        out = capsys.readouterr().out
        assert "boom" in out
        assert "Errno" in out

    def test_tracebacks_empty(self, client, capsys):
        client._visualize_task_tracebacks(T, [])
        out = capsys.readouterr().out
        assert "No tracebacks" in out

    def test_status_palette_unchanged(self, client, capsys):
        # the refactor of _visualize_status onto the shared palette keeps all
        # six statuses in order
        client._visualize_status({"complete": 2, "error": 1}, Scope("o", "c", "p"))
        out = capsys.readouterr().out
        for status in ("complete", "running", "waiting", "error", "invalid", "deleted"):
            assert status in out
