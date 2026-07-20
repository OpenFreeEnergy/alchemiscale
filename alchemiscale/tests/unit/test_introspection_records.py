"""Unit tests for the v0.8.0 introspection data models: the ``TaskProvenance``
node model and the client-facing ``*Rec``/``Task*`` record models, focusing on
``to_dict``/``from_dict`` round-trips (including the ``None`` branches) that the
API and client rely on for wire transport.
"""

import datetime
import json

import pytest
from gufe.tokenization import GufeKey

from alchemiscale.models import Scope, ScopedKey
from alchemiscale.storage.models import (
    ComputeEnvironment,
    ComputeServiceID,
    ProtocolDAGResultRec,
    ProtocolUnitResultRec,
    ProtocolUnitResultRef,
    TaskAttempt,
    TaskClaim,
    TaskDetails,
    TaskOutcomeEnum,
    TaskProvenance,
    TaskStatusEnum,
    TaskTracebacks,
    TaskUnitTraceback,
    _coerce_datetime,
    _iso,
)

NOW = datetime.datetime(2026, 7, 10, 12, 0, 0, tzinfo=datetime.UTC)
LATER = datetime.datetime(2026, 7, 10, 13, 30, 0, tzinfo=datetime.UTC)
CSID = ComputeServiceID("svc-" + "0" * 32)
PDRR_SK = ScopedKey.from_str("ProtocolDAGResultRef-abc123-org-camp-proj")
PURR_SK = ScopedKey.from_str("ProtocolUnitResultRef-def456-org-camp-proj")
TASK_SK = ScopedKey.from_str("Task-aaa111-org-camp-proj")


class TestHelpers:
    def test_coerce_datetime_none(self):
        assert _coerce_datetime(None) is None

    def test_coerce_datetime_iso_string(self):
        assert _coerce_datetime(NOW.isoformat()) == NOW

    def test_coerce_datetime_passthrough(self):
        assert _coerce_datetime(NOW) == NOW

    def test_coerce_datetime_neo4j_like(self):
        class FakeNeo4jDT:
            def to_native(self_inner):
                return NOW

        assert _coerce_datetime(FakeNeo4jDT()) == NOW

    def test_iso(self):
        assert _iso(None) is None
        assert _iso(NOW) == NOW.isoformat()


class TestTaskProvenance:
    def test_roundtrip_full(self):
        tp = TaskProvenance(
            compute_service_id=CSID,
            hostname="host-a",
            manager_name="mgr",
            datetime_claimed=NOW,
            datetime_end=LATER,
            outcome=TaskOutcomeEnum.complete,
            units_completed=3,
            units_total=5,
        )
        d = tp.to_dict()
        assert d["compute_service_id"] == str(CSID)
        assert d["outcome"] == "complete"
        tp2 = TaskProvenance.from_dict(d)
        assert isinstance(tp2.compute_service_id, ComputeServiceID)
        assert tp2.outcome is TaskOutcomeEnum.complete
        assert tp2.units_completed == 3
        assert tp2.datetime_end == LATER

    def test_roundtrip_open(self):
        # an open attempt: no end / outcome / progress
        tp = TaskProvenance(compute_service_id=CSID, datetime_claimed=NOW)
        d = tp.to_dict()
        assert d["outcome"] is None
        assert d["datetime_end"] is None
        tp2 = TaskProvenance.from_dict(d)
        assert tp2.outcome is None
        assert tp2.datetime_end is None
        assert tp2.hostname is None


class TestTaskAttempt:
    @pytest.mark.parametrize(
        "outcome,pdrr",
        [
            (TaskOutcomeEnum.complete, PDRR_SK),
            (TaskOutcomeEnum.error, PDRR_SK),
            (TaskOutcomeEnum.expired, None),
            (TaskOutcomeEnum.released, None),
            (None, None),
        ],
    )
    def test_roundtrip(self, outcome, pdrr):
        env = {"tool": "conda", "packages": {"gufe": "1.10.0"}, "captured_at": None}
        ta = TaskAttempt(
            compute_service_id=str(CSID),
            hostname="h",
            manager_name=None,
            datetime_claimed=NOW,
            datetime_end=LATER if outcome is not None else None,
            outcome=outcome,
            units_completed=1 if outcome else None,
            units_total=4 if outcome else None,
            protocoldagresultref=pdrr,
            environment=env if outcome is not None else None,
        )
        ta2 = TaskAttempt.from_dict(ta.to_dict())
        assert ta2.compute_service_id == str(CSID)
        assert ta2.outcome is outcome
        assert ta2.protocoldagresultref == pdrr
        assert ta2.environment == (env if outcome is not None else None)
        assert ta2.datetime_claimed == NOW


class TestTaskClaim:
    def test_roundtrip(self):
        tc = TaskClaim(
            compute_service_id=str(CSID),
            hostname="h",
            datetime_claimed=NOW,
            units_completed=2,
            units_total=6,
        )
        tc2 = TaskClaim.from_dict(tc.to_dict())
        assert tc2.hostname == "h"
        assert tc2.units_completed == 2
        assert tc2.datetime_claimed == NOW

    def test_roundtrip_minimal(self):
        tc = TaskClaim(compute_service_id=str(CSID))
        tc2 = TaskClaim.from_dict(tc.to_dict())
        assert tc2.datetime_claimed is None
        assert tc2.units_total is None


class TestTaskDetails:
    def test_roundtrip_with_claim_and_attempt(self):
        claim = TaskClaim(
            compute_service_id=str(CSID),
            hostname="h",
            datetime_claimed=NOW,
            units_completed=1,
            units_total=3,
        )
        attempt = TaskAttempt(
            compute_service_id=str(CSID),
            datetime_claimed=NOW,
            outcome=None,
        )
        td = TaskDetails(
            task=TASK_SK,
            status=TaskStatusEnum.running,
            datetime_status_changed=NOW,
            reason="because",
            num_claims=2,
            current_claim=claim,
            most_recent_attempt=attempt,
        )
        td2 = TaskDetails.from_dict(td.to_dict())
        assert td2.task == TASK_SK
        assert td2.status is TaskStatusEnum.running
        assert td2.reason == "because"
        assert td2.num_claims == 2
        assert td2.current_claim.compute_service_id == str(CSID)
        assert td2.most_recent_attempt.compute_service_id == str(CSID)

    def test_roundtrip_minimal(self):
        td = TaskDetails(task=TASK_SK, status=TaskStatusEnum.waiting)
        td2 = TaskDetails.from_dict(td.to_dict())
        assert td2.status is TaskStatusEnum.waiting
        assert td2.num_claims == 0
        assert td2.current_claim is None
        assert td2.most_recent_attempt is None
        assert td2.reason is None


class TestTaskTracebacks:
    def test_roundtrip(self):
        units = [
            TaskUnitTraceback(
                failure_key=GufeKey("ProtocolUnitFailure-f1"),
                source_key=GufeKey("ProtocolUnit-u1"),
                traceback="boom",
                protocolunitresultref=PURR_SK,
            ),
            TaskUnitTraceback(
                failure_key=GufeKey("ProtocolUnitFailure-f2"),
                source_key=GufeKey("ProtocolUnit-u2"),
                traceback="kaboom",
                protocolunitresultref=None,
            ),
        ]
        tt = TaskTracebacks(
            protocoldagresultref=PDRR_SK,
            datetime_created=NOW,
            creator=str(CSID),
            tracebacks=units,
        )
        tt2 = TaskTracebacks.from_dict(tt.to_dict())
        assert tt2.protocoldagresultref == PDRR_SK
        assert tt2.creator == str(CSID)
        assert [u.traceback for u in tt2.tracebacks] == ["boom", "kaboom"]
        assert tt2.tracebacks[0].protocolunitresultref == PURR_SK
        assert tt2.tracebacks[1].protocolunitresultref is None
        assert isinstance(tt2.tracebacks[0].failure_key, GufeKey)


class TestProtocolDAGResultRec:
    @pytest.mark.parametrize("ok", [True, False])
    def test_roundtrip(self, ok):
        rec = ProtocolDAGResultRec(
            scoped_key=PDRR_SK,
            ok=ok,
            datetime_created=NOW,
            creator=str(CSID),
        )
        rec2 = ProtocolDAGResultRec.from_dict(rec.to_dict())
        assert rec2.scoped_key == PDRR_SK
        assert rec2.ok is ok
        assert rec2.datetime_created == NOW
        assert rec2.creator == str(CSID)

    def test_roundtrip_minimal(self):
        rec = ProtocolDAGResultRec(scoped_key=PDRR_SK, ok=True)
        rec2 = ProtocolDAGResultRec.from_dict(rec.to_dict())
        assert rec2.datetime_created is None
        assert rec2.creator is None


class TestProtocolUnitResultRec:
    def test_roundtrip_full(self):
        rec = ProtocolUnitResultRec(
            scoped_key=PURR_SK,
            obj_key=GufeKey("ProtocolUnitResult-r1"),
            source_key=GufeKey("ProtocolUnit-u1"),
            name="unit one",
            ok=True,
            start_time=NOW,
            end_time=LATER,
            has_logs=True,
            has_stdout=True,
            has_stderr=False,
        )
        rec2 = ProtocolUnitResultRec.from_dict(rec.to_dict())
        assert rec2.scoped_key == PURR_SK
        assert isinstance(rec2.obj_key, GufeKey)
        assert isinstance(rec2.source_key, GufeKey)
        assert rec2.name == "unit one"
        assert rec2.ok is True
        assert rec2.start_time == NOW
        assert rec2.end_time == LATER
        assert rec2.has_logs is True
        assert rec2.has_stdout is True
        assert rec2.has_stderr is False

    def test_roundtrip_minimal(self):
        rec = ProtocolUnitResultRec(
            scoped_key=PURR_SK,
            obj_key=GufeKey("ProtocolUnitFailure-r2"),
            source_key=GufeKey("ProtocolUnit-u2"),
            ok=False,
        )
        rec2 = ProtocolUnitResultRec.from_dict(rec.to_dict())
        assert rec2.ok is False
        assert rec2.name is None
        assert rec2.start_time is None
        assert rec2.end_time is None
        assert rec2.has_logs is False
        assert rec2.has_stdout is False
        assert rec2.has_stderr is False


class TestProtocolUnitResultRefNode:
    """The state-store node type; verify it tokenizes and its _to_dict/_from_dict
    round-trip through gufe's keyed-chain machinery."""

    def test_gufe_roundtrip(self):
        purr = ProtocolUnitResultRef(
            location="protocoldagresult/o/c/p/T/results/K/units/R",
            obj_key=GufeKey("ProtocolUnitResult-r1"),
            source_key=GufeKey("ProtocolUnit-u1"),
            scope=Scope("o", "c", "p"),
            ok=True,
            name="u",
            start_time=NOW,
            end_time=LATER,
            has_logs=True,
        )
        # deterministic key computed once at creation
        key1 = purr.key
        # round-trip through the keyed chain (as the state store does)
        from gufe.tokenization import KeyedChain

        purr2 = KeyedChain.from_gufe(purr).to_gufe()
        assert purr2.obj_key == purr.obj_key
        assert purr2.source_key == purr.source_key
        assert purr2.ok is True
        assert purr2.has_logs is True
        assert purr2.start_time == NOW
        assert str(purr2.key) == str(key1)


class TestComputeEnvironment:
    CAP = {
        "tool": "conda",
        "packages": {"gufe": "1.10.0", "python": "3.11.9"},
        "captured_at": "2026-07-20T00:00:00+00:00",
    }

    def test_content_hash_order_independent(self):
        h1 = ComputeEnvironment.content_hash("conda", {"a": "1", "b": "2"})
        h2 = ComputeEnvironment.content_hash("conda", {"b": "2", "a": "1"})
        assert h1 == h2

    def test_content_hash_distinguishes_tool_and_versions(self):
        base = ComputeEnvironment.content_hash("conda", {"a": "1"})
        assert base != ComputeEnvironment.content_hash("pip", {"a": "1"})
        assert base != ComputeEnvironment.content_hash("conda", {"a": "2"})

    def test_from_capture(self):
        ce = ComputeEnvironment.from_capture(self.CAP)
        assert ce.tool == "conda"
        assert ce.packages == self.CAP["packages"]
        assert ce.hash == ComputeEnvironment.content_hash("conda", self.CAP["packages"])
        assert ce.captured_at == datetime.datetime.fromisoformat(
            self.CAP["captured_at"]
        )

    def test_to_capture_dict_roundtrip(self):
        ce = ComputeEnvironment.from_capture(self.CAP)
        assert ce.to_capture_dict() == self.CAP

    def test_from_node(self):
        ce = ComputeEnvironment.from_capture(self.CAP)

        class FakeNode(dict):
            def get(self, k, d=None):
                return super().get(k, d)

        node = FakeNode(
            hash=ce.hash,
            tool="conda",
            packages=json.dumps(ce.packages),
            captured_at=self.CAP["captured_at"],
        )
        ce2 = ComputeEnvironment.from_node(node)
        assert ce2.hash == ce.hash
        assert ce2.packages == ce.packages
        assert ce2.tool == "conda"
