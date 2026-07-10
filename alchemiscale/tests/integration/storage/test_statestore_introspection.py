"""Integration tests for the v0.8.0 Task-introspection state-store surface.

Covers the ``Neo4jStore`` additions from the introspection work: durable
``TaskProvenance`` records (creation at claim; finalization at result/expiry/
deregistration/release; the late-result overwrite rules), the
``datetime_status_changed``/``reason`` indicators, per-unit
``ProtocolUnitResultRef``s, live progress, and compute share --- exercised
against a real Neo4j instance via the same harness as ``test_statestore.py``.
"""

import datetime
from datetime import timedelta

import pytest
from gufe.protocols import ProtocolUnitFailure

from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.models import (
    ComputeServiceID,
    ComputeServiceRegistration,
    ProtocolDAGResultRef,
    TaskAttempt,
    TaskDetails,
    TaskOutcomeEnum,
    TaskStatusEnum,
    TaskTracebacks,
)
from alchemiscale.models import Scope, ScopedKey


def _register(
    n4js: Neo4jStore,
    compute_service_id: ComputeServiceID,
    hostname: str | None = "host-a",
    manager_name: str | None = None,
) -> ComputeServiceID:
    """Register a compute service carrying a ``hostname`` (and optional manager)."""
    now = datetime.datetime.now(tz=datetime.UTC)
    registration = ComputeServiceRegistration(
        identifier=compute_service_id,
        registered=now,
        heartbeat=now,
        failure_times=[],
        hostname=hostname,
        manager_name=manager_name,
    )
    return n4js.register_computeservice(registration)


def _provenance_nodes(n4js: Neo4jStore, task: ScopedKey) -> list:
    """Return the raw ``TaskProvenance`` nodes for a Task, newest claim first.

    ``PROVENANCE_OF`` points from the provenance node to the ``Task``.
    """
    q = """
    MATCH (tp:TaskProvenance)-[:PROVENANCE_OF]->(t:Task {_scoped_key: $task})
    RETURN tp
    ORDER BY tp.datetime_claimed DESC
    """
    return [rec["tp"] for rec in n4js.execute_query(q, task=str(task)).records]


class TestStateStoreIntrospection:

    @pytest.fixture
    def n4js(self, n4js_fresh):
        return n4js_fresh

    def _claimed_task(
        self,
        n4js: Neo4jStore,
        network,
        transformation,
        scope_test,
        compute_service_id: ComputeServiceID,
        hostname: str | None = "host-a",
        manager_name: str | None = None,
    ):
        """Assemble a network, create+action a single Task, and claim it.

        Returns ``(task_sk, taskhub_sk)`` with the Task now ``running`` and an
        open ``TaskProvenance`` attempt created at claim.
        """
        _, taskhub_sk, _ = n4js.assemble_network(network, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)
        task_sk = n4js.create_task(transformation_sk)
        n4js.action_tasks([task_sk], taskhub_sk)
        _register(n4js, compute_service_id, hostname, manager_name)
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, compute_service_id)
        assert claimed[0] == task_sk
        return task_sk, taskhub_sk

    # --- provenance creation at claim -------------------------------------

    def test_provenance_created_at_claim(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.claim")
        task_sk, _ = self._claimed_task(
            n4js,
            network_tyk2,
            transformation,
            scope_test,
            csid,
            hostname="cluster-node-7",
            manager_name=None,
        )

        nodes = _provenance_nodes(n4js, task_sk)
        assert len(nodes) == 1
        tp = nodes[0]
        assert tp["compute_service_id"] == str(csid)
        assert tp["hostname"] == "cluster-node-7"
        assert tp.get("manager_name") is None
        assert tp.get("datetime_claimed") is not None
        # open attempt: not yet finalized
        assert tp.get("datetime_end") is None
        assert tp.get("outcome") is None

        # the Task itself flipped to running with a status-change timestamp
        task_node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        assert task_node["status"] == TaskStatusEnum.running.value
        assert task_node.get("datetime_status_changed") is not None

    # --- finalization: complete / error via set_task_result ---------------

    def test_provenance_finalized_complete(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.complete")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=True
        )
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.complete.value
        assert tp.get("datetime_end") is not None

        # PROVENANCE_OF edge to the produced ProtocolDAGResultRef
        linked = n4js.execute_query(
            """
            MATCH (tp:TaskProvenance {compute_service_id: $csid})-[:PROVENANCE_OF]->(pdrr:ProtocolDAGResultRef)
            RETURN pdrr._scoped_key AS sk
            """,
            csid=str(csid),
        ).records
        assert linked[0]["sk"] == str(pdrr_sk)

    def test_provenance_finalized_error(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.error")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=False
        )
        n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.error.value
        assert tp.get("datetime_end") is not None

    # --- finalization: expired / released ---------------------------------

    def test_provenance_expired_on_expire_registrations(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.expire")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        # force the heartbeat into the past so the registration expires
        n4js.execute_query(
            """
            MATCH (csreg:ComputeServiceRegistration {identifier: $csid})
            SET csreg.heartbeat = datetime($past)
            """,
            csid=str(csid),
            past=(
                datetime.datetime.now(tz=datetime.UTC) - timedelta(hours=1)
            ).isoformat(),
        )
        n4js.expire_registrations(
            datetime.datetime.now(tz=datetime.UTC) - timedelta(minutes=1)
        )

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.expired.value
        assert tp.get("datetime_end") is not None
        # units_* last-reported values would remain here; the Task returns to waiting
        task_node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        assert task_node["status"] == TaskStatusEnum.waiting.value

    def test_provenance_expired_on_deregister(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.dereg")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        n4js.deregister_computeservice(csid)

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.expired.value
        assert tp.get("datetime_end") is not None

    @pytest.mark.parametrize("force_status", ("waiting", "invalid", "deleted"))
    def test_provenance_released_on_user_status_change(
        self, n4js, network_tyk2, transformation, scope_test, force_status
    ):
        csid = ComputeServiceID.new_from_name(f"prov.release.{force_status}")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        n4js.set_task_status([task_sk], TaskStatusEnum(force_status))

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.released.value
        assert tp.get("datetime_end") is not None

    # --- late-result race rules (M4) --------------------------------------

    def test_late_result_overwrites_expired_record(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.late.expired")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        # registration expires; the open attempt closes as expired
        n4js.deregister_computeservice(csid)
        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.expired.value

        # a late result for the SAME service arrives; the attempt did finish, so
        # its expired record is overwritten to complete
        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=True
        )
        n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.complete.value

    def test_late_result_does_not_overwrite_released_record(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("prov.late.released")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        # user forces the running Task back to waiting; the attempt is released
        n4js.set_task_waiting([task_sk])
        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.released.value

        # a late result for the same service must NOT resurrect the released
        # record (the immutable history that a user ended the attempt stands)
        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=True
        )
        n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.released.value

    def test_restart_churn_yields_distinct_records(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        # two attempts by two services: first expires, second completes
        csid1 = ComputeServiceID.new_from_name("prov.attempt.one")
        task_sk, taskhub_sk = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid1
        )
        n4js.deregister_computeservice(csid1)  # attempt 1 -> expired, back to waiting

        csid2 = ComputeServiceID.new_from_name("prov.attempt.two")
        _register(n4js, csid2, hostname="host-b")
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, csid2)
        assert claimed[0] == task_sk
        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=True
        )
        n4js.set_task_result(task_sk, pdrr, compute_service_id=csid2)

        nodes = _provenance_nodes(n4js, task_sk)
        assert len(nodes) == 2
        outcomes = {n["compute_service_id"]: n["outcome"] for n in nodes}
        assert outcomes[str(csid1)] == TaskOutcomeEnum.expired.value
        assert outcomes[str(csid2)] == TaskOutcomeEnum.complete.value

    # --- status-change indicator + reason ---------------------------------

    def test_datetime_status_changed_and_reason_on_change(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        _, taskhub_sk, _ = n4js.assemble_network(network_tyk2, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)
        task_sk = n4js.create_task(transformation_sk)

        # a genuine transition sets the timestamp and the reason
        n4js.set_task_invalid([task_sk], reason="operator marked bad input")
        node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        assert node["status"] == TaskStatusEnum.invalid.value
        assert node["reason"] == "operator marked bad input"
        ts1 = node["datetime_status_changed"]
        assert ts1 is not None

    def test_status_write_idempotent_noop_preserves_indicators(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        """A no-op re-set must not reset datetime_status_changed or wipe reason.

        This validates the ``_status_write`` CASE guards against real Neo4j SET
        semantics (the guard relies on the CASE reading the pre-clause status).
        """
        _, taskhub_sk, _ = n4js.assemble_network(network_tyk2, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)
        task_sk = n4js.create_task(transformation_sk)

        n4js.set_task_invalid([task_sk], reason="first reason")
        node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        ts1 = node["datetime_status_changed"]

        # re-assert the SAME status with a DIFFERENT reason: no-op transition
        n4js.set_task_invalid([task_sk], reason="second reason")
        node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]

        assert node["status"] == TaskStatusEnum.invalid.value
        # neither the change-timestamp nor the reason was clobbered (compare the
        # native datetimes to avoid any neo4j DateTime equality quirk)
        assert node["datetime_status_changed"].to_native() == ts1.to_native()
        assert node["reason"] == "first reason"

    def test_reason_cleared_on_transition_to_waiting(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("reason.clear")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )
        # error the running Task with a reason (DAG-creation-failure path)
        n4js.set_task_error(
            [task_sk], reason="boom during create", compute_service_id=csid
        )
        node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        assert node["reason"] == "boom during create"

        # transition back to waiting clears the reason (describes current status)
        n4js.set_task_waiting([task_sk])
        node = n4js.execute_query(
            "MATCH (t:Task {_scoped_key: $task}) RETURN t", task=str(task_sk)
        ).records[0]["t"]
        assert node["status"] == TaskStatusEnum.waiting.value
        assert node.get("reason") is None

    def test_set_task_error_with_reason_finalizes_provenance(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        # the DAG-creation-failure path: error + reason + provenance error, no PDRR
        csid = ComputeServiceID.new_from_name("error.reason.prov")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )
        n4js.set_task_error([task_sk], reason="traceback text", compute_service_id=csid)
        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["outcome"] == TaskOutcomeEnum.error.value
        assert tp.get("datetime_end") is not None

    # --- get_task_history --------------------------------------------------

    def test_get_task_history(self, n4js, network_tyk2, transformation, scope_test):
        csid1 = ComputeServiceID.new_from_name("hist.one")
        task_sk, taskhub_sk = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid1, hostname="h1"
        )
        n4js.deregister_computeservice(csid1)  # attempt 1 -> expired (no result)

        csid2 = ComputeServiceID.new_from_name("hist.two")
        _register(n4js, csid2, hostname="h2")
        assert n4js.claim_taskhub_tasks(taskhub_sk, csid2)[0] == task_sk
        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=True
        )
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid2)

        history = n4js.get_task_history(task_sk)
        assert len(history) == 2
        assert all(isinstance(a, TaskAttempt) for a in history)

        # most recent first: the completing attempt
        assert history[0].compute_service_id == str(csid2)
        assert history[0].hostname == "h2"
        assert history[0].outcome == TaskOutcomeEnum.complete
        assert history[0].protocoldagresultref == pdrr_sk

        # the earlier expired attempt has no result ref
        assert history[1].compute_service_id == str(csid1)
        assert history[1].outcome == TaskOutcomeEnum.expired
        assert history[1].protocoldagresultref is None

        # limit
        assert len(n4js.get_task_history(task_sk, limit=1)) == 1

    # --- get_tasks_details -------------------------------------------------

    def test_get_tasks_details(self, n4js, network_tyk2, transformation, scope_test):
        csid = ComputeServiceID.new_from_name("details.svc")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid, hostname="dhost"
        )

        # a second, unclaimed task, plus a nonexistent one
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)
        task_sk2 = n4js.create_task(transformation_sk)
        missing = ScopedKey(gufe_key="Task-doesnotexist", **scope_test.to_dict())

        details = n4js.get_tasks_details([task_sk, task_sk2, missing])
        assert len(details) == 3

        d0 = details[0]
        assert isinstance(d0, TaskDetails)
        assert d0.task == task_sk
        assert d0.status == TaskStatusEnum.running
        assert d0.datetime_status_changed is not None
        assert d0.num_claims == 1
        assert d0.current_claim is not None
        assert d0.current_claim.compute_service_id == str(csid)
        assert d0.current_claim.hostname == "dhost"
        assert d0.most_recent_attempt is not None
        assert d0.most_recent_attempt.compute_service_id == str(csid)

        # unclaimed waiting task: no claim, no attempts
        d1 = details[1]
        assert d1.status == TaskStatusEnum.waiting
        assert d1.num_claims == 0
        assert d1.current_claim is None
        assert d1.most_recent_attempt is None

        # missing task -> None in place, order preserved
        assert details[2] is None

    # --- get_task_tracebacks ----------------------------------------------

    def test_get_task_tracebacks(self, n4js, network_tyk2, transformation, scope_test):
        csid = ComputeServiceID.new_from_name("tb.svc")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=task_sk.gufe_key, ok=False
        )
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        pufs = [
            ProtocolUnitFailure(
                source_key=f"FakeProtocolUnitKey-{i}",
                inputs={},
                outputs={},
                exception=("RuntimeError", ("boom",)),
                traceback=f"traceback number {i}",
            )
            for i in range(2)
        ]
        n4js.add_protocol_dag_result_ref_tracebacks(pufs, pdrr_sk)

        tbs = n4js.get_task_tracebacks(task_sk)
        assert len(tbs) == 1
        assert isinstance(tbs[0], TaskTracebacks)
        assert tbs[0].protocoldagresultref == pdrr_sk
        returned = {t.traceback for t in tbs[0].tracebacks}
        assert returned == {"traceback number 0", "traceback number 1"}
        # no unit refs stored for this synthetic result, so no unit ref link
        assert all(t.protocolunitresultref is None for t in tbs[0].tracebacks)

    # --- per-unit result refs (real ProtocolDAGResult) --------------------

    def test_add_and_get_unit_result_refs(
        self, n4js, network_tyk2, transformation, scope_test, protocoldagresults
    ):
        csid = ComputeServiceID.new_from_name("units.svc")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        pdr = protocoldagresults[0]
        location = (
            f"protocoldagresult/{'/'.join(task_sk.scope.to_tuple())}/"
            f"{transformation.key}/results/{pdr.key}/obj.json.zst"
        )
        pdrr = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key=pdr.key, ok=True, location=location
        )
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        refs_map = n4js.add_protocol_unit_result_refs(pdrr, pdrr_sk, pdr)
        assert len(refs_map) == len(pdr.protocol_unit_results)

        # CONTAINS edges from the pdrr to each unit ref
        contains = n4js.execute_query(
            """
            MATCH (pdrr:ProtocolDAGResultRef {_scoped_key: $pdrr})-[:CONTAINS]->(purr:ProtocolUnitResultRef)
            RETURN count(purr) AS n
            """,
            pdrr=str(pdrr_sk),
        ).records[0]["n"]
        assert contains == len(pdr.protocol_unit_results)

        # UNIT_DEPENDS_ON edges reproduce the execution topology (a DummyProtocol
        # DAG has at least one dependency edge), and are NOT DEPENDS_ON
        unit_depends = n4js.execute_query(
            """
            MATCH (:ProtocolUnitResultRef)-[r:UNIT_DEPENDS_ON]->(:ProtocolUnitResultRef)
            RETURN count(r) AS n
            """,
        ).records[0]["n"]
        assert unit_depends >= 1

        # records come back one per unit result, in dependency order
        recs = n4js.get_result_unit_recs(pdrr_sk)
        assert len(recs) == len(pdr.protocol_unit_results)
        # location recorded under the unit prefix
        assert all(
            "units/" in r_location
            for r_location in [n4js.get_gufe(rec.scoped_key).location for rec in recs]
        )

    def test_add_protocol_unit_result_refs_idempotent(
        self, n4js, network_tyk2, transformation, scope_test, protocoldagresults
    ):
        csid = ComputeServiceID.new_from_name("units.idem")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )
        pdr = protocoldagresults[0]
        pdrr = ProtocolDAGResultRef(scope=task_sk.scope, obj_key=pdr.key, ok=True)
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)

        refs_map = n4js.add_protocol_unit_result_refs(pdrr, pdrr_sk, pdr)
        # a later, separate request flips has_logs on one unit ref
        some_purr = next(iter(refs_map.values()))
        n4js.set_protocol_unit_result_ref_artifacts(some_purr, has_logs=True)

        # a duplicate result push (same deterministic pdrr) must NOT wipe flags
        refs_map2 = n4js.add_protocol_unit_result_refs(pdrr, pdrr_sk, pdr)
        assert set(map(str, refs_map2.values())) == set(map(str, refs_map.values()))

        purr = n4js.get_gufe(some_purr)
        assert purr.has_logs is True

    def test_get_task_result_recs_ok_filter(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("recs.svc")
        task_sk, taskhub_sk = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )
        ok_ref = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key="ProtocolDAGResult-okresult", ok=True
        )
        fail_ref = ProtocolDAGResultRef(
            scope=task_sk.scope, obj_key="ProtocolDAGResult-failresult", ok=False
        )
        n4js.set_task_result(task_sk, ok_ref, compute_service_id=csid)
        n4js.set_task_result(task_sk, fail_ref, compute_service_id=csid)

        all_recs = n4js.get_task_result_recs(task_sk)
        assert len(all_recs) == 2
        assert {r.ok for r in all_recs} == {True, False}

        assert all(r.ok for r in n4js.get_task_result_recs(task_sk, ok=True))
        assert all(not r.ok for r in n4js.get_task_result_recs(task_sk, ok=False))

    # --- live progress -----------------------------------------------------

    def test_update_and_get_tasks_progress(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        csid = ComputeServiceID.new_from_name("progress.svc")
        task_sk, _ = self._claimed_task(
            n4js, network_tyk2, transformation, scope_test, csid
        )

        n4js.update_task_progress(csid, {str(task_sk): (3, 10)})
        progress = n4js.get_tasks_progress([task_sk])
        assert progress == [(3, 10)]

        # progress also lands on the open provenance record
        tp = _provenance_nodes(n4js, task_sk)[0]
        assert tp["units_completed"] == 3
        assert tp["units_total"] == 10

    def test_progress_dropped_when_claim_absent(
        self, n4js, network_tyk2, transformation, scope_test
    ):
        # a task with no claim by this service: the update is dropped
        _, taskhub_sk, _ = n4js.assemble_network(network_tyk2, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)
        task_sk = n4js.create_task(transformation_sk)

        csid = ComputeServiceID.new_from_name("progress.noclaim")
        _register(n4js, csid)
        n4js.update_task_progress(csid, {str(task_sk): (1, 5)})

        # waiting (unclaimed) task reports no progress
        assert n4js.get_tasks_progress([task_sk]) == [None]

    # --- compute share -----------------------------------------------------

    def test_get_scope_compute_share(self, n4js, network_tyk2, transformation):
        # set up two orgs with running tasks in a shared campaign/project space
        scope_a = Scope("orgA", "camp", "proj")
        scope_b = Scope("orgB", "camp", "proj")

        def running_tasks(scope, count, name):
            _, taskhub_sk, _ = n4js.assemble_network(
                network_tyk2.copy_with_replacements(name=network_tyk2.name + name),
                scope,
            )
            tf_sk = n4js.get_scoped_key(transformation, scope)
            task_sks = n4js.create_tasks([tf_sk] * count)
            n4js.action_tasks(task_sks, taskhub_sk)
            csid = ComputeServiceID.new_from_name(f"share{name}")
            _register(n4js, csid)
            claimed = n4js.claim_taskhub_tasks(taskhub_sk, csid, count=count)
            assert all(c is not None for c in claimed)

        running_tasks(scope_a, 3, "a")  # 3 running in orgA
        running_tasks(scope_b, 1, "b")  # 1 running in orgB

        # orgA's share of all running tasks across orgs: 3 / (3 + 1)
        share = n4js.get_scope_compute_share(Scope(org="orgA"))
        assert share == pytest.approx(0.75)

        # empty population -> 0.0
        assert n4js.get_scope_compute_share(Scope(org="orgC")) == 0.0
