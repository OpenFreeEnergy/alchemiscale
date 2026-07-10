"""Integration tests for the v0.8.0 Task-introspection client surface, exercised
end-to-end through the ``AlchemiscaleClient`` -> API -> Neo4j/S3 round trip.

State (provenance, results, per-unit refs, artifacts) is set up directly via the
``n4js``/``s3os`` handles (as the existing interface result tests do), then the
new client methods are called and asserted.
"""

import datetime
from pathlib import Path

import pytest
from gufe.tokenization import TOKENIZABLE_REGISTRY
from gufe.protocols.protocoldag import execute_DAG

from alchemiscale.compression import compress_gufe_zstd
from alchemiscale.models import Scope
from alchemiscale.storage.models import (
    ComputeServiceID,
    ComputeServiceRegistration,
    ProtocolDAGResultRec,
    ProtocolUnitResultRec,
    TaskAttempt,
    TaskDetails,
    TaskOutcomeEnum,
    TaskStatusEnum,
    TaskTracebacks,
)
from alchemiscale.interface import client


def _register(n4js, name, hostname="host-a"):
    csid = ComputeServiceID.new_from_name(name)
    now = datetime.datetime.now(tz=datetime.UTC)
    n4js.register_computeservice(
        ComputeServiceRegistration(
            identifier=csid,
            registered=now,
            heartbeat=now,
            failure_times=[],
            hostname=hostname,
        )
    )
    return csid


class TestClientIntrospection:

    def _claim_and_run(
        self, user_client, n4js, s3os, transformation, network_sk, scope_test, csid
    ):
        """Create+action a task, claim it (creating provenance), execute its DAG,
        push the result finalizing provenance, and derive per-unit refs.

        Returns ``(task_sk, pdrr_sk, pdr)``.
        """
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)
        task_sk = user_client.create_tasks(transformation_sk, count=1)[0]
        user_client.action_tasks([task_sk], network_sk)

        taskhub_sk = n4js.get_taskhub(network_sk)
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, csid)
        assert claimed[0] == task_sk

        # execute the DAG as a compute service would
        protocoldag = transformation.create(name=str(task_sk))
        shared = Path("shared").absolute() / str(protocoldag.key)
        shared.mkdir(parents=True)
        scratch = Path("scratch").absolute() / str(protocoldag.key)
        scratch.mkdir(parents=True)
        pdr = execute_DAG(
            protocoldag,
            shared_basedir=shared,
            scratch_basedir=scratch,
            raise_error=False,
        )

        pdrr = s3os.push_protocoldagresult(
            compress_gufe_zstd(pdr),
            pdr.ok(),
            pdr.key,
            transformation=transformation_sk,
        )
        pdrr_sk = n4js.set_task_result(task_sk, pdrr, compute_service_id=csid)
        n4js.add_protocol_unit_result_refs(pdrr, pdrr_sk, pdr)
        return task_sk, pdrr_sk, pdr

    def test_history_details_and_result_recs(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server_fresh,
        user_client_no_cache: client.AlchemiscaleClient,
        network_tyk2,
        tmpdir,
    ):
        user_client = user_client_no_cache
        n4js = n4js_preloaded
        s3os = s3os_server_fresh

        an = network_tyk2
        transformation = [t for t in an.edges if "_solvent" in t.name][0]
        network_sk = user_client.get_scoped_key(an, scope_test)

        csid = _register(n4js, "history.svc", hostname="node-1")
        with tmpdir.as_cwd():
            task_sk, pdrr_sk, pdr = self._claim_and_run(
                user_client, n4js, s3os, transformation, network_sk, scope_test, csid
            )
        n4js.set_task_complete([task_sk])

        # get_task_history
        history = user_client.get_task_history(task_sk)
        assert len(history) == 1
        assert isinstance(history[0], TaskAttempt)
        assert history[0].compute_service_id == str(csid)
        assert history[0].hostname == "node-1"
        assert history[0].outcome is TaskOutcomeEnum.complete
        assert history[0].protocoldagresultref == pdrr_sk

        # get_tasks_details
        details = user_client.get_tasks_details([task_sk])
        assert len(details) == 1
        assert isinstance(details[0], TaskDetails)
        assert details[0].task == task_sk
        assert details[0].status is TaskStatusEnum.complete
        assert details[0].num_claims == 1
        assert details[0].most_recent_attempt is not None
        assert details[0].most_recent_attempt.outcome is TaskOutcomeEnum.complete

        # get_task_result_recs
        recs = user_client.get_task_result_recs(task_sk)
        assert len(recs) == 1
        assert isinstance(recs[0], ProtocolDAGResultRec)
        assert recs[0].scoped_key == pdrr_sk
        assert recs[0].ok is True
        assert user_client.get_task_result_recs(task_sk, ok=True)
        assert user_client.get_task_result_recs(task_sk, ok=False) == []

        # get_result_unit_recs, in dependency order; count matches the PDR
        unit_recs = user_client.get_result_unit_recs(recs[0])
        assert len(unit_recs) == len(pdr.protocol_unit_results)
        assert all(isinstance(r, ProtocolUnitResultRec) for r in unit_recs)
        assert all(r.ok for r in unit_recs)

    def test_unit_artifacts_retrieval(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server_fresh,
        user_client_no_cache: client.AlchemiscaleClient,
        network_tyk2,
        tmpdir,
    ):
        user_client = user_client_no_cache
        n4js = n4js_preloaded
        s3os = s3os_server_fresh

        an = network_tyk2
        transformation = [t for t in an.edges if "_solvent" in t.name][0]
        network_sk = user_client.get_scoped_key(an, scope_test)

        csid = _register(n4js, "artifacts.svc")
        with tmpdir.as_cwd():
            task_sk, pdrr_sk, pdr = self._claim_and_run(
                user_client, n4js, s3os, transformation, network_sk, scope_test, csid
            )
        n4js.set_task_complete([task_sk])

        # manually attach artifacts to the first unit result and flip its flags
        unit_recs = n4js.get_result_unit_recs(pdrr_sk)
        purr_sk = unit_recs[0].scoped_key
        location = n4js.get_gufe(purr_sk).location

        s3os.push_protocol_unit_result_logs(location, "hello from the unit log\n")
        n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_logs=True)
        s3os.push_protocol_unit_result_streams(
            location, "stdout", {"out.txt": b"captured stdout\n"}
        )
        n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_stdout=True)
        s3os.push_protocol_unit_result_streams(
            location, "stderr", {"err.txt": b"captured stderr\n"}
        )
        n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_stderr=True)

        # single-unit retrieval (accepts a ScopedKey or a *Rec)
        assert user_client.get_result_unit_logs(purr_sk) == "hello from the unit log\n"
        assert user_client.get_result_unit_stdout(purr_sk) == {
            "out.txt": "captured stdout\n"
        }
        assert user_client.get_result_unit_stderr(purr_sk) == {
            "err.txt": "captured stderr\n"
        }

        # a unit with no logs returns None
        no_logs = [r for r in unit_recs if r.scoped_key != purr_sk]
        if no_logs:
            assert user_client.get_result_unit_logs(no_logs[0].scoped_key) is None

        # rendered aggregations include the captured content
        rendered = user_client.get_result_logs(pdrr_sk)
        assert "hello from the unit log" in rendered
        assert "captured stdout" in user_client.get_task_stdout(task_sk)
        assert "captured stderr" in user_client.get_task_stderr(task_sk)

    def test_tracebacks(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server_fresh,
        user_client_no_cache: client.AlchemiscaleClient,
        network_tyk2_failure,
        tmpdir,
    ):
        user_client = user_client_no_cache
        n4js = n4js_preloaded
        s3os = s3os_server_fresh

        network_sk = user_client.create_network(network_tyk2_failure, scope_test)
        transformation = [t for t in network_tyk2_failure.edges if t.name == "broken"][
            0
        ]

        csid = _register(n4js, "tb.svc")
        with tmpdir.as_cwd():
            task_sk, pdrr_sk, pdr = self._claim_and_run(
                user_client, n4js, s3os, transformation, network_sk, scope_test, csid
            )
        # record the failure's tracebacks (as the compute API does) and error it
        n4js.add_protocol_dag_result_ref_tracebacks(pdr.protocol_unit_failures, pdrr_sk)
        n4js.set_task_error([task_sk])

        for pdr_ in [pdr]:
            TOKENIZABLE_REGISTRY.pop(pdr_.key, None)

        tracebacks = user_client.get_task_tracebacks(task_sk)
        assert len(tracebacks) == 1
        assert isinstance(tracebacks[0], TaskTracebacks)
        assert tracebacks[0].protocoldagresultref == pdrr_sk
        assert len(tracebacks[0].tracebacks) == len(pdr.protocol_unit_failures)
        assert all(tb.traceback for tb in tracebacks[0].tracebacks)
        # the unit-ref link is populated (unit refs were derived above)
        assert any(
            tb.protocolunitresultref is not None for tb in tracebacks[0].tracebacks
        )

    def test_progress_and_compute_share(
        self,
        scope_test,
        n4js_preloaded,
        user_client_no_cache: client.AlchemiscaleClient,
        network_tyk2,
    ):
        user_client = user_client_no_cache
        n4js = n4js_preloaded

        an = network_tyk2
        transformation = [t for t in an.edges if "_solvent" in t.name][0]
        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        task_sk = user_client.create_tasks(transformation_sk, count=1)[0]
        user_client.action_tasks([task_sk], network_sk)
        taskhub_sk = n4js.get_taskhub(network_sk)
        csid = _register(n4js, "progress.svc")
        assert n4js.claim_taskhub_tasks(taskhub_sk, csid)[0] == task_sk

        # no progress reported yet
        assert user_client.get_tasks_progress([task_sk]) == [None]

        n4js.update_task_progress(csid, {str(task_sk): (2, 5)})
        assert user_client.get_tasks_progress([task_sk]) == [(2, 5)]

        # compute share: this scope's running Tasks over same-level siblings.
        # only this scope has a running Task, so the org-level share is 1.0
        share = user_client.get_scope_compute_share(Scope(org=scope_test.org))
        assert share == pytest.approx(1.0)

    def test_set_tasks_status_with_reason(
        self,
        scope_test,
        n4js_preloaded,
        user_client_no_cache: client.AlchemiscaleClient,
        network_tyk2,
    ):
        user_client = user_client_no_cache
        an = network_tyk2
        transformation = [t for t in an.edges if "_solvent" in t.name][0]
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        task_sk = user_client.create_tasks(transformation_sk, count=1)[0]
        user_client.set_tasks_status([task_sk], "invalid", reason="bad inputs")

        details = user_client.get_tasks_details([task_sk])
        assert details[0].status is TaskStatusEnum.invalid
        assert details[0].reason == "bad inputs"
