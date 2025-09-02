import pytest
import json
import os
import datetime
from time import sleep

from gufe.tokenization import JSON_HANDLER

from alchemiscale.compute import client
from alchemiscale.models import ScopedKey
from alchemiscale.storage.models import (
    TaskStatusEnum,
    ProtocolDAGResultRef,
)
from alchemiscale.tests.integration.compute.utils import get_compute_settings_override


class TestComputeClient:
    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        compute_client_wrong_credential: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        with pytest.raises(client.AlchemiscaleComputeClientError):
            compute_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        settings = get_compute_settings_override()
        assert compute_client._jwtoken is None
        compute_client._get_token()

        token = compute_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        compute_client.get_info()
        assert token == compute_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        compute_client.get_info()
        assert token != compute_client._jwtoken

    def test_api_check(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        compute_client._api_check()

    def test_register(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
        compute_service_id,
    ):
        out = compute_client.register(compute_service_id)
        assert out == compute_service_id

        csreg = n4js_preloaded.graph.execute_query(
            f"""
            match (csreg:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})
            return csreg
            """
        )

        assert csreg.records
        assert (
            csreg.records[0]["csreg"]["registered"]
            == csreg.records[0]["csreg"]["heartbeat"]
        )

    def test_deregister(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
        compute_service_id,
    ):
        out = compute_client.register(compute_service_id)
        assert out == compute_service_id

        out = compute_client.deregister(compute_service_id)
        assert out == compute_service_id

        q = f"""
            match (csreg:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})
            return csreg
            """

        csreg = n4js_preloaded.graph.execute_query(q)

        assert not csreg.records

    def test_heartbeat(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
        compute_service_id,
    ):
        compute_client.register(compute_service_id)

        out = compute_client.heartbeat(compute_service_id)
        assert out == compute_service_id

        q = f"""
            match (csreg:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})
            return csreg
            """

        csreg = n4js_preloaded.graph.execute_query(q)

        assert csreg.records
        assert len(csreg.records) == 1

        csreg = csreg.records[0]["csreg"]

        assert csreg["registered"] < csreg["heartbeat"]

    def test_list_scope(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
        scope_test,
    ):
        scopes = compute_client.list_scopes()
        # scope_test matches identity used to initialise the client in conftest
        assert scopes == [scope_test]

    ### compute

    def test_query_taskhubs(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        taskhub_sks = compute_client.query_taskhubs([scope_test])

        assert len(taskhub_sks) == 2

        taskhubs = compute_client.query_taskhubs([scope_test], return_gufe=True)
        assert all([tq.weight == 0.5 for tq in taskhubs.values()])

    def test_claim_taskhub_task(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        uvicorn_server,
    ):
        # register compute service id
        compute_client.register(compute_service_id)

        taskhub_sks = compute_client.query_taskhubs([scope_test])

        remaining_tasks = n4js_preloaded.get_taskhub_unclaimed_tasks(taskhub_sks[0])
        assert len(remaining_tasks) == 3

        # claim a single task; should get highest priority task
        task_sks = compute_client.claim_taskhub_tasks(
            taskhub_sks[0], compute_service_id=compute_service_id
        )

        remaining_tasks = n4js_preloaded.get_taskhub_unclaimed_tasks(taskhub_sks[0])
        assert len(remaining_tasks) == 2

        all_tasks = n4js_preloaded.get_taskhub_tasks(taskhub_sks[0], return_gufe=True)

        assert len(task_sks) == 1
        assert task_sks[0] in all_tasks.keys()
        assert [t.gufe_key for t in task_sks] == [
            task.key for task in all_tasks.values() if task.priority == 1
        ]

        remaining_tasks = n4js_preloaded.get_taskhub_unclaimed_tasks(taskhub_sks[0])
        # claim two more tasks
        task_sks2 = compute_client.claim_taskhub_tasks(
            taskhub_sks[0], count=2, compute_service_id=compute_service_id
        )
        assert task_sks2[0] in remaining_tasks
        assert task_sks2[1] in remaining_tasks

    def test_claim_tasks(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        uvicorn_server,
    ):
        # register compute service id
        compute_client.register(compute_service_id)

        # claim a single task; should get highest priority task
        task_sks = compute_client.claim_tasks(
            scopes=[scope_test],
            compute_service_id=compute_service_id,
        )
        all_tasks = n4js_preloaded.query_tasks(scope=scope_test)
        priorities = {
            task_sk: priority
            for task_sk, priority in zip(
                all_tasks, n4js_preloaded.get_task_priority(all_tasks)
            )
        }

        assert len(task_sks) == 1
        assert task_sks[0] in all_tasks
        assert [t.gufe_key for t in task_sks] == [
            t.gufe_key for t in all_tasks if priorities[t] == 1
        ]

    def test_get_task_transformation(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        network_tyk2,
        transformation,
        uvicorn_server,
    ):
        # register compute service id
        compute_client.register(compute_service_id)

        an_sk = ScopedKey(gufe_key=network_tyk2.key, **scope_test.to_dict())

        taskhub_sk = n4js_preloaded.get_taskhub(an_sk)

        # claim our first task
        task_sks = compute_client.claim_taskhub_tasks(
            taskhub_sk, compute_service_id=compute_service_id
        )

        # get the transformation corresponding to this task
        (
            transformation_,
            extends_protocoldagresult,
        ) = compute_client.retrieve_task_transformation(task_sks[0])

        assert transformation_ == transformation
        assert extends_protocoldagresult is None

    ### results

    def test_set_task_result(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        network_tyk2,
        transformation,
        protocoldagresults,
        uvicorn_server,
    ):
        # register compute service id
        compute_client.register(compute_service_id)

        an_sk = ScopedKey(gufe_key=network_tyk2.key, **scope_test.to_dict())
        tf_sk = ScopedKey(gufe_key=transformation.key, **scope_test.to_dict())
        taskhub_sk = n4js_preloaded.get_taskhub(an_sk)

        # claim our first task
        task_sks = compute_client.claim_taskhub_tasks(
            taskhub_sk, compute_service_id=compute_service_id
        )

        # get the transformation corresponding to this task
        (
            transformation_,
            extends_protocoldagresult,
        ) = compute_client.retrieve_task_transformation(task_sks[0])

        assert transformation_ == transformation
        assert extends_protocoldagresult is None

        # push a result for the task
        _ = compute_client.set_task_result(task_sks[0], protocoldagresults[0])

        # now, create a task that extends the one we just "performed"
        task_sk2 = n4js_preloaded.create_task(tf_sk, extends=task_sks[0])

        # get the transformation and the protocoldagresult for the task this extends
        # no need to claim to actually do this
        (
            transformation2,
            extends_protocoldagresult2,
        ) = compute_client.retrieve_task_transformation(task_sk2)

        assert transformation2 == transformation_
        assert extends_protocoldagresult2 == protocoldagresults[0]

    # TODO: Remove in next major release where old to_dict protocoldagresults storage is removed
    def test_set_task_result_legacy(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        network_tyk2,
        transformation,
        protocoldagresults,
        uvicorn_server,
        s3os_server_fresh,
    ):
        s3os_server = s3os_server_fresh
        # register compute service id
        compute_client.register(compute_service_id)

        an_sk = ScopedKey(gufe_key=network_tyk2.key, **scope_test.to_dict())
        tf_sk = ScopedKey(gufe_key=transformation.key, **scope_test.to_dict())
        taskhub_sk = n4js_preloaded.get_taskhub(an_sk)

        # claim our first task
        task_sks = compute_client.claim_taskhub_tasks(
            taskhub_sk, compute_service_id=compute_service_id
        )

        # get the transformation corresponding to this task
        (
            transformation_,
            extends_protocoldagresult,
        ) = compute_client.retrieve_task_transformation(task_sks[0])

        assert transformation_ == transformation
        assert extends_protocoldagresult is None

        # push a result for the task
        protocoldagresult = protocoldagresults[0]
        task_sk = task_sks[0]

        # we need to replicate the behavior of set_task_result:
        #
        #     pdr_sk = compute_client.set_task_result(task_sks[0], protocoldagresults[0])
        #
        # This involves pushing the protocoldagresult in the legacy
        # to_dict() -> json -> utf-8 encoded form, set the task result
        # in the statestore, set the task to complete in the
        # statestore
        #
        #
        # step 1: Push the protocoldagresult. This needs to be done
        # manually since the old behavior was overwritten.

        pdr_bytes_push = json.dumps(
            protocoldagresult.to_dict(), cls=JSON_HANDLER.encoder
        ).encode("utf-8")
        route = "results" if protocoldagresult.ok() else "failures"

        location = os.path.join(
            "protocoldagresult",
            *tf_sk.scope.to_tuple(),
            tf_sk.gufe_key,
            route,
            protocoldagresult.key,
            "obj.json",
        )

        s3os_server._store_bytes(location, pdr_bytes_push)

        pdrr = ProtocolDAGResultRef(
            location=location,
            obj_key=protocoldagresult.key,
            scope=tf_sk.scope,
            ok=protocoldagresult.ok(),
            datetime_created=datetime.datetime.now(tz=datetime.UTC),
            creator=None,
        )

        # step 2: set the task result in the statestore to reflect the
        # protocoldagresult in the objectstore

        _ = n4js_preloaded.set_task_result(task=task_sk, protocoldagresultref=pdrr)

        # step 3: set the task to complete in the statestore

        if pdrr.ok:
            n4js_preloaded.set_task_complete(tasks=[task_sk])
        else:
            n4js_preloaded.set_task_error(tasks=[task_sk])

        # continue normally and show the protocoldagresult stored in
        # the legacy format is properly fetched and decoded

        # create a task that extends the one we just "performed"
        task_sk2 = n4js_preloaded.create_task(tf_sk, extends=task_sks[0])

        # get the transformation and the protocoldagresult for the task this extends
        # no need to claim to actually do this
        (
            transformation2,
            extends_protocoldagresult2,
        ) = compute_client.retrieve_task_transformation(task_sk2)

        assert transformation2 == transformation_
        assert extends_protocoldagresult2 == protocoldagresults[0]

    def test_set_task_result_failure(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id,
        network_tyk2_failure,
        transformation_failure,
        protocoldagresults_failure,
        uvicorn_server,
    ):
        # register compute service id
        compute_client.register(compute_service_id)

        tf_sk = ScopedKey(gufe_key=transformation_failure.key, **scope_test.to_dict())

        # add a network with a transformation that will always fail
        an_sk, taskhub_sk, _ = n4js_preloaded.assemble_network(
            network_tyk2_failure, scope_test
        )

        # create and action a task for the failing transformation
        task_sks = n4js_preloaded.create_tasks([tf_sk])
        n4js_preloaded.action_tasks(task_sks, taskhub_sk)

        # claim the task
        task_sks = compute_client.claim_taskhub_tasks(
            taskhub_sk, compute_service_id=compute_service_id
        )

        # get the transformation corresponding to this task
        (
            transformation_,
            extends_protocoldagresult,
        ) = compute_client.retrieve_task_transformation(task_sks[0])

        assert transformation_ == transformation_failure
        assert extends_protocoldagresult is None

        # push a failed result for the task
        _ = compute_client.set_task_result(task_sks[0], protocoldagresults_failure[0])

        assert n4js_preloaded.get_task_status(task_sks)[0] == TaskStatusEnum.error
