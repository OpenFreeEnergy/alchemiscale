import pytest
from time import sleep

from gufe.tokenization import GufeTokenizable

from fah_alchemy.models import ScopedKey
from fah_alchemy.compute import client

from fah_alchemy.tests.integration.compute.utils import get_compute_settings_override


class TestComputeClient:
    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        compute_client_wrong_credential: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):
        with pytest.raises(client.FahAlchemyComputeClientError):
            compute_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):

        settings = get_compute_settings_override()
        assert compute_client._jwtoken == None
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
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):
        compute_client._api_check()

    ### compute

    def test_query_taskqueues(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):

        taskqueue_sks = compute_client.query_taskqueues([scope_test])

        assert len(taskqueue_sks) == 2

        taskqueues = compute_client.query_taskqueues([scope_test], return_gufe=True)
        assert all([tq.weight == 0.5 for tq in taskqueues.values()])

    def test_claim_taskqueue_task(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):
        taskqueue_sks = compute_client.query_taskqueues([scope_test])

        # claim our first task
        task_sks = compute_client.claim_taskqueue_tasks(taskqueue_sks[0], claimant='me')

        # check that we got the task we expected given order
        all_task_sks = n4js_preloaded.get_taskqueue_tasks(taskqueue_sks[0])

        assert len(task_sks) == 1
        assert task_sks[0] == all_task_sks[0]
        
        # claim two more tasks
        task_sks2 = compute_client.claim_taskqueue_tasks(taskqueue_sks[0], count=2, claimant='me')

        assert task_sks2 == all_task_sks[1:]

    def test_get_task_transformation(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        network_tyk2,
        transformation,
        uvicorn_server,
    ):
        an_sk = ScopedKey(gufe_key=network_tyk2.key, **scope_test.dict())
        taskqueue_sk = n4js_preloaded.get_taskqueue(an_sk)

        # claim our first task
        task_sks = compute_client.claim_taskqueue_tasks(taskqueue_sk, claimant='me')

        # get the transformation corresponding to this task
        transformation_, extends_protocoldagresult = compute_client.get_task_transformation(task_sks[0])

        assert transformation_ == transformation
        assert extends_protocoldagresult is None

    ### results

    def test_set_task_result(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        network_tyk2,
        transformation,
        protocoldagresults,
        uvicorn_server,
    ):

        an_sk = ScopedKey(gufe_key=network_tyk2.key, **scope_test.dict())
        tf_sk = ScopedKey(gufe_key=transformation.key, **scope_test.dict())
        taskqueue_sk = n4js_preloaded.get_taskqueue(an_sk)

        # claim our first task
        task_sks = compute_client.claim_taskqueue_tasks(taskqueue_sk, claimant='me')

        # get the transformation corresponding to this task
        transformation_, extends_protocoldagresult = compute_client.get_task_transformation(task_sks[0])

        assert transformation_ == transformation
        assert extends_protocoldagresult is None

        # push a result for the task
        pdr_sk = compute_client.set_task_result(task_sks[0], protocoldagresults[0])

        # now, create a task that extends the one we just "performed"
        task_sk2 = n4js_preloaded.create_task(tf_sk, extends=task_sks[0])

        # get the transformation and the protocoldagresult for the task this extends
        # no need to claim to actually do this
        transformation2, extends_protocoldagresult2 = compute_client.get_task_transformation(task_sk2)

        assert transformation2 == transformation_
        assert extends_protocoldagresult2 == protocoldagresults[0]


