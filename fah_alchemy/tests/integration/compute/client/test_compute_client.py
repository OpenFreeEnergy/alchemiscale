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

    def test_query_taskqueues(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):

        taskqueues = compute_client.query_taskqueues(scope_test)

        assert len(taskqueues) == 2

        taskqueues = compute_client.query_taskqueues(scope_test, return_gufe=True)
        assert all([tq.weight == 0.5 for tq in taskqueues.values()])

    def test_claim_taskqueue_task(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.FahAlchemyComputeClient,
        uvicorn_server,
    ):

        ...
        # task = compute_client.claim_taskqueue_task(scope_test)
