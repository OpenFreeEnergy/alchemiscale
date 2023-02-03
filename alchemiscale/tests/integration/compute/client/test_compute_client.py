import pytest
from time import sleep

from gufe.tokenization import GufeTokenizable

from alchemiscale.models import ScopedKey
from alchemiscale.compute import client

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

    def test_query_taskhubs(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        taskhubs = compute_client.query_taskhubs([scope_test])

        assert len(taskhubs) == 2

        taskhubs = compute_client.query_taskhubs([scope_test], return_gufe=True)
        assert all([tq.weight == 0.5 for tq in taskhubs.values()])

    def test_claim_taskhub_task(
        self,
        scope_test,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        ...
        # task = compute_client.claim_taskhub_task(scope_test)

    def test_api_check(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        uvicorn_server,
    ):
        compute_client._api_check()
