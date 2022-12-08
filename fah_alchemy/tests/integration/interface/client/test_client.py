import pytest
from time import sleep

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from fah_alchemy.models import ScopedKey
from fah_alchemy.interface import client

from fah_alchemy.tests.integration.interface.utils import get_user_settings_override


class TestClient:
    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        user_client_wrong_credential: client.FahAlchemyClient,
        uvicorn_server,
    ):
        with pytest.raises(client.FahAlchemyClientError):
            user_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        uvicorn_server,
    ):

        settings = get_user_settings_override()
        assert user_client._jwtoken == None
        user_client._get_token()

        token = user_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        user_client.get_info()
        assert token == user_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        user_client.get_info()
        assert token != user_client._jwtoken

    def test_create_network(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        uvicorn_server,
    ):
        ...

