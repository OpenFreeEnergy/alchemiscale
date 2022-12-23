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
    ):
        with pytest.raises(client.FahAlchemyClientError):
            user_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
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
        network_tyk2
    ):
        # make a smaller network that overlaps with an existing one in DB
        an = AlchemicalNetwork(edges=list(network_tyk2.edges)[4:-2], name="smaller")
        an_sk = user_client.create_network(an, scope_test)

        network_sks = user_client.query_networks()
        assert an_sk in network_sks

        # TODO: make a network in a scope that doesn't have any components in
        # common with an existing network
        #user_client.create_network(

    def test_query_networks():
        ...

    def test_get_network():
        ...

    def test_get_transformation():
        ...

    def test_get_chemicalsystem():
        ...

    def test_get_transformation_result(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        network_tyk2
        ):

        # select the transformation we want to compute
        an = network_tyk2
        transformation = an.transformations[0]

        # create a tree of tasks for the transformation

        # action the tasks for execution

        # execute the tasks

        # pull transformation results, evaluate



