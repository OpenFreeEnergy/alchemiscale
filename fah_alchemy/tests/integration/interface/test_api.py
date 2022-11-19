import pytest
import json

from gufe.tokenization import GufeTokenizable
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from fah_alchemy.models import ScopedKey
from fah_alchemy.interface import client


# api tests


class TestAPI:
    def test_info(self, test_client):

        response = test_client.get("/info")
        assert response.status_code == 200

    def test_create_network(self, 
                            n4js_preloaded, 
                            test_client,
                            network_tyk2, 
                            scope_test):

        n4js = n4js_preloaded
        an = network_tyk2

        an2 = AlchemicalNetwork(edges=list(an.edges)[:-3], 
                               nodes=an.nodes,
                               name='incomplete 2')

        headers = {"Content-type": "application/json"}
        data = dict(network=an2.to_dict(), scope=scope_test.dict())
        jsondata = json.dumps(data)

        response = test_client.post("/networks", data=jsondata, headers=headers)
        assert response.status_code == 200

        sk = ScopedKey(**response.json())

        assert sk.gufe_key == an2.key
        assert sk.scope == scope_test

        # check presence of network in database
        assert n4js.check_existence(sk)


# client tests


class TestClient:
    def test_create_network(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        uvicorn_server,
    ):
        ...
