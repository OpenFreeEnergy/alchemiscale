import pytest
import json

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.tokenization import JSON_HANDLER, GufeTokenizable

from fah_alchemy.models import ScopedKey


def pre_load_payload(network, scope, name="incomplete 2"):
    """Helper function to spin up networks for testing"""
    new_network = AlchemicalNetwork(
        edges=list(network.edges)[:-3], nodes=network.nodes, name=name
    )
    headers = {"Content-type": "application/json"}
    data = dict(network=new_network.to_dict(), scope=scope.dict())
    jsondata = json.dumps(data)
    return new_network, headers, jsondata


class TestAPI:
    @pytest.fixture
    def prepared_network(self, n4js_preloaded, test_client, network_tyk2, scope_test):
        """
        Effectively does what create network test does, but as a fixture for other tests
        This could probably be pushed to the n4js_preloaded fixture
        """
        an2, headers, jsondata = pre_load_payload(
            network_tyk2, scope_test, name="Incomplete API Test"
        )

        response = test_client.post("/networks", data=jsondata, headers=headers)
        assert response.status_code == 200
        sk = ScopedKey(**response.json())
        # check presence of network in database
        assert n4js_preloaded.check_existence(sk)

        return an2, sk

    def test_info(self, test_client):
        response = test_client.get("/info")
        assert response.status_code == 200

    def test_check(self, test_client):
        response = test_client.get("/check")
        assert response.status_code == 200

    def test_create_network(
        self, n4js_preloaded, test_client, network_tyk2, scope_test
    ):

        n4js = n4js_preloaded
        an = network_tyk2

        an2, headers, jsondata = pre_load_payload(an, scope_test)

        response = test_client.post("/networks", data=jsondata, headers=headers)
        assert response.status_code == 200

        sk = ScopedKey(**response.json())

        assert sk.gufe_key == an2.key
        assert sk.scope == scope_test

        # check presence of network in database
        assert n4js.check_existence(sk)

    def test_create_network_bad_scope(
        self, test_client, network_tyk2, scope_test, multiple_scopes
    ):
        an = network_tyk2

        # test_client doesn't have this scope in its token
        bad_scope = multiple_scopes[1]

        an2, headers, jsondata = pre_load_payload(an, bad_scope)

        # so we expect to be denied here
        response = test_client.post("/networks", data=jsondata, headers=headers)
        assert response.status_code == 401
        details = response.json()

        # Check our error is expected
        assert "detail" in details
        details = details["detail"]

        # Check our error details are expected
        assert str(bad_scope) in details
        assert str(scope_test) in details

    def test_get_network(self, prepared_network, test_client):
        network, scoped_key = prepared_network
        response = test_client.get(f"/networks/{scoped_key}")

        assert response.status_code == 200

        content = json.loads(response.text, cls=JSON_HANDLER.decoder)
        network_ = GufeTokenizable.from_dict(content)

        assert network_.key == network.key
        assert network_ is network

    def test_get_network_bad_scope(
        self, n4js_preloaded, network_tyk2, test_client, multiple_scopes
    ):
        """Test that having the wrong scoped key denies you access"""
        # Get the preloaded key:
        auth_scope = multiple_scopes[0]  # Should also be the scope_test fixture
        unauthenticated_scope = multiple_scopes[1]
        sk_unauthenticated = n4js_preloaded.get_scoped_key(
            network_tyk2, multiple_scopes[1]
        )
        response = test_client.get(f"/networks/{sk_unauthenticated}")
        assert response.status_code == 401
        details = response.json()

        # Check our error is expected
        assert "detail" in details
        details = details["detail"]

        # Check our error details are expected
        assert str(sk_unauthenticated.scope) in details
        assert str(auth_scope) in details
