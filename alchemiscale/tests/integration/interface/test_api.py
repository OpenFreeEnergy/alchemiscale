import pytest
import json

from gufe import AlchemicalNetwork
from gufe.tokenization import JSON_HANDLER, KeyedChain

from alchemiscale.models import ScopedKey


def pre_load_payload(network, scope, name="incomplete 2"):
    """Helper function to spin up networks for testing"""
    new_network = AlchemicalNetwork(
        edges=list(network.edges)[:-3], nodes=network.nodes, name=name
    )
    headers = {"Content-type": "application/json"}
    data = dict(
        network=KeyedChain.gufe_to_keyed_chain_rep(new_network),
        scope=scope.to_dict(),
        state="active",
    )
    jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

    return new_network, headers, jsondata


# TODO: switch approach to mocking out requests usage in interface client with testclient instead
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

    def test_check(self, test_client, n4js):
        response = test_client.get("/check")
        assert response.status_code == 200

    def test_scopes(self, n4js_preloaded, test_client, fully_scoped_credentialed_user):
        response = test_client.get(
            f"/identities/{fully_scoped_credentialed_user.identifier}/scopes"
        )
        assert response.status_code == 200
        scopes = response.json()
        assert scopes == fully_scoped_credentialed_user.scopes

    ### inputs

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

    def test_merge_networks(
        self, n4js_preloaded, test_client, network_tyk2, scope_test
    ):
        n4js = n4js_preloaded

        # source networks in scope_test (pre-loaded by n4js_preloaded)
        source_sks = n4js.query_networks(scope=scope_test)
        assert len(source_sks) >= 2

        # destination scope: a new project under the same org/campaign so the
        # test_client's scope_test token has access
        merge_scope_dict = {
            "org": scope_test.org,
            "campaign": scope_test.campaign,
            "project": scope_test.project,
        }

        headers = {"Content-type": "application/json"}
        data = dict(
            networks=[str(sk) for sk in source_sks],
            name="api_merged",
            scope=merge_scope_dict,
        )
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post("/networks/merge", data=jsondata, headers=headers)
        assert response.status_code == 200

        merged_sk = ScopedKey(**response.json())
        assert merged_sk.scope == scope_test
        assert merged_sk.gufe_key.startswith("AlchemicalNetwork-")

        # network should now be present in the database
        assert n4js.check_existence(merged_sk)

        # merged network's union-of-edges equals network_tyk2's edge set,
        # since one of the sources is network_tyk2 and the other is a strict subset
        merged_network = n4js.get_gufe(merged_sk)
        assert merged_network.name == "api_merged"
        assert {t.key for t in merged_network.edges} == {
            t.key for t in network_tyk2.edges
        }

    def test_merge_networks_bad_scope(
        self, n4js_preloaded, test_client, scope_test, multiple_scopes
    ):
        # destination scope the test_client's token does not have access to
        bad_scope = multiple_scopes[1]
        assert bad_scope != scope_test

        source_sks = n4js_preloaded.query_networks(scope=scope_test)
        assert source_sks

        headers = {"Content-type": "application/json"}
        data = dict(
            networks=[str(sk) for sk in source_sks],
            name="should_fail",
            scope=bad_scope.to_dict(),
        )
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post("/networks/merge", data=jsondata, headers=headers)
        assert response.status_code == 401
        details = response.json()
        assert "detail" in details
        assert str(bad_scope) in details["detail"]

    def test_merge_networks_bad_source_scope(
        self, n4js_preloaded, test_client, scope_test, multiple_scopes
    ):
        # source network in a scope the test_client's token does not authorize
        unauth_scope = multiple_scopes[1]
        assert unauth_scope != scope_test

        unauth_source_sks = n4js_preloaded.query_networks(scope=unauth_scope)
        assert unauth_source_sks

        headers = {"Content-type": "application/json"}
        data = dict(
            networks=[str(sk) for sk in unauth_source_sks],
            name="should_fail",
            scope=scope_test.to_dict(),
        )
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post("/networks/merge", data=jsondata, headers=headers)
        assert response.status_code == 401
        details = response.json()
        assert "detail" in details
        assert str(unauth_scope) in details["detail"]

    def test_merge_networks_bad_qualname(self, n4js_preloaded, test_client, scope_test):
        """POST /networks/merge with a non-AlchemicalNetwork ScopedKey in
        ``networks`` must be rejected at the store layer (422)."""
        # a Transformation ScopedKey masquerading as a network SK
        tf_sks = n4js_preloaded.query_transformations(scope=scope_test)
        assert tf_sks

        headers = {"Content-type": "application/json"}
        data = dict(
            networks=[str(tf_sks[0])],
            name="should_fail",
            scope=scope_test.to_dict(),
        )
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post("/networks/merge", data=jsondata, headers=headers)
        assert response.status_code == 422
        assert "AlchemicalNetwork" in response.json()["detail"]

    def test_copy_network(self, n4js_preloaded, test_client, network_tyk2, scope_test):
        n4js = n4js_preloaded

        # pick a source network in scope_test (pre-loaded)
        source_sks = n4js.query_networks(scope=scope_test)
        assert source_sks
        source_sk = source_sks[0]

        # destination scope: reuse scope_test since the test_client's token
        # only authorizes it; the copy dedups onto the preexisting node
        # (name preserved, so gufe_key preserved, so ScopedKey preserved)
        headers = {"Content-type": "application/json"}
        data = dict(scope=scope_test.to_dict())
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post(
            f"/networks/{source_sk}/copy", data=jsondata, headers=headers
        )
        assert response.status_code == 200

        copied_sk = ScopedKey(**response.json())
        assert copied_sk.scope == scope_test
        assert copied_sk.gufe_key == source_sk.gufe_key
        assert n4js.check_existence(copied_sk)

    def test_copy_network_bad_target_scope(
        self, n4js_preloaded, test_client, scope_test, multiple_scopes
    ):
        """POST /networks/{sk}/copy with a destination scope the token
        does not authorize must be denied (401)."""
        # source in the authorized scope, destination in an unauthorized one
        source_sks = n4js_preloaded.query_networks(scope=scope_test)
        assert source_sks
        source_sk = source_sks[0]

        bad_scope = multiple_scopes[1]
        assert bad_scope != scope_test

        headers = {"Content-type": "application/json"}
        data = dict(scope=bad_scope.to_dict())
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post(
            f"/networks/{source_sk}/copy", data=jsondata, headers=headers
        )
        assert response.status_code == 401
        details = response.json()
        assert "detail" in details
        assert str(bad_scope) in details["detail"]

    def test_copy_network_bad_source_scope(
        self, n4js_preloaded, test_client, scope_test, multiple_scopes
    ):
        """POST /networks/{sk}/copy with a source network in a scope the
        token does not authorize must be denied (401)."""
        unauth_scope = multiple_scopes[1]
        assert unauth_scope != scope_test

        unauth_source_sks = n4js_preloaded.query_networks(scope=unauth_scope)
        assert unauth_source_sks
        unauth_source_sk = unauth_source_sks[0]

        headers = {"Content-type": "application/json"}
        data = dict(scope=scope_test.to_dict())
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post(
            f"/networks/{unauth_source_sk}/copy", data=jsondata, headers=headers
        )
        assert response.status_code == 401
        details = response.json()
        assert "detail" in details
        assert str(unauth_scope) in details["detail"]

    def test_copy_network_bad_qualname(self, n4js_preloaded, test_client, scope_test):
        """POST /networks/{sk}/copy with a non-AlchemicalNetwork ScopedKey
        must be rejected at the store layer (422)."""
        tf_sks = n4js_preloaded.query_transformations(scope=scope_test)
        assert tf_sks

        headers = {"Content-type": "application/json"}
        data = dict(scope=scope_test.to_dict())
        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)

        response = test_client.post(
            f"/networks/{tf_sks[0]}/copy", data=jsondata, headers=headers
        )
        assert response.status_code == 422
        assert "AlchemicalNetwork" in response.json()["detail"]

    def test_get_network(self, prepared_network, test_client):
        network, scoped_key = prepared_network
        response = test_client.get(f"/networks/{scoped_key}")

        assert response.status_code == 200

        network_ = KeyedChain(
            json.loads(response.text, cls=JSON_HANDLER.decoder)
        ).to_gufe()

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
            network_tyk2, unauthenticated_scope
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

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_query_transformations(
        self, n4js_preloaded, network_tyk2, test_client, scope_test
    ):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_transformation(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_query_chemicalsystems(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_chemicalsystem(self):
        raise NotImplementedError

    ### compute

    def test_set_network_strategy(self, n4js_preloaded, test_client, prepared_network):
        """Test setting and removing network strategy via API."""
        import json
        from alchemiscale.tests.integration.conftest import DummyStrategy
        from gufe.tokenization import KeyedChain, JSON_HANDLER

        an2, sk = prepared_network

        # Test setting strategy
        strategy = DummyStrategy()
        strategy_data = KeyedChain.gufe_to_keyed_chain_rep(strategy)
        headers = {"Content-type": "application/json"}

        data = json.dumps(
            {
                "strategy": strategy_data,
                "max_tasks_per_transformation": 3,
            },
            cls=JSON_HANDLER.encoder,
        )

        response = test_client.post(
            f"/networks/{sk}/strategy", data=data, headers=headers
        )
        assert response.status_code == 200

        # Verify strategy was set in database
        stored_strategy = n4js_preloaded.get_network_strategy(sk)
        assert stored_strategy is not None
        assert type(stored_strategy) == DummyStrategy

        # Test getting strategy via API
        response = test_client.get(f"/networks/{sk}/strategy")
        assert response.status_code == 200
        retrieved_strategy = KeyedChain(
            json.loads(response.text, cls=JSON_HANDLER.decoder)
        ).to_gufe()
        assert type(retrieved_strategy) == DummyStrategy
        assert retrieved_strategy == strategy

        # Test getting strategy state
        response = test_client.get(f"/networks/{sk}/strategy/state")
        assert response.status_code == 200
        state_data = response.json()
        assert "status" in state_data
        assert "iterations" in state_data

        # Test getting strategy status only
        response = test_client.get(f"/networks/{sk}/strategy/status")
        assert response.status_code == 200
        status = response.json()
        assert status in ["awake", "dormant", "disabled", "error"]

        # Test removing strategy (setting to null)
        data = json.dumps({"strategy": None}, cls=JSON_HANDLER.encoder)
        response = test_client.post(
            f"/networks/{sk}/strategy", data=data, headers=headers
        )
        assert response.status_code == 200

        # Verify strategy was removed
        assert n4js_preloaded.get_network_strategy(sk) is None

    def test_set_network_strategy_awake(
        self, n4js_preloaded, test_client, prepared_network
    ):
        """Test waking up dormant/error strategies."""
        import json
        from alchemiscale.tests.integration.conftest import DummyStrategy
        from alchemiscale.storage.models import StrategyState, StrategyStatusEnum
        from gufe.tokenization import KeyedChain, JSON_HANDLER

        an2, sk = prepared_network

        # First set a strategy
        strategy = DummyStrategy()
        strategy_data = KeyedChain.gufe_to_keyed_chain_rep(strategy)
        headers = {"Content-type": "application/json"}
        data = json.dumps(
            {
                "strategy": strategy_data,
                "mode": "full",
                "sleep_interval": 600,
            },
            cls=JSON_HANDLER.encoder,
        )

        response = test_client.post(
            f"/networks/{sk}/strategy", data=data, headers=headers
        )
        assert response.status_code == 200

        # Put it into dormant state
        state = StrategyState(status=StrategyStatusEnum.dormant, iterations=5)
        n4js_preloaded.update_strategy_state(sk, state)

        # Test waking it up
        response = test_client.post(f"/networks/{sk}/strategy/awake")
        assert response.status_code == 200

        # Verify it's now awake
        response = test_client.get(f"/networks/{sk}/strategy/status")
        assert response.status_code == 200
        status = response.json()
        assert status == "awake"

    def test_strategy_validation_error(self, test_client, prepared_network):
        """Test API validation error handling for invalid strategy data."""
        import json

        an2, sk = prepared_network
        headers = {"Content-type": "application/json"}

        # Test with invalid JSON structure
        invalid_data = json.dumps({"invalid": "data"})
        response = test_client.post(
            f"/networks/{sk}/strategy", data=invalid_data, headers=headers
        )
        assert response.status_code == 422  # Validation error

        # Test with malformed strategy object
        malformed_data = json.dumps({"strategy": {"invalid": "strategy_object"}})
        response = test_client.post(
            f"/networks/{sk}/strategy", data=malformed_data, headers=headers
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_create_tasks(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_tasks(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_action_tasks(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_cancel_tasks(self):
        raise NotImplementedError

    ### results

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_transformation_results(self):
        raise NotImplementedError

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_protocoldagresult(self):
        raise NotImplementedError
