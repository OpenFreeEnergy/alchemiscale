from copy import copy

import pytest
from fastapi.testclient import TestClient

from gufe import AlchemicalNetwork

from fah_alchemy.settings import get_jwt_settings
from fah_alchemy.storage import Neo4jStore, get_n4js
from fah_alchemy.compute import api, client
from fah_alchemy.security.models import CredentialedComputeIdentity, TokenData
from fah_alchemy.security.auth import hash_key
from fah_alchemy.base.api import get_token_data_depends

from fah_alchemy.tests.integration.compute.utils import get_compute_settings_override


## compute api


@pytest.fixture(scope="module")
def compute_identity():
    return dict(identifier="test-compute-identity", key="strong passphrase lol")


@pytest.fixture
def n4js_preloaded(n4js_fresh, network_tyk2, scope_test, compute_identity):
    n4js = n4js_fresh

    # set starting contents for many of the tests in this module
    sk1 = n4js.create_network(network_tyk2, scope_test)

    # create another alchemical network
    an2 = AlchemicalNetwork(edges=list(network_tyk2.edges)[:-2], name="incomplete")
    sk2 = n4js.create_network(an2, scope_test)

    # add a taskqueue for each network
    n4js.create_taskqueue(sk1)
    n4js.create_taskqueue(sk2)

    n4js.create_credentialed_entity(
        CredentialedComputeIdentity(
            identifier=compute_identity["identifier"],
            hashed_key=hash_key(compute_identity["key"]),
        )
    )

    return n4js


def get_token_data_depends_override():
    token_data = TokenData(entity="carl", scopes=["*-*-*"])
    return token_data


@pytest.fixture
def compute_api_no_auth(n4js):
    def get_n4js_override():
        return n4js

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_n4js] = get_n4js_override
    api.app.dependency_overrides[get_jwt_settings] = get_compute_settings_override
    api.app.dependency_overrides[get_token_data_depends] = get_token_data_depends_override
    yield api.app
    api.app.dependency_overrides = overrides


@pytest.fixture
def test_client(compute_api_no_auth):
    client = TestClient(compute_api_no_auth)
    return client
