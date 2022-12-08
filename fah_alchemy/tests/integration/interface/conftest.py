from copy import copy

import pytest
from fastapi.testclient import TestClient

from gufe import AlchemicalNetwork

from fah_alchemy.settings import get_jwt_settings
from fah_alchemy.interface import api
from fah_alchemy.security.models import CredentialedUserIdentity, TokenData
from fah_alchemy.security.auth import hash_key
from fah_alchemy.base.api import (
    get_token_data_depends,
    get_n4js_depends,
    get_s3os_depends,
)


from fah_alchemy.tests.integration.interface.utils import get_user_settings_override

## user api


@pytest.fixture(scope="module")
def user_identity():
    return dict(identifier="test-user-identity", key="strong passphrase lol")


@pytest.fixture
def n4js_preloaded(n4js_fresh, network_tyk2, scope_test, user_identity):
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
        CredentialedUserIdentity(
            identifier=user_identity["identifier"],
            hashed_key=hash_key(user_identity["key"]),
        )
    )

    return n4js


def get_token_data_depends_override():
    token_data = TokenData(entity="karen", scopes=["*-*-*"])
    return token_data


@pytest.fixture(scope="module")
def user_api_no_auth(n4js, s3os):
    def get_n4js_override():
        return n4js

    def get_s3os_override():
        return s3os

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_n4js_depends] = get_n4js_override
    api.app.dependency_overrides[get_s3os_depends] = get_s3os_override
    api.app.dependency_overrides[get_jwt_settings] = get_user_settings_override
    api.app.dependency_overrides[
        get_token_data_depends
    ] = get_token_data_depends_override
    yield api.app
    api.app.dependency_overrides = overrides


@pytest.fixture(scope="module")
def test_client(user_api_no_auth):
    client = TestClient(user_api_no_auth)
    return client
