from multiprocessing import Process
from time import sleep

import pytest
import uvicorn
import requests
from fastapi.testclient import TestClient

from gufe import AlchemicalNetwork

from fah_alchemy.settings import APISettings, get_jwt_settings
from fah_alchemy.storage import Neo4jStore, get_n4js
from fah_alchemy.interface import api, client
from fah_alchemy.security.models import CredentialedUserIdentity, TokenData
from fah_alchemy.security.auth import hash_key
from fah_alchemy.base.api import get_token_data_depends


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


def get_user_settings_override():
    # settings overrides for test suite
    return APISettings(
        NEO4J_USER="neo4j",
        NEO4J_PASS="password",
        NEO4J_URL="bolt://localhost:7687",
        FA_API_HOST="127.0.0.1",
        FA_API_PORT=8000,
        JWT_SECRET_KEY="3f072449f5f496d30c0e46e6bc116ba27937a1482c3a4e41195be899a299c7e4",
    )


def get_token_data_depends_override():
    token_data = TokenData(entity="karen", scopes="*-*-*")
    return token_data


@pytest.fixture(scope="module")
def user_api(n4js):
    def get_n4js_override():
        return n4js

    api.app.dependency_overrides[get_n4js] = get_n4js_override
    api.app.dependency_overrides[get_jwt_settings] = get_user_settings_override
    api.app.dependency_overrides[
        get_token_data_depends
    ] = get_token_data_depends_override
    return api.app


@pytest.fixture(scope="module")
def test_client(user_api):
    client = TestClient(user_api)
    return client


## user client


def run_server(fastapi_app, settings):
    uvicorn.run(
        fastapi_app,
        host=settings.FA_API_HOST,
        port=settings.FA_API_PORT,
        log_level=settings.FA_API_LOGLEVEL,
    )


@pytest.fixture(scope="module")
def uvicorn_server(user_api):
    settings = get_user_settings_override()
    proc = Process(target=run_server, args=(user_api, settings), daemon=True)
    proc.start()

    timeout = True
    for _ in range(40):
        try:
            ping = requests.get(f"http://127.0.0.1:8000/info")
            ping.raise_for_status()
        except IOError:
            sleep(0.25)
            continue
        timeout = False
        break
    if timeout:
        raise RuntimeError("The test server could not be reached.")

    yield

    proc.kill()  # Cleanup after test


@pytest.fixture(scope="module")
def user_client(uvicorn_server, user_identity):

    return client.FahAlchemyClient(
        api_url="http://127.0.0.1:8000/",
        identifier=user_identity["identifier"],
        key=user_identity["key"],
    )
