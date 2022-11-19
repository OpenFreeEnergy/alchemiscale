from multiprocessing import Process
from time import sleep

import pytest
import uvicorn
import requests
from fastapi.testclient import TestClient
from passlib.context import CryptContext

from gufe import AlchemicalNetwork

from fah_alchemy.settings import ComputeAPISettings, get_compute_api_settings
from fah_alchemy.storage import Neo4jStore
from fah_alchemy.compute import api, client
from fah_alchemy.security.models import CredentialedComputeIdentity
from fah_alchemy.security.auth import hash_key, generate_secret_key


## compute api

@pytest.fixture(scope='module')
def compute_identity():
    return dict(identifier='test-compute-identity', key='strong passphrase lol')


@pytest.fixture
def n4js_preloaded(n4js_fresh, network_tyk2, scope_test, compute_identity):
    n4js = n4js_fresh

    # set starting contents for many of the tests in this module
    sk1 = n4js.create_network(network_tyk2, scope_test)

    # create another alchemical network
    an2 = AlchemicalNetwork(edges=list(network_tyk2.edges)[:-2], name='incomplete')
    sk2 = n4js.create_network(an2, scope_test)

    # add a taskqueue for each network
    n4js.create_taskqueue(sk1)
    n4js.create_taskqueue(sk2)

    n4js.create_credentialed_entity(CredentialedComputeIdentity(
            identifier=compute_identity['identifier'],
            hashed_key=hash_key(compute_identity['key'])))
    
    return n4js


def get_settings_override():
    # settings overrides for test suite
    return ComputeAPISettings(
            NEO4J_USER='neo4j',
            NEO4J_PASS='password',
            NEO4J_URL="bolt://localhost:7687",
            FA_COMPUTE_API_HOST="127.0.0.1",
            FA_COMPUTE_API_PORT=8000,
            JWT_SECRET_KEY='98d11ba9ca329a4e5a6626faeffc6a9b9fb04e2745cff030f7d6793751bb8245',
            )

@pytest.fixture(scope='module')
def compute_api(n4js):

    def get_n4js_override():
        return n4js

    api.app.dependency_overrides[api.get_n4js] = get_n4js_override
    api.app.dependency_overrides[get_compute_api_settings] = get_settings_override
    return api.app


@pytest.fixture(scope='module')
def test_client(compute_api):
    client = TestClient(compute_api)
    return client


## compute client

def run_server(fastapi_app, settings):
    uvicorn.run(
            fastapi_app,
            host=settings.FA_COMPUTE_API_HOST,
            port=settings.FA_COMPUTE_API_PORT,
            log_level=settings.FA_COMPUTE_API_LOGLEVEL
            )


@pytest.fixture(scope='module')
def uvicorn_server(compute_api):
    settings = get_settings_override()
    proc = Process(target=run_server, args=(compute_api, settings), daemon=True)
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

    proc.kill() # Cleanup after test


@pytest.fixture(scope='module')
def compute_client(uvicorn_server, compute_identity):
    
    return client.FahAlchemyComputeClient(
            compute_api_url="http://127.0.0.1:8000/",
            identifier=compute_identity['identifier'],
            key=compute_identity['key']
            )
