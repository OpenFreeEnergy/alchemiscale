from multiprocessing import Process
from time import sleep

import pytest
from fastapi.testclient import TestClient
import uvicorn
import requests

from gufe import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable

from fah_alchemy.models import ScopedKey
from fah_alchemy.storage import Neo4jStore
from fah_alchemy.compute import api
from fah_alchemy.compute import client


@pytest.fixture
def n4js_clear(graph, network_tyk2, scope_test):
    # clear graph contents; want a fresh state for database
    graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

    # set starting contents for all tests in this module
    n4js = Neo4jStore(graph)
    sk1 = n4js.create_network(network_tyk2, scope_test)

    # create another alchemical network
    an2 = AlchemicalNetwork(edges=list(network_tyk2.edges)[:-2], name='incomplete')
    sk2 = n4js.create_network(an2, scope_test)

    # add a taskqueue for each network
    n4js.create_taskqueue(sk1)
    n4js.create_taskqueue(sk2)
    
    return n4js

@pytest.fixture(scope='module')
def n4js(graph, network_tyk2, scope_test):
    return Neo4jStore(graph)


#def get_settings_override():
#    # settings overrides for test suite
#    return api.Settings(
#            neo4j_url = "bolt://localhost:7687",
#            neo4j_user = "neo4j",
#            neo4j_user = "password"
#            )

def get_settings_override():
    # settings overrides for test suite
    return api.Settings(
            NEO4J_USER='neo4j',
            NEO4J_PASS='password',
            NEO4J_URL="bolt://localhost:7687",
            FA_COMPUTE_API_HOST="127.0.0.1",
            FA_COMPUTE_API_PORT=8000,
            )

@pytest.fixture(scope='module')
def compute_api(n4js):

    def get_n4js_override():
        return n4js

    api.app.dependency_overrides[api.get_n4js] = get_n4js_override
    api.app.dependency_overrides[api.get_settings] = get_settings_override
    return api.app


@pytest.fixture(scope='module')
def test_client(compute_api):
    client = TestClient(compute_api)
    return client


# api tests

class TestComputeAPI:

    def test_info(self, test_client):
    
        response = test_client.get("/info")
        assert response.status_code == 200
    
    
    def test_query_taskqueues(self, n4js_clear, test_client):
    
        response = test_client.get("/taskqueues")
        assert response.status_code == 200
    
        tq_sks = [ScopedKey.from_str(i) for i in response.json()]
        assert len(tq_sks) == 2

        # try getting back actual gufe objects
        response = test_client.get("/taskqueues?return_gufe=True")
        assert response.status_code == 200

        tq_dict = {ScopedKey.from_str(k): GufeTokenizable.from_dict(v) for k, v in response.json().items()}
        assert len(tq_dict) == 2
        assert all([i.weight == .5 for i in tq_dict.values()])


# client tests

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
def compute_client(uvicorn_server):
    
    return client.FahAlchemyComputeClient(
            compute_api_url="http://127.0.0.1:8000/",
            compute_api_key=None
            )


class TestComputeClient:

    def test_query_taskqueues(self, 
                              scope_test, 
                              n4js_clear,
                              compute_client: client.FahAlchemyComputeClient, 
                              uvicorn_server
                              ):

        taskqueues = compute_client.query_taskqueues(scope_test)

        assert len(taskqueues) == 2

        taskqueues = compute_client.query_taskqueues(scope_test, return_gufe=True)
        assert all([tq.weight == .5 for tq in taskqueues.values()])

    def test_claim_taskqueue_task(self, 
                              scope_test, 
                              n4js_clear,
                              compute_client: client.FahAlchemyComputeClient, 
                              uvicorn_server
                              ):

        ...
        #task = compute_client.claim_taskqueue_task(scope_test)
