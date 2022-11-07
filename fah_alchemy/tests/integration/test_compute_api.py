import pytest
from fastapi.testclient import TestClient

from gufe import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable

from fah_alchemy.storage import Neo4jStore
from fah_alchemy.compute import api


@pytest.fixture
def n4js(graph, network_tyk2, scope_test):
    # clear graph contents; want a fresh state for database
    graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

    # set starting contents for all tests in this module
    n4js = Neo4jStore(graph)
    sk1 = n4js.create_network(network_tyk2, scope_test)

    # create another alchemical network
    an2 = AlchemicalNetwork(edges=list(network_tyk2.edges)[:-2], name='incomplete')
    sk2 = n4js.create_network(an2, scope_test)

    # add a taskqueue for each network
    n4js.create_taskqueue(sk1, scope_test)
    n4js.create_taskqueue(sk2, scope_test)
    
    return n4js


#def get_settings_override():
#    # settings overrides for test suite
#    return api.Settings(
#            neo4j_url = "bolt://localhost:7687",
#            neo4j_user = "neo4j",
#            neo4j_user = "password"
#            )


@pytest.fixture
def compute_api(n4js):

    def get_n4js_override(settings: api.Settings = api.get_settings):
        return n4js

    api.app.dependency_overrides[api.get_n4js] = get_n4js_override
    client = TestClient(api.app)
    return client


# api tests

def test_info(compute_api):

    response = compute_api.get("/info")
    assert response.status_code == 200


def test_query_taskqueues(network_tyk2, scope_test, n4js, compute_api):

    scoped_key = n4js.create_taskqueue(network_tyk2, scope_test)

    scope2 = scope_test.dict()
    scope2['org'] = 'another_org'

    response = compute_api.get("/taskqueues")
    assert response.status_code == 200

    taskqueues = [GufeTokenizable.from_dict(i) for i in response.json()]
    assert len(taskqueues) == 2
