import pytest

from fah_alchemy.storage import Neo4jStore
from fah_alchemy.models import Scope


class TestStateStore:
    ...


class TestNeo4jStore(TestStateStore):
    ...

    @pytest.fixture
    def n4js(self, graph):
        return Neo4jStore(graph)

    def test_server(self, graph):
        graph.service.system_graph.call("dbms.security.listUsers")

    def create_network(self, n4js, network_tyk2):
        an = network_tyk2

        scope = Scope(org="test-org", campaign="test-campaign", project="test-project")

        n4js.create_network(an, scope)
