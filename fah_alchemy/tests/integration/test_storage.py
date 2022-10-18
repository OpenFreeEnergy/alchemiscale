import pytest

from fah_alchemy.storage import Neo4jStore
from fah_alchemy.models import Scope, ScopedKey


class TestStateStore:
    ...


class TestNeo4jStore(TestStateStore):
    ...

    @pytest.fixture
    def n4js(self, graph):
        return Neo4jStore(graph)

    def test_server(self, graph):
        graph.service.system_graph.call("dbms.security.listUsers")

    def test_create_network(self, n4js, network_tyk2):
        with n4js.as_tempdb():
            an = network_tyk2

            scope = Scope(org="test-org", campaign="test-campaign", project="test-project")

            sk: ScopedKey = n4js.create_network(an, scope)

            out = n4js.graph.run(
                    f"""
                    match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                                 _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                                 _project: '{sk.project}'}}) 
                    return n
                    """)
            n = out.to_subgraph()

            assert n["name"] == 'tyk2_relative_benchmark'
