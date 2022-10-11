
import pytest

from fah_alchemy.storage import Neo4jStore



class TestStateStore:
    ...


class TestNeo4jStore(TestStateStore):
    ...

    def test_server(self, graph):
        graph.service.system_graph.call('dbms.security.listUsers')
