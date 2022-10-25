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

    ### gufe otject handling

    def test_create_network(self, n4js, network_tyk2, scope_test):
        with n4js.as_tempdb():
            an = network_tyk2

            sk: ScopedKey = n4js.create_network(an, scope_test)

            out = n4js.graph.run(
                    f"""
                    match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                                 _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                                 _project: '{sk.project}'}}) 
                    return n
                    """)
            n = out.to_subgraph()

            assert n["name"] == 'tyk2_relative_benchmark'

    def test_update_network(self, n4js, network_tyk2, scope_test):
        with n4js.as_tempdb():
            an = network_tyk2

            sk: ScopedKey = n4js.create_network(an, scope_test)

            n = n4js.graph.run(
                    f"""
                    match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                                 _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                                 _project: '{sk.project}'}}) 
                    return n
                    """).to_subgraph()

            assert n["name"] == 'tyk2_relative_benchmark'

            with pytest.raises(ValueError):
                n4js.create_network(an, scope_test)

            sk2: ScopedKey = n4js.update_network(an, scope_test)

            assert sk2 == sk

            n2 = n4js.graph.run(
                    f"""
                    match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                                 _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                                 _project: '{sk.project}'}}) 
                    return n
                    """).to_subgraph()

            assert n2["name"] == 'tyk2_relative_benchmark'

            assert n2.identity == n.identity

    def test_delete_network(self):
        ...

    def test_get_network(self):
        ...

    def test_query_networks(self):
        ...

    def test_query_transformations(self):
        ...

    def test_query_chemicalsystems(self):
        ...

    ### compute

    def test_create_task(self, n4js, network_tyk2, scope_test):
        with n4js.as_tempdb():
            an = network_tyk2
            transformation = list(an.edges)[0]

            # try creating a task for a transformation that is not present
            with pytest.raises(ValueError):
                task = n4js.create_task(
                            transformation=transformation,
                            scope=scope_test)

            # add alchemical network, then try generating task
            n4js.create_network(an, scope_test)

            task_sk: ScopedKey = n4js.create_task(
                        transformation=transformation,
                        scope=scope_test)

            n = n4js.graph.run(
                    f"""
                    match (n:Task {{_gufe_key: '{task_sk.gufe_key}', 
                                                 _org: '{task_sk.org}', _campaign: '{task_sk.campaign}', 
                                                 _project: '{task_sk.project}'}})-[:PERFORMS]->(m:Transformation)
                    return m
                    """).to_subgraph()

            assert n['_gufe_key'] == transformation.key

            # add another task, this time with the scoped key for the transformation

    def test_create_taskqueue(self, n4js, network_tyk2, scope_test):
        with n4js.as_tempdb():
            an = network_tyk2

            # try creating a taskqueue for a network that is not present
            with pytest.raises(ValueError):
                task = n4js.create_taskqueue(
                            network=an,
                            scope=scope_test)

            # add alchemical network, then try adding a taskqueue
            network_sk = n4js.create_network(an, scope_test)

            # create taskqueue
            taskqueue_sk: ScopedKey = n4js.create_taskqueue(
                        network=an,
                        scope=scope_test)

            # verify creation looks as we expect
            n = n4js.graph.run(
                    f"""
                    match (n:TaskQueue {{_gufe_key: '{taskqueue_sk.gufe_key}', 
                                                 _org: '{taskqueue_sk.org}', _campaign: '{taskqueue_sk.campaign}', 
                                                 _project: '{taskqueue_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                    return m
                    """).to_subgraph()

            assert n['_gufe_key'] == an.key

            # try adding the task queue again; this should yield exactly the same node
            taskqueue_sk2: ScopedKey = n4js.create_taskqueue(
                        network=an,
                        scope=scope_test)

            assert taskqueue_sk2 == taskqueue_sk

            records = n4js.graph.run(
                    f"""
                    match (n:TaskQueue {{network: '{network_sk}', 
                                                 _org: '{taskqueue_sk.org}', _campaign: '{taskqueue_sk.campaign}', 
                                                 _project: '{taskqueue_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                    return n
                    """)

            assert len(list(records)) == 1

    def test_create_taskqueue_weight(self, n4js, network_tyk2, scope_test):
        with n4js.as_tempdb():
            an = network_tyk2

            # add alchemical network, then try adding a taskqueue
            network_sk = n4js.create_network(an, scope_test)

            # create taskqueue
            taskqueue_sk: ScopedKey = n4js.create_taskqueue(
                        network=an,
                        scope=scope_test)

            n = n4js.graph.run(
                    f"""
                    match (n:TaskQueue)
                    return n
                    """).to_subgraph()

            assert n['weight'] == .5

            # change the weight
            n4js.set_taskqueue_weight(an, scope_test, .7)

            n = n4js.graph.run(
                    f"""
                    match (n:TaskQueue)
                    return n
                    """).to_subgraph()

            assert n['weight'] == .7
