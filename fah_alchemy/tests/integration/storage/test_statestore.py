import random
from time import sleep
from typing import List, Dict
from pathlib import Path

import pytest
from gufe import AlchemicalNetwork
from gufe.tokenization import TOKENIZABLE_REGISTRY
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from fah_alchemy.storage import Neo4jStore
from fah_alchemy.storage.models import Task, TaskQueue, ObjectStoreRef
from fah_alchemy.models import Scope, ScopedKey
from fah_alchemy.security.models import (
    CredentialedEntity,
    CredentialedUserIdentity,
    CredentialedComputeIdentity,
)
from fah_alchemy.security.auth import hash_key


class TestStateStore:
    ...


class TestNeo4jStore(TestStateStore):
    ...

    @pytest.fixture
    def n4js(self, n4js_fresh):
        return n4js_fresh

    def test_server(self, graph):
        graph.service.system_graph.call("dbms.security.listUsers")

    ### gufe otject handling

    def test_create_network(self, n4js, network_tyk2, scope_test):
        an = network_tyk2

        sk: ScopedKey = n4js.create_network(an, scope_test)

        out = n4js.graph.run(
            f"""
                match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                             _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                             _project: '{sk.project}'}}) 
                return n
                """
        )
        n = out.to_subgraph()

        assert n["name"] == "tyk2_relative_benchmark"

    def test_create_overlapping_networks(self, n4js, network_tyk2, scope_test):
        an = network_tyk2

        sk: ScopedKey = n4js.create_network(an, scope_test)

        n = n4js.graph.run(
            f"""
                match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                             _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                             _project: '{sk.project}'}}) 
                return n
                """
        ).to_subgraph()

        assert n["name"] == "tyk2_relative_benchmark"

        # add the same network twice
        sk2: ScopedKey = n4js.create_network(an, scope_test)
        assert sk2 == sk

        n2 = n4js.graph.run(
            f"""
                match (n:AlchemicalNetwork {{_gufe_key: '{an.key}', 
                                             _org: '{sk.org}', _campaign: '{sk.campaign}', 
                                             _project: '{sk.project}'}}) 
                return n
                """
        ).to_subgraph()

        assert n2["name"] == "tyk2_relative_benchmark"
        assert n2.identity == n.identity

        # add a slightly different network
        an2 = AlchemicalNetwork(
            edges=list(an.edges)[:-1], name="tyk2_relative_benchmark_-1"
        )
        sk3 = n4js.create_network(an2, scope_test)
        assert sk3 != sk

        n3 = n4js.graph.run(
            f"""
                match (n:AlchemicalNetwork) 
                return n
                """
        ).to_subgraph()

        assert len(n3.nodes) == 2

    def test_delete_network(self):
        ...

    def test_get_network(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        sk: ScopedKey = n4js.create_network(an, scope_test)

        an2 = n4js.get_gufe(sk)

        assert an2 == an
        assert an2 is an

        TOKENIZABLE_REGISTRY.clear()

        an3 = n4js.get_gufe(sk)

        assert an3 == an2 == an

    def test_query_network(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        an2 = AlchemicalNetwork(edges=list(an.edges)[:-2], name="incomplete")

        sk: ScopedKey = n4js.create_network(an, scope_test)
        sk2: ScopedKey = n4js.create_network(an2, scope_test)

        networks_sk: List[ScopedKey] = n4js.query_networks()

        assert sk in networks_sk
        assert sk2 in networks_sk
        assert len(networks_sk) == 2

        # add in a scope test

        # add in a name test

    def test_query_transformations(self):
        ...

    def test_query_chemicalsystems(self):
        ...

    ### compute

    def test_create_task(self, n4js, network_tyk2, scope_test):
        # add alchemical network, then try generating task
        an = network_tyk2
        n4js.create_network(an, scope_test)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        task_sk: ScopedKey = n4js.create_task(transformation_sk)

        m = n4js.graph.run(
            f"""
                match (n:Task {{_gufe_key: '{task_sk.gufe_key}', 
                                             _org: '{task_sk.org}', _campaign: '{task_sk.campaign}', 
                                             _project: '{task_sk.project}'}})-[:PERFORMS]->(m:Transformation)
                return m
                """
        ).to_subgraph()

        assert m["_gufe_key"] == transformation.key

    def test_create_taskqueue(self, n4js, network_tyk2, scope_test):
        # add alchemical network, then try adding a taskqueue
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)

        # create taskqueue
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        # verify creation looks as we expect
        m = n4js.graph.run(
            f"""
                match (n:TaskQueue {{_gufe_key: '{taskqueue_sk.gufe_key}', 
                                             _org: '{taskqueue_sk.org}', _campaign: '{taskqueue_sk.campaign}', 
                                             _project: '{taskqueue_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return m
                """
        ).to_subgraph()

        assert m["_gufe_key"] == an.key

        # try adding the task queue again; this should yield exactly the same node
        taskqueue_sk2: ScopedKey = n4js.create_taskqueue(network_sk)

        assert taskqueue_sk2 == taskqueue_sk

        records = n4js.graph.run(
            f"""
                match (n:TaskQueue {{network: '{network_sk}', 
                                             _org: '{taskqueue_sk.org}', _campaign: '{taskqueue_sk.campaign}', 
                                             _project: '{taskqueue_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return n
                """
        )

        assert len(list(records)) == 1

    def test_create_taskqueue_weight(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)

        # create taskqueue
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        n = n4js.graph.run(
            f"""
                match (n:TaskQueue)
                return n
                """
        ).to_subgraph()

        assert n["weight"] == 0.5

        # change the weight
        n4js.set_taskqueue_weight(network_sk, 0.7)

        n = n4js.graph.run(
            f"""
                match (n:TaskQueue)
                return n
                """
        ).to_subgraph()

        assert n["weight"] == 0.7

    def test_query_taskqueues(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        # add a slightly different network
        an2 = AlchemicalNetwork(
            edges=list(an.edges)[:-1], name="tyk2_relative_benchmark_-1"
        )
        network_sk2 = n4js.create_network(an2, scope_test)
        taskqueue_sk2: ScopedKey = n4js.create_taskqueue(network_sk2)

        tq_sks: List[ScopedKey] = n4js.query_taskqueues()
        assert len(tq_sks) == 2
        assert all([isinstance(i, ScopedKey) for i in tq_sks])

        tq_dict: Dict[ScopedKey, TaskQueue] = n4js.query_taskqueues(return_gufe=True)
        assert len(tq_dict) == 2
        assert all([isinstance(i, TaskQueue) for i in tq_dict.values()])

    def test_queue_task(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # queue the tasks
        n4js.queue_taskqueue_tasks(task_sks, taskqueue_sk)

        # count tasks in queue
        queued_task_sks = n4js.get_taskqueue_tasks(taskqueue_sk)
        assert task_sks == queued_task_sks

        # add a second network, with the transformation above missing
        # try to add a task from that transformation to the new network's queue
        # this should fail
        an2 = AlchemicalNetwork(
            edges=list(an.edges)[1:], name="tyk2_relative_benchmark_-1"
        )
        assert transformation not in an2.edges

        network_sk2 = n4js.create_network(an2, scope_test)
        taskqueue_sk2: ScopedKey = n4js.create_taskqueue(network_sk2)

        with pytest.raises(ValueError, match="not found in same network"):
            task_sks_fail = n4js.queue_taskqueue_tasks(task_sks, taskqueue_sk2)

    def test_claim_task(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # shuffle the tasks; want to check that order of claiming is actually
        # based on order in queue
        random.shuffle(task_sks)

        # try to claim from an empty queue
        nothing = n4js.claim_taskqueue_tasks(taskqueue_sk, "early bird task handler")
        assert nothing[0] is None

        # queue the tasks
        n4js.queue_taskqueue_tasks(task_sks, taskqueue_sk)

        # claim a single task; we expect this should be the first in the list
        claimed = n4js.claim_taskqueue_tasks(taskqueue_sk, "the best task handler")
        assert claimed[0] == task_sks[0]

        # set all tasks to priority 5, fourth task to priority 1; claim should
        # yield fourth task
        for task_sk in task_sks[1:]:
            n4js.set_task_priority(task_sk, 5)
        n4js.set_task_priority(task_sks[3], 1)

        claimed2 = n4js.claim_taskqueue_tasks(taskqueue_sk, "another task handler")
        assert claimed2[0] == task_sks[3]

        # next task claimed should be the second task in line
        claimed3 = n4js.claim_taskqueue_tasks(taskqueue_sk, "yet another task handler")
        assert claimed3[0] == task_sks[1]

        # try to claim multiple tasks
        claimed4 = n4js.claim_taskqueue_tasks(
            taskqueue_sk, "last task handler", count=4
        )
        assert claimed4[0] == task_sks[2]
        assert claimed4[1:] == task_sks[4:7]

        # exhaust the queue
        claimed5 = n4js.claim_taskqueue_tasks(
            taskqueue_sk, "last task handler", count=3
        )

        # try to claim from a queue with no tasks available
        claimed6 = n4js.claim_taskqueue_tasks(
            taskqueue_sk, "last task handler", count=2
        )
        assert claimed6 == [None] * 2

    def test_set_task_result(self, n4js: Neo4jStore, network_tyk2, scope_test, tmpdir):
        # need to understand why ProtocolDAGResult fails to be represented as
        # subgraph

        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a task; compute its result, and attempt to submit it
        task_sk = n4js.create_task(transformation_sk)

        transformation, protocoldag_prev = n4js.get_task_transformation(task_sk)
        protocoldag = transformation.protocol.create(
            stateA=transformation.stateA,
            stateB=transformation.stateB,
            mapping=transformation.mapping,
            extend_from=protocoldag_prev,
            name=str(task_sk),
        )

        # execute the task
        with tmpdir.as_cwd():
            protocoldagresult = execute_DAG(protocoldag, shared=Path(".").absolute())

        osr = ObjectStoreRef(location="protocoldagresult/{protocoldagresult.key}")

        # try to push the result
        n4js.set_task_result(task_sk, osr)

        n = n4js.graph.run(
            f"""
                match (n:ObjectStoreRef)<-[:RESULTS_IN]-(t:Task)
                return n
                """
        ).to_subgraph()

        assert n["location"] == osr.location

    ### authentication

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    def test_create_credentialed_entity(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity
    ):

        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        cls_name = credential_type.__name__

        n4js.create_credentialed_entity(user)

        n = n4js.graph.run(
            f"""
            match (n:{cls_name} {{identifier: '{user.identifier}'}})
            return n
            """
        ).to_subgraph()

        assert n["identifier"] == user.identifier
        assert n["hashed_key"] == user.hashed_key

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    def test_get_credentialed_entity(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity
    ):

        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        # get the user back
        user_g = n4js.get_credentialed_entity(user.identifier, credential_type)

        assert user_g == user

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    def test_list_credentialed_entities(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity
    ):

        identities = ("bill", "ted", "napoleon")

        for ident in identities:
            user = credential_type(
                identifier=ident,
                hashed_key=hash_key("a string for a key"),
            )

            n4js.create_credentialed_entity(user)

        # get the user back
        identities_ = n4js.list_credentialed_entities(credential_type)

        assert set(identities) == set(identities_)

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    def test_remove_credentialed_entity(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity
    ):

        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        # get the user back
        user_g = n4js.get_credentialed_entity(user.identifier, credential_type)

        assert user_g == user

        n4js.remove_credentialed_identity(user.identifier, credential_type)
        with pytest.raises(KeyError):
            n4js.get_credentialed_entity(user.identifier, credential_type)

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    @pytest.mark.parametrize("scope_str", ("*-*-*", "a-*-*", "a-b-*", "a-b-c"))
    def test_list_scope(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity, scope_str: str
    ):

        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        scope = Scope.from_str(scope_str)

        n4js.add_scope(user.identifier, credential_type, scope)

        scopes = n4js.list_scopes(user.identifier, credential_type)

        assert scope in scopes

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    @pytest.mark.parametrize("scope_str", ("*-*-*", "a-*-*", "a-b-*", "a-b-c"))
    def test_add_scope(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity, scope_str: str
    ):

        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        scope = Scope.from_str(scope_str)

        n4js.add_scope(user.identifier, credential_type, scope)

        q = f"""
        MATCH (n:{credential_type.__name__} {{identifier: '{user.identifier}'}})
        MERGE (n)-[r:HAS_SCOPE]->(s:Scope)
        RETURN s
        """

        s = n4js.graph.run(q).to_subgraph()
        org = s["org"]
        campaign = s["campaign"]
        project = s["project"]

        new_scope = Scope.from_str_tuple((org, campaign, project))

        assert new_scope == scope
