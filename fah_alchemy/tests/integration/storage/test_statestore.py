import random
from time import sleep
from typing import List, Dict
from pathlib import Path

import pytest
from gufe import AlchemicalNetwork
from gufe.tokenization import TOKENIZABLE_REGISTRY
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from fah_alchemy.storage import Neo4jStore
from fah_alchemy.storage.models import Task, TaskQueue, ProtocolDAGResultRef
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

    def test_get_transformation_results(self):
        # TODO: ADD TEST
        ...

    def test_get_transformation_failures(self):
        # TODO: ADD TEST
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

    def test_get_tasks(self):
        # TODO: ADD TEST
        ...

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

    def test_action_task(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # queue the tasks
        n4js.action_tasks(task_sks, taskqueue_sk)

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

        task_sks_fail = n4js.action_tasks(task_sks, taskqueue_sk2)
        assert all([i is None for i in task_sks_fail])

    def test_cancel_task(self):
        # TODO: ADD TEST
        ...

    def test_get_taskqueue_tasks(self):
        # TODO: ADD TEST
        ...

    def test_claim_taskqueue_tasks(self, n4js: Neo4jStore, network_tyk2, scope_test):
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
        n4js.action_tasks(task_sks, taskqueue_sk)

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

    def test_get_task_transformation(
            self,
            n4js: Neo4jStore, 
            network_tyk2, 
            scope_test, 
            protocoldagresult,
            ):
        # create a network with just the transformation we care about
        transformation = list(network_tyk2.edges)[0]
        network_sk = n4js.create_network(AlchemicalNetwork(edges=[transformation]), scope_test)

        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a task; use its scoped key to get its transformation
        # this should be the same one we used to spawn the task
        task_sk = n4js.create_task(transformation_sk)

        # get transformations back as both gufe objects and scoped keys
        tf, _ = n4js.get_task_transformation(task_sk)
        tf_sk, _ = n4js.get_task_transformation(task_sk, return_gufe=False)

        assert tf == transformation
        assert tf_sk == transformation_sk

        # pretend we completed this one, and we have a protocoldagresult for it
        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresult.key,
            success=True
        )

        # try to push the result
        pdr_ref_sk = n4js.set_task_result(task_sk, pdr_ref)

        # create a task that extends the previous one
        task_sk2 = n4js.create_task(transformation_sk, extends=task_sk)

        # get transformations and protocoldagresultrefs as both gufe objects and scoped keys
        tf, protocoldagresultref = n4js.get_task_transformation(task_sk2)
        tf_sk, protocoldagresultref_sk = n4js.get_task_transformation(task_sk2, return_gufe=False)

        assert pdr_ref == protocoldagresultref
        assert pdr_ref_sk == protocoldagresultref_sk

        assert tf == transformation
        assert tf_sk == transformation_sk


    def test_set_task_result(self, n4js: Neo4jStore, network_tyk2, scope_test, tmpdir):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskqueue_sk: ScopedKey = n4js.create_taskqueue(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a task; compute its result, and attempt to submit it
        task_sk = n4js.create_task(transformation_sk)

        transformation, protocoldag_prev = n4js.get_task_transformation(task_sk)
        protocoldag = transformation.create(
            extends=protocoldag_prev,
            name=str(task_sk),
        )

        # execute the task
        with tmpdir.as_cwd():
            protocoldagresult = execute_DAG(protocoldag, shared=Path(".").absolute())

        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresult.key
        )

        # try to push the result
        n4js.set_task_result(task_sk, pdr_ref)

        n = n4js.graph.run(
            f"""
                match (n:ProtocolDAGResultRef)<-[:RESULTS_IN]-(t:Task)
                return n
                """
        ).to_subgraph()

        assert n["location"] == pdr_ref.location
        assert n["obj_key"] == str(protocoldagresult.key)

    def test_get_task_results():
        # TODO: ADD TEST
        ...

    def test_get_task_failures():
        # TODO: ADD TEST
        ...

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
