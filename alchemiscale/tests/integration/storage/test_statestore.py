import random
from time import sleep
from typing import List, Dict
from pathlib import Path

import pytest
from gufe import AlchemicalNetwork
from gufe.tokenization import TOKENIZABLE_REGISTRY
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from alchemiscale.storage import Neo4jStore
from alchemiscale.storage.models import Task, TaskHub, ProtocolDAGResultRef
from alchemiscale.models import Scope, ScopedKey
from alchemiscale.security.models import (
    CredentialedEntity,
    CredentialedUserIdentity,
    CredentialedComputeIdentity,
)
from alchemiscale.security.auth import hash_key


class TestStateStore:
    ...


class TestNeo4jStore(TestStateStore):
    ...

    @pytest.fixture
    def n4js(self, n4js_fresh):
        return n4js_fresh

    def test_server(self, graph):
        graph.service.system_graph.call("dbms.security.listUsers")

    ### gufe object handling

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

        # TODO: add in a scope test

        # TODO: add in a name test

    def test_query_transformations(self):
        ...

    def test_query_chemicalsystems(self):
        ...

    def test_get_transformation_results(
        self,
        n4js: Neo4jStore,
        network_tyk2,
        scope_test,
        transformation,
        protocoldagresults,
    ):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a task; pretend we computed it, submit reference for pre-baked
        # result
        task_sk = n4js.create_task(transformation_sk)

        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults[0].key,
            ok=protocoldagresults[0].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref)

        # get the result back, at the transformation level
        pdr_refs = n4js.get_transformation_results(transformation_sk)

        assert len(pdr_refs) == 1
        assert pdr_ref in pdr_refs

        # try adding a new task, then adding the same result to it
        # should result in two tasks pointing to the same result, and yield
        # only one
        task_sk2 = n4js.create_task(transformation_sk)
        n4js.set_task_result(task_sk2, pdr_ref)
        pdr_refs2 = n4js.get_transformation_results(transformation_sk)

        assert len(pdr_refs2) == 1
        assert pdr_ref in pdr_refs2

        # try adding additional unique results to one of the tasks
        for pdr in protocoldagresults[1:]:
            pdr_ref_ = ProtocolDAGResultRef(
                scope=task_sk.scope, obj_key=pdr.key, ok=pdr.ok()
            )
            # push the result
            n4js.set_task_result(task_sk, pdr_ref_)

        # now get all results back for this transformation
        pdr_refs3 = n4js.get_transformation_results(transformation_sk)

        assert len(pdr_refs3) == 3
        assert set([p.obj_key for p in pdr_refs3]) == set(
            [p.key for p in protocoldagresults]
        )

    def test_get_transformation_failures(
        self,
        n4js: Neo4jStore,
        network_tyk2_failure,
        scope_test,
        transformation_failure,
        protocoldagresults_failure,
    ):
        an = network_tyk2_failure
        network_sk = n4js.create_network(an, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation_failure, scope_test)

        # create a task; pretend we computed it, submit reference for pre-baked
        # result
        task_sk = n4js.create_task(transformation_sk)

        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults_failure[0].key,
            ok=protocoldagresults_failure[0].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref)

        # try to get the result back, at the transformation level
        pdr_refs = n4js.get_transformation_results(transformation_sk)

        assert len(pdr_refs) == 0

        # try to get failure back
        failure_pdr_refs = n4js.get_transformation_failures(transformation_sk)

        assert len(failure_pdr_refs) == 1
        assert pdr_ref in failure_pdr_refs

        # try adding a new task, then adding the same result to it
        # should result in two tasks pointing to the same result, and yield
        # only one
        task_sk2 = n4js.create_task(transformation_sk)
        n4js.set_task_result(task_sk2, pdr_ref)
        pdr_refs2 = n4js.get_transformation_failures(transformation_sk)

        assert len(pdr_refs2) == 1
        assert pdr_ref in pdr_refs2

        # try adding additional unique results to one of the tasks
        for pdr in protocoldagresults_failure[1:]:
            pdr_ref_ = ProtocolDAGResultRef(
                scope=task_sk.scope, obj_key=pdr.key, ok=pdr.ok()
            )
            # push the result
            n4js.set_task_result(task_sk, pdr_ref_)

        # should still get 0 results back for this transformation
        assert len(n4js.get_transformation_results(transformation_sk)) == 0

        # but should get 3 failures back if we ask for those
        pdr_refs3 = n4js.get_transformation_failures(transformation_sk)

        assert len(pdr_refs3) == 3
        assert set([p.obj_key for p in pdr_refs3]) == set(
            [p.key for p in protocoldagresults_failure]
        )

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

    def test_get_tasks(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a tree of tasks for a selected transformation
        task_sks = []
        for i in range(3):
            task_i: ScopedKey = n4js.create_task(transformation_sk)
            task_sks.append(task_i)
            for j in range(3):
                task_j = n4js.create_task(transformation_sk, extends=task_i)
                task_sks.append(task_j)
                for k in range(3):
                    task_k = n4js.create_task(transformation_sk, extends=task_j)
                    task_sks.append(task_k)

        # get all tasks for the transformation
        all_task_sks: List[ScopedKey] = n4js.get_tasks(transformation_sk)

        f = lambda x, y: x**y + x ** (y - 1) + x ** (y - 2)

        assert len(all_task_sks) == f(3, 3)
        assert set(task_sks) == set(all_task_sks)

        # try getting back only tasks extending from a given one
        subtree = n4js.get_tasks(transformation_sk, extends=task_sks[0])

        assert len(subtree) == f(3, 2) - 1
        assert set(subtree) == set(task_sks[1:13])

        # try getting tasks back in the "graph" representation instead
        # this is a mapping of each Task to the Task they extend, if applicable
        graph = n4js.get_tasks(transformation_sk, return_as="graph")

        assert len(graph) == len(task_sks)
        assert set(graph.keys()) == set(task_sks)
        assert all([graph[t] == task_sks[0] for t in task_sks[1:13:4]])

    def test_create_taskhub(self, n4js, network_tyk2, scope_test):
        # add alchemical network, then try adding a taskhub
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)

        # create taskhub
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        # verify creation looks as we expect
        m = n4js.graph.run(
            f"""
                match (n:TaskHub {{_gufe_key: '{taskhub_sk.gufe_key}', 
                                             _org: '{taskhub_sk.org}', _campaign: '{taskhub_sk.campaign}', 
                                             _project: '{taskhub_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return m
                """
        ).to_subgraph()

        assert m["_gufe_key"] == an.key

        # try adding the task hub again; this should yield exactly the same node
        taskhub_sk2: ScopedKey = n4js.create_taskhub(network_sk)

        assert taskhub_sk2 == taskhub_sk

        records = n4js.graph.run(
            f"""
                match (n:TaskHub {{network: '{network_sk}', 
                                             _org: '{taskhub_sk.org}', _campaign: '{taskhub_sk.campaign}', 
                                             _project: '{taskhub_sk.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return n
                """
        )

        assert len(list(records)) == 1

    def test_create_taskhub_weight(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)

        # create taskhub
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        n = n4js.graph.run(
            f"""
                match (n:TaskHub)
                return n
                """
        ).to_subgraph()

        assert n["weight"] == 0.5

        # change the weight
        n4js.set_taskhub_weight(network_sk, 0.7)

        n = n4js.graph.run(
            f"""
                match (n:TaskHub)
                return n
                """
        ).to_subgraph()

        assert n["weight"] == 0.7

    def test_query_taskhubs(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        # add a slightly different network
        an2 = AlchemicalNetwork(
            edges=list(an.edges)[:-1], name="tyk2_relative_benchmark_-1"
        )
        network_sk2 = n4js.create_network(an2, scope_test)
        taskhub_sk2: ScopedKey = n4js.create_taskhub(network_sk2)

        tq_sks: List[ScopedKey] = n4js.query_taskhubs()
        assert len(tq_sks) == 2
        assert all([isinstance(i, ScopedKey) for i in tq_sks])

        tq_dict: Dict[ScopedKey, TaskHub] = n4js.query_taskhubs(return_gufe=True)
        assert len(tq_dict) == 2
        assert all([isinstance(i, TaskHub) for i in tq_dict.values()])

    def test_action_task(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # action the tasks
        n4js.action_tasks(task_sks, taskhub_sk)

        # count tasks actioned
        actioned_task_sks = n4js.get_taskhub_tasks(taskhub_sk)
        assert set(task_sks) == set(actioned_task_sks)
        assert len(task_sks) == 10

        # add a second network, with the transformation above missing
        # try to add a task from that transformation to the new network's hub
        # this should fail
        an2 = AlchemicalNetwork(
            edges=list(an.edges)[1:], name="tyk2_relative_benchmark_-1"
        )
        assert transformation not in an2.edges

        network_sk2 = n4js.create_network(an2, scope_test)
        taskhub_sk2: ScopedKey = n4js.create_taskhub(network_sk2)

        task_sks_fail = n4js.action_tasks(task_sks, taskhub_sk2)
        assert all([i is None for i in task_sks_fail])

    def test_action_task_extends(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks that extend in an EXTENDS chain
        first_task = n4js.create_task(transformation_sk)
        collected_sks = [first_task]
        prev = first_task
        for i in range(9):
            curr = n4js.create_task(transformation_sk, extends=prev)
            collected_sks.append(curr)
            prev = curr

        # action the tasks
        actioned_task_sks = n4js.action_tasks(collected_sks, taskhub_sk)
        assert set(actioned_task_sks) == set(collected_sks)

    def test_get_unclaimed_tasks(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # action the tasks
        n4js.action_tasks(task_sks, taskhub_sk)

        # claim a single task; There is no deterministic ordering of tasks, so
        # simply test that the claimed task is one of the actioned tasks
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, "the best task handler")

        assert claimed[0] in task_sks

        # query for unclaimed tasks
        unclaimed = n4js.get_taskhub_unclaimed_tasks(taskhub_sk)

        assert set(unclaimed) == set(task_sks) - set(claimed)
        assert len(unclaimed) == 9

    def test_get_set_weights(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]
        n4js.action_tasks(task_sks, taskhub_sk)

        # weights should all be the default 1.0
        weights = n4js.get_task_weights(task_sks, taskhub_sk)
        assert all([w == 1.0 for w in weights])

        # set weights on the tasks to be all 10
        n4js.set_task_weights(task_sks, taskhub_sk, weight=10)
        weights = n4js.get_task_weights(task_sks, taskhub_sk)
        assert all([w == 10 for w in weights])

    def test_cancel_task(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # action the tasks
        actioned = n4js.action_tasks(task_sks, taskhub_sk)

        # cancel the second and third task we created
        canceled = n4js.cancel_tasks(task_sks[1:3], taskhub_sk)

        # check that the hub has the contents we expect
        tasks = n4js.graph.run(
            f"""
                MATCH (tq:TaskHub {{_scoped_key: '{taskhub_sk}'}})-[:ACTIONS]->(task:Task)
                return task
                """
        )
        tasks = [record["task"] for record in tasks]

        assert len(tasks) == 8
        assert set([ScopedKey.from_str(t["_scoped_key"]) for t in tasks]) == set(
            actioned
        ) - set(canceled)

    def test_get_taskhub_tasks(self, n4js, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # action the tasks
        actioned = n4js.action_tasks(task_sks, taskhub_sk)
        assert len(actioned) == 10

        # get the full hub back; no particular order
        task_sks = n4js.get_taskhub_tasks(taskhub_sk)

        # no particular order so must check that the sets are equal
        assert set(actioned) == set(task_sks)

        # try getting back as gufe objects instead
        tasks = n4js.get_taskhub_tasks(taskhub_sk, return_gufe=True)

        assert all([t.key == tsk.gufe_key for t, tsk in zip(tasks.values(), task_sks)])

    def test_claim_taskhub_tasks(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # shuffle the tasks; want to check that order of claiming is unrelated
        # to order created
        random.shuffle(task_sks)

        # try to claim from an empty hub
        nothing = n4js.claim_taskhub_tasks(taskhub_sk, "early bird task handler")

        assert nothing[0] is None

        # action the tasks
        n4js.action_tasks(task_sks, taskhub_sk)

        # claim a single task; there is no deterministic ordering of tasks, so
        # simply test that the claimed task is one of the actioned tasks
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, "the best task handler")

        assert claimed[0] in task_sks

        # filter out the claimed task so that we have clean list of remaining
        # tasks
        remaining_tasks = n4js.get_taskhub_unclaimed_tasks(taskhub_sk)

        # set all tasks to priority 5, first task to priority 1; claim should
        # yield first task
        for task_sk in remaining_tasks:
            n4js.set_task_priority(task_sk, 5)
        n4js.set_task_priority(remaining_tasks[0], 1)

        claimed2 = n4js.claim_taskhub_tasks(taskhub_sk, "another task handler")
        assert claimed2[0] == remaining_tasks[0]

        remaining_tasks = n4js.get_taskhub_unclaimed_tasks(taskhub_sk)

        # next task claimed should be one of the remaining tasks
        claimed3 = n4js.claim_taskhub_tasks(taskhub_sk, "yet another task handler")
        assert claimed3[0] in remaining_tasks

        remaining_tasks = n4js.get_taskhub_unclaimed_tasks(taskhub_sk)

        # try to claim multiple tasks
        claimed4 = n4js.claim_taskhub_tasks(taskhub_sk, "last task handler", count=4)
        assert len(claimed4) == 4
        for sk in claimed4:
            assert sk in remaining_tasks

        # exhaust the hub
        claimed5 = n4js.claim_taskhub_tasks(taskhub_sk, "last task handler", count=3)

        # try to claim from a hub with no tasks available
        claimed6 = n4js.claim_taskhub_tasks(taskhub_sk, "last task handler", count=2)
        assert claimed6 == [None] * 2

    def test_action_claim_task_extends(
        self, n4js: Neo4jStore, network_tyk2, scope_test
    ):
        # tests the ability to action and claim a set of tasks in an
        # EXTENDS chain
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks that extend in an EXTENDS chain
        first_task = n4js.create_task(transformation_sk)
        collected_sks = [first_task]
        prev = first_task
        for i in range(9):
            curr = n4js.create_task(transformation_sk, extends=prev)
            collected_sks.append(curr)
            prev = curr

        # action the tasks
        actioned_task_sks = n4js.action_tasks(collected_sks, taskhub_sk)
        assert set(actioned_task_sks) == set(collected_sks)

        # claim the first task
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler")

        assert claimed_task_sks == collected_sks[:1]

        # claim the next 9 tasks
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=9)
        # oops the extends task is still running!
        assert claimed_task_sks == [None] * 9

        # complete the extends task
        n4js.set_task_complete(first_task)

        # claim the next task again
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=1)
        assert claimed_task_sks == collected_sks[1:2]

    def test_action_claim_task_extends_non_extends(
        self, n4js: Neo4jStore, network_tyk2, scope_test
    ):
        # tests the ability to action and claim a set of tasks that have a mix of
        # EXTENDS and non-EXTENDS tasks
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks that extend in an EXTENDS chain
        first_task = n4js.create_task(transformation_sk)
        collected_sks = [first_task]
        prev = first_task
        for i in range(9):
            curr = n4js.create_task(transformation_sk, extends=prev)
            collected_sks.append(curr)
            prev = curr

        # create another two tasks that don't extend anything
        extra_task_1 = n4js.create_task(transformation_sk)
        extra_task_2 = n4js.create_task(transformation_sk)
        extra_tasks = [extra_task_1, extra_task_2]
        collected_sks.extend(extra_tasks)

        # action the tasks
        actioned_task_sks = n4js.action_tasks(collected_sks, taskhub_sk)
        assert set(actioned_task_sks) == set(collected_sks)

        # claim the first task **3** tasks, this set should be the first extends
        # task and the two non-extends tasks
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=3)

        assert set(claimed_task_sks) == set([first_task] + extra_tasks)

        # claim the next 10 tasks
        claimed_task_sks = n4js.claim_taskhub_tasks(
            taskhub_sk, "task handler", count=10
        )
        # oops the extends task is still running and there should be no other tasks to grab
        assert claimed_task_sks == [None] * 10

        # complete the extends task
        n4js.set_task_complete(first_task)

        # claim the next task again
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=1)
        assert claimed_task_sks == collected_sks[1:2]

    def test_action_claim_task_extends_bifuricating(
        self, n4js: Neo4jStore, network_tyk2, scope_test
    ):
        # tests the ability to action and claim a set of tasks in an
        # EXTENDS chain
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 7 tasks that extend in a bifuricating  EXTENDS chain

        first_task = n4js.create_task(transformation_sk)

        layer_two_1 = n4js.create_task(transformation_sk, extends=first_task)
        layer_two_2 = n4js.create_task(transformation_sk, extends=first_task)

        layer_three_1 = n4js.create_task(transformation_sk, extends=layer_two_1)
        layer_three_2 = n4js.create_task(transformation_sk, extends=layer_two_1)
        layer_three_3 = n4js.create_task(transformation_sk, extends=layer_two_2)
        layer_three_4 = n4js.create_task(transformation_sk, extends=layer_two_2)

        collected_sks = [
            first_task,
            layer_two_1,
            layer_two_2,
            layer_three_1,
            layer_three_2,
            layer_three_3,
            layer_three_4,
        ]
        # action the tasks
        actioned_task_sks = n4js.action_tasks(collected_sks, taskhub_sk)
        assert set(actioned_task_sks) == set(collected_sks)

        # claim the first task
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler")

        assert claimed_task_sks == [first_task]
        # complete the first task
        n4js.set_task_complete(first_task)

        # claim the next layer of tasks, should be all of layer two
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=2)
        assert set(claimed_task_sks) == set([layer_two_1, layer_two_2])

        # complete the layer two tasks
        n4js.set_task_complete(layer_two_1)
        n4js.set_task_complete(layer_two_2)

        # claim the next layer of tasks, should be all of layer three
        claimed_task_sks = n4js.claim_taskhub_tasks(taskhub_sk, "task handler", count=4)
        assert set(claimed_task_sks) == set(
            [layer_three_1, layer_three_2, layer_three_3, layer_three_4]
        )

    def test_claim_task_byweight(self, n4js: Neo4jStore, network_tyk2, scope_test):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

        transformation = list(an.edges)[0]
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create 10 tasks
        task_sks = [n4js.create_task(transformation_sk) for i in range(10)]

        # action the tasks
        n4js.action_tasks(task_sks, taskhub_sk)

        # shuffle the tasks; want to check that order of claiming is unrelated
        # to order actioned
        random.shuffle(task_sks)

        # set weights on the tasks to be all 0, disabling them
        n4js.set_task_weights(task_sks, taskhub_sk, weight=0)

        # set the weight of the first task to be 10
        weight_dict = {task_sks[0]: 10}
        n4js.set_task_weights(weight_dict, taskhub_sk)

        # check that the claimed task is the first task
        claimed = n4js.claim_taskhub_tasks(taskhub_sk, "the best task handler")
        assert claimed[0] == task_sks[0]

        # claim again; should get None as no other tasks have any weight
        claimed_again = n4js.claim_taskhub_tasks(taskhub_sk, "the best task handler")
        assert claimed_again[0] == None

    def test_get_task_transformation(
        self,
        n4js: Neo4jStore,
        network_tyk2,
        scope_test,
        protocoldagresults,
    ):
        # create a network with just the transformation we care about
        transformation = list(network_tyk2.edges)[0]
        network_sk = n4js.create_network(
            AlchemicalNetwork(edges=[transformation]), scope_test
        )

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
            scope=task_sk.scope, obj_key=protocoldagresults[0].key, ok=True
        )

        # try to push the result
        pdr_ref_sk = n4js.set_task_result(task_sk, pdr_ref)

        # create a task that extends the previous one
        task_sk2 = n4js.create_task(transformation_sk, extends=task_sk)

        # get transformations and protocoldagresultrefs as both gufe objects and scoped keys
        tf, protocoldagresultref = n4js.get_task_transformation(task_sk2)
        tf_sk, protocoldagresultref_sk = n4js.get_task_transformation(
            task_sk2, return_gufe=False
        )

        assert pdr_ref == protocoldagresultref
        assert pdr_ref_sk == protocoldagresultref_sk

        assert tf == transformation
        assert tf_sk == transformation_sk

    def test_set_task_result(self, n4js: Neo4jStore, network_tyk2, scope_test, tmpdir):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        taskhub_sk: ScopedKey = n4js.create_taskhub(network_sk)

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
            scope=task_sk.scope, obj_key=protocoldagresult.key, ok=True
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

    def test_get_task_results(
        self,
        n4js: Neo4jStore,
        network_tyk2,
        scope_test,
        transformation,
        protocoldagresults,
    ):
        an = network_tyk2
        network_sk = n4js.create_network(an, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation, scope_test)

        # create a task; pretend we computed it, submit reference for pre-baked
        # result
        task_sk = n4js.create_task(transformation_sk)

        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults[0].key,
            ok=protocoldagresults[0].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref)

        # get the result back
        pdr_refs = n4js.get_task_results(task_sk)

        assert len(pdr_refs) == 1
        assert pdr_ref in pdr_refs

        # try doing it again; should be idempotent
        n4js.set_task_result(task_sk, pdr_ref)
        pdr_refs = n4js.get_task_results(task_sk)

        assert len(pdr_refs) == 1
        assert pdr_ref in pdr_refs

        # if we add a different result, should now have 2
        pdr_ref2 = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults[1].key,
            ok=protocoldagresults[1].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref2)

        # get the result back
        pdr_refs = n4js.get_task_results(task_sk)

        assert len(pdr_refs) == 2
        assert pdr_ref in pdr_refs
        assert pdr_ref2 in pdr_refs

    def test_get_task_failures(
        self,
        n4js: Neo4jStore,
        network_tyk2_failure,
        scope_test,
        transformation_failure,
        protocoldagresults_failure,
    ):
        an = network_tyk2_failure
        network_sk = n4js.create_network(an, scope_test)
        transformation_sk = n4js.get_scoped_key(transformation_failure, scope_test)

        # create a task; pretend we computed it, submit reference for pre-baked
        # result
        task_sk = n4js.create_task(transformation_sk)

        pdr_ref = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults_failure[0].key,
            ok=protocoldagresults_failure[0].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref)

        # try get results back
        pdr_refs = n4js.get_task_results(task_sk)

        assert len(pdr_refs) == 0
        assert pdr_ref not in pdr_refs

        # try to get failure back
        failure_pdr_refs = n4js.get_task_failures(task_sk)

        assert len(failure_pdr_refs) == 1
        assert pdr_ref in failure_pdr_refs

        # try doing it again; should be idempotent
        n4js.set_task_result(task_sk, pdr_ref)
        failure_pdr_refs = n4js.get_task_failures(task_sk)

        assert len(failure_pdr_refs) == 1
        assert pdr_ref in failure_pdr_refs

        # if we add a different failure, should now have 2
        pdr_ref2 = ProtocolDAGResultRef(
            scope=task_sk.scope,
            obj_key=protocoldagresults_failure[1].key,
            ok=protocoldagresults_failure[1].ok(),
        )

        # push the result
        n4js.set_task_result(task_sk, pdr_ref2)

        # get the result back
        failure_pdr_refs2 = n4js.get_task_failures(task_sk)

        assert len(failure_pdr_refs2) == 2
        assert pdr_ref in failure_pdr_refs2
        assert pdr_ref2 in failure_pdr_refs2

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
    @pytest.mark.parametrize(
        "scope_strs", (["*-*-*"], ["a-*-*"], ["a-b-*"], ["a-b-c", "a-b-d"])
    )
    def test_list_scope(
        self,
        n4js: Neo4jStore,
        credential_type: CredentialedEntity,
        scope_strs: List[str],
    ):
        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)
        ref_scopes = []
        for scope_str in scope_strs:
            scope = Scope.from_str(scope_str)
            ref_scopes.append(scope)
            n4js.add_scope(user.identifier, credential_type, scope)

        scopes = n4js.list_scopes(user.identifier, credential_type)
        assert set(scopes) == set(ref_scopes)

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
        RETURN n
        """
        scopes_qr = n4js.graph.run(q).to_subgraph()
        scopes = scopes_qr.get("scopes")
        assert len(scopes) == 1

        new_scope = Scope.from_str(scopes[0])
        assert new_scope == scope

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    def test_add_scope_duplicate(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity
    ):
        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        scope1 = Scope.from_str("*-*-*")
        scope2 = Scope.from_str("*-*-*")

        n4js.add_scope(user.identifier, credential_type, scope1)
        n4js.add_scope(user.identifier, credential_type, scope2)

        q = f"""
        MATCH (n:{credential_type.__name__} {{identifier: '{user.identifier}'}})
        RETURN n
        """
        scopes_qr = n4js.graph.run(q).to_subgraph()
        scopes = scopes_qr.get("scopes")
        assert len(scopes) == 1

        new_scope = Scope.from_str(scopes[0])
        assert new_scope == scope1

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    @pytest.mark.parametrize("scope_str", ("*-*-*", "a-*-*", "a-b-*", "a-b-c"))
    def test_remove_scope(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity, scope_str: str
    ):
        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        scope = Scope.from_str(scope_str)
        not_removed = Scope.from_str("scope-not-removed")

        n4js.add_scope(user.identifier, credential_type, scope)
        n4js.add_scope(user.identifier, credential_type, not_removed)

        n4js.remove_scope(user.identifier, credential_type, scope)

        scopes = n4js.list_scopes(user.identifier, credential_type)
        assert scope not in scopes
        assert not_removed in scopes

    @pytest.mark.parametrize(
        "credential_type", [CredentialedUserIdentity, CredentialedComputeIdentity]
    )
    @pytest.mark.parametrize("scope_str", ("*-*-*", "a-*-*", "a-b-*", "a-b-c"))
    def test_remove_scope_duplicate(
        self, n4js: Neo4jStore, credential_type: CredentialedEntity, scope_str: str
    ):
        user = credential_type(
            identifier="bill",
            hashed_key=hash_key("and ted"),
        )

        n4js.create_credentialed_entity(user)

        scope = Scope.from_str(scope_str)
        not_removed = Scope.from_str("scope-not-removed")

        n4js.add_scope(user.identifier, credential_type, scope)
        n4js.add_scope(user.identifier, credential_type, not_removed)

        n4js.remove_scope(user.identifier, credential_type, scope)
        n4js.remove_scope(user.identifier, credential_type, scope)

        scopes = n4js.list_scopes(user.identifier, credential_type)
        assert scope not in scopes
        assert not_removed in scopes
