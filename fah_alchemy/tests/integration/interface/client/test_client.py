import pytest
from time import sleep

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
import networkx as nx

from fah_alchemy.models import ScopedKey
from fah_alchemy.interface import client

from fah_alchemy.tests.integration.interface.utils import get_user_settings_override


class TestClient:
    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        user_client_wrong_credential: client.FahAlchemyClient,
    ):
        with pytest.raises(client.FahAlchemyClientError):
            user_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
    ):

        settings = get_user_settings_override()
        assert user_client._jwtoken == None
        user_client._get_token()

        token = user_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        user_client.get_info()
        assert token == user_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        user_client.get_info()
        assert token != user_client._jwtoken

    ### inputs

    def test_create_network(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        network_tyk2,
    ):
        # make a smaller network that overlaps with an existing one in DB
        an = AlchemicalNetwork(edges=list(network_tyk2.edges)[4:-2], name="smaller")
        an_sk = user_client.create_network(an, scope_test)

        network_sks = user_client.query_networks()
        assert an_sk in network_sks

        # TODO: make a network in a scope that doesn't have any components in
        # common with an existing network
        # user_client.create_network(

    def test_query_networks(self):
        ...

    def test_get_network(self):
        ...

    def test_get_transformation(self):
        ...

    def test_get_chemicalsystem(self):
        ...

    ### compute

    def test_create_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        network_tyk2,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]
        sk = user_client.get_scoped_key(transformation, scope_test)

        task_sks = user_client.create_tasks(sk, count=3)

        assert set(task_sks) == set(n4js.get_tasks(sk))

        # try creating tasks that extend one of those we just created
        task_sks_e = user_client.create_tasks(sk, count=4, extends=task_sks[0])

        # check that we now get additional tasks
        assert set(task_sks + task_sks_e) == set(n4js.get_tasks(sk))

        # check that tasks are structured as we expect
        assert set(task_sks_e) == set(n4js.get_tasks(sk, extends=task_sks[0]))
        assert set() == set(n4js.get_tasks(sk, extends=task_sks[1]))
        

    def test_get_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        network_tyk2,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]
        sk = user_client.get_scoped_key(transformation, scope_test)

        task_sks = user_client.create_tasks(sk, count=3)

        assert set(task_sks) == set(user_client.get_tasks(sk))

        # try creating tasks that extend one of those we just created
        task_sks_e = user_client.create_tasks(sk, count=4, extends=task_sks[0])

        # check that we now get additional tasks
        assert set(task_sks + task_sks_e) == set(user_client.get_tasks(sk))

        # check that tasks are structured as we expect
        assert set(task_sks_e) == set(user_client.get_tasks(sk, extends=task_sks[0]))
        assert set() == set(user_client.get_tasks(sk, extends=task_sks[1]))

        # check graph form of output
        graph: nx.DiGraph = user_client.get_tasks(sk, return_as='graph')

        for task_sk in task_sks:
            assert len(list(graph.successors(task_sk))) == 0

        for task_sk in task_sks_e:
            assert graph.has_edge(task_sk, task_sks[0])

    def test_action_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.FahAlchemyClient,
        network_tyk2,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        task_sks = user_client.create_tasks(transformation_sk, count=3)

        # action these task for this network, in reverse order
        actioned_sks = user_client.action_tasks(task_sks[::-1], network_sk)

        # check that the taskqueue looks as we expect
        taskqueue_sk = n4js.get_taskqueue(network_sk)
        queued_sks = n4js.get_taskqueue_tasks(taskqueue_sk)

        assert actioned_sks == queued_sks
        assert actioned_sks == task_sks[::-1]

        # create extending tasks; try to action one of them
        # this should yield `None` in results, since it shouldn't be possible to action these tasks
        # if they extend a task that isn't 'complete'
        task_sks_e = user_client.create_tasks(transformation_sk, count=4, extends=task_sks[0])
        actioned_sks_e = user_client.action_tasks(task_sks_e, network_sk)

        assert all([i is None for i in actioned_sks_e])

    ### results

    def test_get_transformation_result(
        self,
        scope_test,
        n4js_preloaded,
        s3os,
        user_client: client.FahAlchemyClient,
        network_tyk2,
    ):

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]

        # user client : create a tree of tasks for the transformation

        # user client : action the tasks for execution

        # execute the tasks and push results directly using statestore and object store

        # user client : pull transformation results, evaluate
