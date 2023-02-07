import pytest
from time import sleep

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.tokenization import TOKENIZABLE_REGISTRY, GufeKey
from gufe.protocols.protocoldag import execute_DAG
from gufe.tests.test_protocol import BrokenProtocol
import networkx as nx

from alchemiscale.models import ScopedKey
from alchemiscale.interface import client

from alchemiscale.tests.integration.interface.utils import get_user_settings_override


class TestClient:
    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        user_client_wrong_credential: client.AlchemiscaleClient,
        uvicorn_server,
    ):
        with pytest.raises(client.AlchemiscaleClientError):
            user_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
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

    def test_create_network(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
    ):
        ...

    def test_api_check(
        self,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
    ):
        user_client._api_check()

    ### inputs

    def test_create_network(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
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
        user_client: client.AlchemiscaleClient,
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
        user_client: client.AlchemiscaleClient,
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
        graph: nx.DiGraph = user_client.get_tasks(sk, return_as="graph")

        for task_sk in task_sks:
            assert len(list(graph.successors(task_sk))) == 0

        for task_sk in task_sks_e:
            assert graph.has_edge(task_sk, task_sks[0])

    def test_action_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
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
        task_sks_e = user_client.create_tasks(
            transformation_sk, count=4, extends=task_sks[0]
        )
        actioned_sks_e = user_client.action_tasks(task_sks_e, network_sk)

        assert all([i is None for i in actioned_sks_e])

    def test_cancel_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
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

        # try canceling one of these tasks
        canceled_sks = user_client.cancel_tasks(task_sks[1:2], network_sk)

        # check that the taskqueue looks as we expect
        taskqueue_sk = n4js.get_taskqueue(network_sk)
        queued_sks = n4js.get_taskqueue_tasks(taskqueue_sk)

        assert [actioned_sks[0], actioned_sks[2]] == queued_sks
        assert canceled_sks == [actioned_sks[1]]

        # try to cancel a task that's not present in the queue
        canceled_sks_2 = user_client.cancel_tasks(task_sks[1:2], network_sk)

        assert canceled_sks_2 == [None]

    ### results

    @staticmethod
    def _execute_tasks(tasks, n4js, s3os_server):
        protocoldagresults = []
        for task_sk in tasks:
            if task_sk is None:
                continue

            # get the transformation and extending protocoldagresult as if we
            # were a compute service
            transformation, extends_protocoldagresult = n4js.get_task_transformation(
                task=task_sk
            )

            protocoldag = transformation.create(
                extends=extends_protocoldagresult,
                name=str(task_sk),
            )

            protocoldagresult = execute_DAG(protocoldag, raise_error=False)

            assert protocoldagresult.transformation_key == transformation.key
            if extends_protocoldagresult:
                assert protocoldagresult.extends_key == extends_protocoldagresult.key

            protocoldagresults.append(protocoldagresult)

            protocoldagresultref = s3os_server.push_protocoldagresult(
                protocoldagresult, scope=task_sk.scope
            )

            n4js.set_task_result(
                task=task_sk, protocoldagresultref=protocoldagresultref
            )

        return protocoldagresults

    def test_get_transformation_results(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server,
        user_client: client.AlchemiscaleClient,
        network_tyk2,
        tmpdir,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        # user client : create a tree of tasks for the transformation
        tasks = user_client.create_tasks(transformation_sk, count=3)
        for task in tasks:
            tasks_2 = user_client.create_tasks(transformation_sk, extends=task, count=3)
            for task2 in tasks_2:
                user_client.create_tasks(transformation_sk, extends=task2, count=3)

        # user client : action the tasks for execution
        all_tasks = user_client.get_tasks(transformation_sk)
        # all_tasks = reversed(list(nx.topological_sort(all_tasks_g)))

        # only tasks that do not extend an incomplete task are actioned
        actioned_tasks = user_client.action_tasks(all_tasks, network_sk)

        # execute the actioned tasks and push results directly using statestore and object store
        with tmpdir.as_cwd():
            protocoldagresults = self._execute_tasks(actioned_tasks, n4js, s3os_server)

        # clear local gufe registry of pdr objects
        # not critical, but ensures we see the objects that are deserialized
        # instead of our instances already in memory post-pull
        for pdr in protocoldagresults:
            TOKENIZABLE_REGISTRY.pop(pdr.key, None)

        # user client : pull transformation results, evaluate
        protocolresult = user_client.get_transformation_results(transformation_sk)

        assert protocolresult.get_estimate() == 95500.0
        assert set(protocolresult.data.keys()) == {"logs", "key_results"}
        assert len(protocolresult.data["key_results"]) == 3

        # get back protocoldagresults instead
        protocoldagresults_r = user_client.get_transformation_results(
            transformation_sk, return_protocoldagresults=True
        )

        assert set(protocoldagresults_r) == set(protocoldagresults)

        for pdr in protocoldagresults_r:
            assert pdr.transformation_key == transformation.key
            assert isinstance(pdr.extends_key, GufeKey) or pdr.extends_key is None
            assert pdr.ok()

    def test_get_transformation_failures(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server,
        user_client: client.AlchemiscaleClient,
        network_tyk2_failure,
        tmpdir,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2_failure
        user_client.create_network(an, scope_test)
        transformation = [
            t for t in list(an.edges) if isinstance(t.protocol, BrokenProtocol)
        ][0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        # user client : create tasks for the transformation
        tasks = user_client.create_tasks(transformation_sk, count=2)

        # user client : action the tasks for execution
        actioned_tasks = user_client.action_tasks(tasks, network_sk)

        # execute the actioned tasks and push results directly using statestore and object store
        with tmpdir.as_cwd():
            protocoldagresults = self._execute_tasks(actioned_tasks, n4js, s3os_server)

        # clear local gufe registry of pdr objects
        # not critical, but ensures we see the objects that are deserialized
        # instead of our instances already in memory post-pull
        for pdr in protocoldagresults:
            TOKENIZABLE_REGISTRY.pop(pdr.key, None)

        # user client : attempt to pull transformation results; should yield nothing
        protocolresult = user_client.get_transformation_results(transformation_sk)

        # with the way BrokenProtocol.gather constructs its output, we expect
        # this to be empty
        assert len(protocolresult.data) == 0

        # user client : instead, pull failures
        protocoldagresults_r = user_client.get_transformation_failures(
            transformation_sk
        )

        assert set(protocoldagresults_r) == set(protocoldagresults)
        assert len(protocoldagresults_r) == 2

        for pdr in protocoldagresults_r:
            assert pdr.transformation_key == transformation.key
            assert isinstance(pdr.extends_key, GufeKey) or pdr.extends_key is None
            assert not pdr.ok()

    def test_get_task_results(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server,
        user_client: client.AlchemiscaleClient,
        network_tyk2,
        tmpdir,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2
        transformation = list(an.edges)[0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        # user client : create tasks for the transformation
        tasks = user_client.create_tasks(transformation_sk, count=2)

        # only tasks that do not extend an incomplete task are actioned
        actioned_tasks = user_client.action_tasks(tasks, network_sk)

        # execute the actioned tasks and push results directly using statestore and object store
        with tmpdir.as_cwd():
            protocoldagresults = self._execute_tasks(actioned_tasks, n4js, s3os_server)

        # clear local gufe registry of pdr objects
        # not critical, but ensures we see the objects that are deserialized
        # instead of our instances already in memory post-pull
        for pdr in protocoldagresults:
            TOKENIZABLE_REGISTRY.pop(pdr.key, None)

        # user client : pull task results, evaluate
        for task in tasks:
            protocoldagresults_r = user_client.get_task_results(task)

            assert len(protocoldagresults_r) == 1
            assert set(protocoldagresults_r).issubset(set(protocoldagresults))

            for pdr in protocoldagresults_r:
                assert pdr.transformation_key == transformation.key
                assert isinstance(pdr.extends_key, GufeKey) or pdr.extends_key is None
                assert pdr.ok()

    def test_get_task_failures(
        self,
        scope_test,
        n4js_preloaded,
        s3os_server,
        user_client: client.AlchemiscaleClient,
        network_tyk2_failure,
        tmpdir,
    ):
        n4js = n4js_preloaded

        # select the transformation we want to compute
        an = network_tyk2_failure
        user_client.create_network(an, scope_test)
        transformation = [
            t for t in list(an.edges) if isinstance(t.protocol, BrokenProtocol)
        ][0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        # user client : create tasks for the transformation
        tasks = user_client.create_tasks(transformation_sk, count=2)

        # only tasks that do not extend an incomplete task are actioned
        actioned_tasks = user_client.action_tasks(tasks, network_sk)

        # execute the actioned tasks and push results directly using statestore and object store
        with tmpdir.as_cwd():
            protocoldagresults = self._execute_tasks(actioned_tasks, n4js, s3os_server)

        # clear local gufe registry of pdr objects
        # not critical, but ensures we see the objects that are deserialized
        # instead of our instances already in memory post-pull
        for pdr in protocoldagresults:
            TOKENIZABLE_REGISTRY.pop(pdr.key, None)

        # user client : attempt to pull task results; should yield nothing
        for task in tasks:
            protocoldagresults_r = user_client.get_task_results(task)

            assert len(protocoldagresults_r) == 0

        for task in tasks:
            protocoldagresults_r = user_client.get_task_failures(task)

            assert len(protocoldagresults_r) == 1
            assert set(protocoldagresults_r).issubset(set(protocoldagresults))

            for pdr in protocoldagresults_r:
                assert pdr.transformation_key == transformation.key
                assert isinstance(pdr.extends_key, GufeKey) or pdr.extends_key is None
                assert not pdr.ok()

        # TODO: can we mix in a success in here somewhere?
        # not possible with current BrokenProtocol, unfortunately
