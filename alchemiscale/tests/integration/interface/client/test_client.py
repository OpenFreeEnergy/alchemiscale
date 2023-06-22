import pytest
from time import sleep
from pathlib import Path

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.tokenization import TOKENIZABLE_REGISTRY, GufeKey
from gufe.protocols.protocoldag import execute_DAG
from gufe.tests.test_protocol import BrokenProtocol
import networkx as nx

from alchemiscale.models import ScopedKey, Scope
from alchemiscale.storage.models import TaskStatusEnum
from alchemiscale.interface import client
from alchemiscale.tests.integration.interface.utils import get_user_settings_override
from alchemiscale.interface.client import AlchemiscaleClientError


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

    def test_api_check(
        self,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
    ):
        user_client._api_check()

    def test_list_scopes(
        self,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
        multiple_scopes,
    ):
        scopes = user_client.list_scopes()
        # multiple scopes matches identity used to initialise the client in conftest
        assert set(scopes) == set(multiple_scopes)

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

    def test_query_networks(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        network_sks = user_client.query_networks()

        assert len(network_sks) == 6
        assert scope_test in [n_sk.scope for n_sk in network_sks]

        assert len(user_client.query_networks(scope=scope_test)) == 2
        assert len(user_client.query_networks(name=network_tyk2.name)) == 3

    def test_query_transformations(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        transformation_sks = user_client.query_transformations()

        assert len(transformation_sks) == len(network_tyk2.edges) * 3
        assert len(user_client.query_transformations(scope=scope_test)) == len(
            network_tyk2.edges
        )
        assert (
            len(
                user_client.query_transformations(
                    name="lig_ejm_31_to_lig_ejm_50_complex"
                )
            )
            == 3
        )
        assert (
            len(
                user_client.query_transformations(
                    scope=scope_test, name="lig_ejm_31_to_lig_ejm_50_complex"
                )
            )
            == 1
        )

    def test_query_chemicalsystems(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        chemicalsystem_sks = user_client.query_chemicalsystems()

        assert len(chemicalsystem_sks) == len(network_tyk2.nodes) * 3
        assert len(user_client.query_chemicalsystems(scope=scope_test)) == len(
            network_tyk2.nodes
        )
        assert len(user_client.query_chemicalsystems(name="lig_ejm_31_complex")) == 3
        assert (
            len(
                user_client.query_chemicalsystems(
                    scope=scope_test, name="lig_ejm_31_complex"
                )
            )
            == 1
        )

    def test_get_network_transformations(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        n_sk = user_client.get_scoped_key(network_tyk2, scope_test)
        tf_sks = user_client.get_network_transformations(n_sk)

        assert len(tf_sks) == len(network_tyk2.edges)
        assert set(tf_sk.gufe_key for tf_sk in tf_sks) == set(
            t.key for t in network_tyk2.edges
        )

    def test_get_transformation_networks(
        self,
        scope_test,
        n4js_preloaded,
        transformation,
        user_client: client.AlchemiscaleClient,
    ):
        tf_sk = user_client.get_scoped_key(transformation, scope_test)
        n_sks = user_client.get_transformation_networks(tf_sk)

        assert len(n_sks) == 2

    def test_get_network_chemicalsystems(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        n_sk = user_client.get_scoped_key(network_tyk2, scope_test)
        cs_sks = user_client.get_network_chemicalsystems(n_sk)

        assert len(cs_sks) == len(network_tyk2.nodes)
        assert set(cs_sk.gufe_key for cs_sk in cs_sks) == set(
            cs.key for cs in network_tyk2.nodes
        )

    def test_get_chemicalsystem_networks(
        self,
        scope_test,
        n4js_preloaded,
        chemicalsystem,
        user_client: client.AlchemiscaleClient,
    ):
        cs_sk = user_client.get_scoped_key(chemicalsystem, scope_test)
        n_sks = user_client.get_chemicalsystem_networks(cs_sk)

        assert len(n_sks) == 2

    def test_get_transformation_chemicalsystems(
        self,
        scope_test,
        n4js_preloaded,
        transformation,
        user_client: client.AlchemiscaleClient,
    ):
        tf_sk = user_client.get_scoped_key(transformation, scope_test)
        cs_sks = user_client.get_transformation_chemicalsystems(tf_sk)

        assert len(cs_sks) == 2
        assert set(cs_sk.gufe_key for cs_sk in cs_sks) == set(
            [transformation.stateA.key, transformation.stateB.key]
        )

    def test_get_chemicalsystem_transformations(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        chemicalsystem,
        user_client: client.AlchemiscaleClient,
    ):
        cs_sk = user_client.get_scoped_key(chemicalsystem, scope_test)
        tf_sks = user_client.get_chemicalsystem_transformations(cs_sk)

        tfs = []
        for tf in network_tyk2.edges:
            if chemicalsystem in (tf.stateA, tf.stateB):
                tfs.append(tf)

        assert set(tf_sk.gufe_key for tf_sk in tf_sks) == set(t.key for t in tfs)

    def test_get_network(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        an_sk = user_client.get_scoped_key(network_tyk2, scope_test)
        an = user_client.get_network(an_sk)

        assert an == network_tyk2
        assert an is network_tyk2

    def test_get_transformation(
        self,
        scope_test,
        n4js_preloaded,
        transformation,
        user_client: client.AlchemiscaleClient,
    ):
        tf_sk = user_client.get_scoped_key(transformation, scope_test)
        tf = user_client.get_transformation(tf_sk)

        assert tf == transformation
        assert tf is transformation

    def test_get_chemicalsystem(
        self,
        scope_test,
        n4js_preloaded,
        chemicalsystem,
        user_client: client.AlchemiscaleClient,
    ):
        cs_sk = user_client.get_scoped_key(chemicalsystem, scope_test)
        cs = user_client.get_chemicalsystem(cs_sk)

        assert cs == chemicalsystem
        assert cs is chemicalsystem

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

        assert set(task_sks) == set(n4js.get_transformation_tasks(sk))

        # try creating tasks that extend one of those we just created
        task_sks_e = user_client.create_tasks(sk, count=4, extends=task_sks[0])

        # check that we now get additional tasks
        assert set(task_sks + task_sks_e) == set(n4js.get_transformation_tasks(sk))

        # check that tasks are structured as we expect
        assert set(task_sks_e) == set(
            n4js.get_transformation_tasks(sk, extends=task_sks[0])
        )
        assert set() == set(n4js.get_transformation_tasks(sk, extends=task_sks[1]))

    def test_query_tasks(
        self,
        scope_test,
        multiple_scopes,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
    ):
        task_sks = user_client.query_tasks()
        assert len(task_sks) == 0

        tf_sks = user_client.query_transformations(scope=scope_test)

        for tf_sk in tf_sks[:10]:
            user_client.create_tasks(tf_sk, count=3)

        task_sks = user_client.query_tasks()
        assert len(task_sks) == 10 * 3

        task_sks = user_client.query_tasks(scope=scope_test)
        assert len(task_sks) == 10 * 3

        task_sks = user_client.query_tasks(scope=multiple_scopes[1])
        assert len(task_sks) == 0

        # check that we can query by status
        task_sks = user_client.query_tasks()
        user_client.set_tasks_status(task_sks[:10], "invalid")

        task_sks = user_client.query_tasks(status="waiting")
        assert len(task_sks) == 10 * 3 - 10

        task_sks = user_client.query_tasks(status="invalid")
        assert len(task_sks) == 10

        task_sks = user_client.query_tasks(status="complete")
        assert len(task_sks) == 0

    def test_get_network_tasks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
    ):
        an_sk = user_client.query_networks(scope=scope_test)[0]
        tf_sks = user_client.get_network_transformations(an_sk)

        task_sks = []
        for tf_sk in tf_sks[:10]:
            task_sks.extend(user_client.create_tasks(tf_sk, count=3))

        task_sks_network = user_client.get_network_tasks(an_sk)
        assert set(task_sks_network) == set(task_sks)
        assert len(task_sks_network) == len(task_sks)

        user_client.set_tasks_status(task_sks[:10], "invalid")

        task_sks = user_client.get_network_tasks(an_sk, status="waiting")
        assert len(task_sks) == len(task_sks_network) - 10

        task_sks = user_client.get_network_tasks(an_sk, status="invalid")
        assert len(task_sks) == 10

        task_sks = user_client.get_network_tasks(an_sk, status="complete")
        assert len(task_sks) == 0

    def test_get_task_networks(
        self,
        scope_test,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
    ):
        an_sk = user_client.query_networks(scope=scope_test)[0]
        tf_sks = user_client.get_network_transformations(an_sk)

        task_sks = []
        for tf_sk in tf_sks[:10]:
            task_sks.extend(user_client.create_tasks(tf_sk, count=3))

        for task_sk in task_sks:
            an_sks = user_client.get_task_networks(task_sk)
            assert an_sk in an_sks
            assert len(an_sks) == 2

    def test_get_transformation_tasks(
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

        assert set(task_sks) == set(user_client.get_transformation_tasks(sk))

        # try creating tasks that extend one of those we just created
        task_sks_e = user_client.create_tasks(sk, count=4, extends=task_sks[0])

        # check that we now get additional tasks
        assert set(task_sks + task_sks_e) == set(
            user_client.get_transformation_tasks(sk)
        )

        # check that tasks are structured as we expect
        assert set(task_sks_e) == set(
            user_client.get_transformation_tasks(sk, extends=task_sks[0])
        )
        assert set() == set(
            user_client.get_transformation_tasks(sk, extends=task_sks[1])
        )

        # check graph form of output
        graph: nx.DiGraph = user_client.get_transformation_tasks(sk, return_as="graph")

        for task_sk in task_sks:
            assert len(list(graph.successors(task_sk))) == 0

        for task_sk in task_sks_e:
            assert graph.has_edge(task_sk, task_sks[0])

        # try filtering on status
        # check that tasks are structured as we expect
        user_client.set_tasks_status(task_sks_e[:2], "invalid")
        assert set(task_sks_e[2:]) == set(
            user_client.get_transformation_tasks(
                sk, extends=task_sks[0], status="waiting"
            )
        )
        assert set(task_sks_e[:2]) == set(
            user_client.get_transformation_tasks(
                sk, extends=task_sks[0], status="invalid"
            )
        )
        assert set(task_sks_e) == set(
            user_client.get_transformation_tasks(sk, extends=task_sks[0])
        )

    def test_get_task_transformation(
        self,
        n4js_preloaded,
        user_client: client.AlchemiscaleClient,
    ):
        tf_sks = user_client.query_transformations()

        task_sks = []
        for tf_sk in tf_sks[:10]:
            task_sks.append(user_client.create_tasks(tf_sk, count=3))

        for tf_sk, tf_task_sks in zip(tf_sks, task_sks):
            for task_sk in tf_task_sks:
                assert user_client.get_task_transformation(task_sk) == tf_sk

    def test_get_scope_status(
        self,
        multiple_scopes,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
    ):
        # for each of the following scopes, create tasks for a single random
        # transformation
        for scope in multiple_scopes:
            tf_sks = user_client.query_transformations(scope=scope)
            user_client.create_tasks(tf_sks[0], count=3)

        # now, get status for each scope; check that we are filtering down
        # properly
        for scope in multiple_scopes:
            status_counts = user_client.get_scope_status(scope)

            for status in status_counts:
                if status == "waiting":
                    assert status_counts[status] == 3
                else:
                    assert status_counts[status] == 0

        # create tasks in a scope we don't have access to
        other_scope = Scope("other_org", "other_campaign", "other_project")
        n4js_preloaded.create_network(network_tyk2, other_scope)
        other_tf_sk = n4js_preloaded.query_transformations(scope=other_scope)[0]
        task_sk = n4js_preloaded.create_task(other_tf_sk)

        # ask for the scope that we don't have access to
        status_counts = user_client.get_scope_status(other_scope)
        assert status_counts == {}

        # try a more general scope
        status_counts = user_client.get_scope_status(Scope())
        for status in status_counts:
            if status == "waiting":
                assert status_counts[status] == 3 * len(multiple_scopes)
            else:
                assert status_counts[status] == 0

    def test_get_network_status(
        self,
        n4js_preloaded,
        multiple_scopes,
        user_client: client.AlchemiscaleClient,
    ):
        # for each of the following scopes, get one of the networks present,
        # create tasks for a single random transformation
        an_sks = []
        for scope in multiple_scopes:
            an_sk = user_client.query_networks(scope=scope)[0]
            tf_sks = user_client.get_network_transformations(an_sk)
            user_client.create_tasks(tf_sks[0], count=3)

            an_sks.append(an_sk)

        # now, get status for each network
        for an_sk in an_sks:
            status_counts = user_client.get_network_status(an_sk)

            for status in status_counts:
                if status == "waiting":
                    assert status_counts[status] == 3
                else:
                    assert status_counts[status] == 0

    def test_get_transformation_status(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
    ):
        network_sk = user_client.get_scoped_key(network_tyk2, scope_test)
        transformation_sk = user_client.get_network_transformations(network_sk)[0]

        all_tasks = user_client.create_tasks(transformation_sk, count=5)

        # check the status of the tasks; should all be waiting
        stat = user_client.get_transformation_status(transformation_sk)
        assert stat == {"waiting": 5}

        # cheat and set the status of all tasks to running
        ret_task = n4js_preloaded.set_task_status(all_tasks, TaskStatusEnum.running)
        assert set(ret_task) == set(all_tasks)
        stat = user_client.get_transformation_status(transformation_sk)
        assert stat == {"running": 5}

        # cheat and set the status of all tasks to complete
        ret_task = n4js_preloaded.set_task_status(all_tasks, TaskStatusEnum.complete)
        assert set(ret_task) == set(all_tasks)
        stat = user_client.get_transformation_status(transformation_sk)
        assert stat == {"complete": 5}

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

        # check that the taskhub looks as we expect
        taskhub_sk = n4js.get_taskhub(network_sk)
        hub_task_sks = n4js.get_taskhub_tasks(taskhub_sk)

        assert set(actioned_sks) == set(hub_task_sks)
        assert actioned_sks == task_sks[::-1]

        # create extending tasks; these should be actioned to the
        # taskhub but not claimable
        task_sks_e = user_client.create_tasks(
            transformation_sk, count=4, extends=task_sks[0]
        )
        actioned_sks_e = user_client.action_tasks(task_sks_e, network_sk)

        assert set(task_sks_e) == set(actioned_sks_e)

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

        # check that the taskhub looks as we expect
        taskhub_sk = n4js.get_taskhub(network_sk)
        hub_task_sks = n4js.get_taskhub_tasks(taskhub_sk)

        assert set([actioned_sks[0], actioned_sks[2]]) == set(hub_task_sks)
        assert canceled_sks == [actioned_sks[1]]

        # try to cancel a task that's not present on the hub
        canceled_sks_2 = user_client.cancel_tasks(task_sks[1:2], network_sk)

        assert canceled_sks_2 == [None]

    @pytest.mark.parametrize(
        "status, should_raise",
        [
            (TaskStatusEnum.waiting, False),
            (TaskStatusEnum.running, True),
            (TaskStatusEnum.complete, True),
            (TaskStatusEnum.error, True),
            (TaskStatusEnum.invalid, False),
            (TaskStatusEnum.deleted, False),
        ],
    )
    def test_set_tasks_status(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
        status,
        should_raise,
    ):
        an = network_tyk2
        transformation = list(an.edges)[0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        all_tasks = user_client.create_tasks(transformation_sk, count=5)

        if should_raise:
            with pytest.raises(AlchemiscaleClientError):
                user_client.set_tasks_status(all_tasks, status)
        else:
            # set the status of a task
            user_client.set_tasks_status(all_tasks, status)

            # check that the status has been set
            # note must be list on n4js side
            statuses = n4js_preloaded.get_task_status(all_tasks)
            assert all([s == status for s in statuses])

    @pytest.mark.parametrize(
        "status, should_raise",
        [
            (TaskStatusEnum.waiting, False),
            (TaskStatusEnum.running, True),
            (TaskStatusEnum.complete, True),
            (TaskStatusEnum.error, True),
            (TaskStatusEnum.invalid, False),
            (TaskStatusEnum.deleted, False),
        ],
    )
    def test_get_tasks_status(
        self,
        scope_test,
        n4js_preloaded,
        network_tyk2,
        user_client: client.AlchemiscaleClient,
        uvicorn_server,
        status,
        should_raise,
    ):
        an = network_tyk2
        transformation = list(an.edges)[0]

        network_sk = user_client.get_scoped_key(an, scope_test)
        transformation_sk = user_client.get_scoped_key(transformation, scope_test)

        all_tasks = user_client.create_tasks(transformation_sk, count=5)

        # set the status of a task
        if should_raise:
            with pytest.raises(AlchemiscaleClientError):
                user_client.set_tasks_status(all_tasks, status)
        else:
            user_client.set_tasks_status(all_tasks, status)

            # check that the status has been set
            statuses = user_client.get_tasks_status(all_tasks)
            assert all([s == status.value for s in statuses])

    def test_get_tasks_priority(self):
        ...

    def test_set_tasks_priority(self):
        ...

    ### results

    @staticmethod
    def _execute_tasks(tasks, n4js, s3os_server):
        shared_basedir = Path("shared").absolute()
        shared_basedir.mkdir()
        scratch_basedir = Path("scratch").absolute()
        scratch_basedir.mkdir()

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

            shared = shared_basedir / str(protocoldag.key)
            shared.mkdir()

            scratch = scratch_basedir / str(protocoldag.key)
            scratch.mkdir()

            protocoldagresult = execute_DAG(
                protocoldag,
                shared_basedir=shared,
                scratch_basedir=scratch,
                raise_error=False,
            )

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

        # user client : create three independent tasks for the transformation
        tasks = user_client.create_tasks(transformation_sk, count=3)

        # user client : action the tasks for execution
        all_tasks = user_client.get_transformation_tasks(transformation_sk)
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
        assert protocolresult is None

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
