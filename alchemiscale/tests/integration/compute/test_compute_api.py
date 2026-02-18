import json

import pytest

from gufe import Transformation

from alchemiscale.base.client import json_to_gufe
from alchemiscale.models import Scope, ScopedKey
from alchemiscale.storage.models import ComputeServiceID


class TestComputeAPI:
    def test_info(self, test_client):
        response = test_client.get("/info")
        assert response.status_code == 200

    def test_check(self, test_client):
        response = test_client.get("/check")
        assert response.status_code == 200

    def test_scopes(
        self, n4js_preloaded, test_client, fully_scoped_credentialed_compute
    ):
        response = test_client.get(
            f"/identities/{fully_scoped_credentialed_compute.identifier}/scopes"
        )
        assert response.status_code == 200
        scopes = response.json()
        assert scopes == fully_scoped_credentialed_compute.scopes

    def test_query_taskhubs(self, n4js_preloaded, test_client):
        response = test_client.get("/taskhubs")
        assert response.status_code == 200

        tq_sks = [ScopedKey.from_str(i) for i in response.json()]
        assert len(tq_sks) == 2

        # try getting back actual gufe objects
        response = test_client.get("/taskhubs?return_gufe=True")
        assert response.status_code == 200

        tq_dict = {
            ScopedKey.from_str(k): json_to_gufe(v) for k, v in response.json().items()
        }
        assert len(tq_dict) == 2
        assert all([i.weight == 0.5 for i in tq_dict.values()])

    @pytest.fixture
    def scoped_keys(self, n4js_preloaded, network_tyk2, scope_test):
        n4js = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, scope_test)
        tq_sk = n4js.get_taskhub(network_sk)
        task_sks = n4js.get_taskhub_tasks(tq_sk)
        assert len(task_sks) > 0
        return {"network": network_sk, "taskhub": tq_sk, "tasks": task_sks}

    @pytest.fixture
    def out_of_scoped_keys(self, n4js_preloaded, network_tyk2, multiple_scopes):
        n4js = n4js_preloaded
        network_sk = n4js.get_scoped_key(network_tyk2, multiple_scopes[-1])
        tq_sk = n4js.get_taskhub(network_sk)
        task_sks = n4js.get_taskhub_tasks(tq_sk)
        assert len(task_sks) > 0
        return {"network": network_sk, "taskhub": tq_sk, "tasks": task_sks}

    def test_retrieve_task_transformation(
        self,
        n4js_preloaded,
        test_client,
        scoped_keys,
    ):
        response = test_client.get(
            f"/tasks/{scoped_keys['tasks'][0]}/transformation/gufe"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        transformation = json_to_gufe(data[0])

        assert isinstance(transformation, Transformation)

    def test_get_task_transformation_bad_scope(
        self,
        n4js_preloaded,
        test_client,
        out_of_scoped_keys,
    ):
        response = test_client.get(
            f"/tasks/{out_of_scoped_keys['tasks'][0]}/transformation"
        )
        assert response.status_code == 401

    # def test_task_result(self, n4js_preloaded, test_client, protocoldagresult):

    #    json.dumps(protocoldagresult.to_dict()

    #    test_client.post("/tasks/{task}/result")

    #    # try to push the result
    #    objstoreref: ObjectStoreRef = s3os.push_protocoldagresult(protocoldagresult)

    #    assert objstoreref.location == os.path.join(
    #        "protocoldagresult", protocoldagresult.key
    #    )

    #    # examine object metadata
    #    objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())

    #    assert len(objs) == 1
    #    assert objs[0].key == os.path.join(s3os.prefix, objstoreref.location)

    def test_claim_tasks_with_scopes_exclude(
        self,
        n4js_preloaded,
        multi_scope_test_client,
        multiple_scopes,
        scope_test,
        compute_service_id,
    ):
        """Test that scopes_exclude filters out taskhubs matching excluded scopes."""
        # register a compute service
        multi_scope_test_client.post(
            f"/computeservice/{compute_service_id}/register",
            content=json.dumps({"compute_manager_id": None}),
        )

        # claim tasks with all scopes, no exclusions
        data = dict(
            scopes=[s.to_dict() for s in multiple_scopes],
            scopes_exclude=None,
            compute_service_id=str(compute_service_id),
            count=1,
            protocols=None,
        )
        response = multi_scope_test_client.post("/claim", content=json.dumps(data))
        assert response.status_code == 200
        tasks_no_exclude = response.json()
        assert len(tasks_no_exclude) == 1
        assert tasks_no_exclude[0] is not None

        # claim tasks excluding scope_test — should still get tasks from other scopes
        data_with_exclude = dict(
            scopes=[s.to_dict() for s in multiple_scopes],
            scopes_exclude=[scope_test.to_dict()],
            compute_service_id=str(compute_service_id),
            count=1,
            protocols=None,
        )
        response = multi_scope_test_client.post(
            "/claim", content=json.dumps(data_with_exclude)
        )
        assert response.status_code == 200
        tasks_with_exclude = response.json()
        assert len(tasks_with_exclude) == 1
        # the claimed task should not be from scope_test
        if tasks_with_exclude[0] is not None:
            claimed_sk = ScopedKey.from_str(tasks_with_exclude[0])
            assert claimed_sk.scope != scope_test

    def test_claim_tasks_scopes_exclude_all(
        self,
        n4js_preloaded,
        multi_scope_test_client,
        multiple_scopes,
        compute_service_id,
    ):
        """Test that excluding all scopes returns an empty list."""
        # register a compute service
        multi_scope_test_client.post(
            f"/computeservice/{compute_service_id}/register",
            content=json.dumps({"compute_manager_id": None}),
        )

        # exclude all scopes — should return empty list
        data = dict(
            scopes=[s.to_dict() for s in multiple_scopes],
            scopes_exclude=[s.to_dict() for s in multiple_scopes],
            compute_service_id=str(compute_service_id),
            count=1,
            protocols=None,
        )
        response = multi_scope_test_client.post("/claim", content=json.dumps(data))
        assert response.status_code == 200
        assert response.json() == []

    def test_claim_tasks_scopes_exclude_wildcard(
        self,
        n4js_preloaded,
        multi_scope_test_client,
        multiple_scopes,
        compute_service_id,
    ):
        """Test that a wildcard exclusion scope filters all matching taskhubs."""
        # register a compute service
        multi_scope_test_client.post(
            f"/computeservice/{compute_service_id}/register",
            content=json.dumps({"compute_manager_id": None}),
        )

        # exclude with a wildcard scope that matches everything
        wildcard_scope = Scope()  # org=None, campaign=None, project=None => *-*-*
        data = dict(
            scopes=[s.to_dict() for s in multiple_scopes],
            scopes_exclude=[wildcard_scope.to_dict()],
            compute_service_id=str(compute_service_id),
            count=1,
            protocols=None,
        )
        response = multi_scope_test_client.post("/claim", content=json.dumps(data))
        assert response.status_code == 200
        assert response.json() == []
