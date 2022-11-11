import pytest

from gufe.tokenization import GufeTokenizable

from fah_alchemy.models import ScopedKey
from fah_alchemy.compute import client


# api tests

class TestComputeAPI:

    def test_info(self, test_client):
    
        response = test_client.get("/info")
        assert response.status_code == 200
    
    
    def test_query_taskqueues(self, n4js_clear, test_client):
    
        response = test_client.get("/taskqueues")
        assert response.status_code == 200
    
        tq_sks = [ScopedKey.from_str(i) for i in response.json()]
        assert len(tq_sks) == 2

        # try getting back actual gufe objects
        response = test_client.get("/taskqueues?return_gufe=True")
        assert response.status_code == 200

        tq_dict = {ScopedKey.from_str(k): GufeTokenizable.from_dict(v) for k, v in response.json().items()}
        assert len(tq_dict) == 2
        assert all([i.weight == .5 for i in tq_dict.values()])


# client tests

class TestComputeClient:

    def test_query_taskqueues(self, 
                              scope_test, 
                              n4js_clear,
                              compute_client: client.FahAlchemyComputeClient, 
                              uvicorn_server
                              ):

        taskqueues = compute_client.query_taskqueues(scope_test)

        assert len(taskqueues) == 2

        taskqueues = compute_client.query_taskqueues(scope_test, return_gufe=True)
        assert all([tq.weight == .5 for tq in taskqueues.values()])

    def test_claim_taskqueue_task(self, 
                              scope_test, 
                              n4js_clear,
                              compute_client: client.FahAlchemyComputeClient, 
                              uvicorn_server
                              ):

        ...
        #task = compute_client.claim_taskqueue_task(scope_test)
