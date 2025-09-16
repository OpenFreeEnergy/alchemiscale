import datetime
from time import sleep

import pytest

from alchemiscale.compute import client
from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerStatus,
    ComputeServiceID,
)
from alchemiscale.tests.integration.compute.utils import get_compute_settings_override


class TestComputeManagerClient:

    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        manager_client_wrong_credential: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        with pytest.raises(client.AlchemiscaleComputeManagerClientError):
            manager_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        settings = get_compute_settings_override()
        assert compute_manager_client._jwtoken is None
        compute_manager_client._get_token()

        token = compute_manager_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        compute_manager_client.get_info()
        assert token == compute_manager_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        compute_manager_client.get_info()
        assert token != compute_manager_client._jwtoken

    def test_api_check(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        compute_manager_client._api_check()

    def test_registration(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        returned_id = compute_manager_client.register(compute_manager_id)

        assert compute_manager_id == returned_id

    def test_deregistration(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)
        returned_id = compute_manager_client.deregister(compute_manager_id)
        assert compute_manager_id == returned_id

    def test_get_instruction(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id: ComputeServiceID,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)

        # no compute services being managed
        instruction, payload = compute_manager_client.get_instruction(
            [], compute_manager_id
        )

        assert instruction == ComputeManagerInstruction.OK
        assert payload == {"compute_service_ids": [], "num_tasks": 3}

        # mock the creation of a compute service
        _compute_service_id = compute_client.register(
            compute_service_id, compute_manager_id=compute_manager_id
        )

        instruction, payload = compute_manager_client.get_instruction(
            [], compute_manager_id
        )

        assert instruction == ComputeManagerInstruction.OK, (instruction, payload)
        assert payload == {
            "compute_service_ids": [_compute_service_id],
            "num_tasks": 3,
        }, (instruction, payload)

        # add in many failures to trigger a SKIP instruction on next request

        add_fake_failures_query = """
        UNWIND $failure_times AS failure_time
        MATCH (csr: ComputeServiceRegistration {identifier: $compute_service_id})
        SET csr.failure_times = datetime(failure_time) + csr.failure_times
        """

        failure_times = [datetime.datetime.now(tz=datetime.UTC)] * 4
        n4js_preloaded.execute_query(
            add_fake_failures_query,
            compute_service_id=compute_service_id,
            failure_times=failure_times,
        )

        instruction, payload = compute_manager_client.get_instruction(
            [], compute_manager_id
        )

        assert instruction == ComputeManagerInstruction.SKIP, (instruction, payload)
        assert payload == {"compute_service_ids": [_compute_service_id]}

        # change the UUID associated with the stored manager to simulate usurping

        usurp_query = """
        MATCH (cmr: ComputeManagerRegistration {name: $name})
        SET cmr.uuid = $new_uuid
        """
        from uuid import uuid4

        new_uuid = uuid4()
        params = {
            "name": compute_manager_id.name,
            "new_uuid": str(new_uuid),
        }
        n4js_preloaded.execute_query(usurp_query, params)

        instruction, payload = compute_manager_client.get_instruction(
            [], compute_manager_id
        )

        assert instruction == ComputeManagerInstruction.SHUTDOWN, (instruction, payload)
        assert payload == {
            "message": "no compute manager was found with the given manager name and UUID"
        }, (instruction, payload)

    def test_update_status(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)

        query = """
        MATCH (cmr: ComputeManagerRegistration {name: $name})
        RETURN cmr.status AS status, cmr.detail as detail
        """

        params = {"name": compute_manager_id.name}

        # status: OK
        returned_id = compute_manager_client.update_status(
            compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=0,
        )

        assert returned_id == compute_manager_id

        results = n4js_preloaded.execute_query(query, params)
        record = results.records[0]
        assert record["status"] == "OK" and record["detail"] is None

        # status: ERROR
        exception = RuntimeError("UnexpectedError")

        returned_id = compute_manager_client.update_status(
            compute_manager_id,
            ComputeManagerStatus.ERROR,
            detail=repr(exception),
        )

        assert returned_id == compute_manager_id

        results = n4js_preloaded.execute_query(query, params)
        record = results.records[0]
        assert record["status"] == "ERROR" and record["detail"] == repr(exception)
