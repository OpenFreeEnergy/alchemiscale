from time import sleep

import pytest

from alchemiscale.compute import client
from alchemiscale.storage.models import ComputeManagerID, ComputeManagerInstruction
from alchemiscale.tests.integration.compute.utils import get_compute_settings_override


class TestComputeManager:

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
        compute_client: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        settings = get_compute_settings_override()
        assert compute_client._jwtoken is None
        compute_client._get_token()

        token = compute_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        compute_client.get_info()
        assert token == compute_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        compute_client.get_info()
        assert token != compute_client._jwtoken

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
        compute_manager_id = ComputeManagerID.from_manager_id("testmanager")
        returned_id = compute_manager_client.register(compute_manager_id)

        assert compute_manager_id == returned_id

    def test_deregistration(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.from_manager_id("testmanager")
        compute_manager_client.register(compute_manager_id)
        returned_id = compute_manager_client.deregister(compute_manager_id)
        assert compute_manager_id == returned_id

    def test_get_instruction(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.from_manager_id("testmanager")
        compute_manager_client.register(compute_manager_id)
        instruction, payload = compute_manager_client.get_instruction(
            compute_manager_id
        )

        assert instruction == ComputeManagerInstruction.OK, (instruction, payload)
        assert payload == {"compute_service_ids": [], "num_registered": 0}

    def test_update_status(
        self,
        n4js_preloaded,
        computer_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        raise NotImplementedError
