import pytest
from uuid import uuid4
from time import sleep

from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerStatus,
    ComputeManagerInstruction,
)
from alchemiscale.compute.manager import ComputeManager, ComputeManagerSettings
from alchemiscale.compute.client import AlchemiscaleComputeManagerClient


class LocalTestingComputeManager(ComputeManager):

    def __init__(self, settings: ComputeManagerSettings):
        self.settings = settings
        self.compute_manager_id = ComputeManagerID.new_from_manager_name(
            self.settings.name
        )

        self.client = AlchemiscaleComputeManagerClient(
            api_url=self.settings.api_url,
            identifier=self.settings.identifier,
            key=self.settings.key,
        )

    def start(self):
        self._register()
        self.cycle()
        self._deregister()

    def cycle(self):
        try:
            while True:

                instruction, data = self.client.get_instruction(self.compute_manager_id)

                match instruction:
                    case ComputeManagerInstruction.OK:
                        break
                    case ComputeManagerInstruction.SKIP:
                        pass
                    case ComputeManagerInstruction.SHUTDOWN:
                        break

                self.client.update_status(
                    self.compute_manager_id, ComputeManagerStatus.OK
                )
                sleep(self.settings.sleep_interval)
        except Exception as e:
            self.client.update_status(
                self.compute_manager_id, ComputeManagerStatus.ERRORED, repr(e)
            )
            raise e
        finally:
            self._deregister()

    def create_compute_service(self):
        raise NotImplementedError


@pytest.fixture()
def manager_settings(compute_manager_client):
    return ComputeManagerSettings(
        api_url=compute_manager_client.api_url,
        identifier=compute_manager_client.identifier,
        key=compute_manager_client.key,
        name="testmanager",
        logfile=None,
        status_update_interval=15,
        max_compute_services=2,
        sleep_interval=3,
    )


@pytest.fixture()
def manager(
    manager_settings,
    tmpdir,
):
    return LocalTestingComputeManager(manager_settings)


class TestComputeManager:

    def test_manager_implementation(
        self, n4js_preloaded, manager: LocalTestingComputeManager
    ):
        manager.start()
