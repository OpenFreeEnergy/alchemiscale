"""
:mod:`alchemiscale.compute.manager` --- compute manager for creating compute services
=====================================================================================

"""

from abc import abstractmethod
from enum import StrEnum

from alchemsicale.storage.models import ComputeManagerID
from .client import AlchemiscaleComputeManagerClient
from .manager import ComputeManagerInstruction
from settings import ComputeManagerSettings


class ComputeManagerInstruction(StrEnum):
    OK = "OK"
    SKIP = "SKIP"
    SHUTDOWN = "SHUTDOWN"


class ComputeManager:

    compute_manager_id: ComputeManagerID
    status_update_interval: int
    sleep_interval: int
    client: AlchemiscaleComputeManagerClient
    service_settings_template: bytes
    manager_settings: ComputeManagerSettings

    def _register(self):
        self.client.register(self.compute_manager_id)

    def _deregister(self):
        self.client.deregister(self.compute_client_id)

    def request_instruction(self) -> ComputeManagerInstruction:
        instruction = self.client._post_resource(
            f"/computemanager/{self.compute_manager_id}/update_status", {}
        )
        return instruction

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def cycle(self):
        raise NotImplementedError

    @abstractmethod
    def create_compute_service(self):
        raise NotImplementedError
