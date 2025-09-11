"""
:mod:`alchemiscale.compute.manager` --- compute manager for creating compute services
=====================================================================================

"""

from abc import abstractmethod
import logging
import time

from ..storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerStatus,
)
from .client import (
    AlchemiscaleComputeManagerClient,
    AlchemiscaleComputeManagerClientError,
)
from .settings import ComputeManagerSettings, ComputeServiceSettings


class ComputeManager:

    compute_manager_id: ComputeManagerID
    client: AlchemiscaleComputeManagerClient
    service_settings: ComputeServiceSettings
    settings: ComputeManagerSettings

    def __init__(
        self, settings: ComputeManagerSettings, service_settings: ComputeServiceSettings
    ):
        self.settings = settings
        self.compute_manager_id = ComputeManagerID.new_from_name(self.settings.name)
        self.service_settings = service_settings
        self.service_settings.compute_manager_id = self.compute_manager_id
        self.client = AlchemiscaleComputeManagerClient(
            api_url=self.service_settings.api_url,
            identifier=self.service_settings.identifier,
            key=self.service_settings.key,
        )

        self._stop = False

        logger = logging.getLogger("AlchemiscaleComputeManager")
        logger.setLevel(self.settings.loglevel)

        extra = {"compute_manager_id": self.compute_manager_id}
        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_manager_id)s] [%(levelname)s] %(message)s"
        )
        formatter.converter = time.gmtime  # use utc time for logging timestamps

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if self.settings.logfile is not None:
            fh = logging.FileHandler(self.settings.logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logging.LoggerAdapter(logger, extra)

    def _register(self):
        try:
            self.client.register(self.compute_manager_id)
        except AlchemiscaleComputeManagerClientError as e:
            self.logger.error(f"Registration failed: '{e.detail}'")
            raise e

    def _deregister(self):
        self.client.deregister(self.compute_manager_id)

    def start(self, max_cycles: int | None = None):
        self.logger.info(f"Starting up compute manager '{self.settings.name}'")
        self._register()
        self.logger.info(f"Registered compute manager '{self.compute_manager_id}'")
        self._stop = False
        try:
            count = 0
            self.logger.info("Starting main loop")
            while self.cycle():
                count += 1
                if max_cycles and count >= max_cycles:
                    self.logger.info("Reached maximum number of cycles")
                    break
                time.sleep(self.settings.sleep_interval)
        except Exception as e:
            self.logger.error(f"Unknown exception raised: '{str(e)}'")
            self.logger.info(f"Updating manager status to 'ERROR'")
            self.client.update_status(
                self.compute_manager_id, ComputeManagerStatus.ERROR, detail=repr(e)
            )
            raise e
        except KeyboardInterrupt:
            self.logger.info("Caught SIGINT/Keyboard interrupt.")
        finally:
            self.logger.info(f"Deregistering '{self.compute_manager_id}'")
            self._deregister()
            self.logger.info(f"Deregistration successful")

    @abstractmethod
    def create_compute_services(self, data: dict) -> int:
        """Method responsible for creating compute services based on
        data returned with an OK ComputeManagerInstruction. This must
        return the number of compute services started.
        """
        raise NotImplementedError

    def stop(self):
        self._stop = True

    def cycle(self) -> bool:

        if self._stop:
            return False

        self.logger.info(f"Requesting instruction from '{self.client.api_url}'")
        instruction, data = self.client.get_instruction(
            self.service_settings.scopes or [], self.compute_manager_id
        )
        self.logger.info(f"Recieved instruction '{instruction}'")
        match instruction:
            case ComputeManagerInstruction.OK:
                total_services = len(data["compute_service_ids"])
                num_tasks = data["num_tasks"]
                if (
                    total_services < self.settings.max_compute_services
                    and num_tasks > 0
                ):
                    new_services = self.create_compute_services(data)
                    total_services += new_services
                    if new_services:
                        self.logger.info(
                            f"Created {new_services} new compute service(s)"
                        )
                    else:
                        self.logger.info("No new compute services created")
            case ComputeManagerInstruction.SKIP:
                total_services = len(data["compute_service_ids"])
                self.logger.info("Received skip instruction")
            case ComputeManagerInstruction.SHUTDOWN:
                shutdown_message = data["message"]
                self.logger.info(f'Received shutdown message: "{shutdown_message}"')
                return False
        self.logger.info(f"Updating manager status at '{self.client.api_url}'")
        self.client.update_status(
            self.compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=total_services / self.settings.max_compute_services,
        )
        return True

    def clear_error(self):
        try:
            self.logger.info(f"Clearing '{self.settings.name}' with the ERROR status")
            self.client.clear_error(self.settings.name)
        except ValueError:
            self.logger.info(
                f"Could not clear '{self.settings.name}', no such compute manager with the ERROR status"
            )
