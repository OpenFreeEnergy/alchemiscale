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
from .client import AlchemiscaleComputeManagerClient
from .settings import ComputeManagerSettings


class ComputeManager:

    compute_manager_id: ComputeManagerID
    client: AlchemiscaleComputeManagerClient
    service_settings_template: bytes
    manager_settings: ComputeManagerSettings

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
        self.client.register(self.compute_manager_id)

    def _deregister(self):
        self.client.deregister(self.compute_manager_id)

    def start(self, max_cycles: int | None = None):
        self._register()
        try:
            count = 0
            while True:
                self.cycle()
                count += 1
                if max_cycles and count >= max_cycles:
                    break
                time.sleep(self.settings.sleep_interval)
        except Exception as e:
            self.client.update_status(
                self.compute_manager_id, ComputeManagerStatus.ERRORED, detail=repr(e)
            )
            raise e
        except KeyboardInterrupt:
            self.logger.info("Caught SIGINT/Keyboard interrupt.")
        finally:
            self._deregister()

    @abstractmethod
    def create_compute_services(self, data: dict) -> int:
        """Method responsible for creating compute services based on
        data returned with an OK ComputeManagerInstruction. This must
        return the number of compute services started.
        """
        raise NotImplementedError

    def cycle(self):
        instruction, data = self.client.get_instruction(self.compute_manager_id)
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
                        self.logger.info(f"No new compute services created")
            case ComputeManagerInstruction.SKIP:
                total_services = len(data["compute_service_ids"])
                self.logger.info(f"Received skip instruction")
            case ComputeManagerInstruction.SHUTDOWN:
                shutdown_message = data["message"]
                self.logger.info(f'Received shutdown message: "{shutdown_message}"')
                return
        self.client.update_status(
            self.compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=total_services / self.settings.max_compute_services,
        )

    @abstractmethod
    def create_compute_service(self):
        raise NotImplementedError
