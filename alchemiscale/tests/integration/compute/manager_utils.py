"""Helpers for running ComputeManager cycles against a real state store
without the compute API."""

import datetime
from datetime import timedelta
from importlib.metadata import version
import logging

from packaging.version import Version

from alchemiscale.compute.client import ComputeManagerNotRegisteredError
from alchemiscale.compute.manager import ComputeManager
from alchemiscale.compute.settings import ComputeManagerSettings, ComputeServiceSettings
from alchemiscale.compute.testing import FakeComputeBackend
from alchemiscale.models import Scope
from alchemiscale.storage.models import (
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerRegistration,
    ComputeManagerState,
    ComputeManagerStateUpdate,
    ComputeManagerStatus,
    ComputeServiceID,
    ComputeServiceRegistration,
)
from alchemiscale.storage.statestore import (
    ComputeManagerNotFoundError,
    Neo4jStore,
)


class StoreClient:
    """Stands in for ``AlchemiscaleComputeManagerClient``, calling the state
    store directly as the compute API would.

    ``num_tasks`` and ``instruction`` override what the server would report.

    """

    api_url = "store://"

    def __init__(self, n4js: Neo4jStore, expire_seconds: int = 3600):
        self.n4js = n4js
        self.expire_seconds = expire_seconds
        self.num_tasks = 10
        self.instruction: ComputeManagerInstruction | None = None
        self.posted: list[tuple[str, ComputeManagerStateUpdate]] = []

    def _expire_time(self):
        return datetime.datetime.now(tz=datetime.UTC) - timedelta(
            seconds=self.expire_seconds
        )

    def get_server_version(self):
        return Version(version("alchemiscale"))

    def register(self, compute_manager_id, steal=False):
        now = datetime.datetime.now(tz=datetime.UTC)
        return self.n4js.register_computemanager(
            ComputeManagerRegistration(
                name=compute_manager_id.name,
                uuid=compute_manager_id.uuid,
                registered=now,
                last_status_update=now,
                status=ComputeManagerStatus.OK,
                saturation=0,
            ),
            steal=steal,
            expire_time=self._expire_time(),
        )

    def deregister(self, compute_manager_id):
        self.n4js.deregister_computemanager(compute_manager_id)
        return compute_manager_id

    def get_instruction(self, scopes, protocols, compute_manager_id):
        try:
            instruction, data = self.n4js.get_computemanager_instruction(
                compute_manager_id,
                datetime.datetime.now(tz=datetime.UTC) - timedelta(minutes=10),
                3,
                [Scope()],
                [],
                expire_time=self._expire_time(),
            )
        except ComputeManagerNotFoundError as e:
            raise ComputeManagerNotRegisteredError(str(e), status_code=404)

        data["state"] = ComputeManagerState.from_dict(data["state"])
        data["compute_service_ids"] = [str(i) for i in data["compute_service_ids"]]
        instruction = self.instruction or instruction
        if instruction == ComputeManagerInstruction.OK:
            data["num_tasks"] = self.num_tasks
        return instruction, data

    def update_state(
        self, compute_manager_id, status, *, detail=None, saturation=None, state_update
    ):
        # round-trip through JSON, as the compute API would
        state_update = ComputeManagerStateUpdate.from_dict(state_update.to_dict())
        self.posted.append((str(status), state_update))
        try:
            return self.n4js.update_compute_manager_status(
                compute_manager_id,
                status,
                detail,
                saturation,
                expire_time=self._expire_time(),
                state_update=state_update,
            )
        except ComputeManagerNotFoundError as e:
            raise ComputeManagerNotRegisteredError(str(e), status_code=404)

    def update_status(
        self, compute_manager_id, status, *, detail=None, saturation=None
    ):
        self.update_state(
            compute_manager_id,
            status,
            detail=detail,
            saturation=saturation,
            state_update=ComputeManagerStateUpdate(),
        )
        return compute_manager_id


def make_manager(
    n4js: Neo4jStore,
    backend: FakeComputeBackend,
    name: str = "testmanager",
    **settings,
) -> ComputeManager:
    """A ComputeManager running on ``backend`` against ``n4js``, registered."""
    settings = {
        "name": name,
        "logfile": None,
        "max_compute_services": 4,
        "max_submit_per_cycle": 2,
        "sleep_interval": 1,
        "loglevel": "INFO",
    } | settings
    service_settings = ComputeServiceSettings(
        api_url="http://unused/",
        identifier="unused",
        key="unused",
        name="unused",
        shared_basedir="./shared",
        scratch_basedir="./scratch",
    )
    manager = ComputeManager(
        ComputeManagerSettings(**settings), service_settings, backend
    )
    manager.client = StoreClient(n4js)
    manager._register()
    return manager


def register_service(n4js: Neo4jStore, job_name: str) -> ComputeServiceID:
    """Register a compute service for ``job_name``, as the job would."""
    now = datetime.datetime.now(tz=datetime.UTC)
    return n4js.register_computeservice(
        ComputeServiceRegistration(
            identifier=ComputeServiceID.new_from_name(job_name),
            registered=now,
            heartbeat=now,
            manager_name=job_name.rsplit(".", 1)[0],
        )
    )


def deregister_services(n4js: Neo4jStore, job_name: str):
    """Deregister the compute services of ``job_name``."""
    records = n4js.execute_query(
        """
        MATCH (csr:ComputeServiceRegistration)
        WHERE csr.identifier STARTS WITH $prefix
        RETURN csr.identifier AS identifier
        """,
        prefix=job_name + "-",
    ).records
    for record in records:
        n4js.deregister_computeservice(ComputeServiceID(record["identifier"]))


def get_state(manager: ComputeManager) -> ComputeManagerState:
    return manager.client.n4js.get_compute_manager_state(manager.compute_manager_id)
