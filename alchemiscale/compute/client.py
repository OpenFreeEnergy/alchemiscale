"""
:mod:`alchemiscale.compute.client` --- client for interacting with compute API
==============================================================================


"""

import json
from urllib.parse import urljoin

import requests
import zstandard as zstd

from gufe import Transformation
from gufe.protocols import ProtocolDAGResult
from gufe.tokenization import JSON_HANDLER

from ..base.client import (
    AlchemiscaleBaseClient,
    AlchemiscaleBaseClientError,
    json_to_gufe,
)
from ..compression import compress_gufe_zstd, decompress_gufe_zstd
from ..models import Scope, ScopedKey
from ..storage.models import (
    TaskHub,
    Task,
    ComputeServiceID,
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerStatus,
)


class AlchemiscaleComputeClientError(AlchemiscaleBaseClientError): ...


class AlchemiscaleComputeClient(AlchemiscaleBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = AlchemiscaleComputeClientError

    def register(
        self,
        compute_service_id: ComputeServiceID,
        compute_manager_id: ComputeManagerID | None = None,
        hostname: str | None = None,
    ):
        res = self._post_resource(
            f"/computeservice/{compute_service_id}/register",
            {"compute_manager_id": compute_manager_id, "hostname": hostname},
        )
        return ComputeServiceID(res)

    def deregister(self, compute_service_id: ComputeServiceID):
        res = self._post_resource(
            f"/computeservice/{compute_service_id}/deregister", {}
        )
        return ComputeServiceID(res)

    def heartbeat(self, compute_service_id: ComputeServiceID):
        res = self._post_resource(f"/computeservice/{compute_service_id}/heartbeat", {})
        return ComputeServiceID(res)

    def list_scopes(self) -> list[Scope]:
        scopes = self._get_resource(
            f"/identities/{self.identifier}/scopes",
        )
        return [Scope.from_str(s) for s in scopes]

    def query_taskhubs(
        self, scopes: list[Scope], return_gufe=False
    ) -> list[ScopedKey] | dict[ScopedKey, TaskHub]:
        """Return all `TaskHub`s corresponding to given `Scope`."""
        if return_gufe:
            taskhubs = {}
        else:
            taskhubs = []

        for scope in scopes:
            params = dict(return_gufe=return_gufe, **scope.to_dict())
            if return_gufe:
                taskhubs.update(self._query_resource("/taskhubs", params=params))
            else:
                taskhubs.extend(self._query_resource("/taskhubs", params=params))

        return taskhubs

    def claim_taskhub_tasks(
        self,
        taskhub: ScopedKey,
        compute_service_id: ComputeServiceID,
        count: int = 1,
        protocols: list[str] | None = None,
    ) -> Task:
        """Claim a `Task` from the specified `TaskHub`"""
        data = dict(
            compute_service_id=str(compute_service_id), count=count, protocols=protocols
        )
        tasks = self._post_resource(f"/taskhubs/{taskhub}/claim", data)

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def claim_tasks(
        self,
        scopes: list[Scope],
        compute_service_id: ComputeServiceID,
        scopes_exclude: list[Scope] | None = None,
        count: int = 1,
        protocols: list[str] | None = None,
    ):
        """Claim Tasks from TaskHubs within a list of Scopes."""
        data = dict(
            scopes=[scope.to_dict() for scope in scopes],
            scopes_exclude=(
                [scope.to_dict() for scope in scopes_exclude]
                if scopes_exclude
                else None
            ),
            compute_service_id=str(compute_service_id),
            count=count,
            protocols=protocols,
        )
        tasks = self._post_resource("/claim", data)

        if tasks is None:
            return None

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def get_task_transformation(self, task: ScopedKey) -> ScopedKey:
        """Get the Transformation associated with the given Task."""
        transformation = self._get_resource(f"/tasks/{task}/transformation")
        return ScopedKey.from_str(transformation)

    def retrieve_task_transformation(
        self, task: ScopedKey
    ) -> tuple[Transformation, ProtocolDAGResult | None]:
        transformation_json, protocoldagresult_latin1 = self._get_resource(
            f"/tasks/{task}/transformation/gufe"
        )

        if (protocoldagresult := protocoldagresult_latin1) is not None:

            protocoldagresult_bytes = protocoldagresult_latin1.encode("latin-1")

            try:
                # Attempt to decompress the ProtocolDAGResult object
                protocoldagresult = decompress_gufe_zstd(protocoldagresult_bytes)
            except zstd.ZstdError:
                # If decompression fails, assume it's a UTF-8 encoded JSON string
                protocoldagresult = json_to_gufe(
                    protocoldagresult_bytes.decode("utf-8")
                )

        return json_to_gufe(transformation_json), protocoldagresult

    def set_task_result(
        self,
        task: ScopedKey,
        protocoldagresult: ProtocolDAGResult,
        compute_service_id: ComputeServiceID | None = None,
    ) -> ScopedKey:

        data = dict(
            protocoldagresult=compress_gufe_zstd(protocoldagresult),
            compute_service_id=str(compute_service_id),
        )

        pdr_sk = self._post_resource(f"/tasks/{task}/results", data)

        return ScopedKey.from_dict(pdr_sk)

    def set_task_error(
        self,
        task: ScopedKey,
        reason: str,
        compute_service_id: ComputeServiceID | None = None,
    ) -> ScopedKey:
        """Set a `Task` to `error` with a human-readable `reason`.

        Used by the compute service when `ProtocolDAG` creation fails, where
        there is no `ProtocolDAGResult` to submit. The `reason` (a traceback)
        is stored on `Task.reason` and the open `TaskProvenance` attempt is
        finalized with `outcome = error`.
        """
        data = dict(reason=reason, compute_service_id=str(compute_service_id))
        task_sk = self._post_resource(f"/tasks/{task}/error", data)
        return ScopedKey.from_str(task_sk)

    def update_task_progress(
        self,
        compute_service_id: ComputeServiceID,
        progress: dict[str, dict[str, int]],
        timeout: float = 5.0,
    ) -> None:
        """Push live progress counts for this service's claimed Tasks.

        `progress` maps `Task` ScopedKey strings to
        ``{"units_completed": int, "units_total": int}``. One batched request
        per push event.

        This is **fire-and-forget**: a single attempt with a short timeout and
        NO retry/backoff (the usual retry machinery would stall the DAG between
        units on a flaky API). Progress is best-effort telemetry, refreshed at
        the next unit boundary. Errors propagate to the caller, which is
        expected to log-and-continue.

        This deliberately does **not** fetch or refresh a JWT: `_get_token`
        issues its request with ``timeout=None`` and would block the execution
        thread indefinitely if the token endpoint hangs. A token is virtually
        always already present by the time a DAG executes (registration/claim
        obtained one); if it is missing or stale, the push is simply skipped and
        the retrying transport (result push, next claim) refreshes it before the
        next boundary.
        """
        if self._jwtoken is None:
            # no token yet and we won't block to get one; skip this push
            return

        url = urljoin(self.api_url, f"/computeservice/{compute_service_id}/progress")
        jsondata = json.dumps(progress, cls=JSON_HANDLER.encoder)

        resp = requests.post(
            url,
            data=jsondata,
            headers=self._headers,
            timeout=timeout,
            verify=self.verify,
        )
        if not 200 <= resp.status_code < 300:
            raise self._exception(
                f"Status Code {resp.status_code} : {resp.reason}",
                status_code=resp.status_code,
            )

    def set_task_result_unit_logs(
        self,
        task: ScopedKey,
        protocoldagresultref: ScopedKey,
        unit_result_key: str,
        logs: str,
    ) -> None:
        """Upload captured log text for a single unit result.

        Uses the normal (retrying) transport --- this happens after DAG
        execution, not between units, so a retry cannot stall execution.
        """
        self._post_resource(
            f"/tasks/{task}/results/{protocoldagresultref}/units/{unit_result_key}/artifacts/logs",
            {"logs": logs},
        )


class AlchemiscaleComputeManagerClientError(AlchemiscaleBaseClientError): ...


class AlchemiscaleComputeManagerClient(AlchemiscaleBaseClient):

    _exception = AlchemiscaleComputeManagerClientError

    def register(
        self, compute_manager_id: ComputeManagerID, steal: bool = False
    ) -> ComputeManagerID:
        res = self._post_resource(
            f"/computemanager/{compute_manager_id}/register", {"steal": steal}
        )
        return ComputeManagerID(res)

    def deregister(self, compute_manager_id: ComputeManagerID) -> ComputeManagerID:
        res = self._post_resource(
            f"/computemanager/{compute_manager_id}/deregister", {}
        )
        return ComputeManagerID(res)

    def get_instruction(
        self,
        scopes: list[Scope],
        protocols: list[str] | None,
        compute_manager_id: ComputeManagerID,
    ) -> tuple[ComputeManagerInstruction, dict]:
        instruction_data = self._post_resource(
            f"/computemanager/{compute_manager_id}/instruction",
            {
                "scopes": [scope.to_dict() for scope in scopes],
                "protocols": protocols if protocols else [],
            },
        )

        match instruction_data:
            case {
                "instruction": "OK",
                "compute_service_ids": ids,
                "num_tasks": num_tasks,
            }:
                return ComputeManagerInstruction.OK, {
                    "compute_service_ids": ids,
                    "num_tasks": num_tasks,
                }
            case {"instruction": "SKIP", "compute_service_ids": ids}:
                return ComputeManagerInstruction.SKIP, {"compute_service_ids": ids}
            case {"instruction": "SHUTDOWN", "message": message}:
                return ComputeManagerInstruction.SHUTDOWN, {"message": message}
            case _:
                raise self._exception(
                    f"Received unknown instruction pattern: {instruction_data}"
                )

    def update_status(
        self,
        compute_manager_id: ComputeManagerID,
        status: ComputeManagerStatus,
        *,
        detail: str | None = None,
        saturation: float | None = None,
    ) -> ComputeManagerID:
        payload = {"detail": detail, "saturation": saturation, "status": str(status)}
        res = self._post_resource(
            f"/computemanager/{compute_manager_id}/status",
            payload,
        )

        return ComputeManagerID(res)

    def clear_error(self, compute_manager_name: str):
        res = self._post_resource(
            f"/computemanager/{compute_manager_name}/clear_error",
            {},
        )
