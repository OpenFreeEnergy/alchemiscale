"""
:mod:`alchemiscale.compute.client` --- client for interacting with compute API
==============================================================================


"""

from typing import List, Tuple, Optional, Dict, Union
import json
from urllib.parse import urljoin
from functools import wraps
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

import zstandard as zstd

from gufe.tokenization import GufeTokenizable, JSON_HANDLER
from gufe import Transformation
from gufe.protocols import ProtocolDAGResult

from ..base.client import (
    AlchemiscaleBaseClient,
    AlchemiscaleBaseClientError,
    json_to_gufe,
)
from ..compression import compress_gufe_zstd, decompress_gufe_zstd
from ..models import Scope, ScopedKey
from ..storage.models import TaskHub, Task, ComputeServiceID, TaskStatusEnum


class AlchemiscaleComputeClientError(AlchemiscaleBaseClientError): ...


class AlchemiscaleComputeClient(AlchemiscaleBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = AlchemiscaleComputeClientError

    def register(self, compute_service_id: ComputeServiceID):
        res = self._post_resource(f"/computeservice/{compute_service_id}/register", {})
        return ComputeServiceID(res)

    def deregister(self, compute_service_id: ComputeServiceID):
        res = self._post_resource(
            f"/computeservice/{compute_service_id}/deregister", {}
        )
        return ComputeServiceID(res)

    def heartbeat(self, compute_service_id: ComputeServiceID):
        res = self._post_resource(f"/computeservice/{compute_service_id}/heartbeat", {})
        return ComputeServiceID(res)

    def list_scopes(self) -> List[Scope]:
        scopes = self._get_resource(
            f"/identities/{self.identifier}/scopes",
        )
        return [Scope.from_str(s) for s in scopes]

    def query_taskhubs(
        self, scopes: List[Scope], return_gufe=False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, TaskHub]]:
        """Return all `TaskHub`s corresponding to given `Scope`."""
        if return_gufe:
            taskhubs = {}
        else:
            taskhubs = []

        for scope in scopes:
            params = dict(return_gufe=return_gufe, **scope.dict())
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
        protocols: Optional[List[str]] = None,
    ) -> Task:
        """Claim a `Task` from the specified `TaskHub`"""
        data = dict(
            compute_service_id=str(compute_service_id), count=count, protocols=protocols
        )
        tasks = self._post_resource(f"/taskhubs/{taskhub}/claim", data)

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def claim_tasks(
        self,
        scopes: List[Scope],
        compute_service_id: ComputeServiceID,
        count: int = 1,
        protocols: Optional[List[str]] = None,
    ):
        """Claim Tasks from TaskHubs within a list of Scopes."""
        data = dict(
            scopes=[scope.dict() for scope in scopes],
            compute_service_id=str(compute_service_id),
            count=count,
            protocols=protocols,
        )
        tasks = self._post_resource("/claim", data)

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def get_task_transformation(self, task: ScopedKey) -> ScopedKey:
        """Get the Transformation associated with the given Task."""
        transformation = self._get_resource(f"/tasks/{task}/transformation")
        return ScopedKey.from_str(transformation)

    def retrieve_task_transformation(
        self, task: ScopedKey
    ) -> tuple[Transformation, Optional[ProtocolDAGResult]]:
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
        compute_service_id: Optional[ComputeServiceID] = None,
    ) -> ScopedKey:

        data = dict(
            protocoldagresult=compress_gufe_zstd(protocoldagresult),
            compute_service_id=str(compute_service_id),
        )

        pdr_sk = self._post_resource(f"/tasks/{task}/results", data)

        return ScopedKey.from_dict(pdr_sk)
