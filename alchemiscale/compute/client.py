"""
Client for interacting with compute API. --- :mod:`alchemiscale.compute.client`
==============================================================================


"""

from typing import List, Tuple, Optional, Dict, Union
import json
from urllib.parse import urljoin
from functools import wraps

import requests
from requests.auth import HTTPBasicAuth

from gufe.tokenization import GufeTokenizable, JSON_HANDLER
from gufe import Transformation
from gufe.protocols import ProtocolDAGResult

from ..base.client import AlchemiscaleBaseClient, AlchemiscaleBaseClientError
from ..models import Scope, ScopedKey
from ..storage.models import TaskHub, Task, TaskStatusEnum


class AlchemiscaleComputeClientError(AlchemiscaleBaseClientError):
    ...


class AlchemiscaleComputeClient(AlchemiscaleBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = AlchemiscaleComputeClientError

    def list_scopes(self) -> List[Scope]:
        scopes = self._get_resource(
            f"/identities/{self.identifier}/scopes",
            return_gufe=False,
        )
        return [Scope.from_str(s) for s in scopes]

    def query_taskhubs(
        self, scopes: List[Scope], return_gufe=False, limit=None, skip=None
    ) -> Union[List[ScopedKey], Dict[ScopedKey, TaskHub]]:
        """Return all `TaskHub`s corresponding to given `Scope`."""
        if return_gufe:
            taskhubs = {}
        else:
            taskhubs = []

        for scope in scopes:
            params = dict(
                return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict()
            )
            if return_gufe:
                taskhubs.update(self._query_resource("/taskhubs", params=params))
            else:
                taskhubs.extend(self._query_resource("/taskhubs", params=params))

        return taskhubs

    def claim_taskhub_tasks(
        self, taskhub: ScopedKey, claimant: str, count: int = 1
    ) -> Task:
        """Claim a `Task` from the specified `TaskHub`"""
        data = dict(claimant=claimant, count=count)
        tasks = self._post_resource(f"taskhubs/{taskhub}/claim", data)

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def get_task_transformation(
        self, task: ScopedKey
    ) -> Tuple[Transformation, Optional[ProtocolDAGResult]]:
        transformation, protocoldagresult_json = self._get_resource(
            f"tasks/{task}/transformation", {}, return_gufe=False
        )

        return (
            GufeTokenizable.from_dict(transformation),
            GufeTokenizable.from_dict(
                json.loads(protocoldagresult_json, cls=JSON_HANDLER.decoder)
            )
            if protocoldagresult_json is not None
            else None,
        )

    def set_task_result(
        self, task: ScopedKey, protocoldagresult: ProtocolDAGResult
    ) -> ScopedKey:
        data = dict(
            protocoldagresult=json.dumps(
                protocoldagresult.to_dict(), cls=JSON_HANDLER.encoder
            )
        )

        pdr_sk = self._post_resource(f"tasks/{task}/results", data)

        return ScopedKey.from_dict(pdr_sk)

    def set_task_status(self, task: ScopedKey, status: TaskStatusEnum) -> None:
        """Set the status of a `Task`."""

        self._post_resource(f"tasks/{task}/status", status.value)
