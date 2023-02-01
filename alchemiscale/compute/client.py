"""
Client for interacting with compute API. --- :mod:`alchemiscale.compute.client`
==============================================================================


"""

from typing import List, Tuple, Optional
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
from ..storage.models import TaskHub, Task


class AlchemiscaleComputeClientError(AlchemiscaleBaseClientError):
    ...


class AlchemiscaleComputeClient(AlchemiscaleBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = AlchemiscaleComputeClientError

    def query_taskhubs(
        self, scopes: List[Scope], return_gufe=False, limit=None, skip=None
    ) -> List[TaskHub]:
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

    def get_taskhub_tasks(self, taskhub: ScopedKey) -> List[Task]:
        """Get list of `Task`s for the given `TaskHub`."""
        self._get_resource(f"taskhubs/{taskhub}/tasks", {})

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

        transformation, protocoldagresult = self._get_resource(
            f"tasks/{task}/transformation", {}, return_gufe=False
        )

        return (
            GufeTokenizable.from_dict(transformation),
            GufeTokenizable.from_dict(protocoldagresult) if protocoldagresult else None,
        )

    def set_task_result(
        self, task: ScopedKey, protocoldagresult: ProtocolDAGResult
    ) -> ScopedKey:
        data = dict(
            protocoldagresult=json.dumps(
                protocoldagresult.to_dict(), cls=JSON_HANDLER.encoder
            )
        )

        pdr_sk = self._post_resource(f"tasks/{task}/result", data)

        return ScopedKey.from_dict(pdr_sk)
