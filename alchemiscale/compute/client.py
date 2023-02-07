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
from ..storage.models import TaskQueue, Task


class AlchemiscaleComputeClientError(AlchemiscaleBaseClientError):
    ...


class AlchemiscaleComputeClient(AlchemiscaleBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = AlchemiscaleComputeClientError

    def query_taskqueues(
        self, scopes: List[Scope], return_gufe=False, limit=None, skip=None
    ) -> Union[List[ScopedKey], Dict[ScopedKey, TaskQueue]]:
        """Return all `TaskQueue`s corresponding to given `Scope`."""
        if return_gufe:
            taskqueues = {}
        else:
            taskqueues = []

        for scope in scopes:
            params = dict(
                return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict()
            )
            if return_gufe:
                taskqueues.update(self._query_resource("/taskqueues", params=params))
            else:
                taskqueues.extend(self._query_resource("/taskqueues", params=params))

        return taskqueues

    def claim_taskqueue_tasks(
        self, taskqueue: ScopedKey, claimant: str, count: int = 1
    ) -> Task:
        """Claim a `Task` from the specified `TaskQueue`"""
        data = dict(claimant=claimant, count=count)
        tasks = self._post_resource(f"taskqueues/{taskqueue}/claim", data)

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