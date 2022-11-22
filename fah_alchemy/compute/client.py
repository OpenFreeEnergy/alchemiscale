"""Client for interacting with compute API.

"""

from typing import List, Tuple, Optional
import json
from urllib.parse import urljoin
from functools import wraps

import requests
from requests.auth import HTTPBasicAuth

from gufe.tokenization import GufeTokenizable
from gufe import Transformation
from gufe.protocols import ProtocolDAGResult

from ..base.client import FahAlchemyBaseClient, FahAlchemyBaseClientError
from ..models import Scope, ScopedKey
from ..storage.models import TaskQueue, Task


class FahAlchemyComputeClientError(FahAlchemyBaseClientError):
    ...


class FahAlchemyComputeClient(FahAlchemyBaseClient):
    """Client for compute service interaction with compute API service."""

    _exception = FahAlchemyComputeClientError

    def query_taskqueues(
        self, scope: Scope, return_gufe=False, limit=None, skip=None
    ) -> List[TaskQueue]:
        """Return all `TaskQueue`s corresponding to given `Scope`."""
        params = dict(return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict())
        taskqueues = self._query_resource("/taskqueues", params=params)

        return taskqueues

    def get_taskqueue_tasks(self, taskqueue: ScopedKey) -> List[Task]:
        """Get list of `Task`s for the given `TaskQueue`."""
        self._get_resource(f"taskqueues/{taskqueue}/tasks")

    def claim_taskqueue_tasks(
        self, taskqueue: ScopedKey, claimant: str, count: int = 1
    ) -> Task:
        """Claim a `Task` from the specified `TaskQueue`"""
        data = dict(claimant=claimant, count=count)
        tasks = self._post_resource(f"taskqueues/{taskqueue}/claim", data)

        return [ScopedKey.from_str(t) if t is not None else None for t in tasks]

    def get_task_transformation(self, task: ScopedKey) -> Tuple[Transformation, Optional[ProtocolDAGResult]]:

        transformation, protocoldagresult = self._get_resource(f"tasks/{task}/transformation", 
                                                                 {}, 
                                                                 return_gufe=False)

        return (GufeTokenizable.from_dict(transformation),
                GufeTokenizable.from_dict(protocoldagresult) if protocoldagresult else None)

    def set_task_result(self, task: ScopedKey, protocoldagresult: ProtocolDAGResult) -> ScopedKey:
        data = dict(protocoldagresult=protocoldagresult.to_dict())
        pdr_sk = self._post_resource(f"tasks/{task}/result", data)

        return ScopedKey.from_str(pdr_sk)

