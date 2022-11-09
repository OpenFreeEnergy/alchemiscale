"""Client for interacting with compute API.

"""

from typing import List
import json
from urllib.parse import urljoin

import requests

from gufe.tokenization import GufeTokenizable

from ..models import Scope, ScopedKey
from ..storage.models import TaskQueue, Task


class FahAlchemyComputeClient:
    ...

    def __init__(self, 
                 compute_api_url,
                 compute_api_key):

        self.compute_api_url = compute_api_url
        self.compute_api_key = compute_api_key

    def __repr__(self):
        
        ret = "FahAlchemyComputeClient('{}')".format(
                self.compute_api_url
        )
        return ret

    def _query_resource(self, resource, params):

        url = urljoin(self.compute_api_url, resource)
        resp = requests.get(url, params=params)

        # iterate through response
        return [GufeTokenizable.from_dict(i) for i in resp.json()]

    def _get_resource(self, resource, params):

        url = urljoin(self.compute_api_url, resource)
        resp = requests.get(url, params=params)

        return GufeTokenizable.from_dict(resp.json())

    def query_taskqueues(self, scope: Scope, limit=None, skip=None) -> List[TaskQueue]:
        """Return all `TaskQueue`s corresponding to given `Scope`.

        """
        params = dict(limit=limit, skip=skip, **scope.dict())
        taskqueues = self._query_resource('/taskqueues', params=params)

        return taskqueues

    def get_taskqueue_tasks(self, scoped_key: ScopedKey) -> List[Task]:
        """Get list of `Task`s for the given `TaskQueue`.

        """
        self._get_resource(f'taskqueues/{scoped_key}/tasks')


    def claim_taskqueue_task(self, scoped_key: ScopedKey, task_scoped_key=None) -> Task:
        """Claim a `Task` from the specified `TaskQueue`

        """
        ...

        self._patch_resource(f'taskqueues/{scoped_key}/tasks')
