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

        if params.get('return_gufe'):
            return {ScopedKey.from_str(k): GufeTokenizable.from_dict(v) for k, v in resp.json().items()}
        else:
            return [ScopedKey.from_str(i) for i in resp.json()]

    def _get_resource(self, resource, params):

        url = urljoin(self.compute_api_url, resource)
        resp = requests.get(url, params=params)

        return GufeTokenizable.from_dict(resp.json())

    def _post_resource(self, resource, data):
        url = urljoin(self.compute_api_url, resource)

        jsondata = json.dumps(data)
        resp = requests.post(
                url, 
                data=jsondata,
                headers={"Content-type": "application/json"})

        return resp.json()

    def query_taskqueues(self, scope: Scope, return_gufe=False, limit=None, skip=None) -> List[TaskQueue]:
        """Return all `TaskQueue`s corresponding to given `Scope`.

        """
        params = dict(return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict())
        taskqueues = self._query_resource('/taskqueues', params=params)

        return taskqueues

    def get_taskqueue_tasks(self, taskqueue: ScopedKey) -> List[Task]:
        """Get list of `Task`s for the given `TaskQueue`.

        """
        self._get_resource(f'taskqueues/{taskqueue}/tasks')


    def claim_taskqueue_tasks(self, taskqueue: ScopedKey, claimant: str, count: int = 1) -> Task:
        """Claim a `Task` from the specified `TaskQueue`

        """
        data = dict(claimant=claimant, count=count)
        tasks = self._post_resource(f'taskqueues/{taskqueue}/claim', data)

        return [ScopedKey.from_str(t['_scoped_key']) if t is not None else None for t in tasks]

