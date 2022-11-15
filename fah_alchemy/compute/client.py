"""Client for interacting with compute API.

"""

from typing import List
import json
from urllib.parse import urljoin
from functools import wraps

import requests
from requests.auth import HTTPBasicAuth

from gufe.tokenization import GufeTokenizable

from ..models import Scope, ScopedKey
from ..storage.models import TaskQueue, Task


class FahAlchemyComputeClientError(Exception):
    ...


class FahAlchemyComputeClient:
    ...

    def __init__(self, 
                 compute_api_url,
                 identifier,
                 key,
                 max_retries=5
        ):

        self.compute_api_url = compute_api_url
        self.identifier = identifier
        self.key = key
        self.max_retries = max_retries

        self._jwtoken = None
        self._headers = None

    def __repr__(self):
        
        ret = "FahAlchemyComputeClient('{}')".format(
                self.compute_api_url
        )
        return ret

    def _get_token(self):

        data = {'username': self.identifier,
                'password': self.key}

        url = urljoin(self.compute_api_url, '/token')
        resp = requests.post(url, data=data)

        self._jwtoken = resp.json()['access_token']
        self._headers = {"Authorization": f"Bearer {self._jwtoken}",
                        "Content-type": "application/json"}

    def _use_token(f):

        @wraps(f)
        def _wrapper(self, *args, **kwargs):
            if self._jwtoken is None:
                self._get_token()

            retries = 0
            while True:
                try:
                    return f(self, *args, **kwargs)
                except FahAlchemyComputeClientError:
                    self._get_token()
                    if retries >= self.max_retries:
                        raise
                    retries += 1

        return _wrapper
    
    @_use_token
    def _query_resource(self, resource, params):

        url = urljoin(self.compute_api_url, resource)
        resp = requests.get(url, 
                            params=params,
                            headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise FahAlchemyComputeClientError(f"Status Code {resp.status_code} : {resp.reason}")

        if params.get('return_gufe'):
            return {ScopedKey.from_str(k): GufeTokenizable.from_dict(v) for k, v in resp.json().items()}
        else:
            return [ScopedKey.from_str(i) for i in resp.json()]

    @_use_token
    def _get_resource(self, resource, params):

        url = urljoin(self.compute_api_url, resource)
        resp = requests.get(url, 
                            params=params,
                            headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise FahAlchemyComputeClientError(f"Status Code {resp.status_code} : {resp.reason}")

        return GufeTokenizable.from_dict(resp.json())

    @_use_token
    def _post_resource(self, resource, data):
        url = urljoin(self.compute_api_url, resource)

        jsondata = json.dumps(data)
        resp = requests.post(
                url, 
                data=jsondata,
                headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise FahAlchemyComputeClientError(f"Status Code {resp.status_code} : {resp.reason}")

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

