"""Base class for API clients.

"""

from typing import List
import json
from urllib.parse import urljoin
from functools import wraps
from fastapi import status, HTTPException

import requests
from requests.auth import HTTPBasicAuth

from gufe.tokenization import GufeTokenizable, JSON_HANDLER

from ..models import Scope, ScopedKey
from ..storage.models import TaskQueue, Task


class FahAlchemyBaseClientError(Exception):
    ...


class FahAlchemyBaseClient:
    """Base class for FahAlchemy API clients."""

    _exception = FahAlchemyBaseClientError

    def __init__(self, api_url, identifier, key, max_retries=5):

        self.api_url = api_url
        self.identifier = identifier
        self.key = key
        self.max_retries = max_retries

        self._jwtoken = None
        self._headers = None

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.api_url}')"

    def _get_token(self):

        data = {"username": self.identifier, "password": self.key}

        url = urljoin(self.api_url, "/token")
        resp = requests.post(url, data=data)

        if not 200 <= resp.status_code < 300:
            raise self._exception(f"Status Code {resp.status_code} : {resp.reason}")

        self._jwtoken = resp.json()["access_token"]
        self._headers = {
            "Authorization": f"Bearer {self._jwtoken}",
            "Content-type": "application/json",
        }

    def _use_token(f):
        @wraps(f)
        def _wrapper(self, *args, **kwargs):
            if self._jwtoken is None:
                self._get_token()

            retries = 0
            while True:
                try:
                    return f(self, *args, **kwargs)
                except self._exception:
                    self._get_token()
                    if retries >= self.max_retries:
                        raise
                    retries += 1

        return _wrapper

    @_use_token
    def _query_resource(self, resource, params):

        url = urljoin(self.api_url, resource)
        resp = requests.get(url, params=params, headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise self._exception(f"Status Code {resp.status_code} : {resp.reason}")

        if params.get("return_gufe"):
            return {
                ScopedKey.from_str(k): GufeTokenizable.from_dict(v)
                for k, v in resp.json().items()
            }
        else:
            return [ScopedKey.from_str(i) for i in resp.json()]

    @_use_token
    def _get_resource(self, resource, params, return_gufe=True):

        url = urljoin(self.api_url, resource)
        resp = requests.get(url, params=params, headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise self._exception(f"Status Code {resp.status_code} : {resp.reason}")

        content = json.loads(resp.text, cls=JSON_HANDLER.decoder)

        if return_gufe:
            return GufeTokenizable.from_dict(content)
        else:
            return content

    @_use_token
    def _post_resource(self, resource, data):
        url = urljoin(self.api_url, resource)

        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)
        resp = requests.post(url, data=jsondata, headers=self._headers)

        if not 200 <= resp.status_code < 300:
            raise self._exception(f"Status Code {resp.status_code} : {resp.reason}")

        return resp.json()

    def get_info(self):
        return self._get_resource("/info", params={}, return_gufe=False)

    @_use_token
    def _api_check(self):
        # Check if the API is up and running and can reach services
        self._get_resource("/check", params={}, return_gufe=False)