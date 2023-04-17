"""
Base class for API clients --- :mod:`alchemiscale.base.client`
==============================================================

"""

import time
import random
from typing import List
import json
from urllib.parse import urljoin
from functools import wraps

import requests

from gufe.tokenization import GufeTokenizable, JSON_HANDLER

from ..models import Scope, ScopedKey
from ..storage.models import TaskHub, Task


def json_to_gufe(jsondata):
    return GufeTokenizable.from_dict(json.loads(jsondata, cls=JSON_HANDLER.decoder))


class AlchemiscaleBaseClientError(Exception):
    def __init__(self, *args, **kwargs):
        self.status_code = kwargs.pop("status_code", None)
        super().__init__(*args, **kwargs)


class AlchemiscaleConnectionError(Exception):
    ...


class AlchemiscaleBaseClient:
    """Base class for Alchemiscale API clients."""

    _exception = AlchemiscaleBaseClientError
    _retry_status_codes = [404, 502, 503, 504]

    def __init__(
        self,
        api_url: str,
        identifier: str,
        key: str,
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
    ):
        """Client class for interfacing with an alchemiscale API service.

        Parameters
        ----------
        api_url
            URL of the API to interact with.
        identifier
            Identifier for the identity used for authentication.
        key
            Credential for the identity used for authentication.
        max_retries
            Maximum number of times to retry a request. In the case the API
            service is unresponsive an exponential backoff is applied with
            retries until this number is reached. A ``self._exception`` is
            raised if retries are exhausted. If set to -1, retries will
            continue indefinitely until success.
        retry_base_seconds
            The base number of seconds to use for exponential backoff.
            Must be greater than 1.0.
        retry_max_seconds
            Maximum number of seconds to sleep between retries; avoids runaway
            exponential backoff while allowing for many retries.

        """
        self.api_url = api_url
        self.identifier = identifier
        self.key = key
        self.max_retries = max_retries

        if retry_base_seconds <= 1.0:
            raise ValueError("'retry_base_seconds' must be greater than 1.0")

        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds

        self._jwtoken = None
        self._headers = None

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.api_url}')"

    def _retry(f):
        """Automatically retry with exponential backoff if API service is
        unreachable or unable to service request.

        Will retry up to ``self.max_retries``, with the time between retries
        increasing by ``self.retry_base_seconds`` ** retries plus a random
        jitter scaled to ``self.retry_base_seconds``. ``self.retry_max_seconds`
        gives an upper bound to the time between retries.

        """

        @wraps(f)
        def _wrapper(self, *args, **kwargs):
            retries = 0
            while True:
                try:
                    return f(self, *args, **kwargs)
                except (self._exception, AlchemiscaleConnectionError) as e:
                    # if we are getting back HTTP errors and the status code is not
                    # one of those we want to retry on, just raise
                    if isinstance(e, self._exception):
                        if e.status_code not in self._retry_status_codes:
                            raise

                    if (self.max_retries != -1) and retries >= self.max_retries:
                        raise
                    retries += 1

                    # apply exponential backoff with random jitter
                    sleep_time = min(
                        self.retry_max_seconds
                        + self.retry_base_seconds * random.random(),
                        self.retry_base_seconds**retries
                        + self.retry_base_seconds * random.random(),
                    )
                    time.sleep(sleep_time)

        return _wrapper

    def _get_token(self):
        data = {"username": self.identifier, "password": self.key}

        url = urljoin(self.api_url, "/token")
        try:
            resp = requests.post(url, data=data)
        except requests.exceptions.RequestException as e:
            raise AlchemiscaleConnectionError(*e.args)

        if not 200 <= resp.status_code < 300:
            raise self._exception(
                f"Status Code {resp.status_code} : {resp.reason}",
                status_code=resp.status_code,
            )

        self._jwtoken = resp.json()["access_token"]
        self._headers = {
            "Authorization": f"Bearer {self._jwtoken}",
            "Content-type": "application/json",
        }

    def _use_token(f):
        @wraps(f)
        def _wrapper(self, *args, **kwargs):
            # if we don't have a token at all, get one
            if self._jwtoken is None:
                self._get_token()

            # execute our function
            # if we get an unauthorized exception, it may be that our token is
            # stale; get a new one if so
            # if it's any other status code, raise it
            try:
                return f(self, *args, **kwargs)
            except self._exception as e:
                if e.status_code == 401:
                    self._get_token()
                else:
                    raise

            # if we made it here, it means we got a fresh token;
            # try the function again with the token in place
            return f(self, *args, **kwargs)

        return _wrapper

    @_retry
    @_use_token
    def _query_resource(self, resource, params):
        url = urljoin(self.api_url, resource)
        try:
            resp = requests.get(url, params=params, headers=self._headers)
        except requests.exceptions.RequestException as e:
            raise AlchemiscaleConnectionError(*e.args)

        if not 200 <= resp.status_code < 300:
            raise self._exception(
                f"Status Code {resp.status_code} : {resp.reason}",
                status_code=resp.status_code,
            )

        if params.get("return_gufe"):
            return {
                ScopedKey.from_str(k): json_to_gufe(v) for k, v in resp.json().items()
            }
        else:
            return [ScopedKey.from_str(i) for i in resp.json()]

    @_retry
    @_use_token
    def _get_resource(self, resource, params=None):
        if params is None:
            params = {}

        url = urljoin(self.api_url, resource)
        try:
            resp = requests.get(url, params=params, headers=self._headers)
        except requests.exceptions.RequestException as e:
            raise AlchemiscaleConnectionError(*e.args)

        if not 200 <= resp.status_code < 300:
            raise self._exception(
                f"Status Code {resp.status_code} : {resp.reason} : {resp.text}",
                status_code=resp.status_code,
            )
        content = json.loads(resp.text, cls=JSON_HANDLER.decoder)
        return content

    @_retry
    @_use_token
    def _post_resource(self, resource, data):
        url = urljoin(self.api_url, resource)

        jsondata = json.dumps(data, cls=JSON_HANDLER.encoder)
        try:
            resp = requests.post(url, data=jsondata, headers=self._headers)
        except requests.exceptions.RequestException as e:
            raise AlchemiscaleConnectionError(*e.args)

        if not 200 <= resp.status_code < 300:
            raise self._exception(
                f"Status Code {resp.status_code} : {resp.reason}",
                status_code=resp.status_code,
            )

        return resp.json()

    @_retry
    def get_info(self):
        return self._get_resource("/info")

    @_retry
    @_use_token
    def _api_check(self):
        # Check if the API is up and running and can reach services
        self._get_resource("/check")
