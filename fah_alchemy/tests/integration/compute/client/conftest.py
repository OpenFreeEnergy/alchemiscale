import pytest
from copy import copy
from time import sleep

import uvicorn
import requests

from fah_alchemy.settings import ComputeAPISettings, get_jwt_settings
from fah_alchemy.storage import Neo4jStore, get_n4js
from fah_alchemy.compute import api, client
from fah_alchemy.security.models import CredentialedComputeIdentity, TokenData
from fah_alchemy.security.auth import hash_key
from fah_alchemy.base.api import get_token_data_depends

from fah_alchemy.tests.integration.compute.utils import get_compute_settings_override
from fah_alchemy.tests.integration.utils import running_service


## compute client


@pytest.fixture(scope="module")
def compute_api(n4js):
    def get_n4js_override():
        return n4js

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_n4js] = get_n4js_override
    api.app.dependency_overrides[get_jwt_settings] = get_compute_settings_override
    yield api.app
    api.app.dependency_overrides = overrides


def run_server(fastapi_app, settings):
    uvicorn.run(
        fastapi_app,
        host=settings.FA_COMPUTE_API_HOST,
        port=settings.FA_COMPUTE_API_PORT,
        log_level=settings.FA_COMPUTE_API_LOGLEVEL,
    )


@pytest.fixture(scope="module")
def uvicorn_server(compute_api):
    settings = get_compute_settings_override()
    with running_service(run_server, port=settings.FA_COMPUTE_API_PORT, args=(compute_api, settings)):
        yield 


@pytest.fixture(scope="module")
def compute_client(uvicorn_server, compute_identity):

    return client.FahAlchemyComputeClient(
        api_url="http://127.0.0.1:8000/",
        identifier=compute_identity["identifier"],
        key=compute_identity["key"],
    )


@pytest.fixture(scope="module")
def compute_client_wrong_credential(uvicorn_server, compute_identity):

    return client.FahAlchemyComputeClient(
        api_url="http://127.0.0.1:8000/",
        identifier=compute_identity["identifier"],
        key="wrong credential",
    )
