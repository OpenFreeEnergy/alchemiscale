import pytest
from copy import copy
from time import sleep
from multiprocessing import Process

import uvicorn
import requests

from fah_alchemy.settings import get_base_api_settings
from fah_alchemy.base.api import get_n4js_depends, get_s3os_depends
from fah_alchemy.interface import api, client

from fah_alchemy.tests.integration.interface.utils import get_user_settings_override
from fah_alchemy.tests.integration.utils import running_service


## user client


@pytest.fixture(scope="module")
def user_api(s3os):
    def get_s3os_override():
        return s3os

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_base_api_settings] = get_user_settings_override
    api.app.dependency_overrides[get_s3os_depends] = get_s3os_override
    yield api.app
    api.app.dependency_overrides = overrides


def run_server(fastapi_app, settings):
    uvicorn.run(
        fastapi_app,
        host=settings.FA_API_HOST,
        port=settings.FA_API_PORT,
        log_level=settings.FA_API_LOGLEVEL,
    )


@pytest.fixture(scope="module")
def uvicorn_server(user_api):
    settings = get_user_settings_override()
    with running_service(
        run_server, port=settings.FA_API_PORT, args=(user_api, settings)
    ):
        yield


@pytest.fixture(scope="module")
def user_client(uvicorn_server, user_identity):

    return client.FahAlchemyClient(
        api_url="http://127.0.0.1:8000/",
        identifier=user_identity["identifier"],
        key=user_identity["key"],
    )


@pytest.fixture(scope="module")
def user_client_wrong_credential(uvicorn_server, user_identity):

    return client.FahAlchemyClient(
        api_url="http://127.0.0.1:8000/",
        identifier=user_identity["identifier"],
        key="incorrect credential",
    )
