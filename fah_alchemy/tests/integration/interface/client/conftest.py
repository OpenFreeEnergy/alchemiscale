import pytest
from copy import copy
from time import sleep
from multiprocessing import Process

import uvicorn
import requests

from fah_alchemy.settings import get_jwt_settings
from fah_alchemy.base.api import get_n4js_depends, get_s3os_depends
from fah_alchemy.interface import api, client

from fah_alchemy.tests.integration.interface.utils import get_user_settings_override


## user client


@pytest.fixture(scope="module")
def user_api(n4js, s3os):
    def get_n4js_override():
        return n4js

    def get_s3os_override():
        return s3os

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_n4js_depends] = get_n4js_override
    api.app.dependency_overrides[get_s3os_depends] = get_s3os_override
    api.app.dependency_overrides[get_jwt_settings] = get_user_settings_override
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
    proc = Process(target=run_server, args=(user_api, settings), daemon=True)
    proc.start()

    timeout = True
    for _ in range(40):
        try:
            ping = requests.get(f"http://127.0.0.1:8000/ping")
            ping.raise_for_status()
        except IOError:
            sleep(0.25)
            continue
        timeout = False
        break
    if timeout:
        raise RuntimeError("The test server could not be reached.")

    yield

    proc.kill()  # Cleanup after test


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
