import pytest
from copy import copy

import uvicorn

from alchemiscale.settings import get_base_api_settings
from alchemiscale.base.api import get_s3os_depends
from alchemiscale.interface import api, client

from alchemiscale.tests.integration.interface.utils import get_user_settings_override
from alchemiscale.tests.integration.utils import running_service


## user client


@pytest.fixture(scope="module")
def user_api(n4jstore_settings, s3os_server, compute_api_port):
    def get_s3os_override():
        return s3os_server

    def get_settings_override():
        return get_user_settings_override(port=compute_api_port)

    overrides = copy(api.app.dependency_overrides)

    api.app.dependency_overrides[get_base_api_settings] = get_settings_override
    api.app.dependency_overrides[get_s3os_depends] = get_s3os_override
    yield api.app
    api.app.dependency_overrides = overrides


def run_server(fastapi_app, settings):
    uvicorn.run(
        fastapi_app,
        host=settings.ALCHEMISCALE_API_HOST,
        port=settings.ALCHEMISCALE_API_PORT,
        log_level=settings.ALCHEMISCALE_API_LOGLEVEL,
    )


@pytest.fixture(scope="module")
def uvicorn_server(user_api, compute_api_port):
    settings = get_user_settings_override(port=compute_api_port)
    with running_service(
        run_server, port=settings.ALCHEMISCALE_API_PORT, args=(user_api, settings)
    ):
        yield


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("alchemiscale-cache")
    return cache_dir


@pytest.fixture(scope="module")
def user_client(uvicorn_server, user_identity, cache_dir, compute_api_port):
    test_client = client.AlchemiscaleClient(
        api_url=f"http://127.0.0.1:{compute_api_port}/",
        identifier=user_identity["identifier"],
        key=user_identity["key"],
        cache_directory=cache_dir,
        cache_size_limit=int(1073741824 / 4),
    )
    test_client._cache.stats(enable=True, reset=True)

    return test_client


@pytest.fixture(scope="module")
def user_client_no_cache(uvicorn_server, user_identity, cache_dir, compute_api_port):
    test_client = client.AlchemiscaleClient(
        api_url=f"http://127.0.0.1:{compute_api_port}/",
        identifier=user_identity["identifier"],
        key=user_identity["key"],
        use_local_cache=False,
    )

    return test_client


@pytest.fixture
def _client_setenv(monkeypatch):
    monkeypatch.setenv("ALCHEMISCALE_URL", "http://env.example.com")
    monkeypatch.setenv("ALCHEMISCALE_ID", "env_id")
    monkeypatch.setenv("ALCHEMISCALE_KEY", "env_key")


@pytest.fixture
def user_client_from_env(_client_setenv):
    return client.AlchemiscaleClient()


@pytest.fixture(scope="module")
def user_client_wrong_credential(uvicorn_server, user_identity, compute_api_port):
    return client.AlchemiscaleClient(
        api_url=f"http://127.0.0.1:{compute_api_port}/",
        identifier=user_identity["identifier"],
        key="incorrect credential",
    )
