import pytest
from click.testing import CliRunner

import contextlib
import os
import traceback

import requests
from fastapi import FastAPI
from fah_alchemy.tests.integration.utils import running_service

from fah_alchemy.cli import (
    get_settings_from_options, cli, ApiApplication
)
from fah_alchemy.settings import Neo4jStoreSettings
from fah_alchemy.storage.statestore import Neo4JStoreError


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


# based on https://stackoverflow.com/a/34333710
@contextlib.contextmanager
def set_env_vars(env):
    old_env = dict(os.environ)
    try:
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


toyapp = FastAPI()
@toyapp.get("/ping")
def read_root():
    return {"Hello": "World"}


def test_api_application():
    # this checks that the gunicorn BaseApplication subclass works correctly
    # with a FastAPI app
    workers = 2
    host = "127.0.0.1"
    port = 50100
    app = ApiApplication.from_parameters(toyapp, workers, host, port)

    expected_ping = {"Hello": "World"}
    with running_service(app.run, port, args=tuple()):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


def test_api():
    workers = 2
    host = "127.0.0.1"
    port = 50100
    invocation = ['api', '--workers', workers, '--host', host, '--port',
                  port]
    runner = CliRunner()
    with running_service(runner.invoke, port, (cli, invocation)):
        response = requests



@pytest.mark.parametrize('cli_vars', [
    {},  # no cli options; all from env
    {
        "NEO4J_URL": "https://foo",
        "NEO4J_USER": "me",
        "NEO4J_PASS": "correct-horse-battery-staple",
    },  # all CLI options given (test without env vars)
    {
        "NEO4J_URL": "https://baz",
    },  # some CLI options given (test that we override)
])
def test_get_settings_from_options(cli_vars):
    env_vars = {
        "NEO4J_URL": "https://bar",
        "NEO4J_USER": "you",
        "NEO4J_PASS": "Tr0ub4dor&3",
    }
    # if we give all by CLI, we don't need the env_vars
    context_vars = {} if len(cli_vars) == 4 else env_vars

    expected = env_vars | cli_vars
    expected["NEO4J_DBNAME"] = "neo4j"  # leave as default

    kwargs = {k: cli_vars[k] if k in cli_vars else None for k in expected}
    with set_env_vars(context_vars):
        settings = get_settings_from_options(kwargs, Neo4jStoreSettings)
        settings_dict = settings.dict()

    for key in expected:
        assert expected[key] == settings_dict[key]


def test_database_init(n4js):
    # ensure the database is empty
    n4js.graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

    with pytest.raises(Neo4JStoreError):
        n4js.check()

    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }

    # run the CLI
    runner = CliRunner()
    with set_env_vars(env_vars):
        result = runner.invoke(cli, ["database", "init"])
        assert click_success(result)

    assert n4js.check() is None


def test_database_check(n4js_fresh, network_tyk2, scope_test):
    n4js = n4js_fresh

    # set starting contents
    n4js.create_network(network_tyk2, scope_test)

    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }

    # run the CLI
    runner = CliRunner()
    with set_env_vars(env_vars):
        result = runner.invoke(cli, ["database", "check"])
        assert click_success(result)

        n4js.reset()

        result = runner.invoke(cli, ["database", "check"])
        assert not click_success(result)

        n4js.initialize()

        result = runner.invoke(cli, ["database", "check"])
        assert click_success(result)


def test_database_reset(n4js_fresh, network_tyk2, scope_test):
    n4js = n4js_fresh

    # set starting contents
    n4js.create_network(network_tyk2, scope_test)

    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }

    # run the CLI
    runner = CliRunner()
    with set_env_vars(env_vars):
        result = runner.invoke(cli, ["database", "reset"])
        assert click_success(result)

    assert n4js.graph.run("MATCH (n) WHERE NOT n:NOPE RETURN n").to_subgraph() is None

    with pytest.raises(Neo4JStoreError):
        n4js.check()
