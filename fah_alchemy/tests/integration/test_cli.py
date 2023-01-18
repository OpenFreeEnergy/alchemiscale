import pytest
from click.testing import CliRunner

import contextlib
import os
import traceback

import requests
from fastapi import FastAPI
from fah_alchemy.tests.integration.utils import running_service

from fah_alchemy.cli import get_settings_from_options, cli, ApiApplication
from fah_alchemy.security.auth import hash_key, authenticate, AuthenticationError
from fah_alchemy.security.models import (
    CredentialedUserIdentity,
    CredentialedComputeIdentity,
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
    workers = 1
    host = "127.0.0.1"
    port = 50100
    app = ApiApplication.from_parameters(toyapp, workers, host, port)

    expected_ping = {"Hello": "World"}
    with running_service(app.run, port, args=tuple()):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


def test_api(n4js, s3os):
    workers = 2
    host = "127.0.0.1"
    port = 50100
    command = ["api"]
    api_opts = ["--workers", workers, "--host", host, "--port", port]
    db_opts = [
        "--url",
        "bolt://localhost:7687",
        "--user",
        "neo4j",
        "--password",
        "password",
    ]
    s3_opts = [
        "--access-key-id",
        "test-key-id",
        "--secret-access-key",
        "test-key",
        "--session-token",
        "test-session-token",
        "--s3-bucket",
        "test-bucket",
        "--s3-prefix",
        "test-prefix",
        "--default-region",
        "us-east-1",
    ]
    jwt_opts = []  # leaving empty, we have default behavior for these

    expected_ping = {"api": "FahAlchemyAPI"}

    runner = CliRunner()
    with running_service(
        runner.invoke, port, (cli, command + api_opts + db_opts + s3_opts + jwt_opts)
    ):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


def test_compute_api(n4js, s3os):
    workers = 2
    host = "127.0.0.1"
    port = 50100
    command = ["compute", "api"]
    api_opts = ["--workers", workers, "--host", host, "--port", port]
    db_opts = [
        "--url",
        "bolt://localhost:7687",
        "--user",
        "neo4j",
        "--password",
        "password",
    ]
    s3_opts = [
        "--access-key-id",
        "test-key-id",
        "--secret-access-key",
        "test-key",
        "--session-token",
        "test-session-token",
        "--s3-bucket",
        "test-bucket",
        "--s3-prefix",
        "test-prefix",
        "--default-region",
        "us-east-1",
    ]
    jwt_opts = []  # leaving empty, we have default behavior for these

    expected_ping = {"api": "FahAlchemyComputeAPI"}

    runner = CliRunner()
    with running_service(
        runner.invoke, port, (cli, command + api_opts + db_opts + s3_opts + jwt_opts)
    ):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


@pytest.mark.parametrize(
    "cli_vars",
    [
        {},  # no cli options; all from env
        {
            "NEO4J_URL": "https://foo",
            "NEO4J_USER": "me",
            "NEO4J_PASS": "correct-horse-battery-staple",
        },  # all CLI options given (test without env vars)
        {
            "NEO4J_URL": "https://baz",
        },  # some CLI options given (test that we override)
    ],
)
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


@pytest.mark.parametrize(
    "identity_type",
    [("user", CredentialedUserIdentity), ("compute", CredentialedComputeIdentity)],
)
def test_identity_add(n4js_fresh, identity_type):
    n4js = n4js_fresh
    identity_type_str, identity_type_cls = identity_type
    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        ident = "bill"
        key = "and ted"
        result = runner.invoke(
            cli,
            [
                "identity",
                "add",
                "--identity-type",
                identity_type_str,
                "--identifier",
                ident,
                "--key",
                key,
            ],
        )
        assert click_success(result)

        cred = authenticate(n4js, identity_type_cls, ident, key)
        assert cred


@pytest.mark.parametrize(
    "identity_type",
    [("user", CredentialedUserIdentity), ("compute", CredentialedComputeIdentity)],
)
def test_identity_remove(n4js_fresh, identity_type):
    n4js = n4js_fresh
    identity_type_str, identity_type_cls = identity_type
    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        ident = "bill"
        key = "and ted"

        identity = identity_type_cls(
            identifier=ident,
            hashed_key=hash_key(key),
        )

        n4js.create_credentialed_entity(identity)

        result = runner.invoke(
            cli,
            [
                "identity",
                "remove",
                "--identity-type",
                identity_type_str,
                "--identifier",
                ident,
            ],
        )
        assert click_success(result)

        with pytest.raises(KeyError, match="No such object in database"):
            cred = n4js.get_credentialed_entity(ident, identity_type_cls)


def test_identity_list(n4js_fresh):
    n4js = n4js_fresh
    env_vars = {
        "NEO4J_URL": n4js.graph.service.uri,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        identities = ("bill", "ted", "napoleon")
        for ident in identities:
            key = "a string for a key"

            identity = CredentialedUserIdentity(
                identifier=ident,
                hashed_key=hash_key(key),
            )

            n4js.create_credentialed_entity(identity)

        result = runner.invoke(
            cli,
            [
                "identity",
                "list",
                "--identity-type",
                "user",
            ],
        )
        assert click_success(result)
        for ident in identities:
            assert ident in result.output
