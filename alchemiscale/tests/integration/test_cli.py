import pytest
from click.testing import CliRunner

import time
import contextlib
import os
import traceback
import multiprocessing
from datetime import datetime, timedelta

import yaml

import requests
from fastapi import FastAPI
from alchemiscale.tests.integration.utils import running_service

from alchemiscale.cli import get_settings_from_options, cli
from alchemiscale.cli_utils import ApiApplication
from alchemiscale.models import Scope
from alchemiscale.security.auth import hash_key, authenticate, AuthenticationError
from alchemiscale.security.models import (
    CredentialedUserIdentity,
    CredentialedComputeIdentity,
)
from alchemiscale.settings import Neo4jStoreSettings
from alchemiscale.storage.statestore import Neo4JStoreError


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

    expected_ping = {"api": "AlchemiscaleAPI"}

    runner = CliRunner()
    with running_service(
        runner.invoke, port, (cli, command + api_opts + db_opts + s3_opts + jwt_opts)
    ):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


@pytest.fixture
def compute_api_args():
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

    return host, port, (command + api_opts + db_opts + s3_opts + jwt_opts)


@pytest.fixture
def compute_service_config(compute_api_args):
    host, port, _ = compute_api_args

    config = {
        "init": {
            "api_url": f"http://{host}:{port}",
            "identifier": "test-compute-user",
            "key": "test-comute-user-key",
            "name": "test-compute-service",
            "shared_basedir": "./shared",
            "scratch_basedir": "./scratch",
            "loglevel": "INFO",
        },
        "start": {"max_time": None},
    }

    return config


def test_compute_api(n4js, s3os, compute_api_args):
    host, port, args = compute_api_args

    expected_ping = {"api": "AlchemiscaleComputeAPI"}

    runner = CliRunner()
    with running_service(runner.invoke, port, (cli, args)):
        response = requests.get(f"http://{host}:{port}/ping")

    assert response.status_code == 200
    assert response.json() == expected_ping


def test_compute_synchronous(
    n4js_fresh, s3os, compute_api_args, compute_service_config, tmpdir
):
    host, port, args = compute_api_args
    n4js = n4js_fresh

    # create compute identity; add all scope access
    identity = CredentialedComputeIdentity(
        identifier=compute_service_config["init"]["identifier"],
        hashed_key=hash_key(compute_service_config["init"]["key"]),
    )

    n4js.create_credentialed_entity(identity)
    n4js.add_scope(identity.identifier, CredentialedComputeIdentity, Scope())

    # start up compute API
    runner = CliRunner()
    with running_service(runner.invoke, port, (cli, args)):
        # start up compute service
        with tmpdir.as_cwd():
            command = ["compute", "synchronous"]
            opts = ["--config-file", "config.yaml"]

            with open("config.yaml", "w") as f:
                yaml.dump(compute_service_config, f)

            multiprocessing.set_start_method("fork", force=True)
            proc = multiprocessing.Process(
                target=runner.invoke, args=(cli, command + opts), daemon=True
            )
            proc.start()

            q = f"""
            match (csreg:ComputeServiceRegistration)
            where csreg.identifier =~ "{compute_service_config['init']['name']}.*"
            return csreg
            """
            while True:
                csreg = n4js.execute_query(q)
                if not csreg.records:
                    time.sleep(1)
                else:
                    break

            assert csreg.records[0]["csreg"][
                "registered"
            ] > datetime.utcnow() - timedelta(seconds=30)

            proc.terminate()
            proc.join()


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


def test_database_init(n4js_fresh):
    n4js = n4js_fresh
    # ensure the database is empty
    n4js.reset()

    with pytest.raises(Neo4JStoreError):
        n4js.check()

    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
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
    n4js.assemble_network(network_tyk2, scope_test)

    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
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
    n4js.assemble_network(network_tyk2, scope_test)

    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }

    # run the CLI
    runner = CliRunner()
    with set_env_vars(env_vars):
        result = runner.invoke(cli, ["database", "reset"])
        assert click_success(result)

    q = "MATCH (n) RETURN n"
    assert not n4js.execute_query(q).records

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
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
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
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
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
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
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


def test_scope_list(n4js_fresh):
    n4js = n4js_fresh
    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        ident = "bill"
        key = "a string for a key"

        identity = CredentialedUserIdentity(
            identifier=ident,
            hashed_key=hash_key(key),
        )

        n4js.create_credentialed_entity(identity)
        n4js.add_scope(
            "bill", CredentialedUserIdentity, Scope.from_str("org1-campaign2-project3")
        )
        n4js.add_scope(
            "bill", CredentialedUserIdentity, Scope.from_str("org4-campaign5-project6")
        )
        n4js.add_scope("bill", CredentialedUserIdentity, Scope.from_str("org7-*-*"))
        result = runner.invoke(
            cli,
            [
                "identity",
                "list-scope",
                "--identity-type",
                "user",
                "--identifier",
                ident,
            ],
        )
        assert click_success(result)
        assert "org1-campaign2-project3" in result.output
        assert "org4-campaign5-project6" in result.output
        assert "org7-*-*" in result.output


@pytest.mark.parametrize(
    "scopes",
    [
        ("org1-campaign2-project3",),
        (
            "org1-campaign2-project3",
            "org1-campaign2-project4",
        ),
    ],
)
def test_scope_add(n4js_fresh, scopes):
    n4js = n4js_fresh
    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        ident = "bill"
        key = "a string for a key"

        identity = CredentialedUserIdentity(
            identifier=ident,
            hashed_key=hash_key(key),
        )

        n4js.create_credentialed_entity(identity)

        scopes_cli = []
        for scope in scopes:
            scopes_cli.append("--scope")
            scopes_cli.append(scope)

        result = runner.invoke(
            cli,
            [
                "identity",
                "add-scope",
                "--identity-type",
                "user",
                "--identifier",
                ident,
            ]
            + scopes_cli,
        )
        assert click_success(result)
        scopes_ = n4js.list_scopes("bill", CredentialedUserIdentity)
        assert len(scopes_) == len(scopes)
        assert set(scopes_) == set([Scope.from_str(scope) for scope in scopes])


@pytest.mark.parametrize(
    "scopes_remove",
    [
        ("org1-campaign2-project3",),
        (
            "org1-campaign2-project3",
            "org1-campaign2-project4",
        ),
    ],
)
def test_scope_remove(n4js_fresh, scopes_remove):
    n4js = n4js_fresh
    env_vars = {
        "NEO4J_URL": "bolt://" + str(n4js.graph.address),
        "NEO4J_USER": "neo4j",
        "NEO4J_PASS": "password",
    }
    runner = CliRunner()
    with set_env_vars(env_vars):
        ident = "bill"
        key = "a string for a key"

        identity = CredentialedUserIdentity(
            identifier=ident,
            hashed_key=hash_key(key),
        )
        n4js.create_credentialed_entity(identity)

        n4js.add_scope(
            ident, CredentialedUserIdentity, Scope.from_str("org1-campaign2-project3")
        )
        n4js.add_scope(
            ident, CredentialedUserIdentity, Scope.from_str("org1-campaign2-project4")
        )
        n4js.add_scope(
            ident, CredentialedUserIdentity, Scope.from_str("org4-campaign5-project6")
        )

        scopes_cli = []
        for scope in scopes_remove:
            scopes_cli.append("--scope")
            scopes_cli.append(scope)

        result = runner.invoke(
            cli,
            [
                "identity",
                "remove-scope",
                "--identity-type",
                "user",
                "--identifier",
                ident,
            ]
            + scopes_cli,
        )
        assert click_success(result)
        scopes = n4js.list_scopes("bill", CredentialedUserIdentity)
        scope_strs = [str(s) for s in scopes]
        for scope in scopes_remove:
            assert scope not in scope_strs
        assert "org4-campaign5-project6" in scope_strs
