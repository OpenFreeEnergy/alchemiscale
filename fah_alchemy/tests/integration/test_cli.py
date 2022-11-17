import pytest
from click.testing import CliRunner

import contextlib
import os

from grolt import Neo4jService
from py2neo import Graph

from fah_alchemy.cli import get_settings_from_options, cli


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


@pytest.mark.parametrize('cli_vars', [
    {},  # no cli options; all from env
    {
        "NEO4J_URL": "https://foo",
        "NEO4J_USER": "me",
        "NEO4J_PASS": "correct-horse-battery-staple",
        "JWT_SECRET_KEY": "setec-astronomy",
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
        "JWT_SECRET_KEY": "too-many-secrets",
    }
    # if we give all by CLI, we don't need the env_vars
    context_vars = {} if len(cli_vars) == 4 else env_vars

    expected = env_vars | cli_vars
    expected["NEO4J_DBNAME"] = "neo4j"  # leave as default

    kwargs = {k: cli_vars[k] if k in cli_vars else None
              for k in expected}
    with set_env_vars(context_vars):
        settings = get_settings_from_options(kwargs)
        settings_dict = settings.dict()

    for key in expected:
        assert expected[key] == settings_dict[key]


def test_database_init(uri):
    # ensure the database is empty
    graph = Graph(uri)
    graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

    env_vars = ...

    # run the CLI
    runner = CliRunner()
