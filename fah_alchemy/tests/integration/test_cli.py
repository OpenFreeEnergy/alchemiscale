import pytest
from click.testing import CliRunner

import contextlib
import os

from grolt import Neo4jService
from py2neo import graph


# based on https://stackoverflow.com/a/34333710
@contextlib.contextmanager
def set_env_vars(**env):
    old_env = dict(os.environ)
    try:
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


@pytest.mark.parametrize('vars_from', ['env', 'cli'])
def test_database_init(uri, vars_from):
    # ensure the database is empty
    graph = Graph(uri)
    graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

    # run the CLI
    runner = CliRunner()
