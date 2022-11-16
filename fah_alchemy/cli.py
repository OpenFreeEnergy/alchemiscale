import click
import uvicorn

from .compute.api import Settings




@click.group()
def cli():
    ...


# settings = @click.Parameter

@cli.command(
    help="Start the client API service."
)
@click.option('--port', help="the port to run this service on",
              type=int, default=None)
def api(port):
    ...


@cli.group(
    help="Subcommands for the compute service"
)
def compute():
    ...


@compute.command(
    help="Start the compute API service."
)
def api():
    ...

@compute.command(
    help="Start the a synchronous compute service."
)
def synchronous():
    ...


@cli.group()
def database():
    ...

@database.command()
@click.argument('neo4j_url', help="database URI")
@click.option('--user', help="database user name")
@click.option('--password', help="database password")
@click.option('--dbname', help="custom database name, default 'neo4j'")
def init(neo4j_url, user, password):
    """Initialize the Neo4j database.

    """
    defaults = Settings()
    graph = ... 
    constraint_q = ("CREATE CONSTRAINT gufe_key FOR (n:GufeTokenizable) "
                    "REQUIRE n._scoped_key is unique")

    try:
        graph.run(constraint_q)
    except:
        pass

    # https://github.com/py2neo-org/py2neo/pull/951
    graph.run("MERGE (:NOPE)")


@cli.group()
def user():
    ...


@user.command()
def add():
    """Add a user to the database.

    """
    ...

@user.command()
def list_scope():
    """List all scopes for the given user.

    """
    ...

@user.command()
def add_scope():
    """Add a scope for the given user(s).

    """
    ...

@user.command()
def remove_scope():
    """Remove a scope for the given user(s).

    """
    ...
