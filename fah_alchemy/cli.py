import click
import uvicorn





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
@click.option('--url', help="database URI", type=str)
@click.option('--user', help="database user name")
@click.option('--password', help="database password")
@click.option('--dbname', help="custom database name, default 'neo4j'")
@click.option('--jwt-secret', help="JSON web token secret")
def init(url, user, password, dbname, jwt_secret):
    """Initialize the Neo4j database.

    """
    from .compute.api import Settings, get_n4js
    selected = {"NEO4J_URL": url, "NEO4J_DBNAME": dbname,
                "NEO4J_USER": user, "NEO4J_PASS": password,
                "JWT_SECRET_KEY": jwt_secret}
    update = {k: v for k, v in selected.items() if v}  # remove the Nones
    settings = Settings(**update)
    store = get_n4js(settings)

    constraint_q = ("CREATE CONSTRAINT gufe_key FOR (n:GufeTokenizable) "
                    "REQUIRE n._scoped_key is unique")

    try:
        store.graph.run(constraint_q)
    except:
        pass

    # https://github.com/py2neo-org/py2neo/pull/951
    store.graph.run("MERGE (:NOPE)")


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
