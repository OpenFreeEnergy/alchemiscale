import click
import gunicorn.app.base


def dictify_callback(ctx, param, value):
    return {param.envvar: value}


SETTINGS_OPTION_KWARGS = {
    'show_envvar': True,
    'callback': dictify_callback,
}


def get_settings_from_options(kwargs):
    from .compute.api import Settings
    update = {k: v for k, v in kwargs.items() if v is not None}
    return Settings(**update)

def api_starting_params(func):
    workers = click.option('--workers', type=int, help="number of workers",
                           default=1)
    host = click.option('--host', type=str, help="IP address of host")
    port = click.option('--port', type=int, help="port")
    return workers(host(port(func)))

class ApiApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, workers, bind):
        self.app = app
        self.workers = workers
        self.bind = bind
        super().__init__()

    def load(self):
        return self.app

    def load_config(self):
        self.cfg.set('workers', self.workers)
        self.cfg.set('bind', self.bind)

def start_api(api_app, workers, host, port):
    gunicorn_app = ApiApplication(api_app, workers, bind=f"{host}:{port}")
    # gunicorn_app.run()



@click.group()
def cli():
    ...


# reusable parameters to ensure consistent naming and help strings
JWT_TOKEN_OPTION = click.option(
    '--jwt-secret', type=str, help="JSON web token secret",
    envvar="JWT_SECRET_KEY", **SETTINGS_OPTION_KWARGS
)
DBNAME_OPTION = click.option(
    '--dbname', type=str, help="custom database name, default 'neo4j'",
    envvar="NEO4J_DBNAME", **SETTINGS_OPTION_KWARGS
)

@cli.command(
    name="api",
    help="Start the client API service",
)
@api_starting_params
def api(workers, host, port):
    from fah_alchemy.interface.api import app
    start_api(app, workers, host, port)

@cli.group(
    help="Compute commands."
)
def compute():
    ...


@compute.command(
    help="Start the compute API service."
)
def api():
    from fah_alchemy.compute.api import app
    start_api(app)

@compute.command(
    help="Start the synchronous compute service."
)
def synchronous():
    ...


@cli.group()
def database():
    ...

@database.command()
@click.option('--url', help="database URI", type=str, envvar="NEO4J_URL",
              **SETTINGS_OPTION_KWARGS)
@click.option('--user', help="database user name", type=str,
                 envvar="NEO4J_USER", **SETTINGS_OPTION_KWARGS)
@click.option('--password', help="database password", type=str,
                 envvar="NEO4J_PASS", **SETTINGS_OPTION_KWARGS)
@DBNAME_OPTION
@JWT_TOKEN_OPTION
def init(url, user, password, dbname, jwt_secret):
    """Initialize the Neo4j database.

    Note that options here can be set by environment variables, as shown on
    each option.
    """
    from .compute.api import get_n4js
    cli_values = url | user | password | dbname | jwt_secret
    settings = get_settings_from_options(cli_values)
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
