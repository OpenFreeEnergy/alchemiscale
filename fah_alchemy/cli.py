import click


def envvar_dictify(ctx, param, value):
    """Callback to return a dict of param's envvar to value.

    This ensures that the envvar name only has to be entered as a string
    once within the click system. It requires that the parameter this
    callback is attached to defines its envvar.
    """
    return {param.envvar: value}

# use these extra kwargs with any option from a settings parameter.
SETTINGS_OPTION_KWARGS = {
    'show_envvar': True,
    'callback': envvar_dictify,
}

def get_settings_from_options(kwargs, settings_cls):
    """Create a settings object from a dict.

    This first strips all items with value None (which will be defaults) so
    that they don't override settings defaults.
    """
    update = {k: v for k, v in kwargs.items() if v is not None}
    return settings_cls(**update)


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
    help="Start the client API service."
)
def api():
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
def init(url, user, password, dbname):
    """Initialize the Neo4j database.

    Note that options here can be set by environment variables, as shown on
    each option.
    """
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname
    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)

    n4js = get_n4js(settings)
    n4js.initialize()


@database.command()
@click.option('--url', help="database URI", type=str, envvar="NEO4J_URL",
              **SETTINGS_OPTION_KWARGS)
@click.option('--user', help="database user name", type=str,
                 envvar="NEO4J_USER", **SETTINGS_OPTION_KWARGS)
@click.option('--password', help="database password", type=str,
                 envvar="NEO4J_PASS", **SETTINGS_OPTION_KWARGS)
@DBNAME_OPTION
def check(url, user, password, dbname):
    """Check consistency of database.

    Note that options here can be set by environment variables, as shown on
    each option.
    """
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname
    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)

    n4js = get_n4js(settings)
    if n4js.check() is None:
        print("No inconsistencies found found in database.")


@database.command()
@click.option('--url', help="database URI", type=str, envvar="NEO4J_URL",
              **SETTINGS_OPTION_KWARGS)
@click.option('--user', help="database user name", type=str,
                 envvar="NEO4J_USER", **SETTINGS_OPTION_KWARGS)
@click.option('--password', help="database password", type=str,
                 envvar="NEO4J_PASS", **SETTINGS_OPTION_KWARGS)
@DBNAME_OPTION
def reset(url, user, password, dbname):
    """Remove all data from database; undo `init`.

    Note that options here can be set by environment variables, as shown on
    each option.
    """
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname
    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)

    n4js = get_n4js(settings)
    n4js.reset()



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
