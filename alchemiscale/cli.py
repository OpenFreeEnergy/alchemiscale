"""
Command line interface --- :mod:`alchemiscale.cli`
=================================================

"""

import click
import gunicorn.app.base
from typing import Type

from .models import Scope
from .security.auth import hash_key, authenticate, AuthenticationError
from .security.models import (
    CredentialedEntity,
    CredentialedUserIdentity,
    CredentialedComputeIdentity,
)


def envvar_dictify(ctx, param, value):
    """Callback to return a dict of param's envvar to value.

    This ensures that the envvar name only has to be entered as a string
    once within the click system. It requires that the parameter this
    callback is attached to defines its envvar.
    """
    return {param.envvar: value}


# use these extra kwargs with any option from a settings parameter.
SETTINGS_OPTION_KWARGS = {
    "show_envvar": True,
    "callback": envvar_dictify,
}


def get_settings_from_options(kwargs, settings_cls):
    """Create a settings object from a dict.

    This first strips all items with value None (which will be defaults) so
    that they don't override settings defaults.
    """
    update = {k: v for k, v in kwargs.items() if v is not None}
    return settings_cls(**update)


def api_starting_params(envvar_host, envvar_port, envvar_loglevel):
    def inner(func):
        workers = click.option(
            "--workers", type=int, help="number of workers", default=1
        )
        host = click.option(
            "--host",
            type=str,
            help="IP address of host",
            envvar=envvar_host,
            **SETTINGS_OPTION_KWARGS,
        )
        port = click.option(
            "--port",
            type=int,
            help="port",
            envvar=envvar_port,
            **SETTINGS_OPTION_KWARGS,
        )
        loglevel = click.option(
            "--loglevel",
            type=str,
            default="info",
            help="logging level",
            envvar=envvar_loglevel,
            **SETTINGS_OPTION_KWARGS,
        )
        return workers(host(port(loglevel(func))))

    return inner


def db_params(func):
    url = click.option(
        "--url",
        help="database URI",
        type=str,
        envvar="NEO4J_URL",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    user = click.option(
        "--user",
        help="database user name",
        type=str,
        envvar="NEO4J_USER",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    password = click.option(
        "--password",
        help="database password",
        type=str,
        envvar="NEO4J_PASS",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    dbname = click.option(
        "--dbname",
        type=str,
        help="custom database name, default 'neo4j'",
        envvar="NEO4J_DBNAME",
        **SETTINGS_OPTION_KWARGS,
    )
    return url(user(password(dbname(func))))


def generate_secret_key(ctx, param, value):
    from alchemiscale.security.auth import generate_secret_key

    return {param.envvar: generate_secret_key()}


def jwt_params(func):
    secret = click.option(
        "--jwt-secret",
        type=str,
        help="JSON web token secret",
        envvar="JWT_SECRET_KEY",
        callback=generate_secret_key,
        show_envvar=True,
    )
    expire_seconds = click.option(
        "--jwt-expire-seconds",
        type=int,
        default=1800,
        envvar="JWT_EXPIRE_SECONDS",
        **SETTINGS_OPTION_KWARGS,
    )
    algo = click.option(
        "--jwt-algorithm",
        type=str,
        default="HS256",
        envvar="JWT_ALGORITHM",
        **SETTINGS_OPTION_KWARGS,
    )
    return secret(expire_seconds(algo(func)))


def s3os_params(func):
    access_key_id = click.option(
        "--access-key-id",
        type=str,
        default=None,
        envvar="AWS_ACCESS_KEY_ID",
        **SETTINGS_OPTION_KWARGS,
    )
    secret_access_key = click.option(
        "--secret-access-key",
        type=str,
        default=None,
        envvar="AWS_SECRET_ACCESS_KEY",
        **SETTINGS_OPTION_KWARGS,
    )
    session_token = click.option(
        "--session-token",
        type=str,
        default=None,
        envvar="AWS_SESSION_TOKEN",
        **SETTINGS_OPTION_KWARGS,
    )
    s3_bucket = click.option(
        "--s3-bucket",
        type=str,
        envvar="AWS_S3_BUCKET",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    s3_prefix = click.option(
        "--s3-prefix",
        type=str,
        envvar="AWS_S3_PREFIX",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    default_region = click.option(
        "--default-region",
        type=str,
        envvar="AWS_DEFAULT_REGION",
        required=True,
        **SETTINGS_OPTION_KWARGS,
    )
    return access_key_id(
        secret_access_key(session_token(s3_bucket(s3_prefix(default_region(func)))))
    )


class ApiApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, workers, bind):
        self.app = app
        self.workers = workers
        self.bind = bind
        super().__init__()

    @classmethod
    def from_parameters(cls, app, workers, host, port):
        return cls(app, workers, bind=f"{host}:{port}")

    def load(self):
        return self.app

    def load_config(self):
        self.cfg.set("workers", self.workers)
        self.cfg.set("bind", self.bind)
        self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")


def start_api(api_app, workers, host, port):
    gunicorn_app = ApiApplication(api_app, workers, bind=f"{host}:{port}")
    gunicorn_app.run()


@click.group()
def cli():
    ...


# reusable parameters to ensure consistent naming and help strings


@cli.command(
    name="api",
    help="Start the user-facing API service",
)
@api_starting_params("FA_API_HOST", "FA_API_PORT", "FA_API_LOGLEVEL")
@db_params
@s3os_params
@jwt_params
def api(
    workers, host, port, loglevel,  # API
    url, user, password, dbname,  # DB
    jwt_secret, jwt_expire_seconds, jwt_algorithm,  # JWT
    access_key_id, secret_access_key, session_token, s3_bucket, s3_prefix, default_region  # AWS
):  # fmt: skip
    from alchemiscale.interface.api import app
    from .settings import APISettings, get_base_api_settings
    from .security.auth import generate_secret_key

    # CONSIDER GENERATING A JWT_SECRET_KEY if none provided with
    # key = generate_secret_key()
    # CONVENIENT FOR THE SINGLE-SERVER CASE HERE
    # HOW-TO: modify the callback in jwt_secret (defined in jwt_params) to
    # do this. See comment there. Use that instead of the callback in
    # SETTINGS_OPTION_KWARGS

    def get_settings_override():
        # inject settings from CLI arguments
        api_dict = host | port | loglevel
        jwt_dict = jwt_secret | jwt_expire_seconds | jwt_algorithm
        db_dict = url | user | password | dbname
        s3_dict = (
            access_key_id
            | secret_access_key
            | session_token
            | s3_bucket
            | s3_prefix
            | default_region
        )
        return get_settings_from_options(
            api_dict | jwt_dict | db_dict | s3_dict, APISettings
        )

    app.dependency_overrides[get_base_api_settings] = get_settings_override

    start_api(app, workers, host["FA_API_HOST"], port["FA_API_PORT"])


@cli.group(help="Subcommands for the compute service")
def compute():
    ...


@compute.command(help="Start the compute API service.")
@api_starting_params(
    "FA_COMPUTE_API_HOST", "FA_COMPUTE_API_PORT", "FA_COMPUTE_API_LOGLEVEL"
)
@db_params
@s3os_params
@jwt_params
def api(
    workers, host, port, loglevel,  # API
    url, user, password, dbname,  # DB
    jwt_secret, jwt_expire_seconds, jwt_algorithm,  #JWT
    access_key_id, secret_access_key, session_token, s3_bucket, s3_prefix, default_region  # AWS
):  # fmt: skip
    from alchemiscale.compute.api import app
    from .settings import ComputeAPISettings, get_base_api_settings
    from .security.auth import generate_secret_key

    # CONSIDER GENERATING A JWT_SECRET_KEY if none provided with
    # key = generate_secret_key()
    # CONVENIENT FOR THE SINGLE-SERVER CASE HERE

    def get_settings_override():
        # inject settings from CLI arguments
        api_dict = host | port | loglevel
        jwt_dict = jwt_secret | jwt_expire_seconds | jwt_algorithm
        db_dict = url | user | password | dbname
        s3_dict = (
            access_key_id
            | secret_access_key
            | session_token
            | s3_bucket
            | s3_prefix
            | default_region
        )
        return get_settings_from_options(
            api_dict | jwt_dict | db_dict | s3_dict, ComputeAPISettings
        )

    app.dependency_overrides[get_base_api_settings] = get_settings_override

    start_api(app, workers, host["FA_COMPUTE_API_HOST"], port["FA_COMPUTE_API_PORT"])


@compute.command(help="Start the synchronous compute service.")
def synchronous():
    ...


@cli.group(help="Subcommands for the database")
def database():
    ...


@database.command()
@db_params
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
@db_params
def check(url, user, password, dbname):
    """Check consistency of database.

    Note that options here can be set by environment variables, as shown on
    each option.
    """
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    db_dict = url | user | password | dbname

    settings = get_settings_from_options(db_dict, Neo4jStoreSettings)

    n4js = get_n4js(settings)
    if n4js.check() is None:
        print("No inconsistencies found in database.")


@database.command()
@db_params
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


def _identity_type_string_to_cls(identity_type: str) -> Type[CredentialedEntity]:
    if identity_type == "user":
        identity_type_cls = CredentialedUserIdentity
    elif identity_type == "compute":
        identity_type_cls = CredentialedComputeIdentity
    else:
        raise RuntimeError(f"Unknown identity type {identity_type}")

    return identity_type_cls


def identity_type(func):
    identity_type = click.option(
        "--identity-type",
        "-t",
        default="user",
        help="User type",
        type=click.Choice(["user", "compute"], case_sensitive=False),
    )
    return identity_type(func)


def identifier(func):
    identifier = click.option(
        "--identifier", "-i", help="identifier", required=True, type=str
    )
    return identifier((func))


def key(func):
    key = click.option("--key", "-k", help="key", required=True, type=str)
    return key(func)


def scope(func):
    scope = click.option("--scope", "-s", help="scope", required=True, type=str)
    return scope(func)


@cli.group(help="Subcommands for managing identities")
def identity():
    ...


@identity.command()
@db_params
@identity_type
@identifier
@key
def add(url, user, password, dbname, identity_type, identifier, key):
    """Add a credentialed identity to the database."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    identity_type_cls = _identity_type_string_to_cls(identity_type)
    identity_model = identity_type_cls(hashed_key=hash_key(key), identifier=identifier)
    n4js.create_credentialed_entity(identity_model)


@identity.command()
@db_params
@identity_type
def list(url, user, password, dbname, identity_type):
    """List all credentialed entities of the given type."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    identity_type_cls = _identity_type_string_to_cls(identity_type)
    click.echo(n4js.list_credentialed_entities(identity_type_cls))


@identity.command()
@db_params
@identity_type
@identifier
def remove(url, user, password, dbname, identity_type, identifier):
    """Remove a credentialed identity from the database."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    identity_type_cls = _identity_type_string_to_cls(identity_type)
    n4js.remove_credentialed_identity(identifier, identity_type_cls)


@identity.command()
@db_params
@identity_type
@identifier
@scope
def add_scope(url, user, password, dbname, identity_type, identifier, scope):
    """Add a scope for the given identity."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    scope = Scope.from_str(scope)
    identity_type_cls = _identity_type_string_to_cls(identity_type)

    n4js.add_scope(identifier, identity_type_cls, scope)


@identity.command()
@db_params
@identity_type
@identifier
def list_scope(url, user, password, dbname, identity_type, identifier):
    """List all scopes for the given identity."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    identity_type_cls = _identity_type_string_to_cls(identity_type)
    scopes = n4js.list_scopes(identifier, identity_type_cls)

    click.echo([str(scope) for scope in scopes])


@identity.command()
@db_params
@identity_type
@identifier
@scope
def remove_scope(url, user, password, dbname, identity_type, identifier, scope):
    """Remove a scope for the given identity(s)."""
    from .storage.statestore import get_n4js
    from .settings import Neo4jStoreSettings

    cli_values = url | user | password | dbname

    settings = get_settings_from_options(cli_values, Neo4jStoreSettings)
    n4js = get_n4js(settings)

    scope = Scope.from_str(scope)
    identity_type_cls = _identity_type_string_to_cls(identity_type)

    n4js.remove_scope(identifier, identity_type_cls, scope)
