import click



@click.group()
def cli():
    ...


@cli.command()
def api(
        help="Start the client API service."
        ):
    ...


@cli.group()
def compute():
    ...


@compute.command()
def api(
        help="Start the compute API service."
        ):
    ...

@compute.command()
def synchronous(
        help="Start the a synchronous compute service."
        ):
    ...


@cli.group()
def storage():
    ...

@storage.command()
def init():
    """Initialize the Neo4j database.

    """
    ...


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
