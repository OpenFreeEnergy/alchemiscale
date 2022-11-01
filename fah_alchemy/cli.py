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
