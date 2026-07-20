"""
:mod:`alchemiscale.migrations.v07_to_v08` --- migration for v0.7 to v0.8
========================================================================

"""

from ..storage.statestore import Neo4jStore


def migrate(n4js: Neo4jStore):
    """Migrate state store from alchemiscale v0.7 to v0.8.

    Adds the uniqueness constraint on the new ``ComputeEnvironment`` node label
    that backs the deduplicated storage of compute-service execution
    environments (issue #106): environments are content-addressed by ``hash``,
    and the constraint is load-bearing --- it makes the ``MERGE`` on
    ``ComputeEnvironment.hash`` (at compute-service registration) correct under
    concurrency and fast, and brings an existing deployment's constraint set in
    line with what ``Neo4jStore.check`` expects.

    Idempotent (``CREATE CONSTRAINT ... IF NOT EXISTS``); no data migration is
    required. Pre-existing ``Task``\\ s simply have no recorded environment.
    """

    n4js.execute_query("""
        CREATE CONSTRAINT compute_environment_hash IF NOT EXISTS
        FOR (n:ComputeEnvironment) REQUIRE n.hash IS UNIQUE
        """)
