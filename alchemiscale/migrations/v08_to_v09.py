"""
:mod:`alchemiscale.migrations.v08_to_v09` --- migration for v0.8 to v0.9
========================================================================

"""

from ..storage.statestore import Neo4jStore


def migrate(n4js: Neo4jStore):
    """Migrate state store from alchemiscale v0.8 to v0.9.

    Changes:
    - keeps only the most recently updated ComputeManagerRegistration for each
      name, since instances of a compute manager now share one registration
    - gives existing ComputeManagerRegistrations a ``version``, so they are
      joined rather than replaced when a v0.9 compute manager registers
    - adds the uniqueness constraint on ComputeManagerRegistration names

    Registrations with the ERROR status are kept: they no longer prevent a
    compute manager from registering, and expire like any other.

    """

    # remove all but the most recently updated registration under each name
    q = """
    MATCH (cmr:ComputeManagerRegistration)
    WITH cmr.name AS name, cmr
    ORDER BY cmr.last_status_update DESC
    WITH name, collect(cmr) AS registrations
    WHERE size(registrations) > 1
    UNWIND registrations[1..] AS duplicate
    DETACH DELETE duplicate
    """
    n4js.execute_query(q)

    q = """
    MATCH (cmr:ComputeManagerRegistration)
    WHERE cmr.version IS NULL
    SET cmr.version = 0
    """
    n4js.execute_query(q)

    # re-attach compute services to the surviving registration for their name
    q = """
    MATCH (csr:ComputeServiceRegistration),
          (cmr:ComputeManagerRegistration {name: csr.manager_name})
    MERGE (cmr)-[:MANAGES]->(csr)
    """
    n4js.execute_query(q)

    n4js.initialize()
