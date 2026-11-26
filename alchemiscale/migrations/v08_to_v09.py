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
    - removes MANAGES relationships, since compute services now belong to the
      registration under their ``manager_name``
    - adds the uniqueness constraint on ComputeManagerRegistration names

    - records the detail of each ERROR registration in the manager's error
      history, so it remains readable after the registration expires

    Registrations with the ERROR status are kept: they no longer prevent a
    compute manager from registering, and expire like any other.

    """

    # keep the errors of ERROR registrations, once each
    q = """
    MATCH (cmr:ComputeManagerRegistration {status: "ERROR"})
    WHERE NOT EXISTS {
        MATCH (e:ComputeManagerError {name: cmr.name})
        WHERE e.time = cmr.last_status_update AND e.detail = cmr.detail
    }
    CREATE (:ComputeManagerError {
        name: cmr.name, time: cmr.last_status_update, seq: 1, status: cmr.status,
        detail: coalesce(cmr.detail, "")
    })
    """
    n4js.execute_query(q)

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

    # compute services now belong to the registration under their
    # manager_name, rather than through a MANAGES relationship
    q = """
    MATCH (:ComputeManagerRegistration)-[rel:MANAGES]->(:ComputeServiceRegistration)
    DELETE rel
    """
    n4js.execute_query(q)

    n4js.initialize()
