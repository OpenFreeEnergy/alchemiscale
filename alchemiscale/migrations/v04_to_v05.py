"""
:mod:`alchemiscale.migrations.v04_to_v05` --- migration for v0.4 to v0.5
========================================================================

"""

from ..storage.statestore import Neo4jStore


def migrate(n4js: Neo4jStore):
    """Migrate state store from alchemiscale v0.4 to v0.5.

    Changes:
    - adds indexes on the new ``TaskProvenance`` node label to support the
      introspection queries introduced in v0.5. ``TaskProvenance`` is a plain
      labeled node (not a ``GufeTokenizable``), identified by its properties and
      reached from a ``Task`` via the ``PROVENANCE_OF`` relationship:
        - ``TaskProvenance.compute_service_id``: provenance records are matched
          by the id of the compute service that produced them, both when
          finalizing an attempt (``set_task_result``, expiry/deregistration) and
          when reading live progress for the current claimant.
        - ``TaskProvenance.datetime_claimed``: attempt histories and
          most-recent-attempt lookups are ordered by claim time.

    (There is nothing to index for the ``PROVENANCE_OF`` traversal itself: the
    ``Task`` endpoint is already covered by the ``GufeTokenizable._scoped_key``
    uniqueness constraint, and the ``PROVENANCE_OF`` relationship carries no
    properties.)

    This migration is idempotent (all indexes are created with
    ``IF NOT EXISTS``) and requires NO data migration. All new properties are
    optional-valued: pre-existing ``Task`` nodes simply have an empty attempt
    history and are unaffected.

    """

    indexes = {
        "TaskProvenance_compute_service_id_index": (
            "TaskProvenance",
            "compute_service_id",
        ),
        "TaskProvenance_datetime_claimed_index": (
            "TaskProvenance",
            "datetime_claimed",
        ),
    }

    for name, (label, property_) in indexes.items():
        n4js.execute_query(f"""
            CREATE INDEX {name} IF NOT EXISTS
            FOR (n:{label}) ON (n.{property_})
        """)
