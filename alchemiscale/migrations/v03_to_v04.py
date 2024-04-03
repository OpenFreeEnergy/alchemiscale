"""
:mod:`alchemiscale.migrations.v03_to_v04` --- migration for v0.3 to v0.4
========================================================================

"""

from ..storage.subgraph import Subgraph, merge_subgraph, record_data_to_node
from ..storage.statestore import Neo4jStore


def migrate(n4js: Neo4jStore):
    """Migrate state store from alchemiscale v0.3 to v0.4.

    Changes:
    - adds a NetworkMark node if not already present for each AlchemicalNetwork
      node; creates a MARKS relationship from the NetworkMark to the
      AlchemicalNetwork, with `active` state as a property

    """

    q = """MATCH (an:AlchemicalNetwork)
           WHERE NOT (an)<-[:MARKS]-(:NetworkMark)
           RETURN an
        """

    res = n4js.execute_query(q)

    subgraph = Subgraph()
    for rec in res.records:
        an_node = record_data_to_node(rec["an"])

        nm_subgraph, _, _ = n4js.create_network_mark_subgraph(network_node=an_node)

        subgraph |= nm_subgraph

    with n4js.transaction() as tx:
        merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")
