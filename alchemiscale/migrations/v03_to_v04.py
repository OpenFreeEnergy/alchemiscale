"""
:mod:`alchemiscale.migrations.v03_to_v04` --- migration for v0.3 to v0.4
========================================================================

"""

from ..storage.subgraph import Subgraph, merge_subgraph
from ..storage.statestore import Neo4jStore


def migrate(n4js: Neo4jStore):

    # for each AlchemicalNetwork present, create a NetworkMark
    an_sks = n4js.query_networks()

    subgraph = Subgraph()
    for an_sk in an_sks:
        an_node = n4js._get_node(an_sk)
        nm_subgraph, nm_node, nm_sk = n4js.create_network_mark_subgraph(
            network_node=an_node
        )

        subgraph |= nm_subgraph

    with n4js.transaction() as tx:
        merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")
