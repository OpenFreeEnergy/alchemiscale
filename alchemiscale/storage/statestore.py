"""
:mod:`alchemiscale.storage.statestore` --- state store interface
================================================================

"""

import abc
import bisect
import datetime
from contextlib import contextmanager
import json
import re
from functools import lru_cache, update_wrapper
from collections import defaultdict
from collections.abc import Iterable
import weakref
import numpy as np

import networkx as nx
from gufe import (
    AlchemicalNetwork,
    Transformation,
    NonTransformation,
    Protocol,
)
from gufe.settings import SettingsBaseModel
from gufe.tokenization import GufeTokenizable, GufeKey, JSON_HANDLER, KeyedChain
from gufe.protocols import ProtocolUnitFailure

from neo4j import Transaction, GraphDatabase, Driver

from stratocaster.base import Strategy

from .models import (
    ComputeServiceID,
    ComputeServiceRegistration,
    ComputeManagerRegistration,
    ComputeManagerInstruction,
    ComputeManagerStatus,
    ComputeManagerID,
    NetworkMark,
    NetworkStateEnum,
    ProtocolDAGResultRef,
    StrategyState,
    StrategyModeEnum,
    StrategyStatusEnum,
    StrategyTaskScalingEnum,
    Task,
    TaskHub,
    TaskRestartPattern,
    TaskStatusEnum,
    Tracebacks,
)

from ..models import Scope, ScopedKey
from .cypher import cypher_or

from ..security.models import CredentialedEntity
from ..settings import Neo4jStoreSettings
from ..validators import validate_network_nonself

from .subgraph import (
    Node,
    Relationship,
    Subgraph,
    create_subgraph,
    merge_subgraph,
    record_data_to_node,
    subgraph_from_path_record,
)


def get_n4js(settings: Neo4jStoreSettings):
    """Convenience function for getting a Neo4jStore directly from settings."""
    return Neo4jStore(settings)


class Neo4JStoreError(Exception): ...


class AlchemiscaleStateStore(abc.ABC): ...


def _select_tasks_from_taskpool(taskpool: list[tuple[str, float]], count) -> list[str]:
    """Select Tasks from a pool of tasks according to the following scheme:

    1. Randomly select N=`count` tasks from the TaskPool based on weighting
    2. Return the string representation of the Task ScopedKeys.

    Parameters
    ----------
    taskpool: List[Tuple[str, float]]
        A list of tuples containing Tasks (string represtnation of their ScopedKeys) of
        equal priority with the weights of their ACTIONS relationships.

    Returns
    -------
    sk: List[str]
        The string representations of the ScopedKeys of the Tasks selected from the taskpool.
    """
    weights = []
    tasks = []
    for t, w in taskpool:
        tasks.append(t)
        weights.append(w)

    weights = np.array(weights)
    prob = weights / weights.sum()

    return list(np.random.choice(tasks, count, replace=False, p=prob))


CLAIM_QUERY = f"""
    // only match the task if it doesn't have an existing CLAIMS relationship
    UNWIND $tasks_list AS task_sk
    MATCH (t:Task {{_scoped_key: task_sk}})
    WHERE NOT (t)<-[:CLAIMS]-(:ComputeServiceRegistration)

    WITH t

    // create CLAIMS relationship with given compute service
    MATCH (csreg:ComputeServiceRegistration {{identifier: $compute_service_id}})
    CREATE (t)<-[cl:CLAIMS {{claimed: datetime($datetimestr)}}]-(csreg)

    SET t.status = '{TaskStatusEnum.running.value}'

    RETURN t
"""


class Neo4jStore(AlchemiscaleStateStore):
    # uniqueness constraints applied to the database; key is node label,
    # 'property' is the property on which uniqueness is guaranteed for nodes
    # with that label
    constraints = {
        "GufeTokenizable": {"name": "scoped_key", "property": "_scoped_key"},
        "CredentialedUserIdentity": {
            "name": "user_identifier",
            "property": "identifier",
        },
        "CredentialedComputeIdentity": {
            "name": "compute_identifier",
            "property": "identifier",
        },
        "ComputeServiceRegistration": {
            "name": "compute_service_registration_identifier",
            "property": "identifier",
        },
    }

    def __init__(self, settings: Neo4jStoreSettings):
        """Initialize Neo4jStore from settings.

        Parameters
        ----------
        settings : Neo4jStoreSettings
            Configuration settings for Neo4j state store.
        """
        self.settings = settings

        self.graph: Driver = GraphDatabase.driver(
            settings.NEO4J_URL, auth=(settings.NEO4J_USER, settings.NEO4J_PASS)
        )
        self.db_name = settings.NEO4J_DBNAME
        self.gufe_nodes = weakref.WeakValueDictionary()

    @contextmanager
    def transaction(self, ignore_exceptions=False) -> Transaction:
        """Context manager for a Neo4j Transaction."""
        with self.graph.session(database=self.db_name) as session:
            tx = session.begin_transaction()
            try:
                yield tx
            except Exception:
                tx.rollback()
                if not ignore_exceptions:
                    raise

            else:
                tx.commit()

    def chainable(func):
        def inner(self, *args, **kwargs):
            if kwargs.get("tx") is not None:
                return func(self, *args, **kwargs)

            with self.transaction() as tx:
                kwargs.update(tx=tx)
                return func(self, *args, **kwargs)

        update_wrapper(inner, func)

        return inner

    def close(self):
        """Close the Neo4j driver for this instance."""
        self.graph.close()

    def execute_query(self, *args, **kwargs):
        kwargs.update({"database_": self.db_name})
        return self.graph.execute_query(*args, **kwargs)

    def initialize(self):
        """Initialize database.

        Ensures that constraints and any other required structures are in place.
        Should be used on any Neo4j database prior to use for Alchemiscale.

        """
        for label, values in self.constraints.items():
            self.execute_query(
                f"""
                CREATE CONSTRAINT {values['name']} IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.{values['property']} is unique
            """
            )

    def check(self):
        """Check consistency of database.

        Will raise `Neo4JStoreError` if any state check fails.
        If no check fails, will return without any exception.

        """
        constraints = {
            rec["name"]: rec for rec in self.execute_query("show constraints").records
        }

        if len(constraints) != len(self.constraints):
            raise Neo4JStoreError(
                f"Number of constraints in database is {len(constraints)}; expected {len(self.constraints)}"
            )

        for label, values in self.constraints.items():
            constraint = constraints[values["name"]]
            if not (
                constraint["labelsOrTypes"] == [label]
                and constraint["properties"] == [values["property"]]
            ):
                raise Neo4JStoreError(
                    f"Constraint {constraint['name']} does not have expected form"
                )

    def _store_check(self):
        """Check that the database is in a state that can be used by the API."""
        try:
            # just list available functions to see if database is working
            self.execute_query("SHOW FUNCTIONS YIELD *")
        except Exception:
            return False
        return True

    def reset(self):
        """Remove all data from database; undo all components in `initialize`."""
        self.execute_query("MATCH (n) DETACH DELETE n")

        for label, values in self.constraints.items():
            self.execute_query(
                f"""
                DROP CONSTRAINT {values['name']} IF EXISTS
            """
            )

    ## gufe object handling

    def _keyed_chain_to_subgraph(
        self,
        keyed_chain: KeyedChain,
        scope: Scope,
    ) -> tuple[Subgraph, Node, str]:
        r"""Construct a Subgraph from a KeyedChain.

        Parameters
        ----------
        keyed_chain
            The keyed chain to convert into a subgraph.
        scope
            The scope to assign to the Subgraph Node objects.

        Returns
        -------
        Subgraph, Node, str
            The Subgraph, the node of the top-level GufeTokenizable, and its
            GufeKey as a string.
        """

        relationships = []
        previous_nodes = {}

        def is_gufe_dict(dct):
            if not isinstance(dct, dict):
                return False
            return ":gufe-key:" in dct.keys()

        def add_previous_node(node_gufe_key, node):
            previous_nodes[
                (node_gufe_key, scope.org, scope.campaign, scope.project)
            ] = node

        def get_previous_node(node_gufe_key):
            if (
                rel_node := previous_nodes.get(
                    (node_gufe_key, scope.org, scope.campaign, scope.project)
                )
            ) is None:
                raise ValueError("Possibly corrupt keyedchain")
            return rel_node

        def update_relationships(node_a, node_b, **kwargs):
            rel = Relationship.type("DEPENDS_ON")(
                node_a,
                node_b,
                _org=scope.org,
                _campaign=scope.campaign,
                _project=scope.project,
                **kwargs,
            )
            relationships.append(rel)

        def handle_dict(node, key, dct):
            # e.g. {'ligand': {':gufe-key:': 'SmallMoleculeComponent-abc123'}}
            if all(map(is_gufe_dict, dct.values())):
                for k, v in dct.items():
                    rel_key = v[":gufe-key:"]
                    update_relationships(
                        node, get_previous_node(rel_key), attribute=key, key=k
                    )
            else:
                node[key] = json.dumps(dct, cls=JSON_HANDLER.encoder)
                node["_json_props"].append(key)

        def handle_list(node, key, values):
            if isinstance(values[0], (int, float, str)) and all(
                (isinstance(x, type(values[0])) for x in values)
            ):
                node[key] = values
                return
            # list of gufe key: [{":gufe-key:": ...}, {":gufe-key:": ...}, {":gufe-key:": ...}]
            elif all(map(is_gufe_dict, values)):
                for i, x in enumerate(values):
                    rel_key = x[":gufe-key:"]
                    update_relationships(
                        node, get_previous_node(rel_key), attribute=key, index=i
                    )
            else:
                node[key] = json.dumps(values, cls=JSON_HANDLER.encoder)
                node["_json_props"].append(key)

        def handle_tuple(node, key, values):
            # currently this won't roundtrip exactly due to `gufe` JSON
            # encoder's non-handling of tuples
            node[key] = json.dumps(values, cls=JSON_HANDLER.encoder)
            node["_json_props"].append(key)

        def handle_settings(node, key, value):
            node[key] = json.dumps(value, cls=JSON_HANDLER.encoder, sort_keys=True)
            node["_json_props"].append(key)

        def process_keyed_dict(gufe_key, kd):
            node = Node("GufeTokenizable", kd["__qualname__"])
            node["_json_props"] = []
            for key, value in kd.items():
                match value:
                    case {":gufe-key:": rel_key}:
                        update_relationships(
                            node, get_previous_node(rel_key), attribute=key
                        )
                    case dict():
                        handle_dict(node, key, value)
                    case list():
                        handle_list(node, key, value)
                    case tuple():
                        handle_tuple(node, key, value)
                    case SettingsBaseModel():
                        handle_settings(node, key, value)
                    case _:
                        node[key] = value

            node["_gufe_key"] = str(gufe_key)
            node["_scoped_key"] = str(
                ScopedKey(gufe_key=str(gufe_key), **scope.to_dict())
            )
            node.update(
                {
                    "_org": scope.org,
                    "_campaign": scope.campaign,
                    "_project": scope.project,
                }
            )

            return node

        for gufe_key, kd in keyed_chain:
            # process each keyed_dict, mutating the relationships list
            node = process_keyed_dict(gufe_key, kd)
            add_previous_node(gufe_key, node)

        subgraph = Subgraph(None, relationships)
        scoped_key = ScopedKey(gufe_key=node["_gufe_key"], **scope.to_dict())
        return subgraph, node, scoped_key

    def _subgraph_to_gufe(
        self, nodes: list[Node], subgraph: Subgraph
    ) -> dict[Node, GufeTokenizable]:
        """Get a Dict `GufeTokenizable` objects within the given subgraph.

        Any `GufeTokenizable` that requires nodes or relationships missing from
        the subgraph will not be returned.

        """
        nxg = self._subgraph_to_networkx(subgraph)
        nodes_to_gufe = {}
        gufe_objs = {}
        for node in nodes:
            gufe_objs[node] = self._node_to_gufe(node, nxg, nodes_to_gufe)

        return gufe_objs

    def _subgraph_to_networkx(self, subgraph: Subgraph):
        g = nx.DiGraph()

        for node in subgraph.nodes:
            g.add_node(node, **dict(node))

        for relationship in subgraph.relationships:
            g.add_edge(
                relationship.start_node, relationship.end_node, **dict(relationship)
            )

        return g

    def _node_to_gufe(
        self, node: Node, g: nx.DiGraph, mapping: dict[Node, GufeTokenizable]
    ):
        # shortcut if we already have this object deserialized
        if gufe_obj := mapping.get(node):
            return gufe_obj

        dct = dict(node)
        # deserialize json-serialized attributes
        for key in dct["_json_props"]:
            dct[key] = json.loads(dct[key], cls=JSON_HANDLER.decoder)

        # inject dependencies
        dep_edges = g.edges(node)
        postprocess = set()
        for edge in dep_edges:
            u, v = edge
            edgedct = g.get_edge_data(u, v)
            if "attribute" in edgedct:
                if "key" in edgedct:
                    if not edgedct["attribute"] in dct:
                        dct[edgedct["attribute"]] = dict()
                    dct[edgedct["attribute"]][edgedct["key"]] = self._node_to_gufe(
                        v, g, mapping
                    )
                elif "index" in edgedct:
                    postprocess.add(edgedct["attribute"])
                    if not edgedct["attribute"] in dct:
                        dct[edgedct["attribute"]] = list()
                    dct[edgedct["attribute"]].append(
                        (edgedct["index"], self._node_to_gufe(v, g, mapping))
                    )
                else:
                    dct[edgedct["attribute"]] = self._node_to_gufe(v, g, mapping)

        # postprocess any attributes that are lists
        # needed because we don't control the order in which a list is built up
        # but can only order it post-hoc
        for attr in postprocess:
            dct[attr] = [j for i, j in sorted(dct[attr], key=lambda x: x[0])]

        # remove all neo4j-specific keys
        dct.pop("_json_props", None)
        dct.pop("_gufe_key", None)
        dct.pop("_org", None)
        dct.pop("_campaign", None)
        dct.pop("_project", None)
        dct.pop("_scoped_key", None)

        mapping[node] = res = GufeTokenizable.from_shallow_dict(dct)
        return res

    def _get_node(
        self,
        scoped_key: ScopedKey,
        return_subgraph=False,
    ) -> Node | tuple[Node, Subgraph]:
        """
        If `return_subgraph = True`, also return subgraph for gufe object.
        """

        # Safety: qualname comes from GufeKey which is validated
        qualname = scoped_key.qualname
        parameters = {"scoped_key": str(scoped_key)}

        q = f"""
        MATCH (n:{qualname} {{ _scoped_key: $scoped_key }})
        """

        if return_subgraph:
            q += """
            OPTIONAL MATCH p = (n)-[r:DEPENDS_ON*]->(m)
            WHERE NOT (m)-[:DEPENDS_ON]->()
            RETURN n, p
            """
        else:
            q += """
            RETURN n
            """

        nodes = set()
        subgraph = Subgraph()

        result = self.execute_query(q, parameters_=parameters)

        for record in result.records:
            node = record_data_to_node(record["n"])
            nodes.add(node)
            if return_subgraph and record.get("p") is not None:
                subgraph = subgraph | subgraph_from_path_record(record["p"])
            else:
                subgraph = node

        if len(nodes) == 0:
            raise KeyError("No such object in database")
        elif len(nodes) > 1:
            raise Neo4JStoreError(
                "More than one such object in database; this should not be possible"
            )

        if return_subgraph:
            return list(nodes)[0], subgraph
        else:
            return list(nodes)[0]

    def _query(
        self,
        *,
        qualname: str,
        additional: dict | None = None,
        key: GufeKey | None = None,
        scope: Scope = Scope(),
        return_gufe=False,
    ):
        properties = {
            "_org": scope.org,
            "_campaign": scope.campaign,
            "_project": scope.project,
        }

        # Remove None values from properties
        properties = {k: v for k, v in properties.items() if v is not None}

        if key is not None:
            properties["_gufe_key"] = str(key)

        if additional is None:
            additional = {}
        properties.update({k: v for k, v in additional.items() if v is not None})

        if not properties:
            prop_string = ""
        else:
            prop_string = ", ".join(f"{key}: ${key}" for key in properties.keys())

            prop_string = f" {{{prop_string}}}"

        q = f"""
        MATCH (n:{qualname}{prop_string})
        """
        if return_gufe:
            q += """
            OPTIONAL MATCH p = (n)-[r:DEPENDS_ON*]->(m)
            WHERE NOT (m)-[:DEPENDS_ON]->()
            RETURN n,p
            """
        else:
            q += """
            RETURN DISTINCT n
            ORDER BY n._org, n._campaign, n._project, n._gufe_key
            """

        with self.transaction() as tx:
            res = tx.run(q, **properties).to_eager_result()

        nodes = list()
        subgraph = Subgraph()

        for record in res.records:
            node = record_data_to_node(record["n"])
            nodes.append(node)
            if return_gufe and record["p"] is not None:
                subgraph = subgraph | subgraph_from_path_record(record["p"])
            else:
                subgraph = node

        if return_gufe:
            return {
                ScopedKey.from_str(k["_scoped_key"]): v
                for k, v in self._subgraph_to_gufe(nodes, subgraph).items()
            }
        else:
            return [ScopedKey.from_str(i["_scoped_key"]) for i in nodes]

    def check_existence(self, scoped_key: ScopedKey):
        try:
            self._get_node(scoped_key=scoped_key)
        except KeyError:
            return False

        return True

    def get_scoped_key(self, obj: GufeTokenizable, scope: Scope):
        qualname = obj.__class__.__qualname__
        res = self._query(qualname=qualname, key=obj.key, scope=scope)

        if len(res) == 0:
            raise KeyError("No such object in database")
        elif len(res) > 1:
            raise Neo4JStoreError(
                "More than one such object in database; this should not be possible"
            )

        return res[0]

    def get_gufe(self, scoped_key: ScopedKey):
        return self.get_keyed_chain(scoped_key).to_gufe()

    def get_keyed_chain(self, scoped_key: ScopedKey) -> KeyedChain:
        """Retrieve the ``KeyedChain`` form of a ``GufeTokenizable`` from the database.

        Parameters
        ----------
        scoped_key
            The ``ScopedKey`` of the ``GufeTokenizable`` to retrieve.

        Returns
        -------
        ``KeyedChain``
            The ``KeyedChain`` form of the tokenizable.
        """

        # find the root node and all nodes that are connected by any number of
        # "DEPENDS_ON" relationships. Discard the path object, just collecting
        # the child nodes into a single set of nodes. Unwind all nodes and get
        # their dependencies, both the data embedded in the relationship,
        # and their keys.
        query = """
        MATCH (root:GufeTokenizable {`_scoped_key`: $scoped_key })
        OPTIONAL MATCH (root)-[:DEPENDS_ON*]->(dep)
        WITH root, COLLECT(DISTINCT dep) AS deps
        UNWIND [root] + deps AS node
        OPTIONAL MATCH (node)-[r:DEPENDS_ON]->(dep)
        WITH node, COLLECT(r) AS rels, COLLECT(dep.`_gufe_key`) AS keys
        RETURN node, rels, keys"""

        results = self.execute_query(query, scoped_key=str(scoped_key))

        # the root node wasn't found
        if len(results.records) == 0:
            raise KeyError("No such object in database")

        # A dictionary whose keys are a node's gufe key, and whose
        # values are a tuple with the node's data in its first element
        # and its dependency keys as its second element.
        graph_data: dict[GufeKey, tuple[Node, list[GufeKey]]] = {}

        # iterate over the nodes, the "DEPENDS_ON" relationships that each node
        # has with other nodes, along with the keys of those child nodes
        for node, rels, keys in results.records:

            # collect attributes for each node and update at the end
            attrs = {}

            # for each pair of relationship and dependency gufe key
            # Note: transform the key into the form used within gufe
            # keyed_dicts
            for depends_on, key in zip(rels, map(lambda k: {":gufe-key:": k}, keys)):
                # determine the attribute type using map destructuring
                match depends_on._properties:
                    # dictionary of tokenizables
                    case {"attribute": _attr, "key": _key}:
                        if _attr not in attrs.keys():
                            attrs[_attr] = {}
                        attrs[_attr][_key] = key
                    # list of tokenizables---since the list is order
                    # dependent, keep the attribute in an intermediate
                    # format, e.g. (index, key) instead of just key,
                    # for sorting later.
                    case {"attribute": _attr, "index": _index}:
                        if _attr not in attrs.keys():
                            attrs[_attr] = []
                        attrs[_attr].append((_index, key))
                    # direct gufe
                    case {"attribute": _attr}:
                        attrs[_attr] = key

            # start updating the properties of the node into a
            # keyed_dict form from the collected relationship
            # attributes and popping out irrelevant data
            properties = node._properties
            gufe_key = properties.pop("_gufe_key")

            # throw away remaining neo4j attributes
            properties.pop("_org", None)
            properties.pop("_campaign", None)
            properties.pop("_project", None)
            properties.pop("_scoped_key", None)

            for key, value in attrs.items():
                # sort the previously unsorted list by the included
                # indices leaving the value as the intended original
                # list
                if isinstance(value, list):
                    value.sort(key=lambda elem: elem[0])
                    value = [elem[1] for elem in value]
                properties[key] = value

            for json_key in properties.pop("_json_props"):
                properties[json_key] = json.loads(
                    properties[json_key], cls=JSON_HANDLER.decoder
                )
            graph_data[gufe_key] = (properties, keys)

        # topological sort before return
        return KeyedChain(self._topological_sort(graph_data))

    @staticmethod
    def _topological_sort(graph_data: dict[GufeKey, tuple[Node, list[GufeKey]]]):
        """Topologically sort graph data using Kahn's algorithm.

        Parameters
        ----------
        graph_data
            A dictionary mapping a ``GufeKey`` to it's ``Node``
            represenation and the keys of the ``GufeTokenizable``
            objects it depends on.

        Returns
        -------
            The ``KeyedChain`` represenation of the graph data.
        """
        L = []

        S = [
            (key, node)
            for key, (node, rel_keys) in graph_data.items()
            if rel_keys == []
        ]
        S.sort(key=lambda x: x[0])
        for rk, _ in S:
            graph_data.pop(rk)

        # while we have nodes with indegree 0
        while S:
            # remove the first node for processing
            key, node = S.pop(0)

            # add to the sorted list
            L.append((key, node))

            # iterate over all graph nodes and their keys
            removal_keys = []
            for pkey in graph_data.keys():

                # if child node isn't a dependency, continue
                if key not in graph_data[pkey][1]:
                    continue

                # remove the child node from the dependencies
                graph_data[pkey][1].remove(key)

                # if we're left with an empty dependency list, remove the node
                if graph_data[pkey][1] == []:
                    pnode = graph_data[pkey][0]
                    bisect.insort(S, (pkey, pnode), key=lambda x: x[0])
                    removal_keys.append(pkey)

            for rk in removal_keys:
                graph_data.pop(rk)
        return L

    def assemble_network(
        self,
        network: AlchemicalNetwork,
        scope: Scope,
        state: NetworkStateEnum | str = NetworkStateEnum.active,
    ) -> tuple[ScopedKey, ScopedKey, ScopedKey]:
        """Create all nodes and relationships needed for an AlchemicalNetwork
        represented in an alchemiscale state store.

        Parameters
        ----------
        network
            The AlchemicalNetwork to submit to the database.
        scope
            The Scope where the AlchemicalNetwork resides.
        state
            The starting state of the network as marked by the NetworkMark.

        Returns
        -------
        A tuple containing the AlchemicalNetwork ScopedKey, the TaskHub
        ScopedKey, and the NetworkMark ScopedKey.
        """

        nw_subgraph, nw_node, nw_sk = self.create_network_subgraph(network, scope)
        th_subgraph, th_node, th_sk = self.create_taskhub_subgraph(nw_node)
        nm_subgraph, nm_node, nm_sk = self.create_network_mark_subgraph(nw_node, state)

        subgraph = nw_subgraph | th_subgraph | nm_subgraph

        with self.transaction() as tx:
            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

        return nw_sk, th_sk, nm_sk

    def create_network_subgraph(self, network: AlchemicalNetwork, scope: Scope):
        """Create a Subgraph for the given AlchemicalNetwork.

        Parameters
        ----------
        network
            An AlchemicalNetwork to generate a Subgraph form of.
        scope
            The Scope where the AlchemicalNetwork resides.

        Returns
        -------
        A tuple containing the AlchemicalNetwork Subgraph, the specific
        AlchemicalNetwork Node within the Subgraph, and the ScopedKey of the
        AlchemicalNetwork.
        """

        validate_network_nonself(network)

        subgraph, node, scoped_key = self._keyed_chain_to_subgraph(
            KeyedChain.from_gufe(network),
            scope=scope,
        )

        return subgraph, node, scoped_key

    def delete_network(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        r"""Delete the given `AlchemicalNetwork` from the database.

        This will not remove any `Transformation`\s or `ChemicalSystem`\s
        associated with the `AlchemicalNetwork`, since these may be associated
        with other `AlchemicalNetwork`\s in the same `Scope`.

        """
        # note: something like the following could perhaps be used to delete everything that is *only*
        # associated with this network
        # not yet tested though
        """
        MATCH p = (n:AlchemicalNetwork {{_scoped_key: "{network_node['_scoped_key']}")-[r:DEPENDS_ON*]->(m),
              (n)-[:DEPENDS_ON]->(t:Transformation|NonTransformation),
              (n)-[:DEPENDS_ON]->(c:ChemicalSystem)
        OPTIONAL MATCH (o:AlchemicalNetwork)
        WHERE NOT o._scoped_key = "{network_node['_scoped_key']}"
          AND NOT (t)<-[:DEPENDS_ON]-(o)
          AND NOT (c)<-[:DEPENDS_ON]-(o)
        return distinct n,m
        """
        # first, delete the network's hub if present
        self.delete_taskhub(network)

        # then delete the network
        _ = """
        MATCH (an:AlchemicalNetwork {_scoped_key: $network})
        DETACH DELETE an
        """
        raise NotImplementedError

    def get_network_state(self, networks: list[ScopedKey]) -> list[str | None]:
        """Get the states of a group of networks.

        Parameters
        ----------
        networks
            The list networks to get the states of.

        Returns
        -------
        List[Optional[str]]
            A list containing the states of the given networks, in the same
            order as they were provided. If a network was not found, ``None``
            is returned at the corresponding index.
        """

        q = """
            UNWIND $networks AS network
            MATCH (an:AlchemicalNetwork {`_scoped_key`: network})<-[:MARKS]-(nm:NetworkMark)
            RETURN an._scoped_key as sk, nm.state AS state
        """

        results = self.execute_query(q, networks=[str(network) for network in networks])

        state_results = {}
        for record in results.records:
            sk = record["sk"]
            state = record["state"]

            state_results[sk] = state

        return [state_results.get(str(network), None) for network in networks]

    def create_network_mark_subgraph(
        self, network_node: Node, state=NetworkStateEnum.active
    ):
        """Create the Subgraph for an AlchemicalNetwork's NetworkMark.

        Parameters
        ----------
        network_node
            A Node representing the target AlchemicalNetwork. This is
            a returned value from the :py:meth:`create_network_subgraph`
            method.
        state
            The starting state for the AlchemicalNetwork.

        Returns
        -------
        A tuple containing the NetworkMark Subgraph, the specific NetworkMark
        Node within the Subgraph, and the ScopedKey of the NetworkMark.
        """

        network_sk = ScopedKey.from_str(network_node["_scoped_key"])
        scope = network_sk.scope

        network_mark = NetworkMark(target=str(network_sk), state=state)

        _, network_mark_node, scoped_key = self._keyed_chain_to_subgraph(
            KeyedChain.from_gufe(network_mark),
            scope=scope,
        )

        subgraph = Relationship.type("MARKS")(
            network_mark_node,
            network_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        return subgraph, network_mark_node, scoped_key

    def set_network_state(
        self, networks: list[ScopedKey], states: list[str]
    ) -> list[ScopedKey | None]:
        """Set the state of a group of AlchemicalNetworks.

        Parameters
        ----------
        networks
            A list networks to set the states for.
        states
            A list of states to set the networks to.

        Returns
        -------
        List[Optional[ScopedKey]]
            The list of ScopedKeys for networks that were updated. If the
            network could not be found in the database, ``None`` is returned at
            the corresponding index.
        """

        if len(networks) != len(states):
            msg = "networks and states must have the same length"
            raise ValueError(msg)

        for network, state in zip(networks, states):
            if network.qualname != "AlchemicalNetwork":
                raise ValueError(
                    "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
                )
            try:
                NetworkStateEnum(state)
            except ValueError:
                valid_states = [state.value for state in NetworkStateEnum]
                msg = f"'{state}' is not a valid state. Valid values include: {valid_states}"
                raise ValueError(msg)

        q = """
            WITH $inputs AS inputs
            UNWIND inputs AS x
            WITH x[0] as network, x[1] as state
            MATCH (:AlchemicalNetwork {`_scoped_key`: network})<-[:MARKS]-(nm:NetworkMark)
            SET nm.state = state
            RETURN network
        """
        inputs = [[str(network), state] for network, state in zip(networks, states)]

        results = self.execute_query(q, inputs=inputs)

        network_results = {}
        for record in results.records:
            network_sk_str = record["network"]
            network_results[network_sk_str] = ScopedKey.from_str(network_sk_str)

        return [network_results.get(str(network), None) for network in networks]

    def query_networks(
        self,
        *,
        name=None,
        key=None,
        scope: Scope | None = None,
        state: str | None = None,
    ) -> list[ScopedKey]:
        r"""Query for `AlchemicalNetwork`\s matching given attributes."""

        if scope is None:
            scope = Scope()

        query_params = dict(
            name_pattern=name,
            org_pattern=scope.org,
            campaign_pattern=scope.campaign,
            project_pattern=scope.project,
            state_pattern=state,
            gufe_key_pattern=None if key is None else str(key),
        )

        where_params = dict(
            name_pattern="an.name",
            org_pattern="an.`_org`",
            campaign_pattern="an.`_campaign`",
            project_pattern="an.`_project`",
            state_pattern="nm.state",
            gufe_key_pattern="an.`_gufe_key`",
        )

        conditions = []

        for k, v in query_params.items():
            if v is not None:
                conditions.append(f"{where_params[k]} =~ ${k}")

        where_clause = "WHERE " + " AND ".join(conditions) if len(conditions) else ""

        q = f"""
            MATCH (an:AlchemicalNetwork)<-[:MARKS]-(nm:NetworkMark)
            {where_clause}
            RETURN an._scoped_key as sk
        """

        results = self.execute_query(
            q,
            parameters_=query_params,
        )

        network_sks = []
        for record in results.records:
            sk = record["sk"]
            network_sks.append(ScopedKey.from_str(sk))

        return network_sks

    def query_transformations(self, *, name=None, key=None, scope: Scope = Scope()):
        r"""Query for `Transformation`\s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="Transformation|NonTransformation",
            additional=additional,
            key=key,
            scope=scope,
        )

    def query_chemicalsystems(self, *, name=None, key=None, scope: Scope = Scope()):
        r"""Query for `ChemicalSystem`\s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="ChemicalSystem", additional=additional, key=key, scope=scope
        )

    def get_network_transformations(self, network: ScopedKey) -> list[ScopedKey]:
        """List ScopedKeys for Transformations associated with the given AlchemicalNetwork."""
        q = """
        MATCH (:AlchemicalNetwork {_scoped_key: $network})-[:DEPENDS_ON]->(t:Transformation|NonTransformation)
        WITH t._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, network=str(network))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_transformation_networks(self, transformation: ScopedKey) -> list[ScopedKey]:
        """List ScopedKeys for AlchemicalNetworks associated with the given Transformation."""
        q = """
        MATCH (:Transformation|NonTransformation {_scoped_key: $transformation})<-[:DEPENDS_ON]-(an:AlchemicalNetwork)
        WITH an._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, transformation=str(transformation))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_network_chemicalsystems(self, network: ScopedKey) -> list[ScopedKey]:
        """List ScopedKeys for ChemicalSystems associated with the given AlchemicalNetwork."""
        q = """
        MATCH (:AlchemicalNetwork {_scoped_key: $network})-[:DEPENDS_ON]->(cs:ChemicalSystem)
        WITH cs._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, network=str(network))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_chemicalsystem_networks(self, chemicalsystem: ScopedKey) -> list[ScopedKey]:
        """List ScopedKeys for AlchemicalNetworks associated with the given ChemicalSystem."""
        q = """
        MATCH (:ChemicalSystem {_scoped_key: $chemicalsystem})<-[:DEPENDS_ON]-(an:AlchemicalNetwork)
        WITH an._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, chemicalsystem=str(chemicalsystem))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_transformation_chemicalsystems(
        self, transformation: ScopedKey
    ) -> list[ScopedKey]:
        """List ScopedKeys for the ChemicalSystems associated with the given Transformation."""
        q = """
        MATCH (:Transformation|NonTransformation {_scoped_key: $transformation})-[deps_on:DEPENDS_ON]->(cs:ChemicalSystem)
        WITH cs._scoped_key as sk
        ORDER BY deps_on.attribute
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, transformation=str(transformation))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_chemicalsystem_transformations(
        self, chemicalsystem: ScopedKey
    ) -> list[ScopedKey]:
        """List ScopedKeys for the Transformations associated with the given ChemicalSystem."""
        q = """
        MATCH (:ChemicalSystem {_scoped_key: $chemicalsystem})<-[:DEPENDS_ON]-(t:Transformation|NonTransformation)
        WITH t._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, chemicalsystem=str(chemicalsystem))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def _get_protocoldagresultrefs(self, q: str, scoped_key: ScopedKey):
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, scoped_key=str(scoped_key))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_transformation_results(self, transformation: ScopedKey) -> list[ScopedKey]:
        # get all task result protocoldagresultrefs corresponding to given transformation
        # returned in no particular order
        q = """
        MATCH (trans:Transformation|NonTransformation {_scoped_key: $scoped_key}),
              (trans)<-[:PERFORMS]-(:Task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = true
        WITH res._scoped_key as sk
        RETURN DISTINCT sk
        """
        return self._get_protocoldagresultrefs(q, transformation)

    def get_transformation_failures(self, transformation: ScopedKey) -> list[ScopedKey]:
        # get all task failure protocoldagresultrefs corresponding to given transformation
        # returned in no particular order
        q = """
        MATCH (trans:Transformation|NonTransformation {_scoped_key: $scoped_key}),
              (trans)<-[:PERFORMS]-(:Task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = false
        WITH res._scoped_key as sk
        RETURN DISTINCT sk
        """
        return self._get_protocoldagresultrefs(q, transformation)

    ## compute

    def set_network_strategy(
        self,
        network: ScopedKey,
        strategy: Strategy | None,
        strategy_state: StrategyState | None = None,
    ) -> ScopedKey | None:
        """Set the compute Strategy for the given AlchemicalNetwork.

        If `strategy` is ``None``, removes the strategy from the network and
        cleans up orphaned `Strategy` nodes.

        Parameters
        ----------
        network
            ScopedKey of the AlchemicalNetwork.
        strategy
            Strategy object (GufeTokenizable) or None to remove strategy.
        strategy_state
            Initial strategy state, if None uses defaults.

        Returns
        -------
        ScopedKey
            ScopedKey of the Strategy that was set, or None if strategy was removed.
        """
        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

        if strategy_state is None:
            strategy_state = StrategyState()

        with self.transaction() as tx:

            # check if there is an existing Strategy for the network; if so, remove it
            q = """
            MATCH (an:AlchemicalNetwork {_scoped_key: $network})<-[r:PROGRESSES]-(s)
            DELETE r

            // Clean up Strategy node if it has no remaining PROGRESSES relationships
            WITH s
            OPTIONAL MATCH (s)-[:PROGRESSES]->()
            WITH s, count(*) as remaining_relationships
            WHERE remaining_relationships = 0
            DELETE s
            """

            tx.run(q, network=str(network))

            # exit early if we didn't want a Strategy for this network
            if strategy is None:
                return None

            # set new Strategy for network, with new relationship
            network_node = self._get_node(network)
            subgraph, strategy_node, scoped_key = self._keyed_chain_to_subgraph(
                KeyedChain.from_gufe(strategy), scope=network.scope
            )

            subgraph = subgraph | Relationship.type("PROGRESSES")(
                strategy_node, network_node, **strategy_state.to_dict()
            )

            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

        return scoped_key

    def get_network_strategy(self, network: ScopedKey) -> Strategy | None:
        """Get the Strategy for the given AlchemicalNetwork.

        Parameters
        ----------
        network
            ScopedKey of the AlchemicalNetwork.

        Returns
        -------
        GufeTokenizable | None
            Strategy object or None if no strategy is set.
        """
        q = """
        MATCH (an:AlchemicalNetwork {_scoped_key: $network})<-[:PROGRESSES]-(s)
        RETURN s
        """

        def _node_to_gufe(node):
            return self._subgraph_to_gufe([node], node)[node]

        with self.transaction() as tx:
            result = tx.run(q, network=str(network))
            record = result.single()

        if record:
            strategy_node = record_data_to_node(record["s"])
            return _node_to_gufe(strategy_node)

        return None

    def get_network_strategy_state(self, network: ScopedKey) -> StrategyState | None:
        """Get the StrategyState for the given AlchemicalNetwork.

        Parameters
        ----------
        network
            ScopedKey of the AlchemicalNetwork.

        Returns
        -------
        StrategyState | None
            Strategy state or None if no strategy is set.
        """
        q = """
        MATCH (an:AlchemicalNetwork {_scoped_key: $network})<-[r:PROGRESSES]-()
        RETURN properties(r) AS state_props
        """

        with self.transaction() as tx:
            result = tx.run(q, network=str(network))
            record = result.single()

        if record:
            state_props = record["state_props"]
            return StrategyState.from_dict(state_props)
        return None

    def get_strategies_for_execution(
        self, scopes: list[Scope] | None = None, min_sleep_interval: int = 0
    ) -> list[tuple[ScopedKey, ScopedKey, StrategyState]]:
        """Get strategies that are ready for execution by the Strategist service.

        Returns strategies that are:
        - Not disabled
        - Not in error status
        - Due for execution based on sleep interval

        Parameters
        ----------
        scopes
            List of scopes to filter by, if None returns all scopes
        min_sleep_interval
            Minimum sleep interval enforced by Strategist service

        Returns
        -------
        list[tuple[ScopedKey, ScopedKey, StrategyState]]
            List of (network_sk, strategy_sk, strategy_state) tuples
        """
        scope_filter = ""
        if scopes:
            scope_conditions = []
            for scope in scopes:
                conditions = []
                if scope.org is not None:
                    conditions.append(f"an._org = '{scope.org}'")
                if scope.campaign is not None:
                    conditions.append(f"an._campaign = '{scope.campaign}'")
                if scope.project is not None:
                    conditions.append(f"an._project = '{scope.project}'")
                if conditions:
                    scope_conditions.append(f"({' AND '.join(conditions)})")

            if scope_conditions:
                scope_filter = f"WHERE {' OR '.join(scope_conditions)}"

        q = f"""
        MATCH (an:AlchemicalNetwork)<-[r:PROGRESSES]-(s)
        {scope_filter}
        
        // Filter out disabled and error strategies
        WHERE r.mode <> 'disabled' AND r.status <> 'error'
        
        // Check if strategy is due for execution
        WITH an, s, r,
             coalesce(r.last_iteration, datetime('1970-01-01T00:00:00Z')) AS last_iter,
             CASE WHEN $min_sleep_interval > r.sleep_interval 
                  THEN $min_sleep_interval 
                  ELSE r.sleep_interval 
             END AS effective_sleep
        WHERE datetime() >= last_iter + duration({{seconds: effective_sleep}})
        
        RETURN an._scoped_key AS network_sk, 
               s._scoped_key AS strategy_sk,
               properties(r) AS state_props
        """

        results = []
        with self.transaction() as tx:
            result = tx.run(q, min_sleep_interval=min_sleep_interval)
            for record in result:
                network_sk = ScopedKey.from_str(record["network_sk"])
                strategy_sk = ScopedKey.from_str(record["strategy_sk"])
                state_props = record["state_props"]
                strategy_state = StrategyState.from_dict(state_props)
                results.append((network_sk, strategy_sk, strategy_state))

        return results

    def update_strategy_state(
        self, network: ScopedKey, strategy_state: StrategyState
    ) -> ScopedKey | None:
        """Update the StrategyState for the given AlchemicalNetwork.

        Parameters
        ----------
        network
            ScopedKey of the AlchemicalNetwork.
        strategy_state
            Updated strategy state.

        Returns
        -------
        ScopedKey | None
            The ScopedKey of the AlchemicalNetwork if StrategyState
            successfully updated; ``None`` otherwise.

        """

        q = """
        MATCH (an:AlchemicalNetwork {_scoped_key: $network})<-[r:PROGRESSES]-()
        SET r = $strategy_state
        RETURN r
        """

        strategy_state_props = strategy_state.to_dict()

        with self.transaction() as tx:
            records = (
                tx.run(q, network=str(network), strategy_state=strategy_state_props)
                .to_eager_result()
                .records
            )

        if not records:
            return None

        return network

    def register_computeservice(
        self, compute_service_registration: ComputeServiceRegistration
    ):
        """Register a ComputeServiceRegistration uniquely identifying a running
        ComputeService.

        A ComputeServiceRegistration node is used for CLAIMS relationships on
        Tasks to avoid collisions in Task execution.

        """

        node = Node(
            "ComputeServiceRegistration", **compute_service_registration.to_dict()
        )

        with self.transaction() as tx:
            create_subgraph(tx, Subgraph() | node)

            if compute_service_registration.manager_name:
                query = """
                MATCH (cmr:ComputeManagerRegistration {name: $manager_name}),
                      (csr:ComputeServiceRegistration {manager_name: $manager_name,
                                                       identifier: $identifier})
                CREATE (cmr)-[rel:MANAGES]->(csr)
                RETURN cmr, rel, csr
                """
                results = tx.run(
                    query,
                    manager_name=compute_service_registration.manager_name,
                    identifier=compute_service_registration.identifier,
                )
                if not len(list(results)):
                    raise ValueError("Could not find ComputeManagerRegistration")

        return compute_service_registration.identifier

    def deregister_computeservice(self, compute_service_id: ComputeServiceID):
        """Remove the registration for the given ComputeServiceID from the
        state store.

        This wil remove the ComputeServiceRegistration node, and all its CLAIMS
        relationships to Tasks.

        All Tasks with CLAIMS relationships to the ComputeServiceRegistration
        and with status `running` will have their status set to `waiting`.

        """

        q = f"""
        MATCH (n:ComputeServiceRegistration {{identifier: $compute_service_id}})

        OPTIONAL MATCH (n)-[cl:CLAIMS]->(t:Task {{status: '{TaskStatusEnum.running.value}'}})
        SET t.status = '{TaskStatusEnum.waiting.value}'

        WITH n, n.identifier as identifier

        DETACH DELETE n

        RETURN identifier
        """

        with self.transaction() as tx:
            res = tx.run(q, compute_service_id=str(compute_service_id))
            identifier = next(res)["identifier"]

        return ComputeServiceID(identifier)

    def heartbeat_computeservice(
        self, compute_service_id: ComputeServiceID, heartbeat: datetime.datetime
    ):
        """Update the heartbeat for the given ComputeServiceID."""

        q = f"""
        MATCH (n:ComputeServiceRegistration {{identifier: $compute_service_id}})
        SET n.heartbeat = datetime('{heartbeat.isoformat()}')

        """
        with self.transaction() as tx:
            tx.run(q, compute_service_id=str(compute_service_id))

        return compute_service_id

    def expire_registrations(self, expire_time: datetime.datetime):
        """Remove all registrations with last heartbeat prior to the given `expire_time`."""
        q = f"""
        MATCH (n:ComputeServiceRegistration)
        WHERE n.heartbeat < datetime('{expire_time.isoformat()}')

        WITH n

        OPTIONAL MATCH (n)-[cl:CLAIMS]->(t:Task {{status: '{TaskStatusEnum.running.value}'}})
        SET t.status = '{TaskStatusEnum.waiting.value}'

        WITH n, n.identifier as ident

        DETACH DELETE n

        RETURN ident
        """
        with self.transaction() as tx:
            res = tx.run(q)

            identities = set()
            for rec in res:
                identities.add(rec["ident"])

        return [ComputeServiceID(i) for i in identities]

    def log_failure_compute_service(
        self,
        compute_service_id: ComputeServiceID,
        failure_time: datetime.datetime,
    ) -> ComputeServiceID:
        """Add a reported compute service failure to the database.

        Parameters
        ----------
        compute_service_id
            The identifier for the compute service that failed.
        failure_time
            The time the failure should be reported as.
        """
        q = """
        MATCH (n:ComputeServiceRegistration {identifier: $compute_service_id})
        SET n.failure_times = [datetime($failure_time)] + n.failure_times
        """

        with self.transaction() as tx:
            tx.run(
                q,
                compute_service_id=str(compute_service_id),
                failure_time=failure_time,
            )

        return compute_service_id

    def compute_services_can_claim(
        self,
        compute_service_ids: list[ComputeServiceID],
        forgive_time: datetime.datetime,
        max_failures: int,
    ) -> list[bool]:
        """Check compute services are able to claim tasks.

        Parameters
        ----------
        compute_service_ids
            The compute services to validate.
        forgive_time
            The time cutoff used to filter failure time reports for the compute
            services. Only entries occuring after this time are considered.
        max_failures
            The number of failures allowed to occur between ``forgive_time``
            and now. Any value greater than this denies the claim request.
        """
        # get the number of failures that occured after `forgive_time`
        query = """
        UNWIND $compute_service_ids as compute_service_id
        MATCH (cs:ComputeServiceRegistration {identifier: compute_service_id})
        SET cs.failure_times = [entry IN cs.failure_times WHERE entry > datetime($forgive_time)]
        RETURN size(cs.failure_times) as n_failures
        """
        results = self.execute_query(
            query, compute_service_ids=compute_service_ids, forgive_time=forgive_time
        )

        return [record["n_failures"] <= max_failures for record in results.records]

    def compute_service_can_claim(
        self,
        compute_service_id: ComputeServiceID,
        forgive_time: datetime.datetime,
        max_failures: int,
    ) -> bool:
        """Check if a compute service is able to claim a ``Task``.

        Parameters
        ----------
        compute_service_id
            The compute service to validate.
        forgive_time
            The time cutoff used to filter failure time reports for the compute
            service. Only entries occuring after this time are considered.
        max_failures
            The number of failures allowed to occur between ``forgive_time``
            and now. Any value greater than this denies the claim request.
        """
        return self.compute_services_can_claim(
            [compute_service_id], forgive_time, max_failures
        )[0]

    ## compute manager

    def register_computemanager(
        self, compute_manager_registration: ComputeManagerRegistration
    ) -> ComputeManagerID:
        """Register a compute manager with the statestore.

        Parameters
        ----------
        compute_manager_registration
            The compute manager registration.

        Returns
        -------
        compute_manager_id
            The compute manager ID string containing the name and the
            UUID.

        Raises
        ------
        ValueError
            Raised when a compute manager is already registered with
            the provided name.

        """
        with self.transaction() as tx:

            # first check if a compute manager with the given name is
            # already registered
            query = """
            MATCH (cmr:ComputeManagerRegistration {name: $name})
            RETURN cmr
            """

            res = tx.run(query, name=compute_manager_registration.name)

            if res.to_eager_result().records:
                raise ValueError("ComputeManager with this name is already registered")

            # create the registrattion for the manager and merge it into
            # the database
            node = Node(
                "ComputeManagerRegistration", **compute_manager_registration.to_dict()
            )

            create_subgraph(tx, Subgraph() | node)

            # check for orphaned compute services that were previously
            # managed by a manager with the same name; reattach if found
            reattach_compute_service_query = """
            MATCH (csr:ComputeServiceRegistration {manager_name: $manager_name}),
                  (cmr:ComputeManagerRegistration {name: $manager_name,
                                                   uuid: $uuid})
            CREATE (cmr)-[rel:MANAGES]->(csr)
            """

            tx.run(
                reattach_compute_service_query,
                manager_name=compute_manager_registration.name,
                uuid=compute_manager_registration.uuid,
            )

        compute_manager_id = ComputeManagerID(
            compute_manager_registration.name + "-" + compute_manager_registration.uuid
        )

        return compute_manager_id

    def deregister_computemanager(self, compute_manager_id: ComputeManagerID):
        """Remove the compute manager registration from the statestore.

        Uses the name and UUID from a ComputeManagerID to deregister a
        compute manager's registration. First, the MANAGES
        relationship with compute services' registration are
        removed. After this, the compute manager registration node is
        removed as long as the registration does not have the ERROR
        status.

        Parameters
        ----------
        compute_manager_id
            The compute manager ID string containing the name and the UUID.

        """
        name, uuid = compute_manager_id.name, compute_manager_id.uuid

        query = """
        MATCH (cmr:ComputeManagerRegistration {name: $name, uuid: $uuid})
        WHERE cmr.status <> "ERROR"
        DETACH DELETE cmr
        """

        self.execute_query(query, name=name, uuid=uuid)

    def expire_computemanager_registrations(
        self, expire_time_ok: datetime.datetime, expire_time_error: datetime.datetime
    ):
        """Remove expired compute managers from the statestore.

        This method checks the status of compute managers and removes
        those that have expired based on their last status update
        time.

        Parameters
        ----------
        expire_time_ok
            The expiration time for "OK" compute managers. Managers with a last
            update time earlier than the expiration cutoff will be removed.

        expire_time_error
            The expiration time for "ERROR" compute managers. Managers
            with a last update time earlier than the expiration cutoff will be
            removed.

        """

        # Match all expired ComputeManagerRegistration nodes based on
        # their status and detach delete them.
        query = """
        MATCH (cmr:ComputeManagerRegistration)
        WHERE (cmr.last_status_update < datetime($expire_time_ok) AND
               cmr.status = $ok_status)
              OR
              (cmr.last_status_update < datetime($expire_time_error) AND
               cmr.status = $error_status)
        DETACH DELETE cmr
        """

        params = {
            "ok_status": ComputeManagerStatus.OK.value,
            "error_status": ComputeManagerStatus.ERROR.value,
            "expire_time_ok": expire_time_ok.isoformat(),
            "expire_time_error": expire_time_error.isoformat(),
        }

        results = self.execute_query(query, **params)

    @chainable
    def get_compute_manager_id(self, name: str, tx=None):
        query = """
        MATCH (cmr:ComputeManagerRegistration {name: $name})
        RETURN cmr.uuid AS uuid
        """
        records = tx.run(query, name=name).to_eager_result().records
        if not records:
            return None
        result = records[0]
        uuid = result["uuid"]
        return ComputeManagerID(f"{name}-{uuid}")

    @chainable
    def clear_errored_computemanager(
        self, compute_manager_id: ComputeManagerID, tx=None
    ):
        """Remove a compute manager with an ERROR status.

        Parameters
        ----------
        compute_manager_id
            The compute manager ID string containing the name and the
            UUID.

        Raises
        ------
        ValueError
            Raised when the ERROR compute manager cannot be found in
            the database
        """

        query = """
        MATCH (cmr:ComputeManagerRegistration {name: $name, uuid: $uuid, status: $status})
        DETACH DELETE cmr
        RETURN cmr
        """

        results = tx.run(
            query, status=ComputeManagerStatus.ERROR, **compute_manager_id.to_dict()
        ).to_eager_result()

        if not results.records:
            raise ValueError(
                "Could not find an ERROR compute manager with the provided name and UUID"
            )

    def get_computemanager_instruction(
        self,
        compute_manager_id: ComputeManagerID,
        forgive_time: datetime.datetime,
        max_failures: int,
        scopes: list[Scope],
    ) -> tuple[ComputeManagerInstruction, dict]:
        """Return an instruction for a compute manager based on the contents of the statestore.

        This method returns one of three instructions along with supporting data:

        1. "OK" with a list of ComputeServiceIDs and the number of available tasks
        2. "SKIP" with a list of ComputeServiceIDs
        3. "SHUTDOWN" with an error message

        Parameters
        ----------
        compute_manager_id
            The compute manager ID string containing the name and the
            UUID.

        forgive_time
            The time at which a failure from a compute service is
            considered forgiven.

         max_failures
            The number of failures a compute service is allowed to
            have (before the forgive time) before it is no longer
            allowed to claim a task. If any managed compute services
            have failues that exceed this value, the returned
            instruction will be SKIP.

        scopes
            The scopes to consider when determining available tasks.

        Returns
        -------
        A tuple with whose first value is the instruction enumeration
        and whose second value is data associated with that
        instruction.

        """

        name, uuid = compute_manager_id.name, compute_manager_id.uuid

        # get the target compute manager along with any compute
        # services it might manage. Expected structure of output when
        # a manager is found:
        #
        # With no managed services:
        # +--------------+-----------+
        # | csm          | csr_id    |
        # +--------------+-----------+
        # | <manager_id> | None      |
        # +--------------+-----------+
        #
        # With N (one or more) managed services
        # +--------------+------------------------+
        # | csm          | csr_id                 |
        # +--------------+------------------------+
        # | <manager_id> | <ComputeServiceID-1>   |
        # | <manager_id> | <ComputeServiceID-2>   |
        # | <manager_id> | <ComputeServiceID-...> |
        # | <manager_id> | <ComputeServiceID-N-1> |
        # | <manager_id> | <ComputeServiceID-N>   |
        # +--------------+------------------------+

        query = """
        MATCH (csm:ComputeManagerRegistration {name: $name,
                                               uuid: $uuid})
        OPTIONAL MATCH (csm)-[rel:MANAGES]->(csr:ComputeServiceRegistration)
        RETURN csm, csr.identifier as csr_id
        """

        results = self.execute_query(query, name=name, uuid=uuid)

        # no compute manager was found the given name and UUID, issue a SHUTDOWN
        if len(results.records) == 0:
            msg = "no compute manager was found with the given manager name and UUID"
            return ComputeManagerInstruction.SHUTDOWN, {"message": msg}

        csr_ids = []
        for record in results.records:
            # if the manager has managed services
            if csr_id := record["csr_id"]:
                csr_ids.append(ComputeServiceID(csr_id))

        if csr_ids:
            if not all(
                self.compute_services_can_claim(csr_ids, forgive_time, max_failures)
            ):
                return ComputeManagerInstruction.SKIP, {"compute_service_ids": csr_ids}

        # determine how many tasks are available
        tasks = []
        for scope in scopes:
            params = {
                "org": scope.org,
                "campaign": scope.campaign,
                "project": scope.project,
                "waiting_status": TaskStatusEnum.waiting.value,
            }
            query = """
            MATCH (task:Task {_org: $org,
                              _campaign: $campaign,
                              _project: $project,
                              status: $waiting_status}),
                  (task)<-[:ACTIONS]-(:TaskHub)
            RETURN task._gufe_key as task
            """
            result = self.execute_query(query, **params)
            tasks += [record["task"] for record in result.records]

        return ComputeManagerInstruction.OK, {
            "compute_service_ids": csr_ids,
            "num_tasks": len(set(tasks)),
        }

    def update_compute_manager_status(
        self,
        compute_manager_id: ComputeManagerID,
        status: ComputeManagerStatus,
        detail: str | None = None,
        saturation: float | None = None,
        update_time: datetime.datetime | None = None,
    ):
        """Update the status of a compute manager.

        Statuses can either be passed in as strings or instances of
        the ComputeManagerStatus enumeration, though the latter is
        safer and preferred.

        Parameters
        ----------
        compute_manager_id
            The compute manager ID string containing the name and the
            UUID.

        status
            An instance of the ComputeManagerStatus string
            enumeration, whose supported values are "OK" and
            "ERROR".

        detail
            A message to be included with the status update. This is
            only allowed and required by the ERROR status. This
            message should indicate to administrators why the compute
            manager entered the ERROR status.

        update_time
            The time to set as the last status update time in the
            database. Defaults to None, which will use the current
            time when updating.

        """
        # validate the status string/enum
        try:
            status = ComputeManagerStatus(status)
        except ValueError:
            raise ValueError(f'"{status}" is not a valid ComputeManagerStatus')

        # only match to valid states
        match status:
            case ComputeManagerStatus.OK:
                # OK requires saturation
                if saturation is None:
                    raise ValueError(
                        f"saturation is required for the '{ComputeManagerStatus.OK}' status"
                    )
                # OK disallows detail
                if detail:
                    raise ValueError(
                        f"detail should only be provided for the '{ComputeManagerStatus.ERROR}' status"
                    )
            case ComputeManagerStatus.ERROR:
                # ERROR disallows saturation
                if saturation is not None:
                    raise ValueError(
                        f"saturation should only be provided for the '{ComputeManagerStatus.OK}' status"
                    )
                # ERROR requires detail
                if not detail:
                    raise ValueError(
                        f"detail is required for the '{ComputeManagerStatus.ERROR}' status"
                    )

        name, uuid = compute_manager_id.name, compute_manager_id.uuid

        if saturation is not None:
            if not 0 <= saturation <= 1:
                raise ValueError("saturation must be between 0 and 1")

        query = """
        MATCH (cmr: ComputeManagerRegistration {name: $name, uuid: $uuid})
        SET cmr.status = $status
        SET cmr.detail = $detail
        SET cmr.saturation = $saturation
        SET cmr.last_status_update = datetime($update_time)
        RETURN cmr
        """

        results = self.execute_query(
            query,
            uuid=uuid,
            name=name,
            status=status.value,
            detail=detail,
            saturation=saturation,
            update_time=(
                update_time or datetime.datetime.now(tz=datetime.UTC)
            ).isoformat(),
        )

        if len(results.records) == 0:
            msg = f"No record for ComputeManager: {compute_manager_id}"
            raise ValueError(msg)

    ## task hubs

    def create_taskhub_subgraph(self, network_node: Node):
        """Create a Subgraph for an AlchemicalNetwork's TaskHub.

        Parameters
        ----------
        network_node
            A Node representing the target AlchemicalNetwork. This is
            a returned value from the :py:meth:`create_network_subgraph`
            method.

        Returns
        -------
        A tuple containing the TaskHub Subgraph, the specific TaskHub
        Node within the Subgraph, and the ScopedKey of the TaskHub.
        """
        network_sk = ScopedKey.from_str(network_node["_scoped_key"])
        scope = network_sk.scope

        taskhub = TaskHub(network=str(network_sk))

        _, taskhub_node, scoped_key = self._keyed_chain_to_subgraph(
            KeyedChain.from_gufe(taskhub),
            scope=scope,
        )

        subgraph = Relationship.type("PERFORMS")(
            taskhub_node,
            network_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        return subgraph, taskhub_node, scoped_key

    def query_taskhubs(
        self, scope: Scope | None = Scope(), return_gufe: bool = False
    ) -> list[ScopedKey] | dict[ScopedKey, TaskHub]:
        r"""Query for `TaskHub`\s matching the given criteria.

        Parameters
        ----------
        return_gufe
            If True, return a dict with `ScopedKey`\s as keys, `TaskHub`
            instances as values. Otherwise, return a list of `ScopedKey`\s.

        """
        return self._query(qualname="TaskHub", scope=scope, return_gufe=return_gufe)

    def get_taskhubs(
        self, network_scoped_keys: list[ScopedKey], return_gufe: bool = False
    ) -> list[ScopedKey | TaskHub]:
        r"""Get the TaskHubs for the given AlchemicalNetworks.

        Parameters
        ----------
        return_gufe
            If True, return `TaskHub` instances.
            Otherwise, return `ScopedKey`\s.

        """

        # TODO: this could fail better, report all instances rather than first
        for network_scoped_key in network_scoped_keys:
            if network_scoped_key.qualname != "AlchemicalNetwork":
                raise ValueError(
                    "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
                )

        query = """
        UNWIND $network_scoped_keys AS network_scoped_key
        MATCH (th:TaskHub {network: network_scoped_key})-[:PERFORMS]->(an:AlchemicalNetwork)
        RETURN th, an
        """

        query_results = self.execute_query(
            query, network_scoped_keys=list(map(str, network_scoped_keys))
        )

        def _node_to_gufe(node):
            return self._subgraph_to_gufe([node], node)[node]

        def _node_to_scoped_key(node):
            return ScopedKey.from_str(node["_scoped_key"])

        transform_function = _node_to_gufe if return_gufe else _node_to_scoped_key
        transform_results = {}
        for record in query_results.records:
            node = record_data_to_node(record["th"])
            network_scoped_key = record["an"]["_scoped_key"]
            transform_results[network_scoped_key] = transform_function(node)

        return [
            transform_results.get(str(network_scoped_key))
            for network_scoped_key in network_scoped_keys
        ]

    def get_taskhub(
        self, network: ScopedKey, return_gufe: bool = False
    ) -> ScopedKey | TaskHub:
        """Get the TaskHub for the given AlchemicalNetwork.

        Parameters
        ----------
        return_gufe
            If True, return a `TaskHub` instance.
            Otherwise, return a `ScopedKey`.

        """

        return self.get_taskhubs([network], return_gufe)[0]

    def delete_taskhub(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Delete a TaskHub for a given AlchemicalNetwork."""

        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

        taskhub = self.get_taskhub(network)

        q = """
        MATCH (th:TaskHub {_scoped_key: $taskhub})
        DETACH DELETE th
        """
        self.execute_query(q, taskhub=str(taskhub))

        return taskhub

    def set_taskhub_weight(
        self, networks: list[ScopedKey], weights: list[float]
    ) -> list[ScopedKey | None]:
        """Set the weights for the TaskHubs associated with the given
        AlchemicalNetworks.

        """

        for weight in weights:
            if not 0 <= weight <= 1:
                raise ValueError("all `weights` must be between 0 and 1 (inclusive)")

        for network in networks:
            if network.qualname != "AlchemicalNetwork":
                raise ValueError(
                    "a `networks` ScopedKey does not correspond to an `AlchemicalNetwork`"
                )

        if len(networks) != len(weights):
            raise ValueError("length of `networks` and `weights` must be the same")

        q = """
        WITH $inputs AS inputs
        UNWIND inputs AS x
        WITH x[0] as network, x[1] as weight
        MATCH (th:TaskHub {network: network})
        SET th.weight = weight
        RETURN network
        """
        inputs = [[str(network), weight] for network, weight in zip(networks, weights)]

        results = self.execute_query(q, inputs=inputs)

        network_results = {}
        for record in results.records:
            network_sk_str = record["network"]
            network_results[network_sk_str] = ScopedKey.from_str(network_sk_str)

        return [network_results.get(str(network), None) for network in networks]

    def get_taskhub_actioned_tasks(
        self,
        taskhubs: list[ScopedKey],
    ) -> list[dict[ScopedKey, float]]:
        """Get the Tasks that the given TaskHubs ACTIONS.

        Parameters
        ----------
        taskhubs
            The ScopedKeys of the TaskHubs to query.

        Returns
        -------
        tasks
            A list of dicts, one per TaskHub, which contains the Task ScopedKeys that are
            actioned on the given TaskHub as keys, with their weights as values.
        """
        th_scoped_keys = [str(taskhub) for taskhub in taskhubs if taskhub is not None]
        q = """
           UNWIND $taskhubs as th_sk
           MATCH (th: TaskHub {_scoped_key: th_sk})-[a:ACTIONS]->(t:Task)
           RETURN t._scoped_key, a.weight, th._scoped_key
        """

        results = self.execute_query(q, taskhubs=th_scoped_keys)

        data = {taskhub: {} for taskhub in taskhubs}
        for record in results.records:
            th_sk = ScopedKey.from_str(record["th._scoped_key"])
            t_sk = ScopedKey.from_str(record["t._scoped_key"])
            weight = record["a.weight"]

            data[th_sk][t_sk] = weight

        return [data[taskhub] for taskhub in taskhubs]

    def get_task_actioned_networks(self, task: ScopedKey) -> dict[ScopedKey, float]:
        """Get all AlchemicalNetwork ScopedKeys whose TaskHub ACTIONS a given Task.

        Parameters
        ----------
        task
            The ScopedKey of the Task to obtain actioned AlchemicalNetworks
            for.

        Returns
        -------
        networks
            A dict with AlchemicalNetwork ScopedKeys whose TaskHub actions a
            given Task as keys, Task weights as values.
        """

        q = """
           MATCH (an:AlchemicalNetwork)<-[:PERFORMS]-(TaskHub)-[a:ACTIONS]->(Task {_scoped_key: $scoped_key})
           RETURN an._scoped_key, a.weight
        """

        with self.transaction() as tx:
            results = tx.run(q, scoped_key=str(task)).to_eager_result()

        return {
            ScopedKey.from_str(record.get("an._scoped_key")): record.get("a.weight")
            for record in results.records
        }

    def get_taskhub_weight(self, networks: list[ScopedKey]) -> list[float]:
        """Get the weight for the TaskHubs associated with the given
        AlchemicalNetworks.

        """

        for network in networks:
            if network.qualname != "AlchemicalNetwork":
                raise ValueError(
                    "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
                )

        networks_scoped_keys = [
            str(network) for network in networks if network is not None
        ]

        q = """
        UNWIND $networks as network
        MATCH (th:TaskHub {network: network})
        RETURN network, th.weight
        """

        results = self.execute_query(q, networks=networks_scoped_keys)

        network_weights = {str(network): None for network in networks}
        for record in results.records:
            weight = record["th.weight"]
            network_sk_str = record["network"]
            network_weights[network_sk_str] = weight

        return [network_weights[str(network)] for network in networks]

    def action_tasks(
        self,
        tasks: list[ScopedKey],
        taskhub: ScopedKey,
    ) -> list[ScopedKey | None]:
        """Add Tasks to the TaskHub for a given AlchemicalNetwork.

        Note: the Tasks must be within the same scope as the AlchemicalNetwork,
        and must correspond to a Transformation in the AlchemicalNetwork.

        A given compute task can be represented in any number of
        AlchemicalNetwork TaskHubs, or none at all.

        Only Tasks with status 'waiting', 'running', or 'error' can be
        actioned.

        """
        # since UNWIND doesn't guarantee order, we need to keep track manually
        # so we can properly return `None` if needed
        task_map = {str(task): None for task in tasks}

        task_scoped_keys = [str(task) for task in tasks if task is not None]

        q = """
        // get our TaskHub
        UNWIND $task_scoped_keys as task_sk
        MATCH (th:TaskHub {_scoped_key: $taskhub_scoped_key})-[:PERFORMS]->(an:AlchemicalNetwork)

        // get the task we want to add to the hub; check that it connects to same network
        MATCH (task:Task {_scoped_key: task_sk})-[:PERFORMS]->(:Transformation|NonTransformation)<-[:DEPENDS_ON]-(an)

        // only proceed for cases where task is not already actioned on hub
        // and where the task is either in 'waiting', 'running', or 'error' status
        WITH th, an, task
        WHERE NOT (th)-[:ACTIONS]->(task)
          AND task.status IN [$waiting, $running, $error]

        // create the connection
        CREATE (th)-[ar:ACTIONS {weight: 0.5}]->(task)

        // set the task property to the scoped key of the Task
        // this is a convenience for when we have to loop over relationships in Python
        SET ar.task = task._scoped_key

        // we want to preserve the list of tasks for the return, so we need to make a subquery
        // since the subsequent WHERE clause could reduce the records in task
        WITH task, th
        CALL {
            WITH task, th
            MATCH (trp: TaskRestartPattern)-[:ENFORCES]->(th)
            WHERE NOT (trp)-[:APPLIES]->(task)

            CREATE (trp)-[:APPLIES {num_retries: 0}]->(task)
        }

        RETURN task
        """

        results = self.execute_query(
            q,
            task_scoped_keys=task_scoped_keys,
            taskhub_scoped_key=str(taskhub),
            waiting=TaskStatusEnum.waiting.value,
            running=TaskStatusEnum.running.value,
            error=TaskStatusEnum.error.value,
        )

        # update our map with the results, leaving None for tasks that aren't found
        for task_record in results.records:
            sk = task_record["task"]["_scoped_key"]
            task_map[str(sk)] = ScopedKey.from_str(sk)

        return [task_map[str(t)] for t in tasks]

    def set_task_weights(
        self,
        tasks: dict[ScopedKey, float] | list[ScopedKey],
        taskhub: ScopedKey,
        weight: float | None = None,
    ) -> list[ScopedKey | None]:
        """Sets weights for the ACTIONS relationship between a TaskHub and a Task.

        This is used to set the relative probabilistic execution order of a
        Task in a TaskHub. Note that this concept is orthogonal to priority in
        that tasks of higher priority will be executed before tasks of lower
        priority, but tasks of the same priority will be distributed according
        to their weights.

        The weights can be set by either a list and a scalar, or a dict of
        {ScopedKey: weight} pairs.

        Must be called after `action_tasks` to have any effect; otherwise, the
        TaskHub will not have an ACTIONS relationship to the Task.

        Parameters
        ----------
        tasks: Union[Dict[ScopedKey, float], List[ScopedKey]]
            If a dict, the keys are the ScopedKeys of the Tasks, and the values are the weights.
            If a list, the weights are set to the scalar value given by the `weight` argument.

        taskhub: ScopedKey
            The ScopedKey of the TaskHub associated with the Tasks.

        weight: Optional[float]
            If `tasks` is a list, this is the weight to set for each Task.

        Returns
        -------
        List[ScopedKey, None]
            A list of ScopedKeys for each Task whose weight was set.
            `None` is given for Tasks that weight was not set for; this could
            be because the TaskHub doesn't have an ACTIONS relationship with the Task,
            or the Task doesn't exist at all

        """
        results = []
        tasks_changed = []
        with self.transaction() as tx:
            if isinstance(tasks, dict):
                if weight is not None:
                    raise ValueError(
                        "Cannot set `weight` to a scalar if `tasks` is a dict"
                    )

                if not all([0 <= weight <= 1 for weight in tasks.values()]):
                    raise ValueError("weights must be between 0 and 1 (inclusive)")

                tasks_list = [{"task": str(t), "weight": w} for t, w in tasks.items()]

                q = """
                UNWIND $tasks_list AS item
                MATCH (th:TaskHub {_scoped_key: $taskhub})-[ar:ACTIONS]->(task:Task {_scoped_key: item.task})
                SET ar.weight = item.weight
                RETURN task, ar
                """
                results.append(
                    tx.run(
                        q, taskhub=str(taskhub), tasks_list=tasks_list
                    ).to_eager_result()
                )

            elif isinstance(tasks, list):
                if weight is None:
                    raise ValueError(
                        "Must set `weight` to a scalar if `tasks` is a list"
                    )

                if not 0 <= weight <= 1:
                    raise ValueError("weight must be between 0 and 1 (inclusive)")

                tasks_list = [str(t) for t in tasks]

                q = """
                UNWIND $tasks_list AS task_sk
                MATCH (th:TaskHub {_scoped_key: $taskhub})-[ar:ACTIONS]->(task:Task {_scoped_key: task_sk})
                SET ar.weight = $weight
                RETURN task, ar
                """
                results.append(
                    tx.run(
                        q, taskhub=str(taskhub), tasks_list=tasks_list, weight=weight
                    ).to_eager_result()
                )

        # return ScopedKeys for Tasks we changed; `None` for tasks we didn't
        for res in results:
            for record in res.records:
                task = record["task"]
                tasks_changed.append(
                    ScopedKey.from_str(task["_scoped_key"])
                    if task is not None
                    else None
                )

        return tasks_changed

    def get_task_weights(
        self,
        tasks: list[ScopedKey],
        taskhub: ScopedKey,
    ) -> list[float | None]:
        """Get weights for the ACTIONS relationship between a TaskHub and a Task.

        Parameters
        ----------
        tasks
            The ScopedKeys of the Tasks to get the weights for.
        taskhub
            The ScopedKey of the TaskHub associated with the Tasks.

        Returns
        -------
        weights
            Weights for the list of Tasks, in the same order.
        """

        with self.transaction() as tx:
            q = """
            UNWIND $tasks_list AS task_scoped_key
            OPTIONAL MATCH (th:TaskHub {_scoped_key: $taskhub})-[ar:ACTIONS]->(task:Task {_scoped_key: task_scoped_key})
            RETURN task_scoped_key, ar.weight AS weight
            """

            result = tx.run(q, taskhub=str(taskhub), tasks_list=list(map(str, tasks)))
            results = result.data()

        weights = [record["weight"] for record in results]

        return weights

    @chainable
    def cancel_tasks(
        self,
        tasks: list[ScopedKey],
        taskhub: ScopedKey,
        tx=None,
    ) -> list[ScopedKey | None]:
        """Remove Tasks from the TaskHub for a given AlchemicalNetwork.

        Note: Tasks must be within the same scope as the AlchemicalNetwork.

        A given Task can be represented in many AlchemicalNetwork TaskHubs, or
        none at all.

        """
        query = """
        UNWIND $task_scoped_keys AS task_scoped_key
        MATCH (th:TaskHub {_scoped_key: $taskhub_scoped_key})-[ar:ACTIONS]->(task:Task {_scoped_key: task_scoped_key})
        DELETE ar

        WITH task, th
        CALL {
            WITH task, th
            MATCH (task)<-[applies:APPLIES]-(:TaskRestartPattern)-[:ENFORCES]->(th)
            DELETE applies
        }

        RETURN task._scoped_key as task_scoped_key
        """
        results = tx.run(
            query,
            task_scoped_keys=list(map(str, tasks)),
            taskhub_scoped_key=str(taskhub),
        ).to_eager_result()

        returned_keys = {record["task_scoped_key"] for record in results.records}
        filtered_tasks = [
            task if str(task) in returned_keys else None for task in tasks
        ]

        return filtered_tasks

    @chainable
    def get_taskhub_tasks(
        self,
        taskhub: ScopedKey,
        return_gufe=False,
        tx=None,
    ) -> list[ScopedKey] | dict[ScopedKey, Task]:
        """Get a list of Tasks on the TaskHub."""

        q = """
        // get list of all tasks associated with the taskhub
        MATCH (th:TaskHub {_scoped_key: $taskhub})-[:ACTIONS]->(task:Task)
        RETURN task
        """
        res = tx.run(q, taskhub=str(taskhub)).to_eager_result()

        tasks = []
        subgraph = Subgraph()
        for record in res.records:
            tasks.append(record_data_to_node(record["task"]))
            subgraph = subgraph | tasks[-1]

        if return_gufe:
            return {
                ScopedKey.from_str(k["_scoped_key"]): v
                for k, v in self._subgraph_to_gufe(tasks, subgraph).items()
            }
        else:
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]

    def get_taskhub_unclaimed_tasks(
        self, taskhub: ScopedKey, return_gufe=False
    ) -> list[ScopedKey] | dict[ScopedKey, Task]:
        """Get a list of unclaimed Tasks in the TaskHub."""

        q = """
        // get list of all unclaimed tasks in the hub
        MATCH (th:TaskHub {_scoped_key: $taskhub})-[:ACTIONS]->(task:Task)
        WHERE NOT (task)<-[:CLAIMS]-(:ComputeServiceRegistration)
        RETURN task
        """
        with self.transaction() as tx:
            res = tx.run(q, taskhub=str(taskhub)).to_eager_result()

        tasks = []
        subgraph = Subgraph()
        for record in res.records:
            node = record_data_to_node(record["task"])
            tasks.append(node)
            subgraph = subgraph | node

        if return_gufe:
            return {
                ScopedKey.from_str(k["_scoped_key"]): v
                for k, v in self._subgraph_to_gufe(tasks, subgraph).items()
            }
        else:
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]

    def claim_taskhub_tasks(
        self,
        taskhub: ScopedKey,
        compute_service_id: ComputeServiceID,
        count: int = 1,
        protocols: list[Protocol | str] | None = None,
    ) -> list[ScopedKey | None]:
        """Claim a TaskHub Task.

        This method will claim Tasks from a TaskHub according to the following process:

        1. `waiting` Tasks with the highest priority are selected for consideration.
        2. Tasks with an `EXTENDS` relationship to an incomplete Task are dropped
           from consideration.
        3. Of those that remain, a Task is claimed stochastically based on the
           `weight` of its ACTIONS relationship with the TaskHub.

        This process is repeated until `count` Tasks have been claimed.
        If no Task is available, then `None` is given in its place.

        Parameters
        ----------
        compute_service_id
            Unique identifier for the compute service claiming the Tasks for execution.
        count
            Claim the given number of Tasks in a single transaction.
        protocols
            Protocols to restrict Task claiming to. `None` means no restriction.
            If an empty list, raises ValueError.

        """
        if protocols is not None and len(protocols) == 0:
            raise ValueError("`protocols` must be either `None` or not empty")

        q = f"""
            MATCH (th:TaskHub {{_scoped_key: $taskhub}})-[actions:ACTIONS]-(task:Task)
            WHERE task.status = '{TaskStatusEnum.waiting.value}'
            AND actions.weight > 0
            OPTIONAL MATCH (task)-[:EXTENDS]->(other_task:Task)

            WITH task, other_task, actions
            """

        # filter down to `protocols`, if specified
        if protocols is not None:
            # need to extract qualnames if given protocol classes
            protocols = [
                protocol.__qualname__ if isinstance(protocol, Protocol) else protocol
                for protocol in protocols
            ]

            q += f"""
            MATCH (task)-[:PERFORMS]->(:Transformation|NonTransformation)-[:DEPENDS_ON]->(protocol:{cypher_or(protocols)})
            WITH task, other_task, actions
            """

        q += f"""
            WHERE other_task.status = '{TaskStatusEnum.complete.value}' OR other_task IS NULL

            RETURN task.`_scoped_key`, task.priority, actions.weight
            ORDER BY task.priority ASC
        """
        _tasks = {}
        with self.transaction() as tx:
            tx.run(
                """
                MATCH (th:TaskHub {_scoped_key: $taskhub})

                // lock the TaskHub to avoid other queries from changing its state while we claim
                SET th._lock = True
                """,
                taskhub=str(taskhub),
            )
            _taskpool = tx.run(q, taskhub=str(taskhub))

            def task_count(task_dict: dict):
                return sum(map(len, task_dict.values()))

            # directly use iterator to avoid pulling more tasks than we need
            # since we will likely stop early
            _task_iter = _taskpool.__iter__()
            while task_count(_tasks) < count:
                try:
                    candidate = next(_task_iter)
                    pr = candidate["task.priority"]

                    # get all tasks and their actions weights at each priority level
                    # until we've reached or surpassed `count`
                    _tasks[pr] = []
                    _tasks[pr].append(
                        (candidate["task.`_scoped_key`"], candidate["actions.weight"])
                    )
                    while True:

                        # if we've run out of tasks, stop immediately
                        if not (next_task := _taskpool.peek()):
                            raise StopIteration

                        # if next task has a different (lower) priority, stop consuming
                        if next_task["task.priority"] != pr:
                            break

                        candidate = next(_task_iter)
                        _tasks[candidate["task.priority"]].append(
                            (
                                candidate["task.`_scoped_key`"],
                                candidate["actions.weight"],
                            )
                        )

                except StopIteration:
                    break

            remaining = count
            tasks = []
            # for each group of tasks at each priority level
            for _, taskgroup in sorted(_tasks.items()):
                # if we want more tasks (or exactly as many tasks) as there are
                # in the group, just add them all
                if len(taskgroup) <= remaining:
                    tasks.extend(map(lambda x: ScopedKey.from_str(x[0]), taskgroup))

                    # immediately stop if we've reached our target count
                    if not (remaining := count - len(tasks)):
                        break

                # otherwise, perform a weighted random selection from the tasks
                # to fill out remaining
                else:
                    tasks.extend(
                        map(
                            ScopedKey.from_str,
                            _select_tasks_from_taskpool(taskgroup, remaining),
                        )
                    )

            # if tasks is not empty, proceed with claiming
            if tasks:
                tx.run(
                    CLAIM_QUERY,
                    tasks_list=[str(task) for task in tasks if task is not None],
                    datetimestr=str(datetime.datetime.now(tz=datetime.UTC).isoformat()),
                    compute_service_id=str(compute_service_id),
                )

            tx.run(
                """
                MATCH (th:TaskHub {_scoped_key: $taskhub})

                // remove lock on the TaskHub now that we're done with it
                SET th._lock = null
                """,
                taskhub=str(taskhub),
            )

        return tasks + [None] * (count - len(tasks))

    ## tasks

    def _validate_extends_tasks(self, task_list) -> dict[str, tuple[Node, str]]:

        if not task_list:
            return {}

        q = """
            UNWIND $task_list AS task
            MATCH (t:Task {_scoped_key: task})-[PERFORMS]->(tf:Transformation|NonTransformation)
            return t, tf._scoped_key as tf_sk
        """

        results = self.execute_query(q, task_list=list(map(str, task_list)))

        nodes = {}

        for record in results.records:
            node = record_data_to_node(record["t"])
            transformation_sk = record["tf_sk"]

            status = node.get("status")

            if status in ("invalid", "deleted"):
                invalid_task_scoped_key = node["_scoped_key"]
                raise ValueError(
                    f"Cannot extend a `deleted` or `invalid` Task: {invalid_task_scoped_key}"
                )

            nodes[node["_scoped_key"]] = (node, transformation_sk)

        return nodes

    def create_tasks(
        self,
        transformations: list[ScopedKey],
        extends: list[ScopedKey | None] | None = None,
        creator: str | None = None,
    ) -> list[ScopedKey]:
        """Create Tasks for the given Transformations.

        Note: this creates Tasks; it does not action them.

        Parameters
        ----------
        transformations
            The Transformations to create Tasks for.
            One Task is created for each Transformation ScopedKey given; to
            create multiple Tasks for a given Transformation, provide its
            ScopedKey multiple times.
        extends
            The ScopedKeys of the Tasks to use as a starting point for the
            created Tasks, in the same order as `transformations`. If ``None``
            given for a given Task, it will not extend any other Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extends` input for the Task's eventual call to `Protocol.create`.
        creator (optional)
            The creator of the Tasks.
        """
        allowed_types = [Transformation.__qualname__, NonTransformation.__qualname__]

        # reshape data to a standard form
        if extends is None:
            extends = [None] * len(transformations)
        elif len(extends) != len(transformations):
            raise ValueError(
                "`extends` must either be `None` or have the same length as `transformations`"
            )

        for i, _extends in enumerate(extends):
            if _extends is not None:
                if not (extended_task_qualname := getattr(_extends, "qualname", None)):
                    raise ValueError(
                        f"`extends` entry for `Task` {transformations[i]} is not valid"
                    )
                if extended_task_qualname != "Task":
                    raise ValueError(
                        f"`extends` ScopedKey ({_extends}) does not correspond to a `Task`"
                    )

        transformation_map = {
            transformation_type: [[], []] for transformation_type in allowed_types
        }
        for i, transformation in enumerate(transformations):
            if transformation.qualname not in allowed_types:
                raise ValueError(
                    f"Got an unsupported `Transformation` type: {transformation.qualname}"
                )
            transformation_map[transformation.qualname][0].append(transformation)
            transformation_map[transformation.qualname][1].append(extends[i])

        extends_nodes = self._validate_extends_tasks(
            [_extends for _extends in extends if _extends is not None]
        )

        subgraph = Subgraph()

        sks = []
        # iterate over all allowed types, unpacking the transformations and extends subsets
        for node_type, (
            transformation_subset,
            extends_subset,
        ) in transformation_map.items():

            if not transformation_subset:
                continue

            q = f"""
            UNWIND $transformation_subset AS sk
            MATCH (n:{node_type} {{`_scoped_key`: sk}})
            RETURN n
            """

            results = self.execute_query(
                q, transformation_subset=list(map(str, transformation_subset))
            )

            transformation_nodes = {}
            for record in results.records:
                node = record_data_to_node(record["n"])
                transformation_nodes[node["_scoped_key"]] = node

            for _transformation, _extends in zip(transformation_subset, extends_subset):

                scope = transformation.scope

                _task = Task(
                    creator=creator,
                    extends=str(_extends) if _extends is not None else None,
                )
                _, task_node, scoped_key = self._keyed_chain_to_subgraph(
                    KeyedChain.from_gufe(_task),
                    scope=scope,
                )

                sks.append(scoped_key)

                if _extends is not None:

                    extends_task_node, extends_transformation_sk = extends_nodes[
                        str(_extends)
                    ]

                    if extends_transformation_sk != str(_transformation):
                        raise ValueError(
                            f"{_extends} extends a Transformation other than {_transformation}"
                        )

                    subgraph |= Relationship.type("EXTENDS")(
                        task_node,
                        extends_task_node,
                        _org=scope.org,
                        _campaign=scope.campaign,
                        _project=scope.project,
                    )

                subgraph |= Relationship.type("PERFORMS")(
                    task_node,
                    transformation_nodes[str(_transformation)],
                    _org=scope.org,
                    _campaign=scope.campaign,
                    _project=scope.project,
                )

        with self.transaction() as tx:
            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

        return sks

    def create_task(
        self,
        transformation: ScopedKey,
        extends: ScopedKey | None = None,
        creator: str | None = None,
    ) -> ScopedKey:
        """Create a single Task for a Transformation.

        This is a convenience method that wraps around the more general
        `create_tasks` method.

        """
        return self.create_tasks(
            [transformation],
            extends=[extends] if extends is not None else [None],
            creator=creator,
        )[0]

    def query_tasks(self, *, status=None, key=None, scope: Scope = Scope()):
        r"""Query for `Task`\s matching given attributes."""
        additional = {"status": status}
        return self._query(qualname="Task", additional=additional, key=key, scope=scope)

    def get_network_tasks(
        self, network: ScopedKey, status: TaskStatusEnum | None = None
    ) -> list[ScopedKey]:
        """List ScopedKeys for all Tasks associated with the given AlchemicalNetwork."""
        q = """
        MATCH (an:AlchemicalNetwork {_scoped_key: $network})-[:DEPENDS_ON]->(tf:Transformation|NonTransformation),
              (tf)<-[:PERFORMS]-(t:Task)
        """

        if status is not None:
            q += """
            WHERE t.status = $status
            """

        q += """
        WITH t._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(
                q, network=str(network), status=status.value if status else None
            )
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_task_networks(self, task: ScopedKey) -> list[ScopedKey]:
        """List ScopedKeys for AlchemicalNetworks associated with the given Task."""
        q = """
        MATCH (t:Task {_scoped_key: $task})-[:PERFORMS]->(tf:Transformation|NonTransformation),
              (tf)<-[:DEPENDS_ON]-(an:AlchemicalNetwork)
        WITH an._scoped_key as sk
        RETURN sk
        """
        sks = []
        with self.transaction() as tx:
            res = tx.run(q, task=str(task))
            for rec in res:
                sks.append(rec["sk"])

        return [ScopedKey.from_str(sk) for sk in sks]

    def get_transformation_tasks(
        self,
        transformation: ScopedKey,
        extends: ScopedKey | None = None,
        return_as: str = "list",
        status: TaskStatusEnum | None = None,
    ) -> list[ScopedKey] | dict[ScopedKey, ScopedKey | None]:
        """Get all Tasks that perform the given Transformation.

        If a Task ScopedKey is given for `extends`, then only those Tasks
        that follow via any number of EXTENDS relationships will be returned.

        `return_as` takes either `list` or `graph` as input.
        `graph` will yield a dict mapping each Task's ScopedKey (as keys) to
        the Task ScopedKey it extends (as values).

        Parameters
        ----------
        transformation
            ScopedKey of the Transformation to retrieve Tasks for.
        extends

        """
        q = """
        MATCH (trans:Transformation|NonTransformation {_scoped_key: $transformation})<-[:PERFORMS]-(task:Task)
        """

        if status is not None:
            q += """
            WHERE task.status = $status
            """

        if extends:
            q += """
            MATCH (trans)<-[:PERFORMS]-(extends:Task {_scoped_key: $extends})
            WHERE (task)-[:EXTENDS*]->(extends)
            RETURN task
            """
        else:
            q += """
            RETURN task
            """

        with self.transaction() as tx:
            res = tx.run(
                q,
                transformation=str(transformation),
                status=status.value if status else None,
                extends=str(extends) if extends else None,
            ).to_eager_result()

        tasks = []
        for record in res.records:
            tasks.append(record["task"])

        if return_as == "list":
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]
        elif return_as == "graph":
            return {
                ScopedKey.from_str(t["_scoped_key"]): (
                    ScopedKey.from_str(t["extends"])
                    if t["extends"] is not None
                    else None
                )
                for t in tasks
            }

    def get_transformation_actioned_tasks(
        self,
        transformation: ScopedKey,
        taskhub: ScopedKey,
    ) -> list[ScopedKey]:
        """Get all Tasks for a Transformation that are actioned by the given TaskHub.

        Parameters
        ----------
        transformation
            ScopedKey of the Transformation to retrieve actioned Tasks for.
        taskhub
            ScopedKey of the TaskHub to check for actioned Tasks.

        Returns
        -------
        tasks
            List of Task ScopedKeys that perform the given Transformation and are
            actioned by the given TaskHub.
        """
        q = """
        MATCH (th:TaskHub {_scoped_key: $taskhub})-[:ACTIONS]->(task:Task),
              (task)-[:PERFORMS]->(trans:Transformation|NonTransformation {_scoped_key: $transformation})
        RETURN task._scoped_key
        """

        with self.transaction() as tx:
            results = tx.run(
                q,
                transformation=str(transformation),
                taskhub=str(taskhub),
            ).to_eager_result()

        return [
            ScopedKey.from_str(record["task._scoped_key"]) for record in results.records
        ]

    def get_task_transformation(
        self,
        task: ScopedKey,
        return_gufe=True,
    ) -> (
        tuple[Transformation, ProtocolDAGResultRef | None]
        | tuple[ScopedKey, ScopedKey | None]
    ):
        r"""Get the `Transformation` and `ProtocolDAGResultRef` to extend from (if
        present) for the given `Task`.

        If `return_gufe` is `True`, returns actual `Transformation` and
        `ProtocolDAGResultRef` object (`None` if not present); if `False`, returns
        `ScopedKey`\s for these instead.

        """
        q = """
        MATCH (task:Task {_scoped_key: $task})-[:PERFORMS]->(trans:Transformation|NonTransformation)
        OPTIONAL MATCH (task)-[:EXTENDS]->(prev:Task)-[:RESULTS_IN]->(result:ProtocolDAGResultRef)
        RETURN trans, result
        """

        with self.transaction() as tx:
            res = tx.run(q, task=str(task)).to_eager_result()

        transformations = []
        results = []
        for record in res.records:
            transformations.append(record["trans"])
            results.append(record["result"])

        if len(transformations) == 0 or len(results) == 0:
            raise KeyError("No such object in database")
        elif len(transformations) > 1 or len(results) > 1:
            raise Neo4JStoreError(
                "More than one such object in database; this should not be possible"
            )

        transformation = ScopedKey.from_str(transformations[0]["_scoped_key"])

        protocoldagresultref = (
            ScopedKey.from_str(results[0]["_scoped_key"])
            if results[0] is not None
            else None
        )

        if return_gufe:
            return (
                self.get_gufe(transformation),
                (
                    self.get_gufe(protocoldagresultref)
                    if protocoldagresultref is not None
                    else None
                ),
            )

        return transformation, protocoldagresultref

    def set_tasks(
        self,
        transformation: ScopedKey,
        extends: Task | None = None,
        count: int = 1,
    ) -> ScopedKey:
        """Set a fixed number of Tasks against the given Transformation if not
        already present.

        Note: Tasks created by this method are not added to any TaskHubs.

        Parameters
        ----------
        transformation
            The Transformation to compute.
        scope
            The scope the Transformation is in; ignored if `transformation` is a ScopedKey.
        extends
            The Task to use as a starting point for this Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extends` input for the Task's eventual call to `Protocol.create`.
        count
            The total number of tasks that should exist corresponding to the
            specified `transformation`, `scope`, and `extends`.
        """
        raise NotImplementedError
        # TODO: finish this one out when we have a reasonable approach to locking
        # too hard to perform in a single Cypher query; unclear how to create many nodes in a loop
        scope = transformation.scope
        _ = self._get_node_from_obj_or_sk(transformation, Transformation, scope)

    def set_task_priority(
        self, tasks: list[ScopedKey], priority: int
    ) -> list[ScopedKey | None]:
        """Set the priority of a list of Tasks.

        Parameters
        ----------
        tasks
            The list of Tasks to set the priority of.
        priority
            The priority to set the Tasks to.

        Returns
        -------
        List[Optional[ScopedKey]]
            A list of the Task ScopedKeys for which priority was changed; `None`
            is given for any Tasks for which the priority could not be changed.
        """
        if not (1 <= priority <= 2**63 - 1):
            raise ValueError("priority must be between 1 and 2**63 - 1, inclusive")

        with self.transaction() as tx:
            q = """
            WITH $scoped_keys AS batch
            UNWIND batch AS scoped_key

            OPTIONAL MATCH (t:Task {_scoped_key: scoped_key})
            SET t.priority = $priority

            RETURN scoped_key, t
            """
            res = tx.run(
                q,
                scoped_keys=list(map(str, tasks)),
                priority=priority,
            ).to_eager_result()

        task_results = []
        for record in res.records:
            task_i = record["t"]
            scoped_key = record["scoped_key"]

            # catch missing tasks
            if task_i is None:
                task_results.append(None)
            else:
                task_results.append(ScopedKey.from_str(scoped_key))
        return task_results

    def get_task_priority(self, tasks: list[ScopedKey]) -> list[int | None]:
        """Get the priority of a list of Tasks.

        Parameters
        ----------
        tasks
            The list of Tasks to get the priority for.

        Returns
        -------
        List[Optional[int]]
            A list of priorities in the same order as the provided Tasks.
            If an element is ``None``, the Task could not be found.
        """
        with self.transaction() as tx:
            q = """
            WITH $scoped_keys AS batch
            UNWIND batch AS scoped_key
            OPTIONAL MATCH (t:Task)
            WHERE t._scoped_key = scoped_key
            RETURN t.priority as priority
            """
            res = tx.run(q, scoped_keys=list(map(str, tasks)))
            priorities = [rec["priority"] for rec in res]

        return priorities

    def delete_task(
        self,
        task: ScopedKey,
    ) -> Task:
        """Remove a compute Task from a Transformation.

        This will also remove the Task from all TaskHubs it is a part of.

        This method is intended for administrator use; generally Tasks should
        instead have their tasks set to 'deleted' and retained.

        """
        raise NotImplementedError

    def get_scope_status(
        self,
        scope: Scope,
        network_state: NetworkStateEnum | str | None = NetworkStateEnum.active,
    ) -> dict[str, int]:
        """Return status counts for all Tasks within the given Scope.

        Parameters
        ----------
        scope
            Scope to get status for; may be non-specific.
        network_state
            Network state to restrict status returns for; may be a regex pattern.

        """

        properties = {
            "_org": scope.org,
            "_campaign": scope.campaign,
            "_project": scope.project,
        }

        prop_string = ", ".join(
            f"{key}: ${key}" for key, value in properties.items() if value is not None
        )

        if isinstance(network_state, NetworkStateEnum):
            network_state = network_state.value

        if network_state is None:
            network_state = ".*"

        q = f"""
        MATCH (n:Task {{{prop_string}}})-[:PERFORMS]->(:Transformation|NonTransformation)<-[:DEPENDS_ON]-(:AlchemicalNetwork)<-[:MARKS]-(nm:NetworkMark)
        WHERE nm.state =~ $state_pattern
        RETURN n.status AS status, count(DISTINCT n) as counts
        """
        with self.transaction() as tx:
            res = tx.run(q, state_pattern=network_state, **properties)
            counts = {rec["status"]: rec["counts"] for rec in res}

        return counts

    def get_network_status(self, networks: list[ScopedKey]) -> list[dict[str, int]]:
        """Return status counts for all Tasks associated with the given AlchemicalNetworks."""
        q = """
        UNWIND $networks AS network
        MATCH (an:AlchemicalNetwork {_scoped_key: network})-[:DEPENDS_ON]->(tf:Transformation|NonTransformation),
              (tf)<-[:PERFORMS]-(t:Task)
        RETURN an._scoped_key AS sk, t.status AS status, count(t) as counts
        """

        network_data = {str(network_sk): {} for network_sk in networks}
        for rec in self.execute_query(q, networks=list(map(str, networks))).records:
            sk = rec["sk"]
            status = rec["status"]
            counts = rec["counts"]
            network_data[sk][status] = counts

        return [network_data[str(an)] for an in networks]

    def get_transformation_status(self, transformation: ScopedKey) -> dict[str, int]:
        """Return status counts for all Tasks associated with the given Transformation."""
        q = """
        MATCH (:Transformation|NonTransformation {_scoped_key: $transformation})<-[:PERFORMS]-(t:Task)
        RETURN t.status AS status, count(t) as counts
        """
        with self.transaction() as tx:
            res = tx.run(q, transformation=str(transformation))
            counts = {rec["status"]: rec["counts"] for rec in res}

        return counts

    def set_task_result(
        self, task: ScopedKey, protocoldagresultref: ProtocolDAGResultRef
    ) -> ScopedKey:
        """Set a `ProtocolDAGResultRef` pointing to a `ProtocolDAGResult` for the given `Task`."""

        if task.qualname != "Task":
            raise ValueError("`task` ScopedKey does not correspond to a `Task`")

        scope = task.scope
        task_node = self._get_node(task)

        subgraph, protocoldagresultref_node, scoped_key = self._keyed_chain_to_subgraph(
            KeyedChain.from_gufe(protocoldagresultref),
            scope=scope,
        )

        subgraph = subgraph | Relationship.type("RESULTS_IN")(
            task_node,
            protocoldagresultref_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        with self.transaction() as tx:
            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

        return scoped_key

    def get_task_results(self, task: ScopedKey) -> list[ProtocolDAGResultRef]:
        # get all task result protocoldagresultrefs corresponding to given task
        # returned in no particular order
        q = """
        MATCH (task:Task {_scoped_key: $scoped_key}),
              (task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = true
        WITH res._scoped_key as sk
        RETURN DISTINCT sk
        """
        return self._get_protocoldagresultrefs(q, task)

    def get_task_failures(self, task: ScopedKey) -> list[ProtocolDAGResultRef]:
        # get all task failure protocoldagresultrefs corresponding to given task
        # returned in no particular order
        q = """
        MATCH (task:Task {_scoped_key: $scoped_key}),
              (task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = false
        WITH res._scoped_key as sk
        RETURN DISTINCT sk
        """
        return self._get_protocoldagresultrefs(q, task)

    def add_protocol_dag_result_ref_tracebacks(
        self,
        protocol_unit_failures: list[ProtocolUnitFailure],
        protocol_dag_result_ref_scoped_key: ScopedKey,
    ):
        subgraph = Subgraph()

        with self.transaction() as tx:

            query = """
            MATCH (pdrr:ProtocolDAGResultRef {`_scoped_key`: $protocol_dag_result_ref_scoped_key})
            RETURN pdrr
            """

            pdrr_result = tx.run(
                query,
                protocol_dag_result_ref_scoped_key=str(
                    protocol_dag_result_ref_scoped_key
                ),
            ).to_eager_result()

            try:
                protocol_dag_result_ref_node = record_data_to_node(
                    pdrr_result.records[0]["pdrr"]
                )
            except IndexError:
                raise KeyError("Could not find ProtocolDAGResultRef in database.")

            failure_keys = []
            source_keys = []
            tracebacks = []

            for puf in protocol_unit_failures:
                failure_keys.append(puf.key)
                source_keys.append(puf.source_key)
                tracebacks.append(puf.traceback)

            tracebacks = Tracebacks(tracebacks, source_keys, failure_keys)

            _, tracebacks_node, _ = self._keyed_chain_to_subgraph(
                KeyedChain.from_gufe(tracebacks),
                scope=protocol_dag_result_ref_scoped_key.scope,
            )

            subgraph |= Relationship.type("DETAILS")(
                tracebacks_node,
                protocol_dag_result_ref_node,
            )

            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

    def set_task_status(
        self, tasks: list[ScopedKey], status: TaskStatusEnum, raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks.

        This is a master method that calls the appropriate method for the
        status.

        Parameters
        ----------
        tasks
            The list of Tasks to set the status of.
        status
            The status to set the Task to.
        raise_error
            If `True`, raise a `ValueError` if the status of a given Task cannot be changed.

        Returns
        -------
        List[Optional[ScopedKey]]
            A list of the Task ScopedKeys for which status was changed; `None`
            is given for any Tasks for which the status could not be changed.

        """
        method = getattr(self, f"set_task_{status.value}")
        return method(tasks, raise_error=raise_error)

    def get_task_status(self, tasks: list[ScopedKey]) -> list[TaskStatusEnum]:
        """Get the status of a list of Tasks.

        Parameters
        ----------
        tasks
            The list of Tasks to get the status for.

        Returns
        -------
            A list of the Task statuses requested; `None` is given for any
            Tasks that do not exist.

        """
        statuses = []
        with self.transaction() as tx:
            q = """
            WITH $scoped_keys AS batch
            UNWIND batch AS scoped_key
            OPTIONAL MATCH (t:Task)
            WHERE t._scoped_key = scoped_key
            RETURN t.status as status
            """
            res = tx.run(q, scoped_keys=[str(t) for t in tasks])

            for rec in res:
                status = rec["status"]
                statuses.append(TaskStatusEnum(status) if status is not None else None)

        return statuses

    def _set_task_status(
        self, tasks, q: str, err_msg_func, raise_error
    ) -> list[ScopedKey | None]:
        tasks_statused = []
        with self.transaction() as tx:
            res = tx.run(q, scoped_keys=[str(t) for t in tasks])

            for record in res:
                task_i = record["t"]
                task_set = record["t_"]
                scoped_key = record["scoped_key"]

                if task_set is None:
                    if raise_error:
                        status = task_i["status"]
                        raise ValueError(err_msg_func(scoped_key, status))
                    tasks_statused.append(None)
                elif task_i is None:
                    if raise_error:
                        raise ValueError("No such task {t}")
                    tasks_statused.append(None)
                else:
                    tasks_statused.append(ScopedKey.from_str(scoped_key))

        return tasks_statused

    def set_task_waiting(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `waiting`.

        Only Tasks with status `error` or `running` can be set to `waiting`.

        """

        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE t_.status IN ['{TaskStatusEnum.waiting.value}', '{TaskStatusEnum.running.value}', '{TaskStatusEnum.error.value}']
        SET t_.status = '{TaskStatusEnum.waiting.value}'

        WITH scoped_key, t, t_

        // if we changed the status to waiting,
        // drop CLAIMS relationship
        OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
        DELETE cl

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `waiting` as it is not currently `error` or `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_running(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `running`.

        Only Tasks with status `waiting` can be set to `running`.

        """

        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE t_.status IN ['{TaskStatusEnum.running.value}', '{TaskStatusEnum.waiting.value}']
        SET t_.status = '{TaskStatusEnum.running.value}'

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `running` as it is not currently `waiting`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_complete(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `complete`.

        Only `running` Tasks can be set to `complete`.

        """

        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE t_.status IN ['{TaskStatusEnum.complete.value}', '{TaskStatusEnum.running.value}']
        SET t_.status = '{TaskStatusEnum.complete.value}'

        WITH scoped_key, t, t_

        // if we changed the status to complete,
        // drop all taskhub ACTIONS and task restart APPLIES relationships
        OPTIONAL MATCH (t_)<-[ar:ACTIONS]-(th:TaskHub)
        OPTIONAL MATCH (t_)<-[applies:APPLIES]-(:TaskRestartPattern)
        DELETE ar
        DELETE applies

        WITH scoped_key, t, t_

        // if we changed the status to complete,
        // drop CLAIMS relationship
        OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
        DELETE cl

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `complete` as it is not currently `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_error(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `error`.

        Only `running` Tasks can be set to `error`.

        """

        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE t_.status IN ['{TaskStatusEnum.error.value}', '{TaskStatusEnum.running.value}']
        SET t_.status = '{TaskStatusEnum.error.value}'

        WITH scoped_key, t, t_

        // if we changed the status to error,
        // drop CLAIMS relationship
        OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
        DELETE cl

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `error` as it is not currently `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_invalid(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `invalid`.

        Any Task can be set to `invalid`; an `invalid` Task cannot change to
        any other status.

        """

        # set the status and delete the ACTIONS relationship
        # make sure we follow the extends chain and set all tasks to invalid
        # and remove actions relationships
        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE NOT t_.status IN ['{TaskStatusEnum.deleted.value}']
        SET t_.status = '{TaskStatusEnum.invalid.value}'

        WITH scoped_key, t, t_

        OPTIONAL MATCH (t_)<-[er:EXTENDS*]-(extends_task:Task)
        SET extends_task.status = '{TaskStatusEnum.invalid.value}'

        WITH scoped_key, t, t_, extends_task

        OPTIONAL MATCH (t_)<-[ar:ACTIONS]-(th:TaskHub)
        OPTIONAL MATCH (extends_task)<-[ar_e:ACTIONS]-(th:TaskHub)
        OPTIONAL MATCH (t_)<-[applies:APPLIES]-(:TaskRestartPattern)
        OPTIONAL MATCH (extends_task)<-[applies_e:APPLIES]-(:TaskRestartPattern)

        DELETE ar
        DELETE ar_e
        DELETE applies
        DELETE applies_e

        WITH scoped_key, t, t_

        // drop CLAIMS relationship if present
        OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
        DELETE cl

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `invalid` as it is `deleted`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_deleted(
        self, tasks: list[ScopedKey], raise_error: bool = False
    ) -> list[ScopedKey | None]:
        """Set the status of a list of Tasks to `deleted`.

        Any Task can be set to `deleted`; a `deleted` Task cannot change to
        any other status.

        """

        # set the status and delete the ACTIONS relationship
        # make sure we follow the extends chain and set all tasks to deleted
        # and remove actions relationships
        q = f"""
        WITH $scoped_keys AS batch
        UNWIND batch AS scoped_key

        OPTIONAL MATCH (t:Task {{_scoped_key: scoped_key}})

        OPTIONAL MATCH (t_:Task {{_scoped_key: scoped_key}})
        WHERE NOT t_.status IN ['{TaskStatusEnum.invalid.value}']
        SET t_.status = '{TaskStatusEnum.deleted.value}'

        WITH scoped_key, t, t_

        OPTIONAL MATCH (t_)<-[er:EXTENDS*]-(extends_task:Task)
        SET extends_task.status = '{TaskStatusEnum.deleted.value}'

        WITH scoped_key, t, t_, extends_task

        OPTIONAL MATCH (t_)<-[ar:ACTIONS]-(th:TaskHub)
        OPTIONAL MATCH (extends_task)<-[ar_e:ACTIONS]-(th:TaskHub)
        OPTIONAL MATCH (t_)<-[applies:APPLIES]-(:TaskRestartPattern)
        OPTIONAL MATCH (extends_task)<-[applies_e:APPLIES]-(:TaskRestartPattern)

        DELETE ar
        DELETE ar_e
        DELETE applies
        DELETE applies_e

        WITH scoped_key, t, t_

        // drop CLAIMS relationship if present
        OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
        DELETE cl

        RETURN scoped_key, t, t_
        """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `deleted` as it is `invalid`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    ## task restart policies

    def add_task_restart_patterns(
        self, taskhub: ScopedKey, patterns: list[str], number_of_retries: int
    ):
        """Add a list of restart policy patterns to a `TaskHub` along with the number of retries allowed.

        Parameters
        ----------
        taskhub : ScopedKey
            TaskHub for the restart patterns to enforce.
        patterns: list[str]
            Regular expression patterns that will be compared to tracebacks returned by ProtocolUnitFailures.
        number_of_retries: int
            The number of times the given patterns will apply to a single Task, attempts to restart beyond
            this value will result in a canceled Task with an error status.

        Raises
        ------
        KeyError
            Raised when the provided TaskHub ScopedKey cannot be associated with a TaskHub in the database.
        """

        # get taskhub node
        q = """
        MATCH (th:TaskHub {`_scoped_key`: $taskhub})
        RETURN th
        """
        results = self.execute_query(q, taskhub=str(taskhub))

        # raise error if taskhub not found
        if not results.records:
            raise KeyError("No such TaskHub in the database")

        record_data = results.records[0]["th"]
        taskhub_node = record_data_to_node(record_data)
        scope = taskhub.scope

        with self.transaction() as tx:
            actioned_tasks_query = """
            MATCH (taskhub: TaskHub {`_scoped_key`: $taskhub_scoped_key})-[:ACTIONS]->(task: Task)
            RETURN task
            """

            actioned_task_records = (
                tx.run(actioned_tasks_query, taskhub_scoped_key=str(taskhub))
                .to_eager_result()
                .records
            )

            subgraph = Subgraph()

            actioned_task_nodes = []

            for actioned_tasks_record in actioned_task_records:
                actioned_task_nodes.append(
                    record_data_to_node(actioned_tasks_record["task"])
                )

            for pattern in patterns:
                task_restart_pattern = TaskRestartPattern(
                    pattern,
                    max_retries=number_of_retries,
                    taskhub_scoped_key=str(taskhub),
                )

                _, task_restart_pattern_node, scoped_key = (
                    self._keyed_chain_to_subgraph(
                        KeyedChain.from_gufe(task_restart_pattern),
                        scope=scope,
                    )
                )

                subgraph |= Relationship.type("ENFORCES")(
                    task_restart_pattern_node,
                    taskhub_node,
                    _org=scope.org,
                    _campaign=scope.campaign,
                    _project=scope.project,
                )

                for actioned_task_node in actioned_task_nodes:
                    subgraph |= Relationship.type("APPLIES")(
                        task_restart_pattern_node,
                        actioned_task_node,
                        num_retries=0,
                    )
            merge_subgraph(tx, subgraph, "GufeTokenizable", "_scoped_key")

            actioned_task_scoped_keys: list[ScopedKey] = []

            for actioned_task_record in actioned_task_records:
                actioned_task_scoped_keys.append(
                    ScopedKey.from_str(actioned_task_record["task"]["_scoped_key"])
                )

            self.resolve_task_restarts(actioned_task_scoped_keys, tx=tx)

    def remove_task_restart_patterns(self, taskhub: ScopedKey, patterns: list[str]):
        """Remove a list of restart patterns enforcing a TaskHub from the database.

        Parameters
        ----------
        taskhub: ScopedKey
            The ScopedKey of the TaskHub that the patterns enforce.
        patterns: list[str]
            The patterns to remove. Patterns not enforcing the TaskHub are ignored.
        """
        q = """
        UNWIND $patterns AS pattern

        MATCH (trp: TaskRestartPattern {pattern: pattern, taskhub_scoped_key: $taskhub_scoped_key})

        DETACH DELETE trp
        """

        self.execute_query(q, patterns=patterns, taskhub_scoped_key=str(taskhub))

    def clear_task_restart_patterns(self, taskhub: ScopedKey):
        """Clear all restart patterns from a TaskHub.

        Parameters
        ----------
        taskhub: ScopedKey
            The ScopedKey of the TaskHub to clear of restart patterns.
        """
        q = """
        MATCH (trp: TaskRestartPattern {taskhub_scoped_key: $taskhub_scoped_key})
        DETACH DELETE trp
        """
        self.execute_query(q, taskhub_scoped_key=str(taskhub))

    def set_task_restart_patterns_max_retries(
        self,
        taskhub_scoped_key: ScopedKey,
        patterns: list[str],
        max_retries: int,
    ):
        """Set the maximum number of retries of a pattern enforcing a TaskHub.

        Parameters
        ----------
        taskhub_scoped_key: ScopedKey
            The ScopedKey of the TaskHub that the patterns enforce.
        patterns: list[str]
            The patterns to change the maximum retries value for.
        max_retries: int
            The new maximum retries value.
        """
        query = """
        UNWIND $patterns AS pattern
        MATCH (trp: TaskRestartPattern {pattern: pattern, taskhub_scoped_key: $taskhub_scoped_key})
        SET trp.max_retries = $max_retries
        """

        self.execute_query(
            query,
            patterns=patterns,
            taskhub_scoped_key=str(taskhub_scoped_key),
            max_retries=max_retries,
        )

    def get_task_restart_patterns(
        self, taskhubs: list[ScopedKey]
    ) -> dict[ScopedKey, set[tuple[str, int]]]:
        """For a list of TaskHub ScopedKeys, get the associated restart
        patterns along with the maximum number of retries for each pattern.

        Parameters
        ----------
        taskhubs: list[ScopedKey]
            The ScopedKeys of the TaskHubs to get the restart patterns of.

        Returns
        -------
        dict[ScopedKey, set[tuple[str, int]]]
            A dictionary with ScopedKeys of the TaskHubs provided as keys, and a
            set of tuples containing the patterns enforcing each TaskHub along
            with their associated maximum number of retries as values.
        """

        q = """
            UNWIND $taskhub_scoped_keys as taskhub_scoped_key
            MATCH (trp: TaskRestartPattern)-[ENFORCES]->(th: TaskHub {`_scoped_key`: taskhub_scoped_key})
            RETURN th, trp
        """

        records = self.execute_query(
            q, taskhub_scoped_keys=list(map(str, taskhubs))
        ).records

        data: dict[ScopedKey, set[tuple[str, int]]] = {
            taskhub: set() for taskhub in taskhubs
        }

        for record in records:
            pattern = record["trp"]["pattern"]
            max_retries = record["trp"]["max_retries"]
            taskhub_sk = ScopedKey.from_str(record["th"]["_scoped_key"])
            data[taskhub_sk].add((pattern, max_retries))

        return data

    @chainable
    def resolve_task_restarts(self, task_scoped_keys: Iterable[ScopedKey], *, tx=None):
        """Determine whether or not Tasks need to be restarted or canceled and perform that action.

        Parameters
        ----------
        task_scoped_keys: Iterable[ScopedKey]
            An iterable of Task ScopedKeys that need to be resolved. Tasks without the error status
            are filtered out and ignored.
        """

        # Given the scoped keys of a list of Tasks, find all tasks that have an
        # error status and have a TaskRestartPattern applied. A subquery is executed
        # to optionally get the latest traceback associated with the task
        query = """
        UNWIND $task_scoped_keys AS task_scoped_key
        MATCH (task:Task {status: $error, `_scoped_key`: task_scoped_key})<-[app:APPLIES]-(trp:TaskRestartPattern)-[:ENFORCES]->(taskhub:TaskHub)
        CALL {
            WITH task
            OPTIONAL MATCH (task:Task)-[:RESULTS_IN]->(pdrr:ProtocolDAGResultRef)<-[:DETAILS]-(tracebacks:Tracebacks)
            RETURN tracebacks
            ORDER BY pdrr.datetime_created DESCENDING
            LIMIT 1
        }
        WITH task, tracebacks, trp, app, taskhub
        RETURN task, tracebacks, trp, app, taskhub
        """

        results = tx.run(
            query,
            task_scoped_keys=list(map(str, task_scoped_keys)),
            error=TaskStatusEnum.error.value,
        ).to_eager_result()

        if not results:
            return

        # iterate over all of the results to determine if an applied pattern needs
        # to be iterated or if the task needs to be cancelled outright

        # Keep track of which task/taskhub pairs would need to be canceled
        # None => the pair never had a matching restart pattern
        # True => at least one patterns max_retries was exceeded
        # False => at least one regex matched, but no pattern max_retries were exceeded
        cancel_map: dict[tuple[str, str], bool | None] = {}
        to_increment: list[tuple[str, str]] = []
        all_task_taskhub_pairs: set[tuple[str, str]] = set()
        for record in results.records:
            task_restart_pattern = record["trp"]
            applies_relationship = record["app"]
            task = record["task"]
            taskhub = record["taskhub"]
            _tracebacks = record["tracebacks"]

            task_taskhub_tuple = (task["_scoped_key"], taskhub["_scoped_key"])

            all_task_taskhub_pairs.add(task_taskhub_tuple)

            # TODO: remove in v1.0.0 tasks that errored, prior to the indtroduction of task restart policies will have
            # no tracebacks in the database

            if _tracebacks is None:
                cancel_map[task_taskhub_tuple] = True

            # we have already determined that the task is to be canceled.
            # this is only ever truthy when we say a task needs to be canceled.
            if cancel_map.get(task_taskhub_tuple):
                continue

            num_retries = applies_relationship["num_retries"]
            max_retries = task_restart_pattern["max_retries"]
            pattern = task_restart_pattern["pattern"]
            tracebacks: list[str] = _tracebacks["tracebacks"]

            compiled_pattern = re.compile(pattern)

            if any([compiled_pattern.search(message) for message in tracebacks]):
                if num_retries + 1 > max_retries:
                    cancel_map[task_taskhub_tuple] = True
                else:
                    to_increment.append(
                        (task["_scoped_key"], task_restart_pattern["_scoped_key"])
                    )
                    cancel_map[task_taskhub_tuple] = False

        increment_query = """
        UNWIND $task_trp_pairs as pairs
        WITH pairs[0] as task_scoped_key, pairs[1] as task_restart_pattern_scoped_key
        MATCH (:Task {`_scoped_key`: task_scoped_key})<-[app:APPLIES]-(:TaskRestartPattern {`_scoped_key`: task_restart_pattern_scoped_key})
        SET app.num_retries = app.num_retries + 1
        """

        tx.run(increment_query, task_trp_pairs=to_increment)

        # cancel all tasks (from a taskhub) that didn't trigger any restart patterns (None)
        # or exceeded a pattern's max_retries value (True)
        cancel_groups: defaultdict[str, list[str]] = defaultdict(list)
        for task_taskhub_pair in all_task_taskhub_pairs:
            cancel_result = cancel_map.get(task_taskhub_pair)
            if cancel_result in (True, None):
                cancel_groups[task_taskhub_pair[1]].append(task_taskhub_pair[0])

        for taskhub, tasks in cancel_groups.items():
            self.cancel_tasks(tasks, taskhub, tx=tx)

        # any tasks that are still associated with a TaskHub and a TaskRestartPattern must then be okay to switch to waiting
        renew_waiting_status_query = """
        UNWIND $task_scoped_keys AS task_scoped_key
        MATCH (task:Task {status: $error, `_scoped_key`: task_scoped_key})<-[app:APPLIES]-(trp:TaskRestartPattern)-[:ENFORCES]->(taskhub:TaskHub)
        SET task.status = $waiting
        """

        tx.run(
            renew_waiting_status_query,
            task_scoped_keys=list(map(str, task_scoped_keys)),
            waiting=TaskStatusEnum.waiting.value,
            error=TaskStatusEnum.error.value,
        )

    ## authentication

    def create_credentialed_entity(self, entity: CredentialedEntity):
        """Create a new credentialed entity, such as a user or compute identity.

        If an entity of this type with the same `identifier` already exists,
        then this will overwrite its properties, including credential.

        """
        node = Node("CredentialedEntity", entity.__class__.__name__, **entity.to_dict())

        with self.transaction() as tx:
            merge_subgraph(
                tx, Subgraph() | node, entity.__class__.__name__, "identifier"
            )

    def get_credentialed_entity(self, identifier: str, cls: type[CredentialedEntity]):
        """Get an existing credentialed entity, such as a user or compute identity."""
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        RETURN n
        """

        with self.transaction() as tx:
            res = tx.run(q).to_eager_result()

        nodes = set()
        for record in res.records:
            nodes.add(record_data_to_node(record["n"]))

        if len(nodes) == 0:
            raise KeyError("No such object in database")
        elif len(nodes) > 1:
            raise Neo4JStoreError(
                "More than one such object in database; this should not be possible"
            )

        return cls(**dict(list(nodes)[0]))

    def list_credentialed_entities(self, cls: type[CredentialedEntity]):
        """Get an existing credentialed entity, such as a user or compute identity."""
        q = f"""
        MATCH (n:{cls.__name__})
        RETURN n
        """
        with self.transaction() as tx:
            res = tx.run(q).to_eager_result()

        nodes = set()
        for record in res.records:
            nodes.add(record_data_to_node(record["n"]))

        return [node["identifier"] for node in nodes]

    def remove_credentialed_identity(
        self, identifier: str, cls: type[CredentialedEntity]
    ):
        """Remove a credentialed entity, such as a user or compute identity."""
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        DETACH DELETE n
        """

        with self.transaction() as tx:
            tx.run(q)

    def add_scope(self, identifier: str, cls: type[CredentialedEntity], scope: Scope):
        """Add a scope to the given entity."""

        # n.scopes is always initialized by the pydantic model so no need to check
        # for existence, however, we do need to check that the scope is not already
        # present
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        WHERE NONE(x IN n.scopes WHERE x = '{scope}')
        SET n.scopes = n.scopes + '{scope}'
        """

        with self.transaction() as tx:
            tx.run(q)

    def list_scopes(
        self, identifier: str, cls: type[CredentialedEntity]
    ) -> list[Scope]:
        """List all scopes for which the given entity has access."""

        # get the scope properties for the given entity
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        RETURN n.scopes as s
        """

        with self.transaction() as tx:
            res = tx.run(q).to_eager_result()

        scopes = []
        for record in res.records:
            scope_rec = record["s"]
            for scope_str in scope_rec:
                scope = Scope.from_str(scope_str)
                scopes.append(scope)
        return scopes

    def remove_scope(
        self, identifier: str, cls: type[CredentialedEntity], scope: Scope
    ):
        """Remove a scope from the given entity."""

        # use a list comprehension to remove the scope from the list
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        SET n.scopes = [scope IN n.scopes WHERE scope <> '{scope}']
        """

        with self.transaction() as tx:
            tx.run(q)
