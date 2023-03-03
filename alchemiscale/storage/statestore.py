"""
Node4js state storage --- :mod:`alchemiscale.storage.statestore`
===============================================================

"""

import abc
from datetime import datetime
from contextlib import contextmanager
import json
from functools import lru_cache
from time import sleep
from typing import Dict, List, Optional, Union, Tuple, Set
import weakref
import numpy as np

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, Settings
from gufe.tokenization import GufeTokenizable, GufeKey, JSON_HANDLER
from py2neo import Graph, Node, Relationship, Subgraph
from py2neo.database import Transaction
from py2neo.matching import NodeMatcher
from py2neo.errors import ClientError

from .models import (
    ComputeKey,
    Task,
    TaskHub,
    TaskArchive,
    TaskStatusEnum,
    ProtocolDAGResultRef,
)
from ..strategies import Strategy
from ..models import Scope, ScopedKey

from ..security.models import CredentialedEntity
from ..settings import Neo4jStoreSettings, get_neo4jstore_settings


@lru_cache()
def get_n4js(settings: Neo4jStoreSettings):
    """Convenience function for getting a Neo4jStore directly from settings."""
    graph = Graph(
        settings.NEO4J_URL,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASS),
        name=settings.NEO4J_DBNAME,
    )
    return Neo4jStore(graph)


class Neo4JStoreError(Exception):
    ...


class AlchemiscaleStateStore(abc.ABC):
    ...


def _select_task_from_taskpool(taskpool: Subgraph) -> Union[ScopedKey, None]:
    """
    Select a Task from a pool of tasks in a neo4j subgraph according to the following scheme:

    PRE: taskpool is a subgraph of Tasks of equal priority with a weight on their ACTIONS relationship.
    The records in the subgraph are :ACTIONS relationships with two properties: 'task' and 'weight'.
    1. Randomly select 1 Task from the TaskPool based on weighting
    2. Return the ScopedKey of the Task.

    Parameters
    ----------
    taskpool: 'subgraph'
        A subgraph of Tasks of equal priority with a weight on their ACTIONS relationship.

    Returns
    -------
    sk: ScopedKey
        The ScopedKey of the Task selected from the taskpool.
    """
    tasks = []
    weights = []
    for actions in taskpool.relationships:
        tasks.append(actions.get("task"))
        weights.append(actions.get("weight"))

    # normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # randomly select a task from the taskpool based on weights without replacement
    # NOTE: if useful could expand this to select multiple tasks
    chosen_one = np.random.choice(tasks, 1, p=weights, replace=False)
    return chosen_one[0]


def _generate_claim_query(task_sk: ScopedKey, claimant: str) -> str:
    """
    Generate a query to claim a single Task.
    Parameters
    ----------
    task_sk: ScopedKey
        The ScopedKey of the Task to claim.
    claimant: str
        The name of the claimant.

    Returns
    -------
    query: str
        The Cypher query to claim the Task.
    """
    query = f"""
    MATCH (t:Task {{_scoped_key: '{task_sk}'}})
    SET t.status = 'running', t.claim = '{claimant}'
    RETURN t
    """
    return query


class Neo4jStore(AlchemiscaleStateStore):
    # uniqueness constraints applied to the database; key is node label,
    # 'property' is the property on which uniqueness is guaranteed for nodes
    # with that label
    constraints = {
        "GufeTokenizable": {"name": "scoped_key", "property": "_scoped_key"},
        "Settings": {"name": "settings_content", "property": "content"},
        "CredentialedUserIdentity": {
            "name": "user_identifier",
            "property": "identifier",
        },
        "CredentialedComputeIdentity": {
            "name": "compute_identifier",
            "property": "identifier",
        },
    }

    def __init__(self, graph: "py2neo.Graph"):
        self.graph: Graph = graph
        self.gufe_nodes = weakref.WeakValueDictionary()

    @contextmanager
    def transaction(self, readonly=False, ignore_exceptions=False) -> Transaction:
        """Context manager for a py2neo Transaction."""
        tx = self.graph.begin(readonly=readonly)
        try:
            yield tx
        except:
            self.graph.rollback(tx)
            if not ignore_exceptions:
                raise
        else:
            self.graph.commit(tx)

    def initialize(self):
        """Initialize database.

        Ensures that constraints and any other required structures are in place.
        Should be used on any Neo4j database prior to use for Alchemiscale.

        """
        for label, values in self.constraints.items():
            self.graph.run(
                f"""
                CREATE CONSTRAINT {values['name']} IF NOT EXISTS 
                FOR (n:{label}) REQUIRE n.{values['property']} is unique
            """
            )

        # make sure we don't get objects with id 0 by creating at least one
        # this is a compensating control for a bug in py2neo, where nodes with id 0 are not properly
        # deduplicated by Subgraph set operations, which we currently rely on
        # see this PR: https://github.com/py2neo-org/py2neo/pull/951
        self.graph.run("MERGE (:NOPE)")

    def check(self):
        """Check consistency of database.

        Will raise `Neo4JStoreError` if any state check fails.
        If no check fails, will return without any exception.

        """
        constraints = {rec["name"]: rec for rec in self.graph.run("show constraints")}

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

        nope = self.graph.run("MATCH (n:NOPE) RETURN n").to_subgraph()
        if nope.identity != 0:
            raise Neo4JStoreError("Identity of NOPE node is not exactly 0")

    def _store_check(self):
        """Check that the database is in a state that can be used by the API."""
        try:
            # just list available functions to see if database is working
            self.graph.run("SHOW FUNCTIONS YIELD *")
        except:
            return False
        return True

    def reset(self):
        """Remove all data from database; undo all components in `initialize`."""
        # we'll keep NOPE around to avoid a case where Neo4j doesn't give it id 0
        # after a series of wipes; appears to happen often enough in tests
        # can remove this once py2neo#951 merged
        # self.graph.run("MATCH (n) DETACH DELETE n")
        self.graph.run("MATCH (n) WHERE NOT n:NOPE DETACH DELETE n")

        for label, values in self.constraints.items():
            self.graph.run(
                f"""
                DROP CONSTRAINT {values['name']} IF EXISTS 
            """
            )

    ## gufe object handling

    def _gufe_to_subgraph(
        self, sdct: Dict, labels: List[str], gufe_key: GufeKey, scope: Scope
    ) -> Tuple[Subgraph, Node, str]:
        subgraph = Subgraph()
        node = Node(*labels)

        # used to keep track of which properties we json-encoded so we can
        # apply decoding efficiently
        node["_json_props"] = []
        node["_gufe_key"] = str(gufe_key)
        node.update(
            {"_org": scope.org, "_campaign": scope.campaign, "_project": scope.project}
        )

        scoped_key = ScopedKey(gufe_key=node["_gufe_key"], **scope.dict())
        node["_scoped_key"] = str(scoped_key)

        for key, value in sdct.items():
            if isinstance(value, dict):
                if all([isinstance(x, GufeTokenizable) for x in value.values()]):
                    for k, v in value.items():
                        node_ = subgraph_ = self.gufe_nodes.get(
                            (v.key, scope.org, scope.campaign, scope.project)
                        )
                        if node_ is None:
                            subgraph_, node_, scoped_key_ = self._gufe_to_subgraph(
                                v.to_shallow_dict(),
                                labels=["GufeTokenizable", v.__class__.__name__],
                                gufe_key=v.key,
                                scope=scope,
                            )
                            self.gufe_nodes[
                                (str(v.key), scope.org, scope.campaign, scope.project)
                            ] = node_
                        subgraph = (
                            subgraph
                            | Relationship.type("DEPENDS_ON")(
                                node,
                                node_,
                                attribute=key,
                                key=k,
                                _org=scope.org,
                                _campaign=scope.campaign,
                                _project=scope.project,
                            )
                            | subgraph_
                        )
                else:
                    node[key] = json.dumps(value, cls=JSON_HANDLER.encoder)
                    node["_json_props"].append(key)
            elif isinstance(value, list):
                # lists can only be made of a single, primitive data type
                # we encode these as strings with a special starting indicator
                if isinstance(value[0], (int, float, str)) and all(
                    [isinstance(x, type(value[0])) for x in value]
                ):
                    node[key] = value
                elif all([isinstance(x, GufeTokenizable) for x in value]):
                    for i, x in enumerate(value):
                        node_ = subgraph_ = self.gufe_nodes.get(
                            (x.key, scope.org, scope.campaign, scope.project)
                        )
                        if node_ is None:
                            subgraph_, node_, scoped_key_ = self._gufe_to_subgraph(
                                x.to_shallow_dict(),
                                labels=["GufeTokenizable", x.__class__.__name__],
                                gufe_key=x.key,
                                scope=scope,
                            )
                            self.gufe_nodes[
                                (x.key, scope.org, scope.campaign, scope.project)
                            ] = node_
                        subgraph = (
                            subgraph
                            | Relationship.type("DEPENDS_ON")(
                                node,
                                node_,
                                attribute=key,
                                index=i,
                                _org=scope.org,
                                _campaign=scope.campaign,
                                _project=scope.project,
                            )
                            | subgraph_
                        )
                else:
                    node[key] = json.dumps(value, cls=JSON_HANDLER.encoder)
                    node["_json_props"].append(key)
            elif isinstance(value, tuple):
                # lists can only be made of a single, primitive data type
                # we encode these as strings with a special starting indicator
                if not (
                    isinstance(value[0], (int, float, str))
                    and all([isinstance(x, type(value[0])) for x in value])
                ):
                    node[key] = json.dumps(value, cls=JSON_HANDLER.encoder)
                    node["_json_props"].append(key)
            elif isinstance(value, GufeTokenizable):
                node_ = subgraph_ = self.gufe_nodes.get(
                    (value.key, scope.org, scope.campaign, scope.project)
                )
                if node_ is None:
                    subgraph_, node_, scoped_key_ = self._gufe_to_subgraph(
                        value.to_shallow_dict(),
                        labels=["GufeTokenizable", value.__class__.__name__],
                        gufe_key=value.key,
                        scope=scope,
                    )
                    self.gufe_nodes[
                        (value.key, scope.org, scope.campaign, scope.project)
                    ] = node_
                subgraph = (
                    subgraph
                    | Relationship.type("DEPENDS_ON")(
                        node,
                        node_,
                        attribute=key,
                        _org=scope.org,
                        _campaign=scope.campaign,
                        _project=scope.project,
                    )
                    | subgraph_
                )
            elif isinstance(value, Settings):
                # TODO: finish up approach here for serializing settings
                # include reverse operation in `subgraph_to_gufe`
                settings_json = json.dumps(
                        value.settings,
                        cls=JSON_HANDLER.encoder,
                        sort_keys=True)

                node_ = Node("Settings")
                node_["content"] = settings_json
                node_["hashdigest"] = hashlib.md5(
                        settings_json.encode(), 
                        usedforsecurity=False).hexdigest()
                subgraph = (
                    subgraph
                    | Relationship.type("DEPENDS_ON")(
                        node,
                        node_,
                        attribute=key,
                        _org=scope.org,
                        _campaign=scope.campaign,
                        _project=scope.project,
                    )
                    | subgraph_
                )

            else:
                node[key] = value

        subgraph = subgraph | node

        return subgraph, node, scoped_key

    def _subgraph_to_gufe(
        self, nodes: List[Node], subgraph: Subgraph
    ) -> Dict[Node, GufeTokenizable]:
        """Get a Dict `GufeTokenizable` objects within the given subgraph.

        Any `GufeTokenizable` that requires nodes or relationships missing from the subgraph will not be returned.

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
        self, node: Node, g: nx.DiGraph, mapping: Dict[Node, GufeTokenizable]
    ):
        # shortcut if we already have this object deserialized
        if gufe_obj := mapping.get(node):
            return gufe_obj

        dct = dict(node)
        for key, value in dict(node).items():
            # deserialize json-serialized attributes
            if key in dct["_json_props"]:
                dct[key] = json.loads(value, cls=JSON_HANDLER.decoder)

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
    ) -> Union[Node, Tuple[Node, Subgraph]]:
        """
        If `return_subgraph = True`, also return subgraph for gufe object.

        """
        qualname = scoped_key.qualname

        properties = {"_scoped_key": str(scoped_key)}
        prop_string = ", ".join(
            "{}: '{}'".format(key, value) for key, value in properties.items()
        )

        prop_string = f" {{{prop_string}}}"

        q = f"""
        MATCH (n:{qualname}{prop_string})
        """

        if return_subgraph:
            q += """
            OPTIONAL MATCH p = (n)-[r:DEPENDS_ON*]->(m) 
            WHERE NOT (m)-[:DEPENDS_ON]->()
            RETURN n,p
            """
        else:
            q += """
            RETURN n
            """

        nodes = set()
        subgraph = Subgraph()

        for record in self.graph.run(q):
            nodes.add(record["n"])
            if return_subgraph and record["p"] is not None:
                subgraph = subgraph | record["p"]
            else:
                subgraph = record["n"]

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
        additional: Dict = None,
        key: GufeKey = None,
        scope: Scope = Scope(),
        return_gufe=False,
    ):
        properties = {
            "_org": scope.org,
            "_campaign": scope.campaign,
            "_project": scope.project,
        }

        for k, v in list(properties.items()):
            if v is None:
                properties.pop(k)

        if key is not None:
            properties["_gufe_key"] = str(key)

        if additional is None:
            additional = {}
        properties.update({k: v for k, v in additional.items() if v is not None})

        if not properties:
            prop_string = ""
        else:
            prop_string = ", ".join(
                "{}: '{}'".format(key, value) for key, value in properties.items()
            )

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
            RETURN n
            """
        with self.transaction() as tx:
            res = tx.run(q)

        nodes = set()
        subgraph = Subgraph()

        for record in res:
            nodes.add(record["n"])
            if return_gufe and record["p"] is not None:
                subgraph = subgraph | record["p"]
            else:
                subgraph = record["n"]

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
        node, subgraph = self._get_node(scoped_key=scoped_key, return_subgraph=True)
        return self._subgraph_to_gufe([node], subgraph)[node]

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
        """Add an `AlchemicalNetwork` to the target neo4j database, even if
        some of its components already exist in the database.

        """

        ndict = network.to_shallow_dict()

        g, n, scoped_key = self._gufe_to_subgraph(
            ndict,
            labels=["GufeTokenizable", network.__class__.__name__],
            gufe_key=network.key,
            scope=scope,
        )
        with self.transaction() as tx:
            tx.merge(g, "GufeTokenizable", "_scoped_key")

        return scoped_key

    def delete_network(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Delete the given `AlchemicalNetwork` from the database.

        This will not remove any `Transformation`s or `ChemicalSystem`s
        associated with the `AlchemicalNetwork`, since these may be associated
        with other `AlchemicalNetwork`s in the same `Scope`.

        """
        # note: something like the following could perhaps be used to delete everything that is *only*
        # associated with this network
        # not yet tested though
        """
        MATCH p = (n:AlchemicalNetwork {{_scoped_key: "{network_node['_scoped_key']}")-[r:DEPENDS_ON*]->(m),
              (n)-[:DEPENDS_ON]->(t:Transformation),
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
        q = f"""
        MATCH (an:AlchemicalNetwork {{_scoped_key: "{network}"}})
        DETACH DELETE an
        """
        return network

    def query_networks(
        self,
        *,
        name=None,
        key=None,
        scope: Optional[Scope] = Scope(),
        return_gufe: bool = False,
    ):
        """Query for `AlchemicalNetwork`s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="AlchemicalNetwork",
            additional=additional,
            key=key,
            scope=scope,
            return_gufe=return_gufe,
        )

    def query_transformations(
        self, *, name=None, key=None, scope: Scope = Scope(), chemical_systems=None
    ):
        """Query for `Transformation`s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="Transformation", additional=additional, key=key, scope=scope
        )

    def query_chemicalsystems(
        self, *, name=None, key=None, scope: Scope = Scope(), transformations=None
    ):
        """Query for `ChemicalSystem`s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="ChemicalSystem", additional=additional, key=key, scope=scope
        )

    def get_transformations_for_chemicalsystem(self):
        ...

    def get_networks_for_transformation(self):
        ...

    def _get_protocoldagresultrefs(self, q):
        with self.transaction() as tx:
            res = tx.run(q)

        protocoldagresultrefs = []
        subgraph = Subgraph()
        for record in res:
            protocoldagresultrefs.append(record["res"])
            subgraph = subgraph | record["res"]

        return list(self._subgraph_to_gufe(protocoldagresultrefs, subgraph).values())

    def get_transformation_results(
        self, transformation: ScopedKey
    ) -> List[ProtocolDAGResultRef]:
        # get all task result protocoldagresultrefs corresponding to given transformation
        # returned in no particular order
        q = f"""
        MATCH (trans:Transformation {{_scoped_key: "{transformation}"}}),
              (trans)<-[:PERFORMS]-(:Task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = true
        RETURN res
        """
        return self._get_protocoldagresultrefs(q)

    def get_transformation_failures(
        self, transformation: ScopedKey
    ) -> List[ProtocolDAGResultRef]:
        # get all task failure protocoldagresultrefs corresponding to given transformation
        # returned in no particular order
        q = f"""
        MATCH (trans:Transformation {{_scoped_key: "{transformation}"}}),
              (trans)<-[:PERFORMS]-(:Task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = false
        RETURN res
        """
        return self._get_protocoldagresultrefs(q)

    ## compute

    def set_strategy(
        self,
        strategy: Strategy,
        network: ScopedKey,
    ) -> ScopedKey:
        """Set the compute Strategy for the given AlchemicalNetwork."""
        ...

    ## task hubs

    def create_taskhub(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Create a TaskHub for the given AlchemicalNetwork.

        An AlchemicalNetwork can have only one associated TaskHub.
        A TaskHub is required to action Tasks for a given AlchemicalNetwork.

        This method will only create a TaskHub for an AlchemicalNetwork if it
        doesn't already exist; it will return the scoped key for the TaskHub
        either way.

        """
        scope = network.scope
        network_node = self._get_node(network)

        # create a taskhub for the supplied network
        # use a PERFORMS relationship
        taskhub = TaskHub(network=str(network))
        _, taskhub_node, scoped_key = self._gufe_to_subgraph(
            taskhub.to_shallow_dict(),
            labels=["GufeTokenizable", taskhub.__class__.__name__],
            gufe_key=taskhub.key,
            scope=scope,
        )

        subgraph = Relationship.type("PERFORMS")(
            taskhub_node,
            network_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        # if the TaskHub already exists, this will rollback transaction
        # automatically
        with self.transaction(ignore_exceptions=True) as tx:
            tx.create(subgraph)

        return scoped_key

    def query_taskhubs(
        self, scope: Optional[Scope] = Scope(), return_gufe: bool = False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, TaskHub]]:
        """Query for `TaskHub`s matching the given criteria.

        Parameters
        ----------
        return_gufe
            If True, return a dict with `ScopedKey`s as keys, `TaskHub`
            instances as values. Otherwise, return a list of `ScopedKey`s.

        """
        return self._query(qualname="TaskHub", scope=scope, return_gufe=return_gufe)

    def get_taskhub(
        self, network: ScopedKey, return_gufe: bool = False
    ) -> Union[ScopedKey, TaskHub]:
        """Get the TaskHub for the given AlchemicalNetwork.

        Parameters
        ----------
        return_gufe
            If True, return a `TaskHub` instance.
            Otherwise, return a `ScopedKey`.

        """
        node = self.graph.run(
            f"""
                match (th:TaskHub {{network: "{network}"}})-[:PERFORMS]->(an:AlchemicalNetwork)
                return th
                """
        ).to_subgraph()

        if return_gufe:
            return self._subgraph_to_gufe([node], node)[node]
        else:
            return ScopedKey.from_str(node["_scoped_key"])

    def delete_taskhub(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Delete a TaskHub for a given AlchemicalNetwork."""
        taskhub = self.get_taskhub(network)

        q = f"""
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}}),
        DETACH DELETE th
        """
        self.graph.run(q)

        return taskhub

    def set_taskhub_weight(self, network: ScopedKey, weight: float):
        q = f"""
        MATCH (th:TaskHub {{network: "{network}"}})
        SET th.weight = {weight}
        RETURN th
        """
        with self.transaction() as tx:
            tx.run(q)

    def action_tasks(
        self,
        tasks: List[ScopedKey],
        taskhub: ScopedKey,
    ) -> List[Union[ScopedKey, None]]:
        """Add Tasks to the TaskHub for a given AlchemicalNetwork.

        Note: the Tasks must be within the same scope as the AlchemicalNetwork,
        and must correspond to a Transformation in the AlchemicalNetwork.

        A given compute task can be represented in any number of
        AlchemicalNetwork TaskHubs, or none at all.

        """
        with self.transaction() as tx:
            actioned_sks = []
            for t in tasks:
                q = f"""
                // get our TaskHub
                MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[:PERFORMS]->(an:AlchemicalNetwork)
                
                // get the task we want to add to the hub; check that it connects to same network
                MATCH (task:Task {{_scoped_key: '{t}'}})-[:PERFORMS]->(tf:Transformation)<-[:DEPENDS_ON]-(an)

                // only proceed for cases where task is not already actioned on hub
                WITH th, an, task
                WHERE NOT (th)-[:ACTIONS]->(task)

                // create the connection
                CREATE (th)-[ar:ACTIONS {{weight: 1.0}}]->(task)

                // set the task property to the scoped key of the Task
                // this is a convenience for when we have to loop over relationships in Python
                SET ar.task = task._scoped_key

                RETURN task
                """
                task = tx.run(q).to_subgraph()
                actioned_sks.append(
                    ScopedKey.from_str(task["_scoped_key"])
                    if task is not None
                    else None
                )
        return actioned_sks

    def set_task_weights(
        self,
        tasks: Union[Dict[ScopedKey, float], List[ScopedKey]],
        taskhub: ScopedKey,
        weight: Optional[float] = None,
    ) -> List[Union[ScopedKey, None]]:
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

                for t, w in tasks.items():
                    q = f"""
                    MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[ar:ACTIONS]->(task:Task {{_scoped_key: '{t}'}})
                    SET ar.weight = {w}
                    RETURN task, ar
                    """
                    results.append(tx.run(q))

            elif isinstance(tasks, list):
                if weight is None:
                    raise ValueError(
                        "Must set `weight` to a scalar if `tasks` is a list"
                    )

                for t in tasks:
                    q = f"""
                    MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[ar:ACTIONS]->(task:Task {{_scoped_key: '{t}'}})
                    SET ar.weight = {weight}
                    RETURN task, ar
                    """
                    tx.run(q)

        # return ScopedKeys for Tasks we changed; `None` for tasks we didn't
        for res in results:
            for record in res:
                task = record["task"]
                tasks_changed.append(
                    ScopedKey.from_str(task["_scoped_key"])
                    if task is not None
                    else None
                )

        return tasks_changed

    def get_task_weights(
        self,
        tasks: List[ScopedKey],
        taskhub: ScopedKey,
    ) -> List[Union[float, None]]:
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
        weights = []
        with self.transaction() as tx:
            for t in tasks:
                q = f"""
                MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[ar:ACTIONS]->(task:Task {{_scoped_key: '{t}'}})
                RETURN ar.weight
                """
                result = tx.run(q)

                weight = [record.get("ar.weight") for record in result]

                # if no match for the given Task, we put a `None` as result
                if len(weight) == 0:
                    weights.append(None)
                else:
                    weights.extend(weight)

        return weights

    def cancel_tasks(
        self,
        tasks: List[ScopedKey],
        taskhub: ScopedKey,
    ) -> List[Union[ScopedKey, None]]:
        """Remove Tasks from the TaskHub for a given AlchemicalNetwork.

        Note: Tasks must be within the same scope as the AlchemicalNetwork.

        A given Task can be represented in many AlchemicalNetwork TaskHubs, or
        none at all.

        """
        canceled_sks = []
        with self.transaction() as tx:
            for t in tasks:
                q = f"""
                // get our task hub, as well as the task :ACTIONS relationship we want to remove
                MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[ar:ACTIONS]->(task:Task {{_scoped_key: '{t}'}})
                DELETE ar
                RETURN task
                """
                task = tx.run(q).to_subgraph()
                canceled_sks.append(
                    ScopedKey.from_str(task["_scoped_key"])
                    if task is not None
                    else None
                )

        return canceled_sks

    def get_taskhub_tasks(
        self, taskhub: ScopedKey, return_gufe=False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, Task]]:
        """Get a list of Tasks on the TaskHub."""

        q = f"""
        // get list of all tasks associated with the taskhub
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[:ACTIONS]->(task:Task)
        RETURN task
        """
        with self.transaction() as tx:
            res = tx.run(q)

        tasks = []
        subgraph = Subgraph()
        for record in res:
            tasks.append(record["task"])
            subgraph = subgraph | record["task"]

        if return_gufe:
            return {
                ScopedKey.from_str(k["_scoped_key"]): v
                for k, v in self._subgraph_to_gufe(tasks, subgraph).items()
            }
        else:
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]

    def get_taskhub_unclaimed_tasks(
        self, taskhub: ScopedKey, return_gufe=False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, Task]]:
        """Get a list of unclaimed Tasks in the TaskHub."""

        q = f"""
        // get list of all unclaimed tasks in the hub 
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[:ACTIONS]->(task:Task)
        WHERE task.claim IS NULL
        RETURN task
        """
        with self.transaction() as tx:
            res = tx.run(q)

        tasks = []
        subgraph = Subgraph()
        for record in res:
            tasks.append(record["task"])
            subgraph = subgraph | record["task"]

        if return_gufe:
            return {
                ScopedKey.from_str(k["_scoped_key"]): v
                for k, v in self._subgraph_to_gufe(tasks, subgraph).items()
            }
        else:
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]

    def claim_taskhub_tasks(
        self, taskhub: ScopedKey, claimant: str, count: int = 1
    ) -> List[Union[ScopedKey, None]]:
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
        claimant
            Unique identifier for the entity claiming the Tasks for execution.
        count
            Claim the given number of Tasks in a single transaction.

        """
        taskpool_q = f"""
        // get list of all eligible 'waiting' tasks in the hub 
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[actions:ACTIONS]-(task:Task)
        WHERE task.status = 'waiting'
        AND actions.weight > 0
        OPTIONAL MATCH (task)-[:EXTENDS]->(other_task:Task)

        // drop tasks from consideration if they EXTENDS an incomplete task
        WITH task, other_task, actions
        WHERE other_task.status = 'complete' OR other_task IS NULL

        // get the highest priority present among these tasks (value nearest to 1)
        WITH MIN(task.priority) as top_priority

        // match again, this time filtering on highest priority
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})-[actions:ACTIONS]-(task:Task)
        WHERE task.status = 'waiting'
        AND actions.weight > 0
        AND task.priority = top_priority
        OPTIONAL MATCH (task)-[:EXTENDS]->(other_task:Task)

        // drop tasks from consideration if they EXTENDS an incomplete task
        WITH task, other_task, actions
        WHERE other_task.status = 'complete' OR other_task IS NULL

        // return the tasks and actions relationships
        RETURN task, actions
        """

        tasks = []
        with self.transaction() as tx:
            tx.run(
                f"""
            MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})

            // lock the TaskHub to avoid other queries from changing its state while we claim
            SET th._lock = True
            """
            )
            for i in range(count):
                taskpool = tx.run(taskpool_q).to_subgraph()
                if taskpool is None:
                    tasks.append(None)
                else:
                    chosen_one = _select_task_from_taskpool(taskpool)
                    claim_query = _generate_claim_query(chosen_one, claimant)
                    tasks.append(tx.run(claim_query).to_subgraph())

            tx.run(
                f"""
            MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}})

            // remove lock on the TaskHub now that we're done with it
            SET th._lock = null
            """
            )

        return [
            ScopedKey.from_str(t["_scoped_key"]) if t is not None else None
            for t in tasks
        ]

    ## tasks

    def create_task(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        creator: Optional[str] = None,
    ) -> ScopedKey:
        """Add a compute Task to a Transformation.

        Note: this creates a compute Task, but does not add it to any TaskHubs.

        Parameters
        ----------
        transformation
            The Transformation to compute.
        scope
            The scope the Transformation is in; ignored if `transformation` is a ScopedKey.
        extends
            The ScopedKey of the Task to use as a starting point for this Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extends` input for the Task's eventual call to `Protocol.create`.

        """
        scope = transformation.scope
        transformation_node = self._get_node(transformation)

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        task = Task(
            creator=creator, extends=str(extends) if extends is not None else None
        )

        _, task_node, scoped_key = self._gufe_to_subgraph(
            task.to_shallow_dict(),
            labels=["GufeTokenizable", task.__class__.__name__],
            gufe_key=task.key,
            scope=scope,
        )

        subgraph = Subgraph()

        if extends is not None:
            previous_task_node = self._get_node(extends)
            subgraph = subgraph | Relationship.type("EXTENDS")(
                task_node,
                previous_task_node,
                _org=scope.org,
                _campaign=scope.campaign,
                _project=scope.project,
            )

        subgraph = subgraph | Relationship.type("PERFORMS")(
            task_node,
            transformation_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        with self.transaction() as tx:
            tx.create(subgraph)

        return scoped_key

    def set_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[Task] = None,
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
        transformation_node = self._get_node_from_obj_or_sk(
            transformation, Transformation, scope
        )

    def set_task_priority(self, task: ScopedKey, priority: int):
        q = f"""
        MATCH (t:Task {{_scoped_key: "{task}"}})
        SET t.priority = {priority}
        RETURN t
        """
        with self.transaction() as tx:
            tx.run(q)

    def get_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        return_as: str = "list",
    ) -> Union[List[ScopedKey], Dict[ScopedKey, Optional[ScopedKey]]]:
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
        q = f"""
        MATCH (trans:Transformation {{_scoped_key: '{transformation}'}})<-[:PERFORMS]-(task:Task)
        """

        if extends:
            q += f"""
            MATCH (trans)<-[:PERFORMS]-(extends:Task {{_scoped_key: '{extends}'}})
            WHERE (task)-[:EXTENDS*]->(extends)
            RETURN task
            """
        else:
            q += f"""
            RETURN task
            """

        with self.transaction() as tx:
            res = tx.run(q)

        tasks = []
        for record in res:
            tasks.append(record["task"])

        if return_as == "list":
            return [ScopedKey.from_str(t["_scoped_key"]) for t in tasks]
        elif return_as == "graph":
            return {
                ScopedKey.from_str(t["_scoped_key"]): ScopedKey.from_str(t["extends"])
                if t["extends"] is not None
                else None
                for t in tasks
            }

    def query_tasks(
        self,
        scope: Optional[Scope] = None,
        network: Optional[ScopedKey] = None,
        transformation: Optional[ScopedKey] = None,
        extends: Optional[ScopedKey] = None,
        status: Optional[List[TaskStatusEnum]] = None,
    ):
        raise NotImplementedError

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

    def get_task_transformation(
        self,
        task: ScopedKey,
        return_gufe=True,
    ) -> Tuple[Transformation, Optional[ProtocolDAGResultRef]]:
        """Get the `Transformation` and `ProtocolDAGResultRef` to extend from (if
        present) for the given `Task`.

        """
        q = f"""
        MATCH (task:Task {{_scoped_key: "{task}"}})-[:PERFORMS]->(trans:Transformation)
        OPTIONAL MATCH (task)-[:EXTENDS]->(prev:Task)-[:RESULTS_IN]->(result:ProtocolDAGResultRef)
        RETURN trans, result
        """

        with self.transaction() as tx:
            res = tx.run(q)

        transformations = []
        results = []
        for record in res:
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
                self.get_gufe(protocoldagresultref)
                if protocoldagresultref is not None
                else None,
            )

        return transformation, protocoldagresultref

    def set_task_result(
        self, task: ScopedKey, protocoldagresultref: ProtocolDAGResultRef
    ) -> ScopedKey:
        """Set a `ProtocolDAGResultRef` pointing to a `ProtocolDAGResult` for the given `Task`."""
        scope = task.scope
        task_node = self._get_node(task)

        subgraph, protocoldagresultref_node, scoped_key = self._gufe_to_subgraph(
            protocoldagresultref.to_shallow_dict(),
            labels=["GufeTokenizable", protocoldagresultref.__class__.__name__],
            gufe_key=protocoldagresultref.key,
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
            tx.merge(subgraph, "GufeTokenizable", "_scoped_key")

        return scoped_key

    def get_task_results(self, task: ScopedKey) -> List[ProtocolDAGResultRef]:
        # get all task result protocoldagresultrefs corresponding to given task
        # returned in no particular order
        q = f"""
        MATCH (task:Task {{_scoped_key: "{task}"}}),
              (task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = true
        RETURN res
        """
        return self._get_protocoldagresultrefs(q)

    def get_task_failures(self, task: ScopedKey) -> List[ProtocolDAGResultRef]:
        # get all task failure protocoldagresultrefs corresponding to given task
        # returned in no particular order
        q = f"""
        MATCH (task:Task {{_scoped_key: "{task}"}}),
              (task)-[:RESULTS_IN]->(res:ProtocolDAGResultRef)
        WHERE res.ok = false
        RETURN res
        """
        return self._get_protocoldagresultrefs(q)

    def set_task_waiting(self, task: ScopedKey, clear_claim=True):
        q = f"""
        MATCH (t:Task {{_scoped_key: "{task}"}})
        SET t.status = 'waiting'
        """

        if clear_claim:
            q += ", t.claimant = null"

        q += """
        RETURN t
        """

        with self.transaction() as tx:
            tx.run(q)

    def set_task_running(self, task: ScopedKey, computekey: ComputeKey):
        ...

    def set_task_complete(
        self,
        task: ScopedKey,
    ):
        q = f"""
        MATCH (t:Task {{_scoped_key: "{task}"}})
        SET t.status = 'complete'
        """
        with self.transaction() as tx:
            tx.run(q)

    def set_task_error(
        self,
        task: Union[Task, ScopedKey],
        cancel=True,
    ):
        ...

    def set_task_invalid(
        self,
        task: Union[Task, ScopedKey],
        cancel=True,
    ):
        ...

    def set_task_deleted(
        self,
        task: Union[Task, ScopedKey],
        cancel=True,
    ):
        ...

    ## authentication

    def create_credentialed_entity(self, entity: CredentialedEntity):
        """Create a new credentialed entity, such as a user or compute identity.

        If an entity of this type with the same `identifier` already exists,
        then this will overwrite its properties, including credential.

        """
        node = Node("CredentialedEntity", entity.__class__.__name__, **entity.dict())

        with self.transaction() as tx:
            tx.merge(
                node, primary_label=entity.__class__.__name__, primary_key="identifier"
            )

    def get_credentialed_entity(self, identifier: str, cls: type[CredentialedEntity]):
        """Get an existing credentialed entity, such as a user or compute identity."""
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        RETURN n
        """

        with self.transaction() as tx:
            res = tx.run(q)

        nodes = set()
        for record in res:
            nodes.add(record["n"])

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
            res = tx.run(q)

        nodes = set()
        for record in res:
            nodes.add(record["n"])

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
    ) -> List[Scope]:
        """List all scopes for which the given entity has access."""

        # get the scope properties for the given entity
        q = f"""
        MATCH (n:{cls.__name__} {{identifier: '{identifier}'}})
        RETURN n.scopes as s
        """

        with self.transaction() as tx:
            res = tx.run(q)

        scopes = []
        for record in res:
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
