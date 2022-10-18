import abc
from contextlib import contextmanager
import json
from typing import Dict, List, Optional, Union
import weakref

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ProtocolDAGResult
from gufe.tokenization import GufeTokenizable, GufeKey
from gufe.storage.metadatastore import MetadataStore
from py2neo import Graph, Node, Relationship, Subgraph
from py2neo.matching import NodeMatcher
from py2neo.errors import ClientError

from .models import ComputeKey, Task, TaskQueue, TaskArchive
from ..strategies import Strategy
from ..models import Scope, ScopeKey


class Neo4JStoreError(Exception):
    ...


class FahAlchemyStateStore(abc.ABC):
    ...


class Neo4jStore(FahAlchemyStateStore):

    def __init__(self, graph: "py2neo.Graph"):
        self.graph = graph
        self.gufe_nodes = weakref.WeakValueDictionary()

    @contextmanager
    def as_tempdb(self):
        try:
            yield
        finally:
            self.graph.delete_all()

    ### gufe object handling

    def _gufe_to_subgraph(
            self, 
            sdct: Dict,
            labels: List[str],
            gufe_key: GufeKey,
            scope: Scope
    ):
        subgraph = Subgraph()
        node = Node(*labels)

        # used to keep track of which properties we json-encoded so we can
        # apply decoding efficiently
        node["_json_props"] = []
        node["_gufe_key"] = str(gufe_key)
        node.update({"_org": scope.org, "_campaign": scope.campaign, "_project": scope.project})

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
                                scope=scope
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
                    node[key] = json.dumps(value)
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
                                scope=scope
                            )
                            self.gufe_nodes[(x.key, scope.org, scope.campaign, scope.project)] = node_
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
                    node[key] = json.dumps(value)
                    node["_json_props"].append(key)
            elif isinstance(value, tuple):
                # lists can only be made of a single, primitive data type
                # we encode these as strings with a special starting indicator
                if not (
                    isinstance(value[0], (int, float, str))
                    and all([isinstance(x, type(value[0])) for x in value])
                ):
                    node[key] = json.dumps(value)
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
                        scope=scope
                    )
                    self.gufe_nodes[(value.key, scope.org, scope.campaign, scope.project)] = node_
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

    def _subgraph_to_gufe(self, nodes: List[Node], subgraph: Subgraph):
        """Get a list of all `GufeTokenizable` objects within the given subgraph.

        Any `GufeTokenizable` that requires nodes or relationships missing from the subgraph will not be returned.

        Returns
        -------
        List[GufeTokenizable]

        """
        nxg = self._subgraph_to_networkx(subgraph)
        # nodes = list(reversed(list(nx.topological_sort(subgraph_to_networkx(sg)))))

        nodes_to_gufe = {}

        gufe_objs = []
        for node in nodes:
            gufe_objs.append(self._node_to_gufe(node, nxg, nodes_to_gufe))

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
                dct[key] = json.loads(value)

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

    def _get_obj(
        self,
        *,
        qualname,
        scoped_key: str
    ):
        properties = {"_scoped_key": scoped_key}
        prop_string = ", ".join(
            "{}: '{}'".format(key, value) for key, value in properties.items()
        )

        prop_string = f" {{{prop_string}}}"

        q = f"""
        MATCH p = (n:{qualname}{prop_string})-[r:DEPENDS_ON*]->(m) 
        WHERE NOT (m)-[:DEPENDS_ON]->()
        RETURN n,p
        """
        nodes = set()
        subgraph = Subgraph()

        for record in self.graph.run(q):
            nodes.add(record["n"])
            subgraph = subgraph | record["p"]

        if len(nodes) > 1:
            raise Neo4JStoreError("More than one result for given `scoped_key`; this should not be possible")

        gufe_objs = self._subgraph_to_gufe(nodes, subgraph)

        if len(gufe_objs) > 1:
            raise Neo4JStoreError("More than one result for given `scoped_key`; this should not be possible")

        return gufe_objs[0]

    def _query_obj(
        self,
        *,
        qualname: str,
        additional: Dict = None,
        key: GufeKey = None,
        scope: Scope = Scope() 
    ):
        # TODO : add pagination
        properties = {"_org": scope.org, "_campaign": scope.campaign, "_project": scope.project}

        for (k, v) in list(properties.items()):
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
        MATCH p = (n:{qualname}{prop_string})-[r:DEPENDS_ON*]->(m) 
        WHERE NOT (m)-[:DEPENDS_ON]->()
        RETURN n,p
        """
        nodes = set()
        subgraph = Subgraph()

        for record in self.graph.run(q):
            nodes.add(record["n"])
            subgraph = subgraph | record["p"]

        return self._subgraph_to_gufe(nodes, subgraph)

    def create_network(self, network: AlchemicalNetwork, scope: Scope) -> ScopedKey:
        """Add an `AlchemicalNetwork` to the target neo4j database.

        Will give a `ValueError` if any components already exist in the database.
        If this is expected, consider using `update_network` instead.

        """
        g, n, scoped_key = self._gufe_to_subgraph(
            network.to_shallow_dict(),
            labels=["GufeTokenizable", network.__class__.__name__],
            gufe_key=network.key,
            scope=scope
        )

        try:
            self.graph.create(g)
        except ClientError:
            raise ValueError(
                "At least one component of the network already exists in the target database; "
                "consider using `update_network` if this is expected."
            )

        return scoped_key

    def update_network(self, network: AlchemicalNetwork, scope: Scope):
        """Add an `AlchemicalNetwork` to the target neo4j database, even if
        some of its components already exist in the database.

        """

        ndict = network.to_shallow_dict()

        g, n, scoped_key = self._gufe_to_subgraph(
            ndict,
            labels=["GufeTokenizable", network.__class__.__name__],
            gufe_key=network.key,
            scope=scope
        )
        self.graph.merge(g, "GufeTokenizable", "_scoped_key")

        return scoped_key

    def get_network(self, scoped_key: ScopedKey):
        """Get a specific `AlchemicalNetwork` using its `scoped_key`."""

        return self._get_obj(
            qualname="AlchemicalNetwork",
            scoped_key=scoped_key
        )

    def delete_network(self, scoped_key: ScopedKey):
        ...

    def query_networks(
        self, *, name=None, key=None, scope: Optional[Scope] = Scope() 
    ):
        """Query for `AlchemicalNetwork`s matching given attributes."""
        additional = {"name": name}
        return self._query_obj(
            qualname="AlchemicalNetwork",
            additional=additional,
            key=key,
            scope=scope
        )

    def query_transformations(
        self, *, name=None, key=None, scope: Scope = Scope(),
        chemical_systems=None
    ):
        """Query for `Transformation`s matching given attributes."""
        additional = {"name": name}
        return self._query_obj(
            qualname="Transformation",
            additional=additional,
            key=key,
            scope=scope
        )

    def query_chemicalsystems(
        self, *, name=None, key=None, scope: Scope = Scope()
    ):
        """Query for `ChemicalSystem`s matching given attributes."""
        additional = {"name": name}
        return self._query_obj(
            qualname="ChemicalSystem",
            additional=additional,
            key=key,
            scope=scope
        )

    def get_transformations_for_chemicalsystem(self):
        ...

    def get_networks_for_transformation(self):
        ...

    def get_transformations_result(self):
        ...

    ### compute

    def set_strategy(
            self,
            strategy: Strategy,
            network: Union[AlchemicalNetwork, ScopedKey],
            scope: Scope,
        ) -> ScopedKey: 
        """Set the compute Strategy for the given AlchemicalNetwork.

        """
        ...

    def create_task(
            self, 
            transformation: Union[Transformation, ScopedKey],
            scope: Scope,
            extend_from: Optional[ProtocolDAGResult] = None
        ) -> Task:
        """Add a compute Task to a Transformation.

        Note: this creates a compute Task, but does not add it to any TaskQueues.

        """

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        ...

    def delete_task(
            self, 
            task: Union[Task, ScopedKey],
        ) -> Task:
        """Remove a compute Task from a Transformation.

        This will also remove the Task from all TaskQueues it is a part of.

        """

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        ...

    def action_task(
            self,
            task: Union[Task, ScopedKey],
            network: Union[AlchemicalNetwork, ScopedKey],
        ) -> Task:
        """Add a compute Task to the TaskQueue for a given AlchemicalNetwork.

        Note: the Task must be within the same scope as the AlchemicalNetwork.

        A given compute task can be represented in any number of
        AlchemicalNetwork queues, or none at all.

        """
        ...

        # add task to the end of the queue; use a cursor to query the queue for last
        # item; add new FOLLOWS relationship 

    def cancel_task(
            self,
            task: Union[Task, ScopedKey],
            network: Union[AlchemicalNetwork, ScopedKey],
        ) -> Task:
        """Remove a compute Task from the TaskQueue for a given AlchemicalNetwork.

        Note: the Task must be within the same scope as the AlchemicalNetwork.

        A given compute task can be represented in many AlchemicalNetwork
        queues, or none at all.

        """
        ...

    def get_taskqueue(
            self,
            network: Union[AlchemicalNetwork, ScopedKey],
            scope: Scope,
        ):
        ...

    def query_tasks(
            self,
            network: Union[AlchemicalNetwork, ScopedKey],
            transformation: Union[Transformation, ScopedKey],
            scope: Scope,
        ):
        ...

    def mark_task_ready(
            self,
            task: Union[Task, ScopedKey],
        ):
        ...

    def mark_task_running(
            self,
            task: Union[Task, ScopedKey],
            computekey: ComputeKey
        ):
        ...

    def mark_task_complete(
            self,
            task: Union[Task, ScopedKey],
        ):
        ...

    def mark_task_failed(
            self,
            task: Union[Task, ScopedKey],
        ):
        ...

    def mark_task_defunct(
            self,
            task: Union[Task, ScopedKey],
        ):
        ...
