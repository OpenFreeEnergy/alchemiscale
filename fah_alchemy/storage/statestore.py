import abc
from contextlib import contextmanager
import json
from functools import lru_cache
from time import sleep
from typing import Dict, List, Optional, Union, Tuple
import weakref

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ProtocolDAGResult
from gufe.tokenization import GufeTokenizable, GufeKey
from gufe.storage.metadatastore import MetadataStore
from py2neo import Graph, Node, Relationship, Subgraph
from py2neo.database import Transaction
from py2neo.matching import NodeMatcher
from py2neo.errors import ClientError

from .models import ComputeKey, Task, TaskQueue, TaskArchive, TaskStatusEnum
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


class FahAlchemyStateStore(abc.ABC):
    ...


class Neo4jStore(FahAlchemyStateStore):

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
        Should be used on any Neo4j database prior to use for fah-alchemy.

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
        # first, delete the network's queue if present
        self.delete_taskqueue(network)

        # then delete the network
        q = f"""
        MATCH (an:AlchemicalNetwork {{_scoped_key: "{network}"}})
        DETACH DELETE an
        """
        return network

    def query_networks(self, *, name=None, key=None, scope: Optional[Scope] = Scope()):
        """Query for `AlchemicalNetwork`s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="AlchemicalNetwork", additional=additional, key=key, scope=scope
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

    def get_transformation_results(self):
        ...

    ## compute

    def set_strategy(
        self,
        strategy: Strategy,
        network: ScopedKey,
    ) -> ScopedKey:
        """Set the compute Strategy for the given AlchemicalNetwork."""
        ...

    ## task queues

    def create_taskqueue(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Create a TaskQueue for the given AlchemicalNetwork.

        An AlchemicalNetwork can have only one associated TaskQueue.
        A TaskQueue is required to queue Tasks for a given AlchemicalNetwork.

        This method will only creat a TaskQueue for an AlchemicalNetwork if it
        doesn't already exist; it will return the scoped key for the TaskQueue
        either way.

        """
        scope = network.scope
        network_node = self._get_node(network)

        # create a taskqueue for the supplied network
        # use a PERFORMS relationship
        taskqueue = TaskQueue(network=str(network))
        _, taskqueue_node, scoped_key = self._gufe_to_subgraph(
            taskqueue.to_shallow_dict(),
            labels=["GufeTokenizable", taskqueue.__class__.__name__],
            gufe_key=taskqueue.key,
            scope=scope,
        )

        subgraph = Relationship.type("PERFORMS")(
            taskqueue_node,
            network_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        # create head and tail node, attach to TaskQueue node
        # head and tail connected via FOLLOWS relationship
        head = Node("TaskQueueHead")
        tail = Node("TaskQueueTail")

        subgraph = subgraph | Relationship.type("TASKQUEUE_HEAD")(taskqueue_node, head)
        subgraph = subgraph | Relationship.type("TASKQUEUE_TAIL")(taskqueue_node, tail)
        subgraph = subgraph | Relationship.type("FOLLOWS")(
            tail, head, taskqueue=str(scoped_key)
        )

        # if the taskqueue already exists, this will rollback transaction
        # automatically
        with self.transaction(ignore_exceptions=True) as tx:
            tx.create(subgraph)

        return scoped_key

    def query_taskqueues(
        self, scope: Optional[Scope] = Scope(), return_gufe: bool = False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, TaskQueue]]:
        """Query for `TaskQueue`s matching the given criteria.

        Parameters
        ----------
        return_gufe
            If True, return a dict with `ScopedKey`s as keys, `TaskQueue`
            instances as values. Otherwise, return a list of `ScopedKey`s.

        """
        return self._query(qualname="TaskQueue", scope=scope, return_gufe=return_gufe)

    def get_taskqueue(
        self, network: ScopedKey, return_gufe: bool = False
    ) -> Union[ScopedKey, TaskQueue]:
        """Get the TaskQueue for the given AlchemicalNetwork.

        Parameters
        ----------
        return_gufe
            If True, return a `TaskQueue` instance.
            Otherwise, return a `ScopedKey`.

        """
        node = self.graph.run(
            f"""
                match (n:TaskQueue {{network: "{network}"}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return n
                """
        ).to_subgraph()

        if return_gufe:
            return self._subgraph_to_gufe([node], node)[node]
        else:
            return ScopedKey.from_str(node["_scoped_key"])

    def delete_taskqueue(
        self,
        network: ScopedKey,
    ) -> ScopedKey:
        """Create a TaskQueue for the given AlchemicalNetwork.

        An AlchemicalNetwork can have only one associated TaskQueue.
        A TaskQueue is required to queue Tasks for a given AlchemicalNetwork.

        This method will only creat a TaskQueue for an AlchemicalNetwork if it
        doesn't already exist; it will return the scoped key for the TaskQueue
        either way.

        """
        taskqueue = self.get_taskqueue(network)

        q = f"""
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue}'}}),
              (tq)-[:TASKQUEUE_HEAD]->(tqh)<-[tqf:FOLLOWS* {{taskqueue: '{taskqueue}'}}]-(task),
              (tq)-[:TASKQUEUE_TAIL]->(tqt)
        FOREACH (i in tqf | delete i)
        DETACH DELETE tq,tqh,tqt
        """
        self.graph.run(q)

        return taskqueue

    def set_taskqueue_weight(self, network: ScopedKey, weight: float):
        q = f"""
        MATCH (t:TaskQueue {{network: "{network}"}})
        SET t.weight = {weight}
        RETURN t
        """
        with self.transaction() as tx:
            tx.run(q)

    def queue_taskqueue_tasks(
        self,
        tasks: List[ScopedKey],
        taskqueue: ScopedKey,
    ) -> List[ScopedKey]:
        """Add Tasks to the TaskQueue for a given AlchemicalNetwork.

        Note: the Tasks must be within the same scope as the AlchemicalNetwork,
        and must correspond to a Transformation in the AlchemicalNetwork.

        A given compute task can be represented in any number of
        AlchemicalNetwork queues, or none at all.

        If this Task has an EXTENDS relationship to another Task, that Task must
        be 'complete' before this Task can be added to *any* TaskQueue.

        """
        for t in tasks:
            q = f"""
            // get our task queue, as well as tail and tail relationship to last in line
            MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue}'}})-[:TASKQUEUE_TAIL]->
                      (tqt)-[tqtl:FOLLOWS {{taskqueue: '{taskqueue}'}}]->
                      (last),
                  (tq)-[:PERFORMS]->(an:AlchemicalNetwork)
            
            // get the task we want to add to the queue; check that it connects to same network
            // if it has an EXTENDS relationship, get the task it extends
            MATCH (tn:Task {{_scoped_key: '{t}'}})-[:PERFORMS*]->(tf:Transformation)<-[:DEPENDS_ON]-(an)
            OPTIONAL MATCH (tn)-[:EXTENDS]->(other_task:Task)

            // only proceed for cases where task is not already in queue and only EXTENDS a 'complete' task
            WHERE NOT (tqt)-[:FOLLOWS* {{taskqueue: '{taskqueue}'}}]->(tn)
              AND other_task.status = 'complete'

            // create the connections that add it to the end of the queue, and delete old queue tail connection
            CREATE (tqt)-[:FOLLOWS {{taskqueue: '{taskqueue}'}}] ->
                      (tn)-[:FOLLOWS {{taskqueue: '{taskqueue}'}}]->(last)
            DELETE tqtl

            RETURN tn
            """
            with self.transaction() as tx:
                task = tx.run(q).to_subgraph()

            if task is None:
                raise ValueError(
                    f"Task '{t}' not found in same network as given TaskQueue"
                )

        return tasks

    def dequeue_taskqueue_tasks(
        self,
        tasks: List[ScopedKey],
        taskqueue: ScopedKey,
    ) -> List[ScopedKey]:
        """Remove a compute Task from the TaskQueue for a given AlchemicalNetwork.

        Note: the Task must be within the same scope as the AlchemicalNetwork.

        A given compute task can be represented in many AlchemicalNetwork
        queues, or none at all.

        """
        for t in tasks:
            q = f"""
            // get our task queue, as well as the task we want to remove, and
            // the nodes ahead and behind it 
            MATCH (task:Task {{_scoped_key: '{t}'}}),
                  (behind)-[behindf:FOLLOWS {{taskqueue: '{taskqueue}'}}]->(task),
                  (task)-[aheadf:FOLLOWS {{taskqueue: '{taskqueue}'}}]->(ahead)
            WITH behind, behindf, task, aheadf, ahead

            // create connection between behind and ahead nodes
            CREATE (behind)-[newf:FOLLOWS {{taskqueue: '{taskqueue}'}}]->(ahead)

            // delete connections between node to remove and behind, ahead nodes
            DELETE behindf, aheadf

            """
            with self.transaction() as tx:
                tx.run(q)

        return tasks

    def get_taskqueue_tasks(
        self, taskqueue: ScopedKey, return_gufe=False
    ) -> Union[List[ScopedKey], Dict[ScopedKey, Task]]:
        """Get a list of Tasks in the TaskQueue, in queued order."""
        q = f"""
        // get list of all 'waiting' tasks in the queue
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue}'}})-->
              (head:TaskQueueHead)<-[:FOLLOWS* {{taskqueue: '{taskqueue}'}}]-(task:Task)
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

    def claim_taskqueue_tasks(
        self, taskqueue: ScopedKey, claimant: str, count: int = 1
    ) -> List[ScopedKey]:
        """Claim a TaskQueue Task.

        This method will claim Tasks from a TaskQueue according to the following scheme:
        1. the first Task in the queue if its priority is equal to that of the highest priority Task.
        2. otherwise, the highest priority Task.

        If no Task is available, then `None` is given in its place.

        Parameters
        ----------
        count
            Claim the given number of Tasks in a single transaction.

        """
        q = f"""
        // get list of all 'waiting' tasks in the queue
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue}'}})-->
              (head:TaskQueueHead)<-[:FOLLOWS* {{taskqueue: '{taskqueue}'}}]-(task1:Task)
        WHERE task1.status = 'waiting'

        // get list of all 'waiting' tasks in the queue, but we'll order by priority
        MATCH (tq)-->(head)<-[:FOLLOWS* {{taskqueue: '{taskqueue}'}}]-(task2:Task)
        WHERE task2.status = 'waiting'

        // build our task lists, order second list by priority
        WITH COLLECT(task1) as tsk1, task2 ORDER BY task2.priority
        WITH tsk1, COLLECT(task2) as tsk2

        // compare the first member of each list
        // select first in line if priority same as highest priority
        // otherwise select highest priority
        WITH CASE
         WHEN tsk1[0].priority = tsk2[0].priority THEN tsk1[0]
         ELSE tsk2[0]
        END AS chosen

        // finally, make the claim
        SET chosen.status = 'running', chosen.claim = '{claimant}'

        RETURN chosen
        """
        tasks = []
        with self.transaction() as tx:
            for i in range(count):
                tasks.append(tx.run(q).to_subgraph())

        return [
            ScopedKey.from_str(t["_scoped_key"]) if t is not None else None
            for t in tasks
        ]

    ## tasks

    def create_task(
        self,
        transformation: ScopedKey,
        extend_from: Optional[ScopedKey] = None,
    ) -> ScopedKey:
        """Add a compute Task to a Transformation.

        Note: this creates a compute Task, but does not add it to any TaskQueues.

        Parameters
        ----------
        transformation
            The Transformation to compute.
        scope
            The scope the Transformation is in; ignored if `transformation` is a ScopedKey.
        extend_from
            The ScopedKey of the Task to use as a starting point for this Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extend_from` input for the Task's eventual call to `Protocol.create`.

        """
        scope = transformation.scope
        transformation_node = self._get_node(transformation)

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        task = Task()
        _, task_node, scoped_key = self._gufe_to_subgraph(
            task.to_shallow_dict(),
            labels=["GufeTokenizable", task.__class__.__name__],
            gufe_key=task.key,
            scope=scope,
        )

        subgraph = Subgraph()

        if extend_from is not None:
            previous_task_node = self._get_node(extend_from)
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
        extend_from: Optional[Task] = None,
        count: int = 1,
    ) -> ScopedKey:
        """Set a fixed number of Tasks against the given Transformation if not
        already present.

        Note: Tasks created by this method are not added to any TaskQueues.

        Parameters
        ----------
        transformation
            The Transformation to compute.
        scope
            The scope the Transformation is in; ignored if `transformation` is a ScopedKey.
        extend_from
            The Task to use as a starting point for this Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extend_from` input for the Task's eventual call to `Protocol.create`.
        count
            The total number of tasks that should exist corresponding to the
            specified `transformation`, `scope`, and `extend_from`.
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

    def query_tasks(
        self,
        scope: Optional[Scope] = None,
        network: Optional[ScopedKey] = None,
        transformation: Optional[ScopedKey] = None,
        extend_from: Optional[ScopedKey] = None,
        status: Optional[List[TaskStatusEnum]] = None,
    ):
        raise NotImplementedError

    def delete_task(
        self,
        task: ScopedKey,
    ) -> Task:
        """Remove a compute Task from a Transformation.

        This will also remove the Task from all TaskQueues it is a part of.

        This method is intended for administrator use; generally Tasks should
        instead have their tasks set to 'deleted' and retained.

        """
        ...

    def get_task_transformation(
        self,
        task: ScopedKey,
    ) -> Tuple[Transformation, Optional[ProtocolDAGResult]]:
        """Get the `Transformation` and `ProtocolDAGResult` to extend from (if
        present) for the given `Task`.

        """
        q = f"""
        MATCH (task:Task {{_scoped_key: "{task}"}})-[:PERFORMS]->(trans:Transformation)
        OPTIONAL MATCH (task)-[:EXTENDS]->(prev:Task)-[:RESULTS_IN]->(result:ProtocolDAGResult)
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

        transformation = self.get_gufe(
            ScopedKey.from_str(transformations[0]["_scoped_key"])
        )
        protocol_dag_result = (
            self.get_gufe(ScopedKey.from_str(results[0]["_scoped_key"]))
            if results[0] is not None
            else None
        )

        return transformation, protocol_dag_result

    def set_task_result(
        self, task: ScopedKey, protocol_dag_result: ProtocolDAGResult
    ) -> ScopedKey:

        scope = task.scope
        task_node = self._get_node(task)

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        subgraph, protocoldagresult_node, scoped_key = self._gufe_to_subgraph(
            protocol_dag_result.to_shallow_dict(),
            labels=["GufeTokenizable", protocol_dag_result.__class__.__name__],
            gufe_key=protocol_dag_result.key,
            scope=scope,
        )

        subgraph = subgraph | Relationship.type("RESULTS_IN")(
            task_node,
            protocoldagresult_node,
            _org=scope.org,
            _campaign=scope.campaign,
            _project=scope.project,
        )

        with self.transaction() as tx:
            tx.create(subgraph)

        return scoped_key

    def set_task_waiting(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    def set_task_running(self, task: Union[Task, ScopedKey], computekey: ComputeKey):
        ...

    def set_task_complete(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    def set_task_error(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    def set_task_cancelled(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    def set_task_invalid(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    def set_task_deleted(
        self,
        task: Union[Task, ScopedKey],
    ):
        ...

    ## authentication

    def create_credentialed_entity(self, entity: CredentialedEntity):
        """Create a new credentialed entity, such as a user or compute service.

        If an entity of this type with the same `identifier` already exists,
        then this will overwrite its properties, including credential.

        """
        node = Node("CredentialedEntity", entity.__class__.__name__, **entity.dict())

        with self.transaction() as tx:
            tx.merge(
                node, primary_label=entity.__class__.__name__, primary_key="identifier"
            )

    def get_credentialed_entity(self, identifier: str, cls: type[CredentialedEntity]):
        """Create a new credentialed entity, such as a user or compute service.

        If an entity of this type with the same `identifier` already exists,
        then this will overwrite its properties, including credential.

        """
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
