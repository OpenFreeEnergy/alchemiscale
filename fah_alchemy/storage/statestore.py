import abc
from contextlib import contextmanager
import json
from time import sleep
from typing import Dict, List, Optional, Union, Tuple
import weakref

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ProtocolDAGResult
from gufe.tokenization import GufeTokenizable, GufeKey
from gufe.storage.metadatastore import MetadataStore
from py2neo import Graph, Node, Relationship, Subgraph
from py2neo.matching import NodeMatcher
from py2neo.errors import ClientError

from .models import ComputeKey, Task, TaskQueue, TaskArchive, TaskStatusEnum
from ..strategies import Strategy
from ..models import Scope, ScopedKey


class Neo4JStoreError(Exception):
    ...


class FahAlchemyStateStore(abc.ABC):
    ...


class Neo4jStore(FahAlchemyStateStore):

    def __init__(self, graph: "py2neo.Graph"):
        self.graph = graph
        self.gufe_nodes = weakref.WeakValueDictionary()

    @contextmanager
    def transaction(self, readonly=False, ignore_exceptions=False):
        """Context manager for a py2neo Transaction.

        """
        tx = self.graph.begin(readonly=readonly)
        try:
            yield tx
        except:
            self.graph.rollback(tx)
            if not ignore_exceptions:
                raise
        else:
            self.graph.commit(tx)

    ## gufe object handling

    def _gufe_to_subgraph(
            self, 
            sdct: Dict,
            labels: List[str],
            gufe_key: GufeKey,
            scope: Scope
    ) -> Tuple[Subgraph, Node, str]:
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

    def _get_node(
        self,
        scoped_key: Union[ScopedKey, str]
    ):
        qualname = scoped_key.qualname

        properties = {"_scoped_key": str(scoped_key)}
        prop_string = ", ".join(
            "{}: '{}'".format(key, value) for key, value in properties.items()
        )

        prop_string = f" {{{prop_string}}}"

        q = f"""
        MATCH (n:{qualname}{prop_string})
        OPTIONAL MATCH p = (n)-[r:DEPENDS_ON*]->(m) 
        WHERE NOT (m)-[:DEPENDS_ON]->()
        RETURN n,p
        """
        nodes = set()
        subgraph = Subgraph()

        for record in self.graph.run(q):
            nodes.add(record["n"])
            if record['p'] is not None:
                subgraph = subgraph | record["p"]
            else:
                subgraph = record['n']

        if len(nodes) == 0:
            raise ValueError("No such object in database")
        elif len(nodes) > 1:
            raise Neo4JStoreError("More than one such object in database; this should not be possible")

        return nodes, subgraph

    def _query(
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
        MATCH (n:{qualname}{prop_string})
        RETURN n
        """
        with self.transaction() as tx:
            res = tx.run(q)

        nodes = set()
        for record in res:
            nodes.add(record['n'])

        return [ScopedKey.from_str(i['_scoped_key']) for i in nodes]

    def check_existence(self, qualname, scoped_key: Union[ScopedKey, str]):
        nodes, subgraph = self._get_node(qualname=qualname, scoped_key=str(scoped_key))

        return len(nodes) > 0

    def get_scoped_key(
            self,
            obj: GufeTokenizable,
            scope: Scope
    ):
        qualname = obj.__class__.__qualname__
        res = self._query(
            qualname=qualname,
            key=obj.key,
            scope=scope
        )

        if len(res) == 0:
            raise ValueError("No such object in database")
        elif len(res) > 1:
            raise Neo4JStoreError("More than one such object in database; this should not be possible")

        return res[0]
    
    def get_gufe(
        self,
        scoped_key: ScopedKey
    ):
        nodes, subgraph = self._get_node(scoped_key=scoped_key)
        gufe_objs = self._subgraph_to_gufe(nodes, subgraph)
        return gufe_objs[0]

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
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
        with self.transaction() as tx:
            tx.delete_taskqueue(network, scope)

        network_node = self._get_node_from_obj_or_sk(network, AlchemicalNetwork, scope)

        # then delete the network
        q = f"""
        MATCH (an:AlchemicalNetwork {{_scoped_key: "{network_node['_scoped_key']}"}})
        DETACH DELETE an
        """
        return ScopedKey.from_str(network_node['_scoped_key'])

    def query_networks(
        self, *, name=None, key=None, scope: Optional[Scope] = Scope() 
    ):
        """Query for `AlchemicalNetwork`s matching given attributes."""
        additional = {"name": name}
        return self._query(
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
        return self._query(
            qualname="Transformation",
            additional=additional,
            key=key,
            scope=scope
        )

    def query_chemicalsystems(
        self, *, name=None, key=None, scope: Scope = Scope(),
        transformations=None
    ):
        """Query for `ChemicalSystem`s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="ChemicalSystem",
            additional=additional,
            key=key,
            scope=scope
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
        """Set the compute Strategy for the given AlchemicalNetwork.

        """
        ...

    ## task queues

    def create_taskqueue(
            self,
            network: ScopedKey,
            scope: Scope,
        ) -> ScopedKey:
        """Create a TaskQueue for the given AlchemicalNetwork.

        An AlchemicalNetwork can have only one associated TaskQueue.
        A TaskQueue is required to queue Tasks for a given AlchemicalNetwork.

        This method will only creat a TaskQueue for an AlchemicalNetwork if it
        doesn't already exist; it will return the scoped key for the TaskQueue
        either way.

        """
        network_node = self._get_node_from_obj_or_sk(network, AlchemicalNetwork, scope)

        # create a taskqueue for the supplied network
        # use a PERFORMS relationship
        taskqueue = TaskQueue(network=network_node['_scoped_key'])
        _, taskqueue_node, scoped_key = self._gufe_to_subgraph(
                taskqueue.to_shallow_dict(),
                labels=["GufeTokenizable", taskqueue.__class__.__name__],
                gufe_key=taskqueue.key,
                scope=scope
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

        subgraph = subgraph | Relationship.type("TASKQUEUE_HEAD")(
                taskqueue_node,
                head)
        subgraph = subgraph | Relationship.type("TASKQUEUE_TAIL")(
                taskqueue_node,
                tail)
        subgraph = subgraph | Relationship.type("FOLLOWS")(
                tail,
                head,
                taskqueue=str(scoped_key))

        # if the taskqueue already exists, this will rollback transaction
        # automatically
        with self.transaction(ignore_exceptions=True) as tx:
            tx.create(subgraph)

        return scoped_key

    #def get_taskqueue(
    #        self,
    #        network: Union[AlchemicalNetwork, ScopedKey],
    #        scope: Scope,
    #        scoped_key: ScopedKey,
    #    ) -> ScopedKey:
    #    """Get a TaskQueue from its AlchemicalNetwork or directly with its `scoped_key`.

    #    """
    #    if scoped_key:

    #    network_node = self._get_node_from_obj_or_sk(network, AlchemicalNetwork, scope)

    #    # if the taskqueue already exists, this will rollback transaction
    #    # automatically
    #    with self.transaction(ignore_exceptions=True) as tx:
    #        tx.run(q)

    #    return scoped_key

    def query_taskqueues(
            self,
            scope: Optional[Scope] = Scope() 
        ) -> List[TaskQueue]:
        """Query for `TaskQueue`s matching the given criteria.

        """
        return self._query(
            qualname="TaskQueue",
            scope=scope
        )

    def _get_taskqueue(
            self,
            network: ScopedKey,
            scope: Scope,
        ):
        """Get the TaskQueue for the given AlchemicalNetwork.

        """
        network_node = self._get_node_from_obj_or_sk(network, AlchemicalNetwork, scope)
        node = self.graph.run(
                f"""
                match (n:TaskQueue {{network: "{network_node['_scoped_key']}", 
                                             _org: '{scope.org}', _campaign: '{scope.campaign}', 
                                             _project: '{scope.project}'}})-[:PERFORMS]->(m:AlchemicalNetwork)
                return n
                """).to_subgraph()

        return node

    def delete_taskqueue(
            self,
            network: ScopedKey,
            scope: Scope,
        ) -> ScopedKey:
        """Create a TaskQueue for the given AlchemicalNetwork.

        An AlchemicalNetwork can have only one associated TaskQueue.
        A TaskQueue is required to queue Tasks for a given AlchemicalNetwork.

        This method will only creat a TaskQueue for an AlchemicalNetwork if it
        doesn't already exist; it will return the scoped key for the TaskQueue
        either way.

        """
        taskqueue_node = self._get_taskqueue(network, scope)

        q = f"""
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue_node['_scoped_key']}'}}),
              (tq)-[:TASKQUEUE_HEAD]->(tqh)<-[tqf:FOLLOWS* {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]-(task),
              (tq)-[:TASKQUEUE_TAIL]->(tqt)
        FOREACH (i in tqf | delete i)
        DETACH DELETE tq,tqh,tqt
        """
        self.graph.run(q)

        return ScopedKey.from_str(taskqueue_node['_scoped_key'])

    def set_taskqueue_weight(
            self,
            network: ScopedKey,
            scope: Scope,
            weight: float
        ):
        network_node = self._get_node_from_obj_or_sk(network, AlchemicalNetwork, scope)

        ## this should be performed in a single cypher query, in place
        q = f"""
        MATCH (t:TaskQueue {{network: "{network_node['_scoped_key']}"}})
        SET t.weight = {weight}
        RETURN t
        """
        with self.transaction() as tx:
            tx.run(q)

    def queue_tasks(
            self,
            tasks: List[ScopedKey],
            taskqueue: ScopedKey,
        ) -> ScopedKey:
        """Add a compute Task to the TaskQueue for a given AlchemicalNetwork.

        Note: the Task must be within the same scope as the AlchemicalNetwork,
        and must correspond to a Transformation in the AlchemicalNetwork.

        A given compute task can be represented in any number of
        AlchemicalNetwork queues, or none at all.

        If this Task has an EXTENDS relationship to another Task, that Task must
        be 'complete' before this Task can be added to *any* TaskQueue.

        """
        # TODO: add in EXTENDS relationship handling documented above
        # TODO: add check that Task corresponds to a Transformation on the given network
        taskqueue_node = self._get_node_from_obj_or_sk(taskqueue, TaskQueue, scope)
        
        scoped_keys = []
        for t in tasks:
            task_node = self._get_node_from_obj_or_sk(t, Task, None, independent=True)

            scope = Scope(org=task_node['_org'], 
                          campaign=task_node['_campaign'],
                          project=task_node['_project'])


            # TODO: add check that Task is connected to the given network via a
            # fixed number of DEPENDS relationships

            q = f"""
            MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue_node['_scoped_key']}'}})-
                [:TASKQUEUE_TAIL]
                ->(tqt)-[tqtl:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(last)
            WITH tqt, last, tqtl
            MATCH (tn:Task {{_scoped_key: '{task_node['_scoped_key']}'}})
            WHERE NOT (tqt)-[:FOLLOWS* {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(tn)
            CREATE (tqt)-[:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]
                ->(tn)-[:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(last)
            DELETE tqtl
            """
            with self.transaction() as tx:
                tx.run(q)

            scoped_keys.append(ScopedKey.from_str(task_node['_scoped_key']))

        return scoped_keys

    def dequeue_tasks(
            self,
            tasks: List[ScopedKey],
            taskqueue: ScopedKey,
        ) -> ScopedKey:
        """Remove a compute Task from the TaskQueue for a given AlchemicalNetwork.

        Note: the Task must be within the same scope as the AlchemicalNetwork.

        A given compute task can be represented in many AlchemicalNetwork
        queues, or none at all.

        """
        task_node = self._get_node_from_obj_or_sk(task, Task, None, independent=True)

        scope = Scope(org=task_node['_org'], 
                      campaign=task_node['_campaign'],
                      project=task_node['_project'])

        taskqueue_node = self._get_taskqueue(network, scope)

        q = f"""
        MATCH (task:Task {{_scoped_key: '{task_node['_scoped_key']}'}}),
              (behind)-[behindf:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(task),
              (task)-[aheadf:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(ahead)
        WITH behind, behindf, task, aheadf, ahead
        CREATE (behind)-[newf:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(ahead)
        DELETE behindf, aheadf
        """
        with self.transaction() as tx:
            tx.run(q)

        return ScopedKey.from_str(task_node['_scoped_key'])

    def get_taskqueue_tasks(
            self,
            taskqueue: Union[TaskQueue, ScopedKey],
        ):
        """Get a list of Tasks in the TaskQueue, in queued order.

        """
        ...

    def claim_task(
            self,
            taskqueue: ScopedKey,
            claimant: str,
            count: int = 1
            ):
        """Claim one or more TaskQueue Tasks.

        """
        taskqueue_node = self._get_taskqueue(network, scope)

        # need to check if path order is guaranteed with approach below

        q = f"""
        // get all tasks in queue
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue_node['_scoped_key']}'}})-[:TASKQUEUE_HEAD]->(head:TaskQueueHead),
              (head:TaskQueueHead)-[:FOLLOWS* {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(task2:Task)
        WHERE task2.status = 'waiting'
        WITH DISTINCT task2 as tsk2 ORDER BY tsk2.priority
            WITH COLLLECT(tsk2) as t2
        CASE
         WHEN task1.priority = t2[0].priority THEN task1
         ELSE t2[0]
        END AS chosen
        SET chosen.status = 'running', chosen.claim = '{claimant}'
        RETURN chosen
        """

        q = f"""
        // get first task in line
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue_node['_scoped_key']}'}}),
              (head:TaskQueueHead)-[:FOLLOWS {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(task1:Task)
        WHERE task1.status = 'waiting'
        // get all tasks
        MATCH (tq:TaskQueue {{_scoped_key: '{taskqueue_node['_scoped_key']}'}}),
              (head:TaskQueueHead)-[:FOLLOWS* {{taskqueue: '{taskqueue_node['_scoped_key']}'}}]->(task2:Task)
        WHERE task2.status = 'waiting'
        WITH DISTINCT task2 as tsk2 ORDER BY tsk2.priority
            WITH COLLLECT(tsk2) as t2
        CASE
         WHEN task1.priority = t2[0].priority THEN task1
         ELSE t2[0]
        END AS chosen
        SET chosen.status = 'running', chosen.claim = '{claimant}'
        RETURN chosen
        """
        with self.transaction() as tx:
            res = tx.run(q)

        node = res.to_subgraph()

        task = self._subgraph_to_gufe([node], node)

        return task

    ## tasks

    def create_task(
            self, 
            transformation: ScopedKey,
            extend_from: Optional[Task] = None,
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
            The Task to use as a starting point for this Task.
            Will use the `ProtocolDAGResult` from the given Task as the
            `extend_from` input for the Task's eventual call to `Protocol.create`.

        """
        transformation_node = self._get_node_from_obj_or_sk(transformation, Transformation, scope)

        # TODO: if we want to inject Protocol information into our Task, we
        # might want to instantiate the `gufe` object here; that should be safe

        # create a new task for the supplied transformation
        # use a PERFORMS relationship
        task = Task()
        _, task_node, scoped_key = self._gufe_to_subgraph(
                task.to_shallow_dict(),
                labels=["GufeTokenizable", task.__class__.__name__],
                gufe_key=task.key,
                scope=scope
                )

        if extend_from:
            # check for existence of `ProtocolDAGResult` and set EXTENDS relationship
            ...

        subgraph = Relationship.type("PERFORMS")(
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
        # TODO: finish this one out when we have a reasonable approach to locking
        # too hard to perform in a single Cypher query; unclear how to create many nodes in a loop
        transformation_node = self._get_node_from_obj_or_sk(transformation, Transformation, scope)

        ## this should be performed in a single cypher query, in place
        ## this is really close to working, but still no dice if there are no existing tasks
        ## might have to cut losses and just call `create_task` above first
        q = f"""
        MATCH (n:Transformtion 
                {{_scoped_key: "{transformation_node['_scoped_key']}"}})
        OPTIONAL MATCH (n)<-[:PERFORMS]-(t:Task)
        WITH count(t) as cnt
        CASE
         WHEN cnt IS NULL THEN 0
         ELSE cnt
        END AS cnt_nonnull
        FOREACH (i in range(0, {count} - cnt_nonnull) | 
          MERGE (n)<-[:PERFORMS]-(t:Task)
          )
        """
        self.graph.run(q)

    def get_task(
            self, 
            task: ScopedKey,
            transformation: ScopedKey,
            extend_from: Optional[ScopedKey] = None,
            status: List[TaskStatusEnum] = None
        ) -> ScopedKey:
        ...

    def query_tasks(
            self,
            scope: Optional[Scope] = None,
            network: Optional[ScopedKey] = None,
            transformation: Optional[ScopedKey] = None,
            extend_from: Optional[ScopedKey] = None,
            status: Optional[List[TaskStatusEnum]] = None
        ):
        ...

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
        ):
        """Get the `Transformation` and `ProtocolDAGResult` to extend from (if
        present) for the given `Task`.

        """
        ...

    def set_task_waiting(
            self,
            task: Union[Task, ScopedKey],
        ):
        ...

    def set_task_running(
            self,
            task: Union[Task, ScopedKey],
            computekey: ComputeKey
        ):
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

    ## admin

    def generate_client_api_key(self):
        ...

    def generate_compute_api_key(self):
        ...
