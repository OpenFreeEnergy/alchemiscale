"""
Node4js state storage --- :mod:`alchemiscale.storage.statestore`
================================================================

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
    ComputeServiceID,
    ComputeServiceRegistration,
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


def _generate_claim_query(
    task_sk: ScopedKey, compute_service_id: ComputeServiceID
) -> str:
    """
    Generate a query to claim a single Task.
    Parameters
    ----------
    task_sk
        The ScopedKey of the Task to claim.
    compute_service_id
        ComputeServiceID of the claiming service.

    Returns
    -------
    query: str
        The Cypher query to claim the Task.
    """
    query = f"""
    // only match the task if it doesn't have an existing CLAIMS relationship
    MATCH (t:Task {{_scoped_key: '{task_sk}'}})
    WHERE NOT (t)<-[:CLAIMS]-(:ComputeServiceRegistration)
    SET t.status = 'running'

    WITH t

    // create CLAIMS relationship with given compute service
    MATCH (csreg:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})
    CREATE (t)<-[cl:CLAIMS {{claimed: localdatetime('{datetime.utcnow().isoformat()}')}}]-(csreg)

    RETURN t
    """
    return query


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
            elif isinstance(value, Settings):
                node[key] = json.dumps(value, cls=JSON_HANDLER.encoder, sort_keys=True)
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
        self, node: Node, g: nx.DiGraph, mapping: Dict[Node, GufeTokenizable]
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

        This will not remove any `Transformation`\s or `ChemicalSystem`\s
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
        """Query for `AlchemicalNetwork`\s matching given attributes."""
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
        """Query for `Transformation`\s matching given attributes."""
        additional = {"name": name}
        return self._query(
            qualname="Transformation", additional=additional, key=key, scope=scope
        )

    def query_chemicalsystems(
        self, *, name=None, key=None, scope: Scope = Scope(), transformations=None
    ):
        """Query for `ChemicalSystem`\s matching given attributes."""
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

        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )
        raise NotImplementedError

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
            tx.create(node)

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
        MATCH (n:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})

        OPTIONAL MATCH (n)-[cl:CLAIMS]->(t:Task {{status: 'running'}})
        SET t.status = 'waiting'

        WITH n, n.identifier as identifier

        DETACH DELETE n

        RETURN identifier
        """

        with self.transaction() as tx:
            res = tx.run(q)
            identifier = next(res)["identifier"]

        return ComputeServiceID(identifier)

    def heartbeat_computeservice(
        self, compute_service_id: ComputeServiceID, heartbeat: datetime
    ):
        """Update the heartbeat for the given ComputeServiceID."""

        q = f"""
        MATCH (n:ComputeServiceRegistration {{identifier: '{compute_service_id}'}})
        SET n.heartbeat = localdatetime('{heartbeat.isoformat()}')

        """

        with self.transaction() as tx:
            tx.run(q)

        return compute_service_id

    def expire_registrations(self, expire_time: datetime):
        """Remove all registrations with last heartbeat prior to the given `expire_time`."""
        q = f"""
        MATCH (n:ComputeServiceRegistration)
        WHERE n.heartbeat < localdatetime('{expire_time.isoformat()}')

        WITH n

        OPTIONAL MATCH (n)-[cl:CLAIMS]->(t:Task {{status: 'running'}})
        SET t.status = 'waiting'

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
        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

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
        """Query for `TaskHub`\s matching the given criteria.

        Parameters
        ----------
        return_gufe
            If True, return a dict with `ScopedKey`s as keys, `TaskHub`
            instances as values. Otherwise, return a list of `ScopedKey`\s.

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
        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

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

        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

        taskhub = self.get_taskhub(network)

        q = f"""
        MATCH (th:TaskHub {{_scoped_key: '{taskhub}'}}),
        DETACH DELETE th
        """
        self.graph.run(q)

        return taskhub

    def set_taskhub_weight(self, network: ScopedKey, weight: float):
        """Set the weight for the TaskHub associated with the given
        AlchemicalNetwork.

        """

        if network.qualname != "AlchemicalNetwork":
            raise ValueError(
                "`network` ScopedKey does not correspond to an `AlchemicalNetwork`"
            )

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
        WHERE NOT (task)<-[:CLAIMS]-(:ComputeServiceRegistration)
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
        self, taskhub: ScopedKey, compute_service_id: ComputeServiceID, count: int = 1
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
        compute_service_id
            Unique identifier for the compute service claiming the Tasks for execution.
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
                    claim_query = _generate_claim_query(chosen_one, compute_service_id)
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
        if transformation.qualname not in ["Transformation", "NonTransformation"]:
            raise ValueError(
                "`transformation` ScopedKey does not correspond to a `Transformation`"
            )

        if extends is not None and extends.qualname != "Task":
            raise ValueError("`extends` ScopedKey does not correspond to a `Task`")

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
            stat = previous_task_node.get("status")
            # do not allow creation of a task that extends an invalid or deleted task.
            if (stat == "invalid") or (stat == "deleted"):
                raise ValueError(
                    f"Cannot extend a `deleted` or `invalid` Task: {previous_task_node}"
                )
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
    ) -> Union[
        Tuple[Transformation, Optional[ProtocolDAGResultRef]],
        Tuple[ScopedKey, Optional[ScopedKey]],
    ]:
        """Get the `Transformation` and `ProtocolDAGResultRef` to extend from (if
        present) for the given `Task`.

        If `return_gufe` is `True`, returns actual `Transformation` and
        `ProtocolDAGResultRef` object (`None` if not present); if `False`, returns
        `ScopedKey`\s for these instead.

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

        if task.qualname != "Task":
            raise ValueError("`task` ScopedKey does not correspond to a `Task`")

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

    def set_task_status(
        self, tasks: List[ScopedKey], status: TaskStatusEnum, raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
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

    def get_task_status(self, tasks: List[ScopedKey]) -> List[TaskStatusEnum]:
        """Get the status of a list of Tasks.

        Parameters
        ----------
        tasks
            The list of Tasks to get the status for.

        Returns
        -------
        Dict[ScopedKey,TaskStatusEnum]
            A dictionary of Tasks and their statuses.
        """

        statuses = []
        with self.transaction() as tx:
            for t in tasks:
                q = f"""
                MATCH (t:Task {{_scoped_key: '{t}'}})
                RETURN t
                """
                task = tx.run(q).to_subgraph()
                if task is None:
                    statuses.append(None)
                else:
                    status = task.get("status")
                    statuses.append(TaskStatusEnum(status))

        return statuses

    def _set_task_status(self, tasks, q_func, err_msg_func, raise_error):
        tasks_statused = []
        with self.transaction() as tx:
            for t in tasks:
                res = tx.run(q_func(t))
                # we only need the first record to get the info we need
                for record in res:
                    task_i = record["t"]
                    task_set = record["t_"]
                    break

                if task_set is None:
                    if raise_error:
                        status = task_i["status"]
                        raise ValueError(err_msg_func(t, status))
                    tasks_statused.append(None)
                elif task_i is None:
                    if raise_error:
                        raise ValueError("No such task {t}")
                    tasks_statused.append(None)
                else:
                    tasks_statused.append(t)

        return tasks_statused

    def set_task_waiting(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `waiting`.

        Only Tasks with status `error` or `running` can be set to `waiting`.

        """

        def q(t):
            return f"""
            MATCH (t:Task {{_scoped_key: '{t}'}})

            OPTIONAL MATCH (t_:Task {{_scoped_key: '{t}'}})
            WHERE t_.status IN ['waiting', 'running', 'error']
            SET t_.status = '{TaskStatusEnum.waiting.value}'

            WITH t, t_

            // if we changed the status to waiting,
            // drop CLAIMS relationship
            OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
            DELETE cl

            RETURN t, t_
            """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `waiting` as it is not currently `error` or `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_running(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `running`.

        Only Tasks with status `waiting` can be set to `running`.

        """

        def q(t):
            return f"""
            MATCH (t:Task {{_scoped_key: '{t}'}})

            OPTIONAL MATCH (t_:Task {{_scoped_key: '{t}'}})
            WHERE t_.status IN ['running', 'waiting']
            SET t_.status = '{TaskStatusEnum.running.value}'

            RETURN t, t_
            """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `running` as it is not currently `waiting`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_complete(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `complete`.

        Only `running` Tasks can be set to `complete`.

        """

        def q(t):
            return f"""
            MATCH (t:Task {{_scoped_key: '{t}'}})

            OPTIONAL MATCH (t_:Task {{_scoped_key: '{t}'}})
            WHERE t_.status IN ['complete', 'running']
            SET t_.status = '{TaskStatusEnum.complete.value}'

            WITH t, t_

            // if we changed the status to complete,
            // drop all ACTIONS relationships
            OPTIONAL MATCH (t_)<-[ar:ACTIONS]-(th:TaskHub)
            DELETE ar

            WITH t, t_

            // if we changed the status to complete,
            // drop CLAIMS relationship
            OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
            DELETE cl

            RETURN t, t_
            """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `complete` as it is not currently `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_error(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `error`.

        Only `running` Tasks can be set to `error`.

        """

        def q(t):
            return f"""
            MATCH (t:Task {{_scoped_key: '{t}'}})

            OPTIONAL MATCH (t_:Task {{_scoped_key: '{t}'}})
            WHERE t_.status IN ['error', 'running']
            SET t_.status = '{TaskStatusEnum.error.value}'

            WITH t, t_

            // if we changed the status to error,
            // drop CLAIMS relationship
            OPTIONAL MATCH (t_)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
            DELETE cl

            RETURN t, t_
            """

        def err_msg(t, status):
            return f"Cannot set task {t} with current status: {status} to `error` as it is not currently `running`."

        return self._set_task_status(tasks, q, err_msg, raise_error=raise_error)

    def set_task_invalid(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `invalid`.

        Any Task can be set to `invalid`; an `invalid` Task cannot change to
        any other status.

        """

        with self.transaction() as tx:
            for t in tasks:
                # set the status and delete the ACTIONS relationship
                # make sure we follow the extends chain and set all tasks to invalid
                # and remove actions relationships
                q = f"""
                MATCH (t:Task {{_scoped_key: '{t}'}})

                // EXTENDS* used to get all tasks in the extends chain
                OPTIONAL MATCH (t)<-[er:EXTENDS*]-(extends_task:Task)
                SET t.status = '{TaskStatusEnum.invalid.value}'
                SET extends_task.status = '{TaskStatusEnum.invalid.value}'
                WITH t, extends_task

                OPTIONAL MATCH (t)<-[ar:ACTIONS]-(th:TaskHub)
                OPTIONAL MATCH (extends_task)<-[are:ACTIONS]-(th:TaskHub)

                DELETE ar
                DELETE are

                WITH t

                // drop CLAIMS relationship if present
                OPTIONAL MATCH (t)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
                DELETE cl
                """
                tx.run(q)

        return tasks

    def set_task_deleted(
        self, tasks: List[ScopedKey], raise_error: bool = False
    ) -> List[Optional[ScopedKey]]:
        """Set the status of a list of Tasks to `deleted`.

        Any Task can be set to `deleted`; a `deleted` Task cannot change to
        any other status.

        """

        with self.transaction() as tx:
            for t in tasks:
                # set the status and delete the ACTIONS relationship
                # make sure we follow the extends chain and set all tasks to deleted
                # and remove actions relationships
                q = f"""
                MATCH (t:Task {{_scoped_key: '{t}'}})

                // EXTENDS* used to get all tasks in the extends chain
                OPTIONAL MATCH (t)<-[er:EXTENDS*]-(extends_task:Task)
                SET t.status = '{TaskStatusEnum.deleted.value}'
                SET extends_task.status = '{TaskStatusEnum.deleted.value}'
                WITH t, extends_task

                OPTIONAL MATCH (t)<-[ar:ACTIONS]-(th:TaskHub)
                OPTIONAL MATCH (extends_task)<-[are:ACTIONS]-(th:TaskHub)

                DELETE ar
                DELETE are

                WITH t

                // drop CLAIMS relationship if present
                OPTIONAL MATCH (t)<-[cl:CLAIMS]-(csreg:ComputeServiceRegistration)
                DELETE cl
                """
                tx.run(q)

        return tasks

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
