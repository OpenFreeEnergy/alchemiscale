"""Client for interacting with user-facing API.

"""

from typing import Union, List, Dict, Optional
import requests
import json

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ChemicalSystem
from gufe.tokenization import GufeTokenizable, JSON_HANDLER, GufeKey
from gufe.protocols import ProtocolResult, ProtocolDAGResult

from ..base.client import FahAlchemyBaseClient, FahAlchemyBaseClientError
from ..models import Scope, ScopedKey
from ..storage.models import Task
from ..strategies import Strategy


class FahAlchemyClientError(FahAlchemyBaseClientError):
    ...


class FahAlchemyClient(FahAlchemyBaseClient):
    """Client for user interaction with API service."""

    _exception = FahAlchemyClientError

    ### inputs

    def get_scoped_key(self, obj: GufeTokenizable, scope: Scope):
        """Given any gufe object and a fully-specified Scope, return corresponding ScopedKey.

        This method does not check that this ScopedKey is represented in the database.
        It is only a convenience for properly constructing a ScopedKey from a
        gufe object and a Scope.

        """
        if scope.specific():
            return ScopedKey(gufe_key=obj.key, **scope.dict())

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
        """Submit an AlchemicalNetwork."""
        data = dict(network=network.to_dict(), scope=scope.dict())
        scoped_key = self._post_resource("/networks", data)
        return ScopedKey.from_dict(scoped_key)

    def query_networks(
        self,
        name: Optional[str] = None,
        scope: Optional[Scope] = None,
        return_gufe=False,
        limit=None,
        skip=None,
    ) -> Union[List[ScopedKey], Dict[ScopedKey, AlchemicalNetwork]]:
        """Query for AlchemicalNetworks, optionally by name or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all AlchemicalNetworks that are within the Scopes this user has access
        to.

        """
        if return_gufe:
            networks = {}
        else:
            networks = []

        if scope is None:
            scope = Scope()

        params = dict(
            name=name, return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict()
        )
        if return_gufe:
            networks.update(self._query_resource("/networks", params=params))
        else:
            networks.extend(self._query_resource("/networks", params=params))

        return networks

    def get_network(self, network: Union[ScopedKey, str]) -> AlchemicalNetwork:
        return self._get_resource(f"/networks/{network}", {}, return_gufe=True)

    def get_transformation(
        self, transformation: Union[ScopedKey, str]
    ) -> Transformation:
        return self._get_resource(
            f"/transformations/{transformation}", {}, return_gufe=True
        )

    def get_chemicalsystem(
        self, chemicalsystem: Union[ScopedKey, str]
    ) -> ChemicalSystem:
        return self._get_resource(
            f"/chemicalsystems/{chemicalsystem}", {}, return_gufe=True
        )

    ### compute

    def set_strategy(self, network: ScopedKey, strategy: Strategy):
        """Set the Strategy for evaluating the given AlchemicalNetwork.

        The Strategy will be applied to create and action tasks for the
        Transformations in the AlchemicalNetwork without user interaction.

        """
        raise NotImplementedError

    def create_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        count: int = 1,
    ) -> List[ScopedKey]:
        """Create Tasks for the given Transformation.

        Parameters
        ----------
        transformation
            The ScopedKey of the Transformation to create Tasks for.
        extends
            The ScopedKey for the Task to use as a starting point for the Tasks created.
        count
            The number of new Tasks to create.

        Returns
        -------
        List[ScopedKey]
            A list giving the ScopedKeys of the new Tasks created.

        """
        if extends:
            extends = extends.dict()

        data = dict(extends=extends, count=count)
        task_sks = self._post_resource(f"/transformations/{transformation}/tasks", data)
        return [ScopedKey.from_str(i) for i in task_sks]

    def get_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        return_as: str =  'list'
    ) -> Union[List[ScopedKey], nx.DiGraph]:
        """Return the Tasks associated with the given Transformation.

        Parameters
        ----------
        transformation
            The ScopedKey of the Transformation to get Tasks for.
        extends
            If given, only return Tasks that extend from the given Task's ScopedKey.
            This will also give any Tasks that extend from those Tasks, recursively.
            Using this keyword argument amounts to choosing the tree of Tasks that
            extend from the given Task.
        return_as : ['list', 'graph']
            If 'list', Tasks will be returned in no particular order.
            If `graph`, Tasks will be returned in a `networkx.DiGraph`, with a
            directed edge pointing from a given Task to the Task it extends.

        """
        if extends:
            extends = str(extends)

        params = dict(extends=extends, return_as=return_as)
        task_sks = self._get_resource(f"/transformations/{transformation}/tasks", params, return_gufe=False)

        if return_as == 'list':
            return [ScopedKey.from_str(i) for i in task_sks]
        elif return_as == 'graph':
            g = nx.DiGraph()
            for task, extends in task_sks.items():
                g.add_node(ScopedKey.from_str(task))
                if extends is not None:
                    g.add_edge(ScopedKey.from_str(task), ScopedKey.from_str(extends))

            return g


    def action_tasks(
            self, 
            tasks: List[ScopedKey],
            network: ScopedKey
        ) -> List[Optional[ScopedKey]]:
        """Action Tasks for execution via the given AlchemicalNetwork's
        TaskQueue.

        A Task cannot be actioned:
            - to an AlchemicalNetwork in a different Scope.
            - if it extends another Task that is not complete.

        Parameters
        ----------
        tasks
            Task ScopedKeys to action for execution.
        network
            The AlchemicalNetwork ScopedKey to action the Tasks for.
            The Tasks will be added to the network's associated TaskQueue.

        Returns
        -------
        List[Optional[ScopedKey]]
            ScopedKeys for Tasks actioned, in the same order as given as
            `tasks` on input. If a Task couldn't be actioned, then `None` will
            be returned in its place.

        """
        data = dict(tasks=[t.dict() for t in tasks])
        actioned_sks = self._post_resource(f"/networks/{network}/tasks/action", data)

        return [ScopedKey.from_str(i) if i is not None else None for i in actioned_sks]


    def cancel_tasks(self, tasks: List[ScopedKey], network: ScopedKey):
        """Cancel Tasks for execution via the given AlchemicalNetwork's
        TaskQueue.

        A Task cannot be canceled:
            - if it is not present in the AlchemicalNetwork's TaskQueue.

        Parameters
        ----------
        tasks
            Task ScopedKeys to cancel for execution.
        network
            The AlchemicalNetwork ScopedKey to cancel the Tasks for.
            The Tasks will be removed from the network's associated TaskQueue.

        Returns
        -------
        List[Optional[ScopedKey]]
            ScopedKeys for Tasks canceled, in the same order as given as
            `tasks` on input. If a Task couldn't be canceled, then `None` will
            be returned in its place.

        """
        data = dict(tasks=[t.dict() for t in tasks])
        canceled_sks = self._post_resource(f"/networks/{network}/tasks/cancel", data)

        return [ScopedKey.from_str(i) if i is not None else None for i in canceled_sks]

    def get_tasks_priority(
        self,
        tasks: List[ScopedKey],
    ):
        ...

    def set_tasks_priority(self, tasks: List[ScopedKey], priority: int):
        ...

    ### results

    def get_transformation_result(
        self,
        transformation: ScopedKey,
        return_protocoldagresults: bool = False,
    ) -> Union[ProtocolResult, List[ProtocolDAGResult]]:
        """Get `ProtocolResult` for the given `Transformation`.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve results for.
        return_protocoldagresults
            If `True`, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`.

        """

        # first, get the transformation; also confirms it exists
        tf: Transformation = self.get_transformation(transformation)

        # get all objectstorerefs for the given transformation
        objectstorerefs = self._get_resource(
            f"/transformations/{transformation}/results",
            return_gufe=False,
        )

        # get each protocoldagresult; could optimize by parallelizing these
        # calls to some extent, or at least using async/await
        pdrs = []
        for objectstoreref in objectstorerefs:
            pdr_key = objectstoreref["obj_key"]
            scope = objectstoreref['scope']

            pdr_sk = ScopedKey(gufe_key=GufeKey(pdr_key), **Scope.from_str(scope).dict())

            pdr_json = self._get_resource(
                f"/protocoldagresults/{pdr_sk}",
                return_gufe=False,
            )[0]

            pdr = GufeTokenizable.from_dict(
                json.loads(pdr_json, cls=JSON_HANDLER.decoder)
            )
            pdrs.append(pdr)

        if return_protocoldagresults:
            return pdrs
        else:
            return tf.protocol.gather(pdrs)

    def get_task_result(self):
        ...
