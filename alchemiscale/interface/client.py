"""
Client for interacting with user-facing API. --- :mod:`alchemiscale.interface.client`
=====================================================================================

"""

from typing import Union, List, Dict, Optional, Tuple
import json
from collections import Counter


import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ChemicalSystem
from gufe.tokenization import GufeTokenizable, JSON_HANDLER, GufeKey
from gufe.protocols import ProtocolResult, ProtocolDAGResult


from ..base.client import (
    AlchemiscaleBaseClient,
    AlchemiscaleBaseClientError,
    json_to_gufe,
)
from ..models import Scope, ScopedKey
from ..storage.models import Task, ProtocolDAGResultRef, TaskStatusEnum
from ..strategies import Strategy
from ..security.models import CredentialedUserIdentity


class AlchemiscaleClientError(AlchemiscaleBaseClientError):
    ...


class AlchemiscaleClient(AlchemiscaleBaseClient):
    """Client for user interaction with API service."""

    _exception = AlchemiscaleClientError

    def list_scopes(self) -> List[Scope]:
        scopes = self._get_resource(
            f"/identities/{self.identifier}/scopes",
        )
        return [Scope.from_str(s) for s in scopes]

    ### inputs

    def get_scoped_key(self, obj: GufeTokenizable, scope: Scope) -> ScopedKey:
        """Given any gufe object and a fully-specified Scope, return corresponding ScopedKey.

        This method does not check that this ScopedKey is represented in the database.
        It is only a convenience for properly constructing a ScopedKey from a
        gufe object and a Scope.

        """
        if scope.specific():
            return ScopedKey(gufe_key=obj.key, **scope.dict())
        else:
            raise ValueError(
                "Scope for a ScopedKey must be specific; it cannot contain wildcards."
            )

    def check_exists(self, scoped_key: Scope) -> bool:
        """Returns `True` if the given ScopedKey represents an object in the database."""
        return self._get_resource("/exists/{scoped_key}")

    def create_network(self, network: AlchemicalNetwork, scope: Scope) -> ScopedKey:
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
        return json_to_gufe(self._get_resource(f"/networks/{network}"))

    def get_transformation(
        self, transformation: Union[ScopedKey, str]
    ) -> Transformation:
        return json_to_gufe(self._get_resource(f"/transformations/{transformation}"))

    def get_chemicalsystem(
        self, chemicalsystem: Union[ScopedKey, str]
    ) -> ChemicalSystem:
        return json_to_gufe(self._get_resource(f"/chemicalsystems/{chemicalsystem}"))

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

    def get_transformation_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        return_as: str = "list",
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
        task_sks = self._get_resource(
            f"/transformations/{transformation}/tasks", params
        )

        if return_as == "list":
            return [ScopedKey.from_str(i) for i in task_sks]
        elif return_as == "graph":
            g = nx.DiGraph()
            for task, extends in task_sks.items():
                g.add_node(ScopedKey.from_str(task))
                if extends is not None:
                    g.add_edge(ScopedKey.from_str(task), ScopedKey.from_str(extends))

            return g
        else:
            raise ValueError(f"`return_as` takes 'list' or 'graph', not '{return_as}'")

    def get_task_transformation(self, task: ScopedKey) -> ScopedKey:
        transformation = self._get_resource(f"tasks/{task}/transformation")
        return ScopedKey.from_str(transformation)

    def get_transformation_status(
        self, transformation: ScopedKey, visualize: Optional[bool] = True
    ) -> bool:
        """Return the status of the given Transformation.

        If visualize is True, counts of Task statuses for the Transformation
        will be printed to the console.

        """
        g = self.get_transformation_tasks(transformation, return_as="graph")
        all_tasks = list(g.nodes)
        statuses = self.get_tasks_status(all_tasks)
        stat_dict = {}
        for stat, task in zip(statuses, all_tasks):
            stat_dict[task] = stat

        # check if everything is finished
        complete = all([i == TaskStatusEnum.complete for i in statuses])

        if visualize:
            from rich import print as rprint

            # tasks status
            value_counts = Counter(stat_dict.values())
            rprint(
                f"[bold yellow]Alchemiscale Task Status for transformation: {transformation}\n"
            )
            rprint(f"[bold green]Complete: {value_counts[TaskStatusEnum.complete]}")
            rprint(f"[bold blue]Waiting:  {value_counts[TaskStatusEnum.waiting]}")
            rprint(f"[bold orange3]Running:  {value_counts[TaskStatusEnum.running]}")
            rprint(f"[bold red]Error:    {value_counts[TaskStatusEnum.error]}")
            rprint(f"[bold magenta1]Invalid:  {value_counts[TaskStatusEnum.invalid]}")
            rprint(f"[bold purple]Deleted:  {value_counts[TaskStatusEnum.deleted]}")
            rprint(f"-----------")
            rprint(
                f"[bold white]Total Complete:  {value_counts[TaskStatusEnum.complete]}/{sum(value_counts.values())}"
            )
            if complete:
                rprint(f"\n[bold green]Transformation complete!")
            else:
                rprint(f"\n[bold red]Transformation incomplete!")
        return complete

    def action_tasks(
        self, tasks: List[ScopedKey], network: ScopedKey
    ) -> List[Optional[ScopedKey]]:
        """Action Tasks for execution via the given AlchemicalNetwork's
        TaskHub.

        A Task cannot be actioned:
            - to an AlchemicalNetwork in a different Scope.
            - if it extends another Task that is not complete.

        Parameters
        ----------
        tasks
            Task ScopedKeys to action for execution.
        network
            The AlchemicalNetwork ScopedKey to action the Tasks for.
            The Tasks will be added to the network's associated TaskHub.

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

    def cancel_tasks(
        self, tasks: List[ScopedKey], network: ScopedKey
    ) -> List[ScopedKey]:
        """Cancel Tasks for execution via the given AlchemicalNetwork's
        TaskHub.

        A Task cannot be canceled:
            - if it is not present in the AlchemicalNetwork's TaskHub.

        Parameters
        ----------
        tasks
            Task ScopedKeys to cancel for execution.
        network
            The AlchemicalNetwork ScopedKey to cancel the Tasks for.
            The Tasks will be removed from the network's associated TaskHub.

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

    def _set_task_status(
        self, task: ScopedKey, status: TaskStatusEnum
    ) -> Optional[ScopedKey]:
        """Set the status of a `Task`."""
        task_sk = self._post_resource(f"tasks/{task}/status", status.value)
        return ScopedKey.from_str(task_sk) if task_sk is not None else None

    def _get_task_status(self, task: ScopedKey) -> TaskStatusEnum:
        """Get the status of a `Task`."""
        status = self._get_resource(f"tasks/{task}/status")
        return TaskStatusEnum(status)

    def set_tasks_status(
        self, tasks: Union[ScopedKey, List[ScopedKey]], status: TaskStatusEnum
    ) -> List[Optional[ScopedKey]]:
        """Set the status of one or multiple `Task`\s.

        Task status can be set to 'waiting' if currently 'error'.
        Status can be set to 'invalid' or 'deleted' from any other status.

        Parameters
        ----------
        tasks: Union[ScopedKey, List[ScopedKey]]
            The `Task` or `Task`\s to set the status of.
        status: TaskStatusEnum
            The status to set the `Task`\s to. Can be one of
            'waiting', 'invalid', or 'deleted'.

        Returns
        -------
        List[Optional[ScopedKey]]
            The ScopedKeys of the `Task`\s that were updated, in the same order
            as given in `tasks`. If a given `Task` doesn't exist, `None` will
            be returned in its place.

        """
        if isinstance(tasks, ScopedKey):
            tasks = [tasks]

        status = TaskStatusEnum(status)

        task_sks = [self._set_task_status(t, status) for t in tasks]
        return task_sks

    def get_tasks_status(
        self, tasks: Union[ScopedKey, List[ScopedKey]]
    ) -> List[TaskStatusEnum]:
        """Get the status of one or multiple `Task`\s.

        Parameters
        ----------
        tasks: Union[ScopedKey, List[ScopedKey]]
            The `Task` or `Task`\s to get the status of.

        Returns
        -------
        List[TaskStatusEnum]
            The status of each `Task` in the same order as given in `tasks`. If
            a given `Task` doesn't exist, `None` will be returned in its place.

        """
        if isinstance(tasks, ScopedKey):
            tasks = [tasks]
        statuses = [self._get_task_status(t) for t in tasks]
        return statuses

    def get_tasks_priority(
        self,
        tasks: List[ScopedKey],
    ):
        raise NotImplementedError

    def set_tasks_priority(self, tasks: List[ScopedKey], priority: int):
        raise NotImplementedError

    ### results

    def _get_prototocoldagresults(
        self,
        protocoldagresultrefs: List[Dict],
        transformation: ScopedKey,
        ok: bool,
    ):
        if ok:
            route = "results"
        else:
            route = "failures"

        # get each protocoldagresult; could optimize by parallelizing these
        # calls to some extent, or at least using async/await
        pdrs = []
        for protocoldagresultref in protocoldagresultrefs:
            pdr_key = protocoldagresultref["obj_key"]
            scope = protocoldagresultref["scope"]

            pdr_sk = ScopedKey(
                gufe_key=GufeKey(pdr_key), **Scope.from_str(scope).dict()
            )

            pdr_json = self._get_resource(
                f"/transformations/{transformation}/{route}/{pdr_sk}",
            )[0]

            pdr = GufeTokenizable.from_dict(
                json.loads(pdr_json, cls=JSON_HANDLER.decoder)
            )
            pdrs.append(pdr)

        return pdrs

    def get_transformation_results(
        self,
        transformation: ScopedKey,
        return_protocoldagresults: bool = False,
    ) -> Union[Optional[ProtocolResult], List[ProtocolDAGResult]]:
        """Get a `ProtocolResult` for the given `Transformation`.

        A `ProtocolResult` object corresponding to the `Protocol` used for this
        `Transformation`. This is constructed from the available
        `ProtocolDAGResult`\s for this `Transformation`. If no
        `ProtocolDAGResult`\s exist for this `Transformation`, ``None`` is
        returned.

        If `return_protocoldagresults` is ``True``, then a list of the
        `ProtocolDAGResult`\s themselves are returned instead.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve results for.
        return_protocoldagresults
            If `True`, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`. Only successful `ProtocolDAGResult`\s
            are returned.

        """

        if not return_protocoldagresults:
            # get the transformation if we intend to return a ProtocolResult
            tf: Transformation = self.get_transformation(transformation)

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/transformations/{transformation}/results",
        )

        pdrs = self._get_prototocoldagresults(
            protocoldagresultrefs, transformation, ok=True
        )

        if return_protocoldagresults:
            return pdrs
        else:
            if len(pdrs) != 0:
                return tf.gather(pdrs)
            else:
                return None

    def get_transformation_failures(
        self,
        transformation: ScopedKey,
    ) -> Union[ProtocolResult, List[ProtocolDAGResult]]:
        """Get failed `ProtocolDAGResult`\s for the given `Transformation`.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve failures for.

        """
        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/transformations/{transformation}/failures",
        )

        pdrs = self._get_prototocoldagresults(
            protocoldagresultrefs, transformation, ok=False
        )

        return pdrs

    def get_task_results(self, task: ScopedKey):
        """Get successful `ProtocolDAGResult`s for the given `Task`."""
        # first, get the transformation; also confirms it exists
        transformation: ScopedKey = self.get_task_transformation(task)

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/tasks/{task}/results",
        )

        pdrs = self._get_prototocoldagresults(
            protocoldagresultrefs, transformation, ok=True
        )

        return pdrs

    def get_task_failures(self, task: ScopedKey):
        """Get failed `ProtocolDAGResult`s for the given `Task`."""
        # first, get the transformation; also confirms it exists
        transformation: ScopedKey = self.get_task_transformation(task)

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/tasks/{task}/failures",
        )

        pdrs = self._get_prototocoldagresults(
            protocoldagresultrefs, transformation, ok=False
        )

        return pdrs
