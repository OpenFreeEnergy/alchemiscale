"""
:mod:`alchemiscale.interface.client` --- client for interacting with user-facing API
====================================================================================

"""

import asyncio
from typing import Union, List, Dict, Optional, Tuple
import json
from itertools import chain
from collections import Counter
from functools import lru_cache

import httpx
from async_lru import alru_cache
import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ChemicalSystem
from gufe.tokenization import GufeTokenizable, JSON_HANDLER, GufeKey
from gufe.protocols import ProtocolResult, ProtocolDAGResult


from ..base.client import (
    AlchemiscaleBaseClient,
    AlchemiscaleBaseClientError,
    json_to_gufe,
    use_session,
)
from ..models import Scope, ScopedKey
from ..storage.models import Task, ProtocolDAGResultRef, TaskStatusEnum
from ..strategies import Strategy
from ..security.models import CredentialedUserIdentity
from ..validators import validate_network_nonself


class AlchemiscaleClientError(AlchemiscaleBaseClientError):
    ...


def _get_transformation_results(client_settings, tf_sk, ok: bool, kwargs):
    client = AlchemiscaleClient(**client_settings)
    if ok:
        return tf_sk, client.get_transformation_results(tf_sk, **kwargs)
    else:
        return tf_sk, client.get_transformation_failures(tf_sk, **kwargs)


class AlchemiscaleClient(AlchemiscaleBaseClient):
    """Client for user interaction with API service."""

    _exception = AlchemiscaleClientError

    def get_scopes(self) -> List[Scope]:
        scopes = self._get_resource(
            f"/identities/{self.identifier}/scopes",
        )
        return sorted([Scope.from_str(s) for s in scopes])

    def list_scopes(self) -> List[Scope]:
        return self.get_scopes()

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
        """Returns ``True`` if the given ScopedKey represents an object in the database."""
        return self._get_resource("/exists/{scoped_key}")

    def create_network(
        self,
        network: AlchemicalNetwork,
        scope: Scope,
        compress: Union[bool, int] = True,
        visualize: bool = True,
    ) -> ScopedKey:
        """Submit an AlchemicalNetwork to a specific Scope.

        Parameters
        ----------
        network
            The AlchemicalNetwork to submit.
        scope
            The Scope in which to submit the AlchemicalNetwork.
            This must be a *specific* Scope; it must not contain wildcards.
        compress
            If ``True``, compress the AlchemicalNetwork client-side before
            shipping to the API service. This can reduce submission time
            depending on the bandwidth of your connection to the API service.
            Set to ``False`` to submit without compressing. This is a
            performance optimization; it has no bearing on the result of this
            method call.

            Use an integer between 0 and 9 for finer control over
            the degree of compression; 0 means no compression, 9 means max
            compression. ``True`` is synonymous with level 5 compression.
        visualize
            If ``True``, show submission progress indicator.

        Returns
        -------
        ScopedKey
            The ScopedKey of the AlchemicalNetwork.

        """
        if not scope.specific():
            raise ValueError(
                f"`scope` '{scope}' contains wildcards ('*'); `scope` must be *specific*"
            )

        validate_network_nonself(network)

        sk = self.get_scoped_key(network, scope)

        def post():
            data = dict(network=network.to_dict(), scope=scope.dict())
            return self._post_resource("/networks", data, compress=compress)

        if visualize:
            from rich.progress import Progress

            with Progress(*self._rich_waiting_columns(), transient=False) as progress:
                task = progress.add_task(
                    f"Submitting [bold]'{sk}'[/bold]...", total=None
                )

                scoped_key = post()
                progress.start_task(task)
                progress.update(task, total=1, completed=1)
        else:
            scoped_key = post()

        return ScopedKey.from_dict(scoped_key)

    def query_networks(
        self,
        name: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> List[ScopedKey]:
        """Query for AlchemicalNetworks, optionally by name or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all AlchemicalNetworks that are within the Scopes this user has access
        to.

        """
        if scope is None:
            scope = Scope()

        params = dict(name=name, **scope.dict())

        return self._query_resource("/networks", params=params)

    def query_transformations(
        self,
        name: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> List[ScopedKey]:
        """Query for Transformations, optionally by name or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all Transformations that are within the Scopes this user has access to.

        """
        if scope is None:
            scope = Scope()

        params = dict(name=name, **scope.dict())

        return self._query_resource("/transformations", params=params)

    def query_chemicalsystems(
        self,
        name: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> List[ScopedKey]:
        """Query for ChemicalSystems, optionally by name or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all ChemicalSystems that are within the Scopes this user has access to.

        """
        if scope is None:
            scope = Scope()

        params = dict(name=name, **scope.dict())

        return self._query_resource("/chemicalsystems", params=params)

    def get_network_transformations(self, network: ScopedKey) -> List[ScopedKey]:
        """List ScopedKeys for Transformations associated with the given AlchemicalNetwork."""
        return self._query_resource(f"/networks/{network}/transformations")

    def get_network_weight(self, network: ScopedKey) -> float:
        """Get the weight of the TaskHub associated with the given AlchemicalNetwork.

        Compute services perform a weighted selection of the AlchemicalNetworks
        visible to them before claiming Tasks actioned on those networks.
        Networks with higher weight are more likely to be selected than those
        with lower weight, and so will generally get more compute attention
        over time.

        A weight of ``0`` means the AlchemicalNetwork will not receive any
        compute for its actioned Tasks.

        """
        return self._get_resource(f"/networks/{network}/weight")

    def set_network_weight(self, network: ScopedKey, weight: float) -> None:
        """Set the weight of the TaskHub associated with the given AlchemicalNetwork.

        Compute services perform a weighted selection of the AlchemicalNetworks
        visible to them before claiming Tasks actioned on those networks.
        Networks with higher weight are more likely to be selected than those
        with lower weight, and so will generally get more compute attention
        over time.

        A weight of ``0`` means the AlchemicalNetwork will not receive any
        compute for its actioned Tasks.

        Parameters
        ----------
        network
            The ScopedKey of the AlchemicalNetwork to set the weight for.
        weight
            The weight to set for the network. This must be between 0 and 1
            (inclusive). Setting the value to 0 will effectively disable
            compute on this network without cancelling its actioned Tasks.

        """
        self._post_resource(f"/networks/{network}/weight", weight)

    def get_transformation_networks(self, transformation: ScopedKey) -> List[ScopedKey]:
        """List ScopedKeys for AlchemicalNetworks associated with the given Transformation."""
        return self._query_resource(f"/transformations/{transformation}/networks")

    def get_network_chemicalsystems(self, network: ScopedKey) -> List[ScopedKey]:
        """List ScopedKeys for the ChemicalSystems associated with the given AlchemicalNetwork."""
        return self._query_resource(f"/networks/{network}/chemicalsystems")

    def get_chemicalsystem_networks(self, chemicalsystem: ScopedKey) -> List[ScopedKey]:
        """List ScopedKeys for the AlchemicalNetworks associated with the given ChemicalSystem."""
        return self._query_resource(f"/chemicalsystems/{chemicalsystem}/networks")

    def get_transformation_chemicalsystems(
        self, transformation: ScopedKey
    ) -> List[ScopedKey]:
        """List ScopedKeys for the ChemicalSystems associated with the given Transformation."""
        return self._query_resource(
            f"/transformations/{transformation}/chemicalsystems"
        )

    def get_chemicalsystem_transformations(
        self, chemicalsystem: ScopedKey
    ) -> List[ScopedKey]:
        """List ScopedKeys for the Transformations associated with the given ChemicalSystem."""
        return self._query_resource(
            f"/chemicalsystems/{chemicalsystem}/transformations"
        )

    @lru_cache(maxsize=100)
    def get_network(
        self,
        network: Union[ScopedKey, str],
        compress: bool = True,
        visualize: bool = True,
    ) -> AlchemicalNetwork:
        """Retrieve an AlchemicalNetwork given its ScopedKey.

        Parameters
        ----------
        network
            The ScopedKey of the AlchemicalNetwork to retrieve.
        compress
            If ``True``, compress the AlchemicalNetwork server-side before
            shipping it to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicator.

        Returns
        -------
        AlchemicalNetwork
            The retrieved AlchemicalNetwork.

        """
        if visualize:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TimeElapsedColumn,
                TextColumn,
            )

            with Progress(*self._rich_waiting_columns(), transient=False) as progress:
                task = progress.add_task(
                    f"Retrieving [bold]'{network}'[/bold]...", total=None
                )

                an = json_to_gufe(
                    self._get_resource(f"/networks/{network}", compress=compress)
                )
                progress.start_task(task)
                progress.update(task, total=1, completed=1)
        else:
            an = json_to_gufe(
                self._get_resource(f"/networks/{network}", compress=compress)
            )

        return an

    @lru_cache(maxsize=10000)
    def get_transformation(
        self,
        transformation: Union[ScopedKey, str],
        compress: bool = True,
        visualize: bool = True,
    ) -> Transformation:
        """Retrieve a Transformation given its ScopedKey.

        Parameters
        ----------
        transformation
            The ScopedKey of the Transformation to retrieve.
        compress
            If ``True``, compress the Transformation server-side before
            shipping it to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicator.

        Returns
        -------
        Transformation
            The retrieved Transformation.

        """
        if visualize:
            from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

            with Progress(*self._rich_waiting_columns(), transient=False) as progress:
                task = progress.add_task(
                    f"Retrieving [bold]'{transformation}'[/bold]...", total=None
                )

                tf = json_to_gufe(
                    self._get_resource(
                        f"/transformations/{transformation}", compress=compress
                    )
                )
                progress.start_task(task)
                progress.update(task, total=1, completed=1)
        else:
            tf = json_to_gufe(
                self._get_resource(
                    f"/transformations/{transformation}", compress=compress
                )
            )

        return tf

    @lru_cache(maxsize=1000)
    def get_chemicalsystem(
        self,
        chemicalsystem: Union[ScopedKey, str],
        compress: bool = True,
        visualize: bool = True,
    ) -> ChemicalSystem:
        """Retrieve a ChemicalSystem given its ScopedKey.

        Parameters
        ----------
        chemicalsystem
            The ScopedKey of the ChemicalSystem to retrieve.
        compress
            If ``True``, compress the ChemicalSystem server-side before
            shipping it to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicator.

        Returns
        -------
        ChemicalSystem
            The retrieved ChemicalSystem.

        """
        if visualize:
            from rich.progress import Progress

            with Progress(*self._rich_waiting_columns(), transient=False) as progress:
                task = progress.add_task(
                    f"Retrieving [bold]'{chemicalsystem}'[/bold]...", total=None
                )

                cs = json_to_gufe(
                    self._get_resource(
                        f"/chemicalsystems/{chemicalsystem}", compress=compress
                    )
                )

                progress.start_task(task)
                progress.update(task, total=1, completed=1)
        else:
            cs = json_to_gufe(
                self._get_resource(
                    f"/chemicalsystems/{chemicalsystem}", compress=compress
                )
            )

        return cs

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

    def query_tasks(
        self,
        scope: Optional[Scope] = None,
        status: Optional[str] = None,
    ) -> List[ScopedKey]:
        """Query for Tasks, optionally by status or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all Tasks that are within the Scopes this user has access to.

        """
        if scope is None:
            scope = Scope()

        params = dict(status=status, **scope.dict())

        return self._query_resource("/tasks", params=params)

    def get_network_tasks(self, network: ScopedKey, status: Optional[str] = None):
        """List ScopedKeys for all Tasks associated with the given AlchemicalNetwork."""
        params = {"status": status}
        return self._query_resource(f"/networks/{network}/tasks", params=params)

    def get_task_networks(self, task: ScopedKey):
        """List ScopedKeys for all AlchemicalNetworks associated with the given Task."""
        return self._query_resource(f"/tasks/{task}/networks")

    def get_transformation_tasks(
        self,
        transformation: ScopedKey,
        extends: Optional[ScopedKey] = None,
        return_as: str = "list",
        status: Optional[str] = None,
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
            If 'graph', Tasks will be returned in a `networkx.DiGraph`, with a
            directed edge pointing from a given Task to the Task it extends.

        """
        if extends:
            extends = str(extends)

        params = dict(extends=extends, return_as=return_as, status=status)
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
        """Get the Transformation associated with the given Task."""
        transformation = self._get_resource(f"/tasks/{task}/transformation")
        return ScopedKey.from_str(transformation)

    def _visualize_status(self, status_counts, status_object):
        from rich import print as rprint
        from rich.table import Table

        title = f"{status_object}"
        table = Table(title=title, title_justify="left", expand=True)
        # table = Table()

        table.add_column("status", justify="left", no_wrap=True)
        table.add_column("count", justify="right")

        table.add_row("complete", f"{status_counts.get('complete', 0)}", style="green")
        table.add_row("running", f"{status_counts.get('running', 0)}", style="orange3")
        table.add_row("waiting", f"{status_counts.get('waiting', 0)}", style="#1793d0")
        table.add_row("error", f"{status_counts.get('error', 0)}", style="#ff073a")
        table.add_row("invalid", f"{status_counts.get('invalid', 0)}", style="magenta1")
        table.add_row("deleted", f"{status_counts.get('deleted', 0)}", style="purple")

        rprint(table)

    def get_scope_status(
        self,
        scope: Optional[Scope] = None,
        visualize: Optional[bool] = True,
    ) -> Dict[str, int]:
        """Return status counts for all Tasks within the given Scope.

        Parameters
        ----------
        scope
            Scope to use for querying status. Non-specific Scopes are allowed,
            and will give back counts for all Tasks within that this user has
            Scope access to. Defaults to all Scopes.
        visualize
            If ``True``, print a table of status counts.

        Returns
        -------
        status_counts
            Dict giving statuses as keys, Task counts as values.
        """
        if scope is None:
            scope = Scope()

        status_counts = self._get_resource(f"/scopes/{scope}/status")

        if visualize:
            self._visualize_status(status_counts, scope)

        return status_counts

    def get_network_status(
        self,
        network: ScopedKey,
        visualize: Optional[bool] = True,
    ) -> Dict[str, int]:
        """Return status counts for all Tasks associated with the given AlchemicalNetwork.

        Parameters
        ----------
        network
            ScopedKey for the Alchemicalnetwork to obtain status counts for.
        visualize
            If ``True``, print a table of status counts.

        Returns
        -------
        status_counts
            Dict giving statuses as keys, Task counts as values.
        """
        status_counts = self._get_resource(f"/networks/{network}/status")

        if visualize:
            self._visualize_status(status_counts, network)

        return status_counts

    def get_transformation_status(
        self,
        transformation: ScopedKey,
        visualize: Optional[bool] = True,
    ) -> Dict[str, int]:
        """Return status counts for all Tasks associated with the given
        Transformation.

        Parameters
        ----------
        transformation
            ScopedKey for the Transformation to obtain status counts for.
        visualize
            If ``True``, print a table of status counts.

        Returns
        -------
        status_counts
            Dict giving statuses as keys, Task counts as values.
        """
        status_counts = self._get_resource(f"/transformations/{transformation}/status")

        if visualize:
            self._visualize_status(status_counts, transformation)

        return status_counts

    def action_tasks(
        self, tasks: List[ScopedKey], network: ScopedKey
    ) -> List[Optional[ScopedKey]]:
        """Action Tasks for execution via the given AlchemicalNetwork's
        TaskHub.

        A Task cannot be actioned:
            - to an AlchemicalNetwork in a different Scope.
            - if it extends another Task that is not complete.
            - if it has any status other than 'waiting', 'running', or 'error'

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
            `tasks` on input. If a Task couldn't be actioned, then ``None`` will
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
            `tasks` on input. If a Task couldn't be canceled, then ``None`` will
            be returned in its place.

        """
        data = dict(tasks=[t.dict() for t in tasks])
        canceled_sks = self._post_resource(f"/networks/{network}/tasks/cancel", data)

        return [ScopedKey.from_str(i) if i is not None else None for i in canceled_sks]

    def _set_task_status(
        self, task: ScopedKey, status: TaskStatusEnum
    ) -> Optional[ScopedKey]:
        """Set the status of a `Task`."""
        task_sk = self._post_resource(f"/tasks/{task}/status", status.value)
        return ScopedKey.from_str(task_sk) if task_sk is not None else None

    async def _set_task_status(
        self, tasks: List[ScopedKey], status: TaskStatusEnum
    ) -> List[Optional[ScopedKey]]:
        """Set the statuses for many Tasks"""
        data = dict(tasks=[t.dict() for t in tasks], status=status.value)
        tasks_updated = await self._post_resource_async(
            f"/bulk/tasks/status/set", data=data
        )
        return [
            ScopedKey.from_str(task_sk) if task_sk is not None else None
            for task_sk in tasks_updated
        ]

    async def _get_task_status(self, tasks: List[ScopedKey]) -> List[TaskStatusEnum]:
        """Get the statuses for many Tasks"""
        data = dict(tasks=[t.dict() for t in tasks])
        statuses = await self._post_resource_async(f"/bulk/tasks/status/get", data=data)
        return statuses

    def set_tasks_status(
        self,
        tasks: List[ScopedKey],
        status: Union[TaskStatusEnum, str],
        batch_size: int = 1000,
    ) -> List[Optional[ScopedKey]]:
        """Set the status of one or multiple Tasks.

        Task status can be set to 'waiting' if currently 'error'.
        Status can be set to 'invalid' or 'deleted' from any other status.

        Parameters
        ----------
        tasks
            The Tasks to set the status of.
        status
            The status to set the Tasks to. Can be one of
            'waiting', 'invalid', or 'deleted'.

        Returns
        -------
        updated
            The ScopedKeys of the Tasks that were updated, in the same order
            as given in `tasks`. If a given Task doesn't exist, ``None`` will
            be returned in its place.

        """
        status = TaskStatusEnum(status)

        tasks = [
            ScopedKey.from_str(task) if isinstance(task, str) else task
            for task in tasks
        ]

        @use_session
        async def async_request(self):
            scoped_keys = await asyncio.gather(
                *[
                    self._set_task_status(task_batch, status)
                    for task_batch in self._batched(tasks, batch_size)
                ]
            )

            return list(chain.from_iterable(scoped_keys))

        coro = async_request(self)

        try:
            return asyncio.run(coro)
        except RuntimeError:
            # we use nest_asyncio to support environments where an event loop
            # is already running, such as in a Jupyter notebook
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(coro)

    def get_tasks_status(
        self, tasks: List[ScopedKey], batch_size: int = 1000
    ) -> List[str]:
        """Get the status of multiple Tasks.

        Parameters
        ----------
        tasks
            The Tasks to get the status of.
        batch_size
            The number of Tasks to include in a single request; use to tune
            method call speed when requesting many statuses at once.

        Returns
        -------
        statuses
            The status of each Task in the same order as given in `tasks`. If a
            given Task doesn't exist, ``None`` will be returned in its place.

        """
        tasks = [
            ScopedKey.from_str(task) if isinstance(task, str) else task
            for task in tasks
        ]

        @use_session
        async def async_request(self):
            statuses = await asyncio.gather(
                *[
                    self._get_task_status(task_batch)
                    for task_batch in self._batched(tasks, batch_size)
                ]
            )

            return list(chain.from_iterable(statuses))

        try:
            return asyncio.run(async_request(self))
        except RuntimeError:
            # we use nest_asyncio to support environments where an event loop
            # is already running, such as in a Jupyter notebook
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(async_request(self))

    def get_tasks_priority(
        self,
        tasks: List[ScopedKey],
    ):
        raise NotImplementedError

    def set_tasks_priority(self, tasks: List[ScopedKey], priority: int):
        raise NotImplementedError

    ### results

    @alru_cache(maxsize=10000)
    async def _async_get_protocoldagresult(
        self, protocoldagresultref, transformation, route, compress
    ):
        pdr_json = await self._get_resource_async(
            f"/transformations/{transformation}/{route}/{protocoldagresultref}",
            compress=compress,
        )

        pdr = GufeTokenizable.from_dict(
            json.loads(pdr_json[0], cls=JSON_HANDLER.decoder)
        )

        return pdr

    def _get_protocoldagresults(
        self,
        protocoldagresultrefs: List[ScopedKey],
        transformation: ScopedKey,
        ok: bool,
        compress: bool = True,
        visualize: bool = True,
    ):
        if ok:
            route = "results"
        else:
            route = "failures"

        @use_session
        async def async_request(self):
            if visualize:
                from rich.progress import Progress

                with Progress(
                    *self._rich_progress_columns(), transient=False
                ) as progress:
                    task = progress.add_task(
                        f"Retrieving [bold]ProtocolDAGResult[/bold]s",
                        total=len(protocoldagresultrefs),
                    )

                    coros = [
                        self._async_get_protocoldagresult(
                            protocoldagresultref,
                            transformation,
                            route,
                            compress,
                        )
                        for protocoldagresultref in protocoldagresultrefs
                    ]
                    pdrs = []
                    for coro in asyncio.as_completed(coros):
                        pdr = await coro
                        pdrs.append(pdr)
                        progress.update(task, advance=1)
                    progress.refresh()
            else:
                coros = [
                    self._async_get_protocoldagresult(
                        protocoldagresultref,
                        transformation,
                        route,
                        compress,
                    )
                    for protocoldagresultref in protocoldagresultrefs
                ]
                pdrs = await asyncio.gather(*coros)

            return pdrs

        coro = async_request(self)

        try:
            return asyncio.run(coro)
        except RuntimeError:
            # we use nest_asyncio to support environments where an event loop
            # is already running, such as in a Jupyter notebook
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(coro)

    def _get_network_results(
        self,
        network: ScopedKey,
        ok: bool = True,
        return_protocoldagresults: bool = False,
        compress: bool = True,
        visualize: bool = True,
    ) -> Dict[str, Union[Optional[ProtocolResult], List[ProtocolDAGResult]]]:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")

        if ok:
            kwargs = dict(
                return_protocoldagresults=return_protocoldagresults,
                compress=compress,
                visualize=False,
            )
            route = "results"
        else:
            kwargs = dict(compress=compress, visualize=False)
            route = "failures"

        with ProcessPoolExecutor(mp_context=ctx) as executor:
            futures = []
            tf_sks = self.get_network_transformations(network)
            for tf_sk in tf_sks:
                futures.append(
                    executor.submit(
                        _get_transformation_results,
                        self._settings(),
                        tf_sk,
                        ok,
                        kwargs,
                    )
                )

            results = {}
            if visualize:
                from rich.progress import Progress

                with Progress(
                    *self._rich_progress_columns(), transient=False
                ) as progress:
                    task = progress.add_task(
                        f"Retrieving [bold]Transformation[/bold] {route}",
                        total=len(tf_sks),
                    )

                    for future in as_completed(futures):
                        tf_sk, result = future.result()
                        results[tf_sk] = result
                        progress.update(task, advance=1)
                    progress.refresh()
            else:
                for future in as_completed(futures):
                    tf_sk, result = future.result()
                    results[tf_sk] = result

        return results

    def get_network_results(
        self,
        network: ScopedKey,
        return_protocoldagresults: bool = False,
        compress: bool = True,
        visualize: bool = True,
    ) -> Dict[str, Union[Optional[ProtocolResult], List[ProtocolDAGResult]]]:
        """Get a `ProtocolResult` for every `Transformation` in the given
        `AlchemicalNetwork`.

        A dict giving the `ScopedKey` of each `Transformation` in the network
        as keys, `ProtocolResult` as values, is returned. If no
        `ProtocolDAGResult`\s exist for a given `Transformation`, ``None`` is
        given for its value.

        If `return_protocoldagresults` is ``True``, then a list of the
        `ProtocolDAGResult`\s themselves is given as values instead of
        `ProtocolResult`\s.

        Parameters
        ----------
        network
            The `ScopedKey` of the `AlchemicalNetwork` to retrieve results for.
        return_protocoldagresults
            If ``True``, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`. Only successful `ProtocolDAGResult`\s
            are returned.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.

        """
        return self._get_network_results(
            network=network,
            ok=True,
            return_protocoldagresults=return_protocoldagresults,
            compress=compress,
            visualize=visualize,
        )

    def get_network_failures(
        self,
        network: ScopedKey,
        compress: bool = True,
        visualize: bool = True,
    ) -> Dict[str, List[ProtocolDAGResult]]:
        """Get all failed `ProtocolDAGResult`s for every `Transformation` in
        the given `AlchemicalNetwork`.

        A dict giving the `ScopedKey` of each `Transformation` in the network
        as keys, a list of the `ProtocolDAGResult`\s as values, is returned.

        Parameters
        ----------
        network
            The `ScopedKey` of the `AlchemicalNetwork` to retrieve results for.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.

        """
        return self._get_network_results(
            network=network, ok=False, compress=compress, visualize=visualize
        )

    def get_transformation_results(
        self,
        transformation: ScopedKey,
        return_protocoldagresults: bool = False,
        compress: bool = True,
        visualize: bool = True,
    ) -> Union[Optional[ProtocolResult], List[ProtocolDAGResult]]:
        """Get a `ProtocolResult` for the given `Transformation`.

        A `ProtocolResult` object corresponds to the `Protocol` used for this
        `Transformation`. This is constructed from the available
        `ProtocolDAGResult`\s for this `Transformation` via
        `Transformation.gather`. If no `ProtocolDAGResult`\s exist for this
        `Transformation`, ``None`` is returned.

        If `return_protocoldagresults` is ``True``, then a list of the
        `ProtocolDAGResult`\s themselves is returned instead.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve results for.
        return_protocoldagresults
            If ``True``, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`. Only successful `ProtocolDAGResult`\s
            are returned.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.


        """

        if not return_protocoldagresults:
            # get the transformation if we intend to return a ProtocolResult
            tf: Transformation = self.get_transformation(
                transformation, visualize=visualize
            )

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/transformations/{transformation}/results",
        )

        pdrs = self._get_protocoldagresults(
            protocoldagresultrefs,
            transformation,
            ok=True,
            compress=compress,
            visualize=visualize,
        )

        if return_protocoldagresults:
            return pdrs
        else:
            if len(pdrs) != 0:
                return tf.gather(pdrs)
            else:
                return None

    def get_transformation_failures(
        self, transformation: ScopedKey, compress: bool = True, visualize: bool = True
    ) -> List[ProtocolDAGResult]:
        """Get failed `ProtocolDAGResult`\s for the given `Transformation`.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve failures for.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.

        """
        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/transformations/{transformation}/failures",
        )

        pdrs = self._get_protocoldagresults(
            protocoldagresultrefs,
            transformation,
            ok=False,
            compress=compress,
            visualize=visualize,
        )

        return pdrs

    def get_task_results(
        self, task: ScopedKey, compress: bool = True, visualize: bool = True
    ) -> List[ProtocolDAGResult]:
        """Get successful `ProtocolDAGResult`s for the given `Task`.

        Parameters
        ----------
        task
            The `ScopedKey` of the `Task` to retrieve results for.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.

        """
        # first, get the transformation; also confirms it exists
        transformation: ScopedKey = self.get_task_transformation(task)

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/tasks/{task}/results",
        )

        pdrs = self._get_protocoldagresults(
            protocoldagresultrefs,
            transformation,
            ok=True,
            compress=compress,
            visualize=visualize,
        )

        return pdrs

    def get_task_failures(
        self, task: ScopedKey, compress: bool = True, visualize: bool = True
    ) -> List[ProtocolDAGResult]:
        """Get failed `ProtocolDAGResult`s for the given `Task`.

        Parameters
        ----------
        task
            The `ScopedKey` of the `Task` to retrieve failures for.
        compress
            If ``True``, compress the ProtocolDAGResults server-side before
            shipping them to the client. This can reduce retrieval time depending
            on the bandwidth of your connection to the API service. Set to
            ``False`` to retrieve without compressing. This is a performance
            optimization; it has no bearing on the result of this method call.
        visualize
            If ``True``, show retrieval progress indicators.

        """
        # first, get the transformation; also confirms it exists
        transformation: ScopedKey = self.get_task_transformation(task)

        # get all protocoldagresultrefs for the given transformation
        protocoldagresultrefs = self._get_resource(
            f"/tasks/{task}/failures",
        )

        pdrs = self._get_protocoldagresults(
            protocoldagresultrefs,
            transformation,
            ok=False,
            compress=compress,
            visualize=visualize,
        )

        return pdrs
