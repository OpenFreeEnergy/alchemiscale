"""
AlchemiscaleComputeService --- :mod:`alchemiscale.compute.service`
===============================================================

"""

import asyncio
import sched
import time
from uuid import uuid4
import random
import threading
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
from threading import Thread

import requests

from gufe import Transformation
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from .client import AlchemiscaleComputeClient
from ..storage.models import Task, TaskHub, ComputeServiceID
from ..models import Scope, ScopedKey


class SleepInterrupted(BaseException):
    """
    Exception class used to signal that an InterruptableSleep was interrupted

    This (like KeyboardInterrupt) derives from BaseException to prevent
    it from being handled with "except Exception".
    """

    pass


class InterruptableSleep:
    """
    A class for sleeping, but interruptable

    This class uses threading Events to wake up from a sleep before the entire sleep
    duration has run. If the sleep is interrupted, then an SleepInterrupted exception is raised.

    This class is a functor, so an instance can be passed as the delay function to a python
    sched.scheduler
    """

    def __init__(self):
        self._event = threading.Event()

    def __call__(self, delay: float):
        interrupted = self._event.wait(delay)
        if interrupted:
            raise SleepInterrupted()

    def interrupt(self):
        self._event.set()

    def clear(self):
        self._event.clear()


class SynchronousComputeService:
    """Fully synchronous compute service.

    This service is intended for use as a reference implementation, and for
    testing/debugging protocols.

    """

    def __init__(
        self,
        api_url: str,
        identifier: str,
        key: str,
        name: str,
        shared_basedir: Path,
        scratch_basedir: Path,
        keep_scratch: bool = False,
        sleep_interval: int = 30,
        heartbeat_frequency: int = 30,
        scopes: Optional[List[Scope]] = None,
        limit: int = 1,
    ):
        """Create a `SynchronousComputeService` instance.

        Parameters
        ----------
        api_url
            URL pointing to the compute API to execute Tasks for.
        identifier
            Identifier for the compute identity used for authentication.
        key
            Credential for the compute identity used for authentication.
        name
            The name to give this compute service; used for Task provenance, so
            typically set to a distinct value to distinguish different compute
            resources, e.g. different hosts or HPC clusters. 
        shared_basedir
            Filesystem path to use for `ProtocolDAG` `shared` space.
        scratch_basedir
            Filesystem path to use for `ProtocolUnit` `scratch` space.
        keep_scratch
            If True, don't remove scratch directories for `ProtocolUnit`s after
            completion.
        sleep_interval
            Time in seconds to sleep if no Tasks claimed from compute API.
        heartbeat_frequency
            Frequency at which to send heartbeats to compute API.
        scopes
            Scopes to limit Task claiming to; defaults to all Scopes accessible
            by compute identity.
        limit
            Maximum number of Tasks to claim at a time from a TaskHub.

        """
        self.api_url = api_url
        self.name = name
        self.sleep_interval = sleep_interval
        self.heartbeat_frequency = heartbeat_frequency
        self.limit = limit

        self.client = AlchemiscaleComputeClient(api_url, identifier, key)

        if scopes is None:
            self.scopes = [Scope()]

        self.shared_basedir = shared_basedir
        self.shared_basedir.mkdir(exist_ok=True)

        self.scratch_basedir = scratch_basedir
        self.scratch_basedir.mkdir(exist_ok=True)
        self.keep_scratch = keep_scratch

        self.scheduler = sched.scheduler(time.monotonic, time.sleep)

        self.counter = 0

        self.computeserviceid = ComputeServiceID(
                identifier=f"{self.name}-{uuid4()}")

        self._stop = False


    def _register(self):
        """Register this compute service with the compute API.

        """
        self.client.register(self.computeserviceid)

    def _deregister(self):
        """Deregister this compute service with the compute API.

        """
        self.client.deregister(self.computeserviceid)

    def heartbeat(self):
        """Deliver a heartbeat to the compute API, indicating this service is still alive."""
        ...

    def claim_tasks(self, count=1) -> List[Optional[ScopedKey]]:
        """Get a Task to execute from compute API.

        Returns `None` if no Task was available matching service configuration.

        """
        taskhubs: Dict[ScopedKey, TaskHub] = self.client.query_taskhubs(
            scopes=self.scopes, return_gufe=True
        )

        # based on weights, choose taskhub to draw from
        taskhub: List[ScopedKey] = random.choices(
            list(taskhubs.keys()), weights=[tq.weight for tq in taskhubs.values()]
        )[0]

        # claim tasks from the taskhub
        tasks = self.client.claim_taskhub_tasks(
            taskhub, computeserviceid=self.computeserviceid, count=count
        )

        return tasks

    def unclaim_tasks(self):
        self.client.unclaim_tasks()

    def task_to_protocoldag(
        self, task: ScopedKey
    ) -> Tuple[ProtocolDAG, Transformation, Optional[ProtocolDAGResult]]:
        """Given a Task, produce a corresponding ProtocolDAG that can be executed.

        Also gives the Transformation that this ProtocolDAG corresponds to.
        If the Task extends another Task, then the ProtocolDAGResult for that
        other Task is also given; otherwise `None` given.

        """

        transformation, extends_protocoldagresult = self.client.get_task_transformation(
            task
        )

        protocoldag = transformation.create(
            extends=extends_protocoldagresult,
            name=str(task),
        )
        return protocoldag, transformation, extends_protocoldagresult

    def push_result(
        self, task: ScopedKey, protocoldagresult: ProtocolDAGResult
    ) -> ScopedKey:
        # TODO: this method should postprocess any paths,
        # leaf nodes in DAG for blob results that should go to object store

        # TODO: ship paths to object store

        # finally, push ProtocolDAGResult
        sk: ScopedKey = self.client.set_task_result(task, protocoldagresult)

        return sk

    def execute(self, task: ScopedKey) -> ScopedKey:
        """Executes given Task.

        Returns ScopedKey of ProtocolDAGResultRef following push to database.

        """
        # obtain a ProtocolDAG from the task
        protocoldag, transformation, extends = self.task_to_protocoldag(task)

        # execute the task; this looks the same whether the ProtocolDAG is a
        # success or failure
        shared = self.shared_basedir / str(protocoldag.key) / self.counter
        shared.mkdir()

        protocoldagresult = execute_DAG(
            protocoldag,
            shared=shared,
            scratch_basdir = self.scratch_basedir,
            keep_scratch=self.keep_scratch
        )

        # push the result (or failure) back to the compute API
        result_sk = self.push_result(task, protocoldagresult)

        return result_sk

    def start(self, task_limit: Optional[int] = None):
        """Start the service.

        Parameters
        ----------
        task_limit
            Number of Tasks to complete before exiting.
            If `None`, the service will continue until told to stop.

        """
        self._register()

        def scheduler_heartbeat():
            self.heartbeat()
            self.scheduler.enter(self.heartbeat_frequency, 1, scheduler_heartbeat)

        self.scheduler.enter(0, 2, scheduler_heartbeat)

        while True:
            if task_limit is not None:
                if self.counter >= task_limit:
                    break

            if self._stop:
                return

            # claim tasks from the compute API
            tasks: List[ScopedKey] = self.claim_tasks(self.limit)

            # if no tasks claimed, sleep
            if all([task is None for task in tasks]):
                if self._stop:
                    return
                time.sleep(self.sleep_interval)
                continue

            # otherwise, process tasks
            for task in tasks:
                if self._stop:
                    return

                if task is None:
                    continue

                # execute each task
                self.execute(task)
                self.counter += 1

    def stop(self):
        self._stop = True

        # TODO: drop claims on tasks
        self.unclaim_tasks()

        # Interrupt the scheduler (will finish if in the middle of an update or
        # something, but will cancel running calculations)
        self.int_sleep.interrupt()

        self._deregister()



class AsynchronousComputeService(SynchronousComputeService):
    """Asynchronous compute service.

    This service can be used in production cases, though it does not make use
    of Folding@Home.

    """

    def __init__(self, api_url):
        self.scheduler = sched.scheduler(time.monotonic, time.sleep)
        # self.loop = asyncio.get_event_loop()

        self._stop = False

    def get_new_tasks(self):
        ...

    def start(self):
        """Start the service; will keep going until told to stop."""
        while True:
            if self._stop:
                return

    def stop(self):
        self._stop = True


class AlchemiscaleComputeService(AsynchronousComputeService):
    """Folding@Home-based compute service.

    This service is designed for production use with Folding@Home.

    """

    def __init__(self, object_store, fah_work_server):
        self.scheduler = sched.scheduler(time.time, self.int_sleep)
        self.loop = asyncio.get_event_loop()

        self._stop = False

    async def get_new_tasks(self):
        ...

    def start(self):
        ...
        while True:
            if self._stop:
                return

    def stop(self):
        ...
