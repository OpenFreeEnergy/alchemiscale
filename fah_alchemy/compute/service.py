"""FahAlchemyComputeService

"""

import asyncio
import sched
import time
import random
from typing import Union, Optional, List, Dict
from pathlib import Path
from threading import Thread

import requests

from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

from .client import FahAlchemyComputeClient
from ..storage.models import Task, TaskQueue
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
        shared_path: Path,
        sleep_interval: int = 30,
        heartbeat_frequency: int = 30,
        scope: Optional[Scope] = None,
        limit: int = 1,
    ):
        """

        Parameters
        ----------
        limit
            Maximum number of Tasks to claim at a time from a TaskQueue.

        """
        self.api_url = api_url
        self.name = name
        self.sleep_interval = sleep_interval
        self.heartbeat_frequency = heartbeat_frequency
        self.limit = limit

        self.client = FahAlchemyComputeClient(api_url, identifier, key)

        if scope is None:
            self.scope = Scope()

        self.shared = shared_path
        self.scheduler = sched.scheduler(time.monotonic, time.sleep)

        self._stop = False

    def heartbeat(self):
        """Deliver a heartbeat to the compute API, indicating this service is still alive."""
        ...

    def get_tasks(self, count=1) -> List[Optional[ScopedKey]]:
        """Get a Task to execute from compute API.

        Returns `None` if no Task was available matching service configuration.

        """
        taskqueues: Dict[ScopedKey, TaskQueue] = self.client.query_taskqueues(
            scope=self.scope, return_gufe=True
        )

        # based on weights, choose taskqueue to draw from
        taskqueue: List[ScopedKey] = random.choices(
            list(taskqueues.keys()), weights=[tq.weight for tq in taskqueues.values()]
        )[0]

        # claim tasks from the taskqueue
        tasks = self.client.claim_taskqueue_tasks(
            taskqueue, claimant=self.name, count=count
        )

        return tasks

    def task_to_protocoldag(self, task: ScopedKey) -> ProtocolDAG:
        """Given a Task, produce a corresponding ProtocolDAG that can be executed."""
        ...

        transformation, protocoldag = self.client.get_task_transformation(task)

        return transformation.protocol.create(
            stateA=transformation.stateA,
            stateB=transformation.stateB,
            mapping=transformation.mapping,
            extend_from=protocoldag,
            name=str(task),
        )

    def push_result(
        self, task: ScopedKey, protocoldagresult: ProtocolDAGResult
    ) -> ScopedKey:

        # TODO: this method should postprocess any paths,
        # leaf nodes in DAG for blob results that should go to object store

        # TODO: add check that this protocoldagresult actually corresponds to
        # the given task
        sk = self.client.set_task_result(task, protocoldagresult)

        # TODO: remove claim on task, set to complete; remove from queues

    def execute(self, task: ScopedKey) -> ScopedKey:
        """Executes given Task.

        Returns ScopedKey of ProtocolDAGResult following push to database.

        """
        # obtain a ProtocolDAG from the task
        protocoldag = self.task_to_protocoldag(task)

        # execute the task
        protocoldagresult = execute_DAG(protocoldag, shared=self.shared)

        # push the result (or failure) back to the compute API
        result = self.push_result(task, protocoldagresult)

    def start(self, task_limit: Optional[int] = None):
        """Start the service.

        Parameters
        ----------
        task_limit
            Number of tasks to complete before exiting.
            If `None`, the service will continue until told to stop.

        """

        def scheduler_heartbeat():
            self.heartbeat()
            self.scheduler.enter(self.heartbeat_frequency, 1, scheduler_heartbeat)

        self.scheduler.enter(0, 2, scheduler_heartbeat)

        counter = 0
        while True:

            if task_limit is not None:
                if counter >= task_limit:
                    break

            if self._stop:
                return

            # get a task from the compute API
            tasks: List[ScopedKey] = self.get_tasks(self.limit)

            if all([task is None for task in tasks]):
                time.sleep(self.sleep_interval)
                continue

            for task in tasks:
                if task is None:
                    continue

                self.execute(task)

                counter += 1

    def stop(self):
        self._stop = True

        # Interrupt the scheduler (will finish if in the middle of an update or something, but will
        # cancel running calculations)
        self.int_sleep.interrupt()


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


class FahComputeService(AsynchronousComputeService):
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
