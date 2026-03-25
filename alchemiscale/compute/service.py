"""
:mod:`alchemiscale.compute.service` --- compute services for FEC execution
==========================================================================

"""

import gc
import sched
import time
import logging
from uuid import uuid4
import threading
from pathlib import Path
import queue
import shutil
from typing import Any

from gufe import Transformation
from gufe.protocols.protocoldag import _pu_to_pur, execute_DAG, ProtocolDAG, ProtocolDAGResult
from gufe.protocols.protocolunit import Context, ProtocolUnitFailure, ProtocolUnitResult, ProtocolUnit
from gufe.tokenization import GufeKey
import networkx as nx

from .client import AlchemiscaleComputeClient
from .settings import ComputeServiceSettings
from ..storage.models import ComputeServiceID
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

    def __init__(self, settings: ComputeServiceSettings):
        """Create a `SynchronousComputeService` instance."""
        self.settings = settings

        self.api_url = self.settings.api_url
        self.name = self.settings.name
        self.compute_manager_id = self.settings.compute_manager_id
        self.sleep_interval = self.settings.sleep_interval
        self.deep_sleep_interval = self.settings.deep_sleep_interval
        self.heartbeat_interval = self.settings.heartbeat_interval
        self.claim_limit = self.settings.claim_limit

        self.client = AlchemiscaleComputeClient(
            api_url=self.settings.api_url,
            identifier=self.settings.identifier,
            key=self.settings.key,
            cache_directory=self.settings.client_cache_directory,
            cache_size_limit=self.settings.client_cache_size_limit,
            use_local_cache=self.settings.client_use_local_cache,
            max_retries=self.settings.client_max_retries,
            retry_base_seconds=self.settings.client_retry_base_seconds,
            retry_max_seconds=self.settings.client_retry_max_seconds,
            verify=self.settings.client_verify,
        )

        if self.settings.scopes is None:
            self.scopes = [Scope()]
        else:
            self.scopes = self.settings.scopes

        self.shared_basedir = Path(self.settings.shared_basedir).absolute()
        self.shared_basedir.mkdir(exist_ok=True)
        self.keep_shared = self.settings.keep_shared

        self.scratch_basedir = Path(self.settings.scratch_basedir).absolute()
        self.scratch_basedir.mkdir(exist_ok=True)
        self.keep_scratch = self.settings.keep_scratch

        self.scheduler = sched.scheduler(time.monotonic, time.sleep)

        self.compute_service_id = ComputeServiceID.new_from_name(self.name)

        self.int_sleep = InterruptableSleep()

        self._stop = False

        # logging
        extra = {"compute_service_id": str(self.compute_service_id)}
        logger = logging.getLogger("AlchemiscaleSynchronousComputeService")
        logger.setLevel(self.settings.loglevel)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_service_id)s] [%(levelname)s] %(message)s"
        )
        formatter.converter = time.gmtime  # use utc time for logging timestamps

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if self.settings.logfile is not None:
            fh = logging.FileHandler(self.settings.logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logging.LoggerAdapter(logger, extra)

    def _register(self):
        """Register this compute service with the compute API."""
        self.client.register(self.compute_service_id, self.settings.compute_manager_id)

    def _deregister(self):
        """Deregister this compute service with the compute API."""
        self.client.deregister(self.compute_service_id)

    def beat(self):
        """Deliver a heartbeat to the compute API, indicating this service is still alive."""
        self.client.heartbeat(self.compute_service_id)
        self.logger.debug("Updated heartbeat")

    def heartbeat(self):
        """Start up the heartbeat, sleeping for `self.heartbeat_interval`"""
        while True:
            if self._stop:
                break
            self.beat()
            time.sleep(self.heartbeat_interval)

    def claim_tasks(self, count=1) -> list[ScopedKey | None]:
        """Get a Task to execute from compute API.

        Returns `None` if no Task was available matching service configuration.

        Parameters
        ----------
        count
            The maximum number of Tasks to claim.
        """

        tasks = self.client.claim_tasks(
            scopes=self.scopes,
            compute_service_id=self.compute_service_id,
            count=count,
            protocols=self.settings.protocols,
        )

        return tasks

    def task_to_protocoldag(
        self, task: ScopedKey
    ) -> tuple[ProtocolDAG, Transformation, ProtocolDAGResult | None]:
        """Given a Task, produce a corresponding ProtocolDAG that can be executed.

        Also gives the Transformation that this ProtocolDAG corresponds to.
        If the Task extends another Task, then the ProtocolDAGResult for that
        other Task is also given; otherwise `None` given.

        """

        (
            transformation,
            extends_protocoldagresult,
        ) = self.client.retrieve_task_transformation(task)

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
        sk: ScopedKey = self.client.set_task_result(
            task, protocoldagresult, self.compute_service_id
        )

        return sk

    def execute(self, task: ScopedKey) -> ScopedKey:
        """Executes given Task.

        Returns ScopedKey of ProtocolDAGResultRef following push to database.

        """
        # obtain a ProtocolDAG from the task
        self.logger.info("Creating ProtocolDAG from '%s'...", task)
        protocoldag, transformation, extends = self.task_to_protocoldag(task)
        self.logger.info(
            "Created '%s' from '%s' performing '%s'",
            protocoldag,
            task,
            transformation.protocol,
        )

        # execute the task; this looks the same whether the ProtocolDAG is a
        # success or failure

        shared = self.shared_basedir / str(protocoldag.key)
        shared.mkdir()
        scratch = self.scratch_basedir / str(protocoldag.key)
        scratch.mkdir()

        self.logger.info("Executing '%s'...", protocoldag)
        try:
            protocoldagresult = execute_DAG(
                protocoldag,
                shared_basedir=shared,
                scratch_basedir=scratch,
                keep_scratch=self.keep_scratch,
                raise_error=False,
                n_retries=self.settings.n_retries,
            )
        finally:
            if not self.keep_shared:
                shutil.rmtree(shared)

            if not self.keep_scratch:
                shutil.rmtree(scratch)

        if protocoldagresult.ok():
            self.logger.info("'%s' -> '%s' : SUCCESS", protocoldag, protocoldagresult)
        else:
            for failure in protocoldagresult.protocol_unit_failures:
                self.logger.info(
                    "'%s' -> '%s' : FAILURE :: '%s' : %s",
                    protocoldag,
                    protocoldagresult,
                    failure,
                    failure.exception,
                )

        # push the result (or failure) back to the compute API
        result_sk = self.push_result(task, protocoldagresult)
        self.logger.info("Pushed result `%s'", protocoldagresult)

        return result_sk

    def _check_max_tasks(self, max_tasks):
        if max_tasks is not None:
            if self._tasks_counter >= max_tasks:
                self.logger.info(
                    "Performed %s tasks; at or beyond max tasks = %s",
                    self._tasks_counter,
                    max_tasks,
                )
                raise KeyboardInterrupt

    def _check_max_time(self, max_time):
        if max_time is not None:
            run_time = time.time() - self._start_time
            if run_time >= max_time:
                self.logger.info(
                    "Ran for %s seconds; at or beyond max time = %s seconds",
                    run_time,
                    max_time,
                )
                raise KeyboardInterrupt

    def cycle(self, max_tasks: int | None = None, max_time: int | None = None):
        self._check_max_tasks(max_tasks)
        self._check_max_time(max_time)

        # claim tasks from the compute API
        self.logger.info("Claiming tasks")
        tasks: list[ScopedKey] | None = self.claim_tasks(count=self.claim_limit)

        if tasks is None:
            self.logger.info("No tasks claimed. Compute API denied request.")
            time.sleep(self.deep_sleep_interval)
            return

        self.logger.info("Claimed %d tasks", len([t for t in tasks if t is not None]))

        # if no tasks claimed, sleep
        if all([task is None for task in tasks]):
            self.logger.info(
                "No tasks claimed; sleeping for %d seconds", self.sleep_interval
            )
            time.sleep(self.sleep_interval)
            return

        # otherwise, process tasks
        self.logger.info("Executing tasks...")
        for task in tasks:
            if task is None:
                continue

            # execute each task
            self.logger.info("Executing task '%s'...", task)
            self.execute(task)
            self.logger.info("Finished task '%s'", task)

            if max_tasks is not None:
                self._tasks_counter += 1

            # stop checks
            self._check_max_tasks(max_tasks)
            self._check_max_time(max_time)

        self._check_max_tasks(max_tasks)
        self._check_max_time(max_time)

    def start(self, max_tasks: int | None = None, max_time: int | None = None):
        """Start the service.

        Limits to the maximum number of executed tasks or seconds to run for
        can be set. The first maximum to be hit will trigger the service to
        exit.

        Parameters
        ----------
        max_tasks
            Max number of Tasks to execute before exiting.
            If `None`, the service will have no task limit.
        max_time
            Max number of seconds to run before exiting.
            If `None`, the service will have no time limit.

        """
        self._stop = False

        # add ComputeServiceRegistration
        self.logger.info("Starting up service '%s'", self.name)
        self._register()
        self.logger.info(
            "Registered service with registration '%s'", str(self.compute_service_id)
        )

        # start up heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.heartbeat_thread.start()

        # stop conditions will use these
        self._tasks_counter = 0
        self._start_time = time.time()

        try:
            self.logger.info("Starting main loop")
            while not self._stop:
                # check that heartbeat is still alive; if not, resurrect it
                if not self.heartbeat_thread.is_alive():
                    self.heartbeat_thread = threading.Thread(
                        target=self.heartbeat, daemon=True
                    )
                    self.heartbeat_thread.start()

                # perform main loop cycle
                self.cycle(max_tasks, max_time)

                # force a garbage collection to avoid consuming too much memory
                gc.collect()
        except KeyboardInterrupt:
            self.logger.info("Caught SIGINT/Keyboard interrupt.")
        except SleepInterrupted:
            self.logger.info("Service stopping.")
        finally:
            # remove ComputeServiceRegistration, drop all claims
            self._deregister()
            self.logger.info(
                "Deregistered service with registration '%s'",
                str(self.compute_service_id),
            )

    def stop(self):
        self.int_sleep.interrupt()
        self._stop = True


from multiprocessing import Process, Queue, Lock

type TaskKey = ScopedKey
type NodeKey = tuple[
    TaskKey | None,
    ProtocolUnit | str,
]  # None covers root node condition


class JailedKeyError(Exception):
    pass


class Executor(Process):

    key: NodeKey
    queue: Queue
    lock: Lock
    unit_context: Context
    inputs: dict
    n_retries: int

    def __init__(self, key, queue, lock, context, inputs, n_retries):
        super().__init__()
        self._key = key
        self._queue = queue
        self._lock = lock
        self._unit_context = context
        self._inputs = inputs
        assert n_retries >= 0
        self._n_retries = n_retries

    def run(self):
        attempt = 0
        while attempt <= self._n_retries:
            shared_dir = self._unit_context.shared / str(attempt)
            scratch_dir = self._unit_context.scratch / str(attempt)
            attempt_context = Context(shared=shared_dir, scratch=scratch_dir)
            attempt_context.shared.mkdir()
            attempt_context.scratch.mkdir()
            result = self.execute_unit(attempt_context)
            if result.ok():
                break
            attempt = attempt + 1
        self.put_result(result)

    @property
    def unit(self) -> ProtocolUnit:
        return self._key[1]

    @property
    def key(self) -> NodeKey:
        return self._key

    @property
    def queue(self) -> Queue:
        return self._queue

    @property
    def lock(self) -> Lock:
        return self._lock

    @property
    def inputs(self) -> dict:
        return self._inputs

    @classmethod
    def from_key(
            cls, key: NodeKey, queue: Queue, lock: Lock, context: Context, inputs: dict, n_retries: int
    ):
        return cls(key=key, queue=queue, lock=lock, context=context, inputs=inputs, n_retries=n_retries)

    def put_result(self, result: ProtocolUnitResult):
        """Acquire lock, push key and result into the queue, release lock."""
        with self.lock:
            self.queue.put((self._key, result))

    def execute_unit(self, context) -> ProtocolUnitResult | ProtocolUnitFailure:
        # this method assumes the context is in place and will be removed correctly
        return self.unit.execute(context=context, **self._inputs)


class ExecutorStack:

    stack_size: int
    stack: list[Executor]
    jail: dict[NodeKey, set[NodeKey]]
    queue: Queue
    lock: Lock

    def __init__(self, stack_size: int):
        self._stack = []
        self._stack_size = stack_size
        self._jail = {}
        self._queue = Queue()
        self._lock = Lock()

    @property
    def stack(self) -> list[Executor]:
        return self._stack

    @property
    def stack_size(self) -> int:
        return self._stack_size

    @property
    def jail(self) -> dict[NodeKey, set[NodeKey]]:
        return self._jail

    @property
    def lock(self) -> Lock:
        return self._lock

    @property
    def queue(self) -> Queue:
        return self._queue

    def terminate_all(self):
        with self.lock:
            for proc in self.stack:
                proc.terminate()

    def terminate_task(self, task_key: TaskKey):
        with self.lock:
            to_remove = set()
            for proc in self.stack:
                _task_key, _ = proc.key
                if task_key == _task_key:
                    to_remove.add(proc)
            for proc in to_remove:
                proc.terminate()
                self._stack.remove(proc)

    def push(self, node: NodeKey, context: Context, inputs: dict, n_retries):
        with self.lock:
            if node in self._jail.keys():
                raise JailedKeyError(node)

            executor = Executor.from_key(node, self.queue, self.lock, context, inputs, n_retries)
            self._stack.append(executor)
            self._stack[-1].start()


    def pop(self):
        """Remove last process in the stack. This also clears the node from the jail."""
        if self._stack_size == 0:
            raise IndexError("pop from empty stack")
        popped_executor = self._stack.pop()
        for key in self._jail.keys():
            self._jail[key] -= popped_executor.key
            if not self._jail[key]:
                self._jail.pop(key)

        return popped_executor

    def get_result(self) -> tuple[NodeKey, ProtocolUnitResult | ProtocolUnitFailure] | None:
        if self.queue.qsize():
            with self.lock:
                # since qsize is not always reliable, we tentatively
                # accept there might be results
                try:
                    res = self.queue.get_nowait()
                except queue.Empty:
                    return None
                node_key, _ = res
                self.remove_by_node_key(node_key)
                return res

    def remove_by_node_key(self, node_key: NodeKey):
        to_remove = None
        for proc in self.stack:
            if proc.key == node_key:
                to_remove = proc
                break

        if to_remove:
            self._stack.remove(proc)

    def _get_statuses(self) -> tuple[set[Executor], set[Executor]]:
        running = set()
        terminated = set()
        for proc in self.stack:
            if proc.is_alive():
                running.add(proc)
            else:
                terminated.add(proc)
        return running, terminated


from dataclasses import dataclass


@dataclass
class TaskData:
    protocol_dag: ProtocolDAG
    results: dict[GufeKey, ProtocolUnitResult]
    context: Context

    def to_ProtocolDAGResult(self) -> ProtocolDAGResult:
        return ProtocolDAGResult(
            name=self.protocol_dag.name,
            protocol_units=self.protocol_dag.protocol_units,
            protocol_unit_results=list(self.results.values()),
            transformation_key=self.protocol_dag.transformation_key,
            extends_key=self.protocol_dag.extends_key,
        )


class AsynchronousComputeService(SynchronousComputeService):
    """Asynchronous compute service.

    This service can be used in production cases, though it does not make use
    of Folding@Home.

    """

    _dag_tree: nx.DiGraph
    _executor_stack: ExecutorStack
    _task_data: dict[TaskKey, TaskData]

    def __init__(self, settings: ComputeServiceSettings):

        self._task_data = dict()
        self._initialize_dag_tree()

        self.settings = settings

        self.api_url = self.settings.api_url
        self.name = self.settings.name
        self.sleep_interval = self.settings.sleep_interval
        self.heartbeat_interval = self.settings.heartbeat_interval
        self.claim_limit = self.settings.claim_limit

        self.scheduler = sched.scheduler(time.monotonic, time.sleep)

        self.client = AlchemiscaleComputeClient(
            self.settings.api_url,
            self.settings.identifier,
            self.settings.key,
            cache_directory=self.settings.client_cache_directory,
            cache_size_limit=self.settings.client_cache_size_limit,
            use_local_cache=self.settings.client_use_local_cache,
            max_retries=self.settings.client_max_retries,
            retry_base_seconds=self.settings.client_retry_base_seconds,
            retry_max_seconds=self.settings.client_retry_max_seconds,
            verify=self.settings.client_verify,
        )

        self._stop = False

        self.scopes = self.settings.scopes or [Scope()]
        self.shared_basedir = Path(self.settings.shared_basedir).absolute()
        self.shared_basedir.mkdir(exist_ok=True)
        self.keep_shared = self.settings.keep_shared

        self.scratch_basedir = Path(self.settings.scratch_basedir).absolute()
        self.scratch_basedir.mkdir(exist_ok=True)
        self.keep_scratch = self.settings.keep_scratch

        self.compute_service_id = ComputeServiceID.new_from_name(self.name)
        self._stop = False

    def _initialize_dag_tree(self):
        self._dag_tree = nx.DiGraph()
        root_node: NodeKey = (None, "ROOT")
        self._dag_tree.add_node(root_node)

    def has_tasks(self) -> bool:
        return bool(self._task_data)

    def cycle(self, max_tasks, max_time) -> bool:
        # TODO check max_tasks and max_time

        match (self._stop, self.has_tasks()):
            # should stop, but has remaining tasks, that COULD be finished
            # 1. terminate all processes
            # 2. Process remaining results in queue
            # 3. Push anything that is completed or failed
            # 4. Remove all tasks
            # 5. break from main loop
            case (True, True):
                raise NotImplementedError
                # should terminate all tasks
                mock_service.terminate_all()
                mock_service.process_results()
                mock_service.remove_all()
                return False
            case (True, False):
                return False
            case (False, True):
                pass
            case (False, False):
                tasks: list[ScopedKey] | None = self.claim_tasks(count=self.claim_limit)
                should_stop = True

                for task in tasks:
                    if task is None:
                        pass
                    else:
                        should_stop = False
                        self.add_task(*task)
                if should_stop:
                    self.stop()
            case _:
                raise RuntimeError("Should never hit this")



        # check for terminating nodes
        for terminating_nodes in self.next_terminating_nodes():
            task_scoped_key, _ = terminating_nodes
            pdr = self._consume_results(task_scoped_key)
            self.push_result(task_scoped_key, pdr)

        # collect unit results
        failed_tasks = set()
        while result := self._executor_stack.get_result():
            node_key, res = result
            task_scoped_key, pu = node_key
            self._task_data[task_scoped_key].results[pu.key] = res
            unit_context = Context(
                shared=self._task_data[task_scoped_key].context.shared / f"{pu.key}",
                scratch=self._task_data[task_scoped_key].context.scratch / f"{pu.key}",
            )

            match res:
                case ProtocolUnitFailure():
                    self._executor_stack.terminate_task(task_scoped_key)
                    failed_tasks.add(task_scoped_key)

                case ProtocolUnitResult():
                    self._dag_tree.remove_node(node_key)
            if not self.keep_scratch:
                shutil.rmtree(unit_context.scratch)

        for failed_task in failed_tasks:
            pdr = self._consume_results(failed_task)
            self.push_result(task_scoped_key, pdr)

        # only submit enough tasks to fill the stack
        n = self._executor_stack._stack_size - len(
            self._executor_stack._stack
        )
        for key in tuple(self.next())[:n]:
            tsk, unit = key

            # TODO `next` should not return TERM or ROOT nodes
            if unit in ("TERM", "ROOT"):
                continue

            task_data = self._task_data[tsk]

            inputs = _pu_to_pur(unit.inputs, task_data.results)
            unit_scratch_dir = task_data.context.scratch / f"{str(unit.key)}"
            unit_shared_dir = task_data.context.shared / f"{str(unit.key)}"
            unit_scratch_dir.mkdir()
            unit_shared_dir.mkdir()
            context = Context(scratch=unit_scratch_dir, shared=unit_shared_dir)
            self._executor_stack.push(key, context, inputs, self.n_retries)
        return True

    def available_units(self) -> set[NodeKey]:
        available = set()
        for node, degree in self._dag_tree.out_degree():
            if degree == 0:
                available.add(node)

        return available

    def next(self) -> set[NodeKey]:
        running, terminated = self._executor_stack._get_statuses()
        running = {r.key for r in running}
        terminated = {t.key for t in terminated}
        return self.available_units() - (running | terminated)

    def next_terminating_nodes(self) -> set[NodeKey]:
        completed = {node for node in self.available_units() if node[1] == "TERM"}
        return completed

    def add_task(self, task_scoped_key: ScopedKey):
        protocol_dag, _, _ = self.task_to_protocoldag(task_scoped_key)
        self.graft_dag(task_scoped_key, protocol_dag)

    def graft_dag(self, task_scoped_key: ScopedKey, dag):
        """Add a ``Task`` to the ``AsynchronousComputeService`` internal DAG."""

        def node_transformation(node: ProtocolUnit) -> (TaskKey, ProtocolUnit):
            nonlocal task_scoped_key
            return (task_scoped_key, node)

        tagged_dag = nx.DiGraph()
        terminating = node_transformation("TERM")
        tagged_dag.add_node(terminating)

        for child, parent in dag.graph.edges:
            tagged_child = node_transformation(child)
            tagged_parent = node_transformation(parent)
            tagged_dag.add_edge(tagged_child, tagged_parent)

        for node, in_degree in dag.graph.in_degree:
            if in_degree == 0:
                tagged_dag.add_edge(terminating, node_transformation(node))

        self._dag_tree.add_edges_from(tagged_dag.edges)
        self._dag_tree.add_edge((None, "ROOT"), terminating)

        context = Context(
            scratch=self.scratch_basedir / str(task_scoped_key),
            shared=self.shared_basedir / str(task_scoped_key),
        )
        context.scratch.mkdir(exist_ok=True)
        context.shared.mkdir(exist_ok=True)
        self._task_data[task_scoped_key] = TaskData(
            results={}, context=context, protocol_dag=dag
        )

    def remove_task(self, task_scoped_key):
        # TODO: check executor stack
        # avoid deleting the root node
        if task_scoped_key is None:
            raise ValueError()

        for node in tuple(self._dag_tree.nodes):
            key, _ = node
            if key == task_scoped_key:
                self._dag_tree.remove_node(node)

        context = self._task_data[task_scoped_key].context

        if not self.keep_shared:
            shutil.rmtree(context.shared)
        if not self.keep_scratch:
            shutil.rmtree(context.scratch)

    def _consume_results(self, task_scoped_key) -> ProtocolDAGResult:
        self.remove_task(task_scoped_key)
        data = self._task_data.pop(task_scoped_key)
        pdr = data.to_ProtocolDAGResult()
        return pdr


    def stop(self):
        self._stop = True
