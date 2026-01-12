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
import shutil
import asyncio
import psutil
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from gufe import Transformation
from gufe.protocols.protocoldag import execute_DAG, ProtocolDAG, ProtocolDAGResult

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


@dataclass
class ResourceMetrics:
    """Container for system resource metrics."""

    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_oversaturated(
        self, cpu_threshold: float, memory_threshold: float, gpu_threshold: float
    ) -> bool:
        """Check if any resource exceeds its threshold."""
        if self.cpu_percent > cpu_threshold:
            return True
        if self.memory_percent > memory_threshold:
            return True
        if self.gpu_percent is not None and self.gpu_percent > gpu_threshold:
            return True
        return False

    def is_underutilized(
        self,
        cpu_threshold: float,
        memory_threshold: float,
        gpu_threshold: float,
        margin: float = 20.0,
    ) -> bool:
        """Check if resources are significantly below thresholds."""
        if self.cpu_percent > cpu_threshold - margin:
            return False
        if self.memory_percent > memory_threshold - margin:
            return False
        if self.gpu_percent is not None and self.gpu_percent > gpu_threshold - margin:
            return False
        return True


class ResourceMonitor:
    """Monitor system resources for reactive scheduling.

    GPU monitoring requires nvidia-ml-py (NVIDIA's official Python bindings for NVML).
    Install with: pip install nvidia-ml-py
    Note: The old 'pynvml' package from gpuopenanalytics is deprecated.
    """

    def __init__(self, enable_gpu: bool = False, logger=None):
        self.enable_gpu = enable_gpu
        self.logger = logger
        self._gpu_available = False

        if self.enable_gpu:
            try:
                # Import pynvml from nvidia-ml-py package
                # Note: nvidia-ml-py is the official NVIDIA package that provides the pynvml module
                import pynvml

                pynvml.nvmlInit()
                self._gpu_available = True
                self._pynvml = pynvml
                if self.logger:
                    self.logger.info("GPU monitoring enabled using nvidia-ml-py")
            except (ImportError, Exception) as e:
                if self.logger:
                    self.logger.warning(
                        f"GPU monitoring requested but unavailable: {e}. "
                        f"Install nvidia-ml-py with: pip install nvidia-ml-py. "
                        f"Continuing without GPU monitoring."
                    )

    def get_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        gpu_percent = None
        if self._gpu_available:
            try:
                # Get utilization from first GPU (can be extended for multi-GPU)
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = float(utilization.gpu)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to get GPU metrics: {e}")

        return ResourceMetrics(
            cpu_percent=cpu_percent, memory_percent=memory_percent, gpu_percent=gpu_percent
        )

    def cleanup(self):
        """Clean up GPU monitoring resources."""
        if self._gpu_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


@dataclass
class TaskExecution:
    """Tracks the execution state of a Task."""

    task: ScopedKey
    started_at: datetime
    process: Optional[asyncio.Task] = None
    terminated: bool = False
    retry_after: Optional[datetime] = None


class AsynchronousComputeService(SynchronousComputeService):
    """Asynchronous compute service with reactive scheduling.

    This service executes multiple Tasks concurrently and dynamically adjusts
    parallelism based on system resource utilization. It monitors CPU, memory,
    and optionally GPU usage to maximize throughput while preventing resource
    saturation.

    Key features:
    - Concurrent execution of multiple Tasks/ProtocolDAGs
    - Reactive scheduling that increases parallelism when resources are available
    - Automatic termination of recently-started tasks when resources become oversaturated
    - Backoff mechanism to avoid immediately retrying terminated tasks

    """

    def __init__(self, settings: ComputeServiceSettings):
        """Create an `AsynchronousComputeService` instance."""
        # Initialize parent class
        super().__init__(settings)

        # Async-specific configuration
        self.max_concurrent = self.settings.max_concurrent_tasks
        self.min_concurrent = self.settings.min_concurrent_tasks
        self.cpu_threshold = self.settings.cpu_threshold
        self.memory_threshold = self.settings.memory_threshold
        self.gpu_threshold = self.settings.gpu_threshold
        self.resource_check_interval = self.settings.resource_check_interval
        self.task_retry_backoff = self.settings.task_retry_backoff

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(
            enable_gpu=self.settings.enable_gpu_monitoring, logger=self.logger
        )

        # Task management
        self.running_tasks: dict[str, TaskExecution] = {}
        self.terminated_tasks: dict[ScopedKey, datetime] = {}
        self.current_concurrency = self.min_concurrent

        # Update logger name
        extra = {"compute_service_id": str(self.compute_service_id)}
        logger = logging.getLogger("AlchemiscaleAsynchronousComputeService")
        logger.setLevel(self.settings.loglevel)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_service_id)s] [%(levelname)s] %(message)s"
        )
        formatter.converter = time.gmtime

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if self.settings.logfile is not None:
            fh = logging.FileHandler(self.settings.logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logging.LoggerAdapter(logger, extra)

    async def execute_task_async(self, task: ScopedKey) -> Optional[ScopedKey]:
        """Execute a single task asynchronously.

        This wraps the synchronous execute method to run in an executor.
        """
        try:
            self.logger.info(f"Starting async execution of task '{task}'")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.execute, task)
            self.logger.info(f"Completed async execution of task '{task}'")
            return result
        except asyncio.CancelledError:
            self.logger.warning(f"Task '{task}' was cancelled due to resource constraints")
            raise
        except Exception as e:
            self.logger.error(f"Task '{task}' failed with exception: {e}", exc_info=True)
            return None

    async def monitor_resources(self):
        """Continuously monitor system resources and adjust concurrency."""
        while not self._stop:
            try:
                await asyncio.sleep(self.resource_check_interval)

                metrics = self.resource_monitor.get_metrics()
                self.logger.debug(
                    f"Resource usage - CPU: {metrics.cpu_percent:.1f}%, "
                    f"Memory: {metrics.memory_percent:.1f}%"
                    + (
                        f", GPU: {metrics.gpu_percent:.1f}%"
                        if metrics.gpu_percent is not None
                        else ""
                    )
                )

                # Check if resources are oversaturated
                if metrics.is_oversaturated(
                    self.cpu_threshold, self.memory_threshold, self.gpu_threshold
                ):
                    await self._handle_oversaturation(metrics)
                # Check if we can increase parallelism
                elif metrics.is_underutilized(
                    self.cpu_threshold, self.memory_threshold, self.gpu_threshold
                ):
                    self._increase_concurrency()

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}", exc_info=True)

    async def _handle_oversaturation(self, metrics: ResourceMetrics):
        """Handle oversaturated resources by terminating recent tasks."""
        self.logger.warning(
            f"Resources oversaturated - CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_percent:.1f}%"
            + (
                f", GPU: {metrics.gpu_percent:.1f}%"
                if metrics.gpu_percent is not None
                else ""
            )
        )

        # Find the most recently started task
        if not self.running_tasks:
            # Reduce target concurrency
            self.current_concurrency = max(
                self.min_concurrent, self.current_concurrency - 1
            )
            self.logger.info(
                f"Reduced target concurrency to {self.current_concurrency}"
            )
            return

        # Sort by start time (most recent first)
        sorted_tasks = sorted(
            self.running_tasks.values(), key=lambda x: x.started_at, reverse=True
        )

        # Terminate the most recent task if we're above minimum concurrency
        if len(self.running_tasks) > self.min_concurrent:
            task_exec = sorted_tasks[0]
            if not task_exec.terminated and task_exec.process:
                self.logger.warning(
                    f"Terminating recently started task '{task_exec.task}' due to resource saturation"
                )
                task_exec.process.cancel()
                task_exec.terminated = True

                # Add to terminated tasks with backoff
                retry_time = datetime.now() + timedelta(seconds=self.task_retry_backoff)
                self.terminated_tasks[task_exec.task] = retry_time
                self.logger.info(
                    f"Task '{task_exec.task}' will be eligible for retry after {retry_time}"
                )

        # Also reduce target concurrency
        self.current_concurrency = max(self.min_concurrent, self.current_concurrency - 1)
        self.logger.info(f"Reduced target concurrency to {self.current_concurrency}")

    def _increase_concurrency(self):
        """Increase target concurrency when resources are underutilized."""
        if self.current_concurrency < self.max_concurrent:
            old_concurrency = self.current_concurrency
            self.current_concurrency = min(
                self.max_concurrent, self.current_concurrency + 1
            )
            if old_concurrency != self.current_concurrency:
                self.logger.info(
                    f"Increased target concurrency from {old_concurrency} to {self.current_concurrency}"
                )

    def _can_retry_task(self, task: ScopedKey) -> bool:
        """Check if a terminated task can be retried."""
        if task not in self.terminated_tasks:
            return True

        retry_time = self.terminated_tasks[task]
        if datetime.now() >= retry_time:
            # Remove from terminated tasks
            del self.terminated_tasks[task]
            self.logger.info(f"Task '{task}' is now eligible for retry")
            return True

        return False

    async def cycle_async(self, max_tasks: Optional[int] = None, max_time: Optional[int] = None):
        """Asynchronous version of the main cycle."""
        self._check_max_tasks(max_tasks)
        self._check_max_time(max_time)

        # Clean up completed tasks
        completed_keys = [
            key
            for key, task_exec in self.running_tasks.items()
            if task_exec.process and task_exec.process.done()
        ]
        for key in completed_keys:
            task_exec = self.running_tasks.pop(key)
            if not task_exec.terminated:
                if max_tasks is not None:
                    self._tasks_counter += 1
                self.logger.info(f"Task '{task_exec.task}' completed and removed from running tasks")

        # Claim new tasks if we're below target concurrency
        available_slots = self.current_concurrency - len(self.running_tasks)
        if available_slots > 0:
            claim_count = min(available_slots, self.claim_limit)
            self.logger.info(
                f"Claiming up to {claim_count} tasks (current: {len(self.running_tasks)}, target: {self.current_concurrency})"
            )

            tasks = self.claim_tasks(count=claim_count)

            if tasks is None:
                self.logger.info("No tasks claimed. Compute API denied request.")
                await asyncio.sleep(self.deep_sleep_interval)
                return

            claimed_count = len([t for t in tasks if t is not None])
            if claimed_count > 0:
                self.logger.info(f"Claimed {claimed_count} tasks")

            # Start new tasks
            for task in tasks:
                if task is None:
                    continue

                # Check if task is in backoff period
                if not self._can_retry_task(task):
                    retry_time = self.terminated_tasks[task]
                    wait_seconds = (retry_time - datetime.now()).total_seconds()
                    self.logger.info(
                        f"Skipping task '{task}' - in backoff period (retry in {wait_seconds:.0f}s)"
                    )
                    continue

                # Create and start task execution
                task_id = str(uuid4())
                process = asyncio.create_task(self.execute_task_async(task))

                task_exec = TaskExecution(
                    task=task, started_at=datetime.now(), process=process
                )
                self.running_tasks[task_id] = task_exec
                self.logger.info(f"Started task '{task}' with ID {task_id}")

        # If no tasks running and none available, sleep
        if len(self.running_tasks) == 0:
            self.logger.info(
                f"No tasks running; sleeping for {self.sleep_interval} seconds"
            )
            await asyncio.sleep(self.sleep_interval)

    async def start_async(self, max_tasks: Optional[int] = None, max_time: Optional[int] = None):
        """Start the asynchronous service."""
        self._stop = False

        # Register service
        self.logger.info("Starting up service '%s'", self.name)
        self._register()
        self.logger.info(
            "Registered service with registration '%s'", str(self.compute_service_id)
        )

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.heartbeat_thread.start()

        # Initialize counters
        self._tasks_counter = 0
        self._start_time = time.time()

        try:
            self.logger.info("Starting main async loop")

            # Start resource monitoring task
            monitor_task = asyncio.create_task(self.monitor_resources())

            # Main loop
            while not self._stop:
                # Check that heartbeat is still alive
                if not self.heartbeat_thread.is_alive():
                    self.heartbeat_thread = threading.Thread(
                        target=self.heartbeat, daemon=True
                    )
                    self.heartbeat_thread.start()

                # Perform main loop cycle
                await self.cycle_async(max_tasks, max_time)

                # Force garbage collection
                gc.collect()

            # Cancel monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        except KeyboardInterrupt:
            self.logger.info("Caught SIGINT/Keyboard interrupt.")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        finally:
            # Cancel all running tasks
            for task_exec in self.running_tasks.values():
                if task_exec.process and not task_exec.process.done():
                    task_exec.process.cancel()

            # Wait for tasks to complete cancellation
            if self.running_tasks:
                self.logger.info("Waiting for running tasks to complete cancellation...")
                await asyncio.gather(
                    *[te.process for te in self.running_tasks.values() if te.process],
                    return_exceptions=True,
                )

            # Clean up resources
            self.resource_monitor.cleanup()

            # Deregister service
            self._deregister()
            self.logger.info(
                "Deregistered service with registration '%s'",
                str(self.compute_service_id),
            )

    def start(self, max_tasks: Optional[int] = None, max_time: Optional[int] = None):
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
        # Run the async event loop
        asyncio.run(self.start_async(max_tasks, max_time))

    def stop(self):
        """Stop the service."""
        self._stop = True
