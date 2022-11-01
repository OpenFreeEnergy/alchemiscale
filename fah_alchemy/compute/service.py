"""FAHAlchemyComputeService


"""

import asyncio
import sched
import time
from pathlib import Path
from threading import Thread

import requests

from gufe.protocols.protocoldag import execute

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
            compute_api_uri: str,
            name: str,
            shared_path: Path,
            sleep_interval: int = 30,
            heartbeat_frequency: int = 30,
        ):
        """

        Parameters
        ----------

        """
        self.compute_api_uri = compute_api_uri
        self.name = name
        self.sleep_interval = sleep_interval
        self.heartbeat_frequency = heartbeat_frequency

        self.shared = shared_path
        self.scheduler = sched.scheduler(time.monotonic, time.sleep)

        self._stop = False


    def heartbeat(self):
        """Deliver a heartbeat to the compute API, indicating this service is still alive.

        """
        ...

    def get_task(self):
        """

        """
        ...

    def push_results(self):
        ...

    def execute(self):
        ...


    def start(self):
        """Start the service; will keep going until told to stop.

        """
        def scheduler_heartbeat():
            self.heartbeat()
            self.scheduler.enter(self.heartbeat_frequency, 1, scheduler_heartbeat)

        self.scheduler.enter(0, 2, scheduler_heartbeat)

        while True:

            if self._stop:
                return

            # get a task from the compute API
            task = self.get_task()

            # obtain a ProtocolDAG from the task
            protocoldag = task.to_protocoldag()

            # execute the task
            if task is not None:
                result = execute(protocoldag, self.shared_path)

            # push the result (or failure) back to the compute API
            self.push_results(task, result)

            time.sleep(self.sleep_interval)




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

    def __init__(
            self,
            compute_api_uri
        ):
        self.scheduler = sched.scheduler(time.monotonic, time.sleep)
        #self.loop = asyncio.get_event_loop()

        self._stop = False


    def get_new_tasks(self):
        ...

    def start(self):
        """Start the service; will keep going until told to stop.

        """
        while True:

            if self._stop:
                return

    def stop(self):
        self._stop = True



class FahAlchemyComputeService(AsynchronousComputeService):
    """Folding@Home-based compute service.

    This service is designed for production use with Folding@Home.

    """
    def __init__(
            self,
            object_store,
            fah_work_server
        ):

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
