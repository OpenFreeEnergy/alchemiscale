"""FAHAlchemyCompute service.

Pulls tasks from storage system and coordinate execution via `dask.distributed`
and FAH.

"""

import asyncio
import sched
import time

from dask import delayed
from distributed import Client


class FahAlchemyComputeServer:
    """Compute server; subscribes to a state store for work, pushes to a
    `distributed.scheduler` and asynchronously handles results.

    """

    def __init__(
            self,
            state_store,
            object_store,
            distributed_scheduler,
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
