"""Client for interacting with compute API.

"""

from typing import List

from ..models import Scope
from ..storage.models import TaskQueue, Task


class FahAlchemyComputeClient:
    ...

    def query_taskqueues(self, scope: Scope) -> List[TaskQueue]:
        ...

    def claim_taskqueue_task(self, taskqueue: TaskQueue) -> Task:
        ...
