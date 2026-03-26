import logging
import time

from alchemiscale.compute.service import AsynchronousComputeService, ExecutorStack, InterruptableSleep
from alchemiscale.storage.models import ComputeServiceID

class MockService(AsynchronousComputeService):

    def __init__(self, scratch_basedir, shared_basedir, stack_size, keep_scratch, keep_shared, n_retries, claim_limit, task_generator):
        self._initialize_dag_tree()
        self._task_data = dict()
        self._executor_stack = ExecutorStack(stack_size)

        self.scratch_basedir = scratch_basedir
        self.shared_basedir = shared_basedir
        self.keep_scratch = keep_scratch
        self.keep_shared = keep_shared
        self.n_retries = n_retries
        self.claim_limit = claim_limit
        self.task_generator = task_generator
        self.pdrs = []
        self.tasks_claimed = 0
        self.tasks_finished = 0

        self.int_sleep = InterruptableSleep()

        self.name = "MockService"
        self.compute_service_id = ComputeServiceID.new_from_name(self.name)

        # logging shim
        extra = {"compute_service_id": "fakeid"}
        logger = logging.getLogger("AlchemiscaleSynchronousComputeService")
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(compute_service_id)s] [%(levelname)s] %(message)s"
        )
        formatter.converter = time.gmtime  # use utc time for logging timestamps

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        self.logger = logging.LoggerAdapter(logger, extra)

    def add_task(self, task_scoped_key, transformation):
        protocol_dag = transformation.create()
        self.graft_dag(task_scoped_key, protocol_dag)

    def _register(self):
        self.logger.info("Fake register")

    def _deregister(self):
        self.logger.info("Fake deregister")

    def heartbeat(self):
        pass

    def claim_tasks(self, count=1):
        claimed_tasks = []
        remaining = count
        if remaining == 0:
            return [None] * count
        for task in self.task_generator:
            claimed_tasks.append(task)
            remaining = remaining - 1
            if remaining == 0:
                return claimed_tasks
        return claimed_tasks + [None] * remaining

    def push_result(self, task_scoped_key, pdr):
        _ = task_scoped_key
        self.logger.info(f"Pushing {pdr}")
        self.pdrs.append(pdr)
        self.tasks_finished = self.tasks_finished + 1
