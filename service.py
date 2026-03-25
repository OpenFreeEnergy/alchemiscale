from alchemiscale.compute.service import AsynchronousComputeService, ExecutorStack

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

        self._stop = False

    def add_task(self, task_scoped_key, transformation):
        protocol_dag = transformation.create()
        self.graft_dag(task_scoped_key, protocol_dag)

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
        self.pdrs.append(pdr)
        self.tasks_finished = self.tasks_finished + 1
