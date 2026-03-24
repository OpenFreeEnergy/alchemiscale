from alchemiscale.compute.service import AsynchronousComputeService, ExecutorStack

class MockService(AsynchronousComputeService):

    def __init__(self, scratch_basedir, shared_basedir, stack_size, keep_scratch, keep_shared):
        self._initialize_dag_tree()
        self._task_data = dict()
        self._executor_stack = ExecutorStack(stack_size)

        self.scratch_basedir = scratch_basedir
        self.shared_basedir = shared_basedir
        self.keep_scratch = keep_scratch
        self.keep_shared = keep_shared

    def add_task(self, task_scoped_key, transformation):
        protocol_dag = transformation.create()
        self.graft_dag(task_scoped_key, protocol_dag)
