from alchemiscale.compute.service import AsynchronousComputeService, ExecutorStack

class MockService(AsynchronousComputeService):

    def __init__(self, scratch_basedir, shared_basedir, stack_size):
        self._initialize_dag_tree()
        self._task_data = dict()
        self._executor_stack = ExecutorStack(stack_size)

        self.scratch_basedir = scratch_basedir
        self.shared_basedir = shared_basedir
