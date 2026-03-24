# Local Variables:
# compile-command: "./env/bin/python local_testing.py"
# python-shell-interpreter: "./env/bin/python"
# End:


from pathlib import Path
import shutil

from alchemiscale.compute.service import (
    Executor,
    ExecutorStack,
    NodeKey,
    TaskKey,
    TaskData,
)
from alchemiscale.models import ScopedKey
from gufe import AlchemicalNetwork
from gufe.protocols.protocolunit import Context, ProtocolUnitResult, ProtocolUnitFailure
from gufe.protocols.protocoldag import _pu_to_pur
from gufe.tokenization import GufeKey
from gufe.tests.test_protocol import BrokenProtocol

import networkx as nx

import service
import utils

SCRATCH_DIR = Path("./acs_testing/scratch")
SHARED_DIR = Path("./acs_testing/shared")
STACKSIZE = 10
N_RETRIES = 2
KEEP_SHARED = False
KEEP_SCRATCH = False

SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR.mkdir(parents=True, exist_ok=True)


def create_tyk2():
    try:
        return AlchemicalNetwork.from_json(file="network.json")
    except FileNotFoundError:
        print("\tCould not load from file, creating new network")
        from network import network_tyk2

        _tyk2 = network_tyk2()
        _tyk2.to_json(file="network.json")
        return _tyk2


if __name__ == "__main__":

    mock_service = service.MockService(SCRATCH_DIR, SHARED_DIR, STACKSIZE, KEEP_SCRATCH, KEEP_SHARED)

    with utils.timer(wrap=True):
        print("Creating network")
        tyk2 = create_tyk2()

    transformations = tuple(tyk2.edges)
    tasks = tuple(
        (utils.new_task_scoped_key(), transformation)
        for transformation in transformations
    )

    for tsk, trans in tasks:
        mock_service.add_task(tsk, trans)

    # collect for final inspection
    pdrs = []
    while mock_service._task_data:
        # ask for terminating nodes
        for completed_node in mock_service.next_terminating_nodes():
            task_scoped_key, _ = completed_node
            pdr = mock_service._consume_results(task_scoped_key)
            pdrs.append(pdr)

        # collect unit results
        failed_tasks = set()
        while result := mock_service._executor_stack.get_result():
            node_key, res = result
            task_scoped_key, pu = node_key
            mock_service._task_data[task_scoped_key].results[pu.key] = res
            unit_context = Context(
                shared=mock_service._task_data[task_scoped_key].context.shared / f"{pu.key}",
                scratch=mock_service._task_data[task_scoped_key].context.scratch / f"{pu.key}",
            )

            match res:
                case ProtocolUnitFailure():
                    mock_service._executor_stack.terminate_task(task_scoped_key)
                    failed_tasks.add(task_scoped_key)

                case ProtocolUnitResult():
                    mock_service._dag_tree.remove_node(node_key)
            if not mock_service.keep_scratch:
                shutil.rmtree(unit_context.scratch)
        for failed_task in failed_tasks:
            pdr = mock_service._consume_results(failed_task)
            pdrs.append(pdr)


        # only submit enough tasks to fill the stack
        n = mock_service._executor_stack._stack_size - len(
            mock_service._executor_stack._stack
        )
        for key in tuple(mock_service.next())[:n]:
            tsk, unit = key

            # TODO `next` should not return TERM or ROOT nodes
            if unit in ("TERM", "ROOT"):
                continue

            task_data = mock_service._task_data[tsk]

            inputs = _pu_to_pur(unit.inputs, task_data.results)
            unit_scratch_dir = task_data.context.scratch / f"{str(unit.key)}"
            unit_shared_dir = task_data.context.shared / f"{str(unit.key)}"
            unit_scratch_dir.mkdir()
            unit_shared_dir.mkdir()
            context = Context(scratch=unit_scratch_dir, shared=unit_shared_dir)
            mock_service._executor_stack.push(key, context, inputs, N_RETRIES)

    print(pdrs)
