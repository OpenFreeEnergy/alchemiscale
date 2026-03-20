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
from gufe.protocols.protocolunit import Context
from gufe.protocols.protocoldag import _pu_to_pur, ProtocolDAGResult
from gufe.tests.test_protocol import DummyProtocol
from gufe.tokenization import GufeKey

import networkx as nx

import service
import utils

SCRATCH_DIR = Path("./acs_testing/scratch")
SHARED_DIR = Path("./acs_testing/shared")

SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR.mkdir(parents=True, exist_ok=True)

STACKSIZE = 5


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

    mock_service = service.MockService(SCRATCH_DIR, SHARED_DIR, STACKSIZE)

    with utils.timer(wrap=True):
        print("Creating network")
        tyk2 = create_tyk2()

    transformations = tuple(tyk2.edges)
    tasks = tuple(
        (utils.new_task_scoped_key(), transformation)
        for transformation in transformations
    )

    for tsk, trans in tasks[:3]:
        mock_service.add_task(tsk, trans)

    # collect for final inspection
    pdrs = []
    while mock_service._task_data:
        # only submit enough tasks to fill the stack
        n = mock_service._executor_stack._stack_size - len(
            mock_service._executor_stack._stack
        )
        for key in tuple(mock_service.next())[:n]:
            tsk, unit = key
            task_data = mock_service._task_data[tsk]

            # TODO: if we hit the terminal node, clean up, this should
            # live outside of this loop
            if unit == "TERM":
                pdr = task_data.to_ProtocolDAGResult()
                print(f"Collected output: {pdr}")
                data = mock_service._task_data.pop(tsk)
                mock_service._dag_tree.remove_node(key)
                shutil.rmtree(data.context.scratch)
                shutil.rmtree(data.context.shared)
                pdrs.append(pdr)
                continue

            inputs = _pu_to_pur(unit.inputs, mock_service._task_data[tsk].results)
            unit_scratch_dir = task_data.context.scratch / f"{str(unit.key)}"
            unit_shared_dir = task_data.context.shared / f"{str(unit.key)}"
            unit_scratch_dir.mkdir()
            unit_shared_dir.mkdir()
            context = Context(scratch=unit_scratch_dir, shared=unit_shared_dir)
            mock_service._executor_stack.push(key, context, inputs)

        # collect results
        while result := mock_service._executor_stack.get_result():
            node_key, res = result
            task_scoped_key, pu = node_key
            mock_service._task_data[task_scoped_key].results[pu.key] = res
            mock_service._dag_tree.remove_node(node_key)

    print(pdrs)
