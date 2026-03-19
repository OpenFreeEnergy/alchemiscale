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

SCRATCH_DIR = Path("./acs_testing/scratch")
SHARED_DIR = Path("./acs_testing/shared")

SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR.mkdir(parents=True, exist_ok=True)

STACKSIZE = 2


def create_tyk2():
    try:
        return AlchemicalNetwork.from_json(file="network.json")
    except FileNotFoundError:
        from network import network_tyk2

        _tyk2 = network_tyk2()
        _tyk2.to_json(file="network.json")
        return _tyk2


if __name__ == "__main__":

    mock = service.MockService(SCRATCH_DIR, SHARED_DIR, STACKSIZE)

    task_key = GufeKey("FakeKey-123456")
    task_scoped_key = ScopedKey(
        gufe_key=task_key, org="MockOrg", campaign="MockCampaign", project="MockProject"
    )

    print("Creating network")
    tyk2 = create_tyk2()
    protocol_dag = list(tyk2.edges)[0].create()

    mock.graft_dag(task_scoped_key, protocol_dag)
    exec_stack = mock._executor_stack
    print("Starting unit loop")
    _task_data = mock._task_data[task_scoped_key]
    for unit in _task_data.protocol_dag.protocol_units:
        inputs = _pu_to_pur(unit.inputs, _task_data.results)
        key: NodeKey = (task_key, unit)

        unit_scratch_dir = _task_data.context.scratch / f"{str(unit.key)}"
        unit_shared_dir = _task_data.context.shared / f"{str(unit.key)}"

        unit_scratch_dir.mkdir()
        unit_shared_dir.mkdir()

        context = Context(scratch=unit_scratch_dir, shared=unit_shared_dir)
        exec_stack.push(key, context, inputs)

        (task_key, pu), res = exec_stack.queue.get()
        if not res.ok():
            raise RuntimeError
        _task_data.results[pu.key] = res

        # clean up scratch (later stderr, stdout)
        shutil.rmtree(unit_scratch_dir)

    mock.remove_task(task_scoped_key)

    pdr = _task_data.to_ProtocolDAGResult()
    print(pdr, pdr.ok())
