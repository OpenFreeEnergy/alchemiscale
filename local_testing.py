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
from gufe.protocols.protocoldag import _pu_to_pur, ProtocolDAGResult
from gufe.tokenization import GufeKey
from gufe.tests.test_protocol import BrokenProtocol

import networkx as nx

import service
import utils

SCRATCH_DIR = Path("./acs_testing/scratch")
SHARED_DIR = Path("./acs_testing/shared")
STACKSIZE = 10
N_RETRIES = 2
MAX_TASKS = 3
MAX_TIME = None
KEEP_SHARED = False
KEEP_SCRATCH = False
CLAIM_LIMIT = 3

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

    print("Creating network")
    tyk2 = create_tyk2()

    task_generator = (
        (utils.new_task_scoped_key(), transformation)
        for transformation in tyk2.edges
    )

    mock_service = service.MockService(SCRATCH_DIR, SHARED_DIR, STACKSIZE, KEEP_SCRATCH, KEEP_SHARED, N_RETRIES, CLAIM_LIMIT, task_generator)


    # for local testing, override how the service pushes results
    pdrs = []
    def push_result(task_scoped_key: NodeKey, pdr: ProtocolDAGResult):
        _ = task_scoped_key
        pdrs.append(pdr)
    mock_service.push_result = push_result

    while mock_service.cycle(MAX_TASKS, MAX_TIME):
        pass

    print(pdrs)
