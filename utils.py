from contextlib import contextmanager
from time import time
from uuid import uuid4

from alchemiscale.models import ScopedKey
from gufe.tokenization import GufeKey

@contextmanager
def timer(*args, **kwargs):

    wrap = kwargs.get("wrap")

    if wrap:
        print("="*20)

    start = time()
    yield
    elapsed = time() - start
    if wrap:
        print("-"*20)

    print(f"Time spent: {elapsed}")

    if wrap:
        print("="*20)

def new_task_scoped_key():
    task_key = GufeKey(f"FakeKey-{uuid4().hex}")
    task_scoped_key = ScopedKey(
        gufe_key=task_key, org="MockOrg", campaign="MockCampaign", project="MockProject"
    )
    return task_scoped_key
