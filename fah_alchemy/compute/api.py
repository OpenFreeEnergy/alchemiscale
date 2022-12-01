"""FahAlchemyComputeAPI

"""


from typing import Any, Dict, List
import os
import json

from fastapi import FastAPI, APIRouter, Body, Depends, HTTPException, status
from gufe.tokenization import GufeTokenizable, JSON_HANDLER

from ..base.api import (
    PermissiveJSONResponse,
    scope_params,
    get_token_data_depends,
    base_router,
    get_cred_entity,
)
from ..settings import ComputeAPISettings, get_compute_api_settings, get_jwt_settings
from ..storage.statestore import Neo4jStore, get_n4js
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedComputeIdentity


# TODO:
# - add periodic removal of task claims from compute services that are no longer alive
#   - can be done with an asyncio.sleeping task added to event loop: https://stackoverflow.com/questions/67154839/fastapi-best-way-to-run-continuous-get-requests-in-the-background
# - on startup,

app = FastAPI(title="FahAlchemyComputeAPI")
app.dependency_overrides[get_jwt_settings] = get_compute_api_settings
app.include_router(base_router)


def get_cred_compute():
    return CredentialedComputeIdentity


app.dependency_overrides[get_cred_entity] = get_cred_compute


router = APIRouter(
    dependencies=[Depends(get_token_data_depends)],
)


@app.get("/ping")
async def ping():
    return {"api": "FahAlchemyComputeAPI"}


@router.get("/info")
async def info():
    return {"message": "nothing yet"}


@router.get("/taskqueues")
async def query_taskqueues(
    *,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js),
):
    taskqueues = n4js.query_taskqueues(scope=scope, return_gufe=return_gufe)

    if return_gufe:
        return {str(sk): tq.to_dict() for sk, tq in taskqueues.items()}
    else:
        return [str(sk) for sk in taskqueues]


# @app.get("/taskqueues/{scoped_key}")
# async def get_taskqueue(scoped_key: str,
#                        *,
#                        n4js: Neo4jStore = Depends(get_n4js)):
#    return


@router.get("/taskqueues/{taskqueue}/tasks")
async def get_taskqueue_tasks():
    return {"message": "nothing yet"}


@router.post("/taskqueues/{taskqueue}/claim")
async def claim_taskqueue_tasks(
    taskqueue,
    *,
    claimant: str = Body(),
    count: int = Body(),
    n4js: Neo4jStore = Depends(get_n4js),
):
    tasks = n4js.claim_taskqueue_tasks(
        taskqueue=taskqueue, claimant=claimant, count=count
    )

    return [str(t) if t is not None else None for t in tasks]


@router.get("/tasks/{task}/transformation", response_class=PermissiveJSONResponse)
async def get_task_transformation(
    task,
    *,
    n4js: Neo4jStore = Depends(get_n4js),
):
    transformation, protocoldagresult = n4js.get_task_transformation(task=task)

    return (
        transformation.to_dict(),
        protocoldagresult.to_dict() if protocoldagresult is not None else None,
    )


@router.post("/tasks/{task}/result", response_model=ScopedKey)
def set_task_result(
    task,
    *,
    protocoldagresult: dict = Body(...),
    n4js: Neo4jStore = Depends(get_n4js),
):
    pdrj = json.dumps(protocoldagresult)
    pdr = json.loads(pdrj, cls=JSON_HANDLER.decoder)
    pdr = GufeTokenizable.from_dict(pdr)

    return n4js.set_task_result(task=ScopedKey.from_str(task), protocol_dag_result=pdr)


@router.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}


app.include_router(router)
