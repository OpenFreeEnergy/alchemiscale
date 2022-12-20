"""FahAlchemyComputeAPI

"""


from typing import Any, Dict, List
import os
import json

from fastapi import FastAPI, APIRouter, Body, Depends, HTTPException, status
from gufe.tokenization import GufeTokenizable, JSON_HANDLER

from ..base.api import (
    GufeJSONResponse,
    QueryGUFEHandler,
    scope_params,
    get_token_data_depends,
    get_n4js_depends,
    get_s3os_depends,
    base_router,
    get_cred_entity,
    validate_scopes,
    validate_scopes_query,
)
from ..settings import get_base_api_settings, get_compute_api_settings
from ..storage.statestore import Neo4jStore
from ..storage.objectstore import S3ObjectStore
from ..storage.models import ObjectStoreRef
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedComputeIdentity


# TODO:
# - add periodic removal of task claims from compute services that are no longer alive
#   - can be done with an asyncio.sleeping task added to event loop: https://stackoverflow.com/questions/67154839/fastapi-best-way-to-run-continuous-get-requests-in-the-background
# - on startup,

app = FastAPI(title="FahAlchemyComputeAPI")
app.dependency_overrides[get_base_api_settings] = get_compute_api_settings
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
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):

    # intersect query scopes with accessible scopes in the token
    query_scopes = validate_scopes_query(scope, token)
    taskqueues_handler = QueryGUFEHandler(return_gufe)

    # query each scope
    # loop might be more removable in the future with a Union like operator on scopes
    for single_query_scope in query_scopes:

        # add new task queues
        taskqueues_handler.update_results(
            n4js.query_taskqueues(
                scope=single_query_scope, return_gufe=taskqueues_handler.return_gufe
            )
        )

    return taskqueues_handler.format_return()


# @app.get("/taskqueues/{scoped_key}")
# async def get_taskqueue(scoped_key: str,
#                        *,
#                        n4js: Neo4jStore = Depends(get_n4js_depends)):
#    return


@router.get("/taskqueues/{taskqueue}/tasks")
async def get_taskqueue_tasks():
    return {"message": "nothing yet"}


@router.post("/taskqueues/{taskqueue_scoped_key}/claim")
async def claim_taskqueue_tasks(
    taskqueue_scoped_key,
    *,
    claimant: str = Body(),
    count: int = Body(),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(taskqueue_scoped_key)
    validate_scopes(sk.scope, token)
    tasks = n4js.claim_taskqueue_tasks(
        taskqueue=taskqueue_scoped_key, claimant=claimant, count=count
    )

    return [str(t) if t is not None else None for t in tasks]


@router.get("/tasks/{task_scoped_key}/transformation", response_class=GufeJSONResponse)
async def get_task_transformation(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)
    transformation, protocoldagresult = n4js.get_task_transformation(
        task=task_scoped_key
    )

    return (
        transformation.to_dict(),
        protocoldagresult.to_dict() if protocoldagresult is not None else None,
    )


@router.post("/tasks/{task_scoped_key}/result", response_model=ScopedKey)
def set_task_result(
    task_scoped_key,
    *,
    protocoldagresult: str = Body(embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    pdr = json.loads(protocoldagresult, cls=JSON_HANDLER.decoder)
    pdr = GufeTokenizable.from_dict(pdr)

    # push the ProtocolDAGResult to the object store
    objectstoreref: ObjectStoreRef = s3os.push_protocoldagresult(pdr)

    # push the reference to the state store
    result_sk: ScopedKey = n4js.set_task_result(
        task=task_sk, protocoldagresult=objectstoreref
    )

    return result_sk


@router.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}


### add router

app.include_router(router)
