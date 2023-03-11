"""
AlchemiscaleComputeAPI --- :mod:`alchemiscale.compute.api`
=======================================================

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
    _check_store_connectivity,
    gufe_to_json,
)
from ..settings import get_base_api_settings, get_compute_api_settings
from ..storage.statestore import Neo4jStore
from ..storage.objectstore import S3ObjectStore
from ..storage.models import ProtocolDAGResultRef, ComputeServiceID, TaskStatusEnum
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import (
    Token,
    TokenData,
    CredentialedComputeIdentity,
)


# TODO:
# - add periodic removal of task claims from compute services that are no longer alive
#   - can be done with an asyncio.sleeping task added to event loop: https://stackoverflow.com/questions/67154839/fastapi-best-way-to-run-continuous-get-requests-in-the-background
# - on startup,

app = FastAPI(title="AlchemiscaleComputeAPI")
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
    return {"api": "AlchemiscaleComputeAPI"}


@router.get("/info")
async def info():
    return {"message": "nothing yet"}


@router.get("/check")
async def check(
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
):
    # check connectivity of storage components
    # if no exception raised, all good
    _check_store_connectivity(n4js, s3os)


@router.get("/identities/{identity_identifier}/scopes")
async def list_scopes(
    *,
    identity_identifier,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[str]:
    scopes = n4js.list_scopes(identity_identifier, CredentialedComputeIdentity)
    return [str(scope) for scope in scopes]


@router.post("/computeservice/{computeservice_identifier}/register")
async def register_computeservice(
    computeservice_identifier,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    n4js.register_computeservice(computeservice_identifier)


@router.post("/computeservice/{computeservice_identifier}/deregister")
async def deregister_computeservice(
    computeservice_identifier,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    n4js.deregister_computeservice(computeservice_identifier)


@router.get("/taskhubs")
async def query_taskhubs(
    *,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    # intersect query scopes with accessible scopes in the token
    query_scopes = validate_scopes_query(scope, token)
    taskhubs_handler = QueryGUFEHandler(return_gufe)

    # query each scope
    # loop might be more removable in the future with a Union like operator on scopes
    for single_query_scope in query_scopes:
        # add new task hubs
        taskhubs_handler.update_results(
            n4js.query_taskhubs(
                scope=single_query_scope, return_gufe=taskhubs_handler.return_gufe
            )
        )

    return taskhubs_handler.format_return()


@router.post("/taskhubs/{taskhub_scoped_key}/claim")
async def claim_taskhub_tasks(
    taskhub_scoped_key,
    *,
    computeserviceid: ComputeServiceID ,
    count: int = Body(),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(taskhub_scoped_key)
    validate_scopes(sk.scope, token)

    tasks = n4js.claim_taskhub_tasks(
        taskhub=taskhub_scoped_key, computeserviceid=computeserviceid, count=count
    )

    return [str(t) if t is not None else None for t in tasks]


@router.get("/tasks/{task_scoped_key}/transformation", response_class=GufeJSONResponse)
async def get_task_transformation(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    transformation, protocoldagresultref = n4js.get_task_transformation(
        task=task_scoped_key
    )

    if protocoldagresultref:
        tf_sk = ScopedKey(gufe_key=transformation.key, **sk.scope.dict())
        pdr_sk = ScopedKey(gufe_key=protocoldagresultref.obj_key, **sk.scope.dict())

        # we keep this as a string to avoid useless deserialization/reserialization here
        pdr: str = s3os.pull_protocoldagresult(pdr_sk, tf_sk, return_as="json", ok=True)
    else:
        pdr = None

    return (gufe_to_json(transformation), pdr)


# TODO: support compression performed client-side
@router.post("/tasks/{task_scoped_key}/results", response_model=ScopedKey)
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
    protocoldagresultref: ProtocolDAGResultRef = s3os.push_protocoldagresult(
        pdr,
        scope=task_sk.scope,
    )

    # push the reference to the state store
    result_sk: ScopedKey = n4js.set_task_result(
        task=task_sk, protocoldagresultref=protocoldagresultref
    )

    # TODO: if success, set task complete, remove from all hubs
    # otherwise, set as errored, leave in hubs

    return result_sk


@router.post("/tasks/{task_scoped_key}/status")
async def set_task_status(
    task_scoped_key,
    status: str = Body(),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    status = TaskStatusEnum(status)
    tasks_statused = n4js.set_task_status([task_sk], status)
    return [str(t) if t is not None else None for t in tasks_statused][0]


@router.get("/tasks/{task_scoped_key}/status")
async def get_task_status(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    status = n4js.get_task_status([task_sk])

    return status[0].value


@router.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}


### add router

app.include_router(router)
