"""
AlchemiscaleClientAPI --- :mod:`alchemiscale.interface.api`
===========================================================


"""


from typing import Any, Dict, List, Optional, Union
import os
import json

from fastapi import FastAPI, APIRouter, Body, Depends, HTTPException, status
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.protocols import ProtocolDAGResult
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
from ..settings import get_api_settings
from ..settings import get_base_api_settings, get_api_settings
from ..storage.statestore import Neo4jStore
from ..storage.objectstore import S3ObjectStore
from ..storage.models import ProtocolDAGResultRef, TaskStatusEnum
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedUserIdentity


app = FastAPI(title="AlchemiscaleAPI")
app.dependency_overrides[get_base_api_settings] = get_api_settings
app.include_router(base_router)


def get_cred_user():
    return CredentialedUserIdentity


app.dependency_overrides[get_cred_entity] = get_cred_user

router = APIRouter(
    dependencies=[Depends(get_token_data_depends)],
)


@app.get("/ping")
def ping():
    return {"api": "AlchemiscaleAPI"}


@router.get("/info")
def info():
    return {"message": "nothing yet"}


@router.get("/check")
def check(
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
):
    # check connectivity of storage components
    # if no exception raised, all good
    _check_store_connectivity(n4js, s3os)


@router.get("/identities/{identity_identifier}/scopes")
def list_scopes(
    *,
    identity_identifier,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[str]:
    scopes = n4js.list_scopes(identity_identifier, CredentialedUserIdentity)
    return [str(scope) for scope in scopes]


### inputs


@router.get("/exists/{scoped_key}", response_class=GufeJSONResponse)
def check_existence(
    scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(scoped_key)
    validate_scopes(sk.scope, token)

    return n4js.check_existence(scoped_key=sk)


@router.get("/networks", response_class=GufeJSONResponse)
def query_networks(
    *,
    name: str = None,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    # Intersect query scopes with accessible scopes in the token
    query_scopes = validate_scopes_query(scope, token)
    networks_handler = QueryGUFEHandler(return_gufe)

    # query each scope
    # loop might be removable in the future with a Union like operator on scopes
    for single_query_scope in query_scopes:
        # add new networks
        networks_handler.update_results(
            n4js.query_networks(
                name=name, scope=single_query_scope, return_gufe=return_gufe
            )
        )

    return networks_handler.format_return()


@router.get("/networks/{network_scoped_key}", response_class=GufeJSONResponse)
def get_network(
    network_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    # Get scope from scoped key provided by user, uniquely identifying the network
    sk = ScopedKey.from_str(network_scoped_key)
    validate_scopes(sk.scope, token)

    network = n4js.get_gufe(scoped_key=sk)
    return gufe_to_json(network)


@router.post("/networks", response_model=ScopedKey)
def create_network(
    *,
    network: Dict = Body(...),
    scope: Scope,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    validate_scopes(scope, token)

    an = AlchemicalNetwork.from_dict(network)
    an_sk = n4js.create_network(network=an, scope=scope)

    # create taskhub for this network
    n4js.create_taskhub(an_sk)

    return an_sk


@router.get("/transformations")
def query_transformations():
    return {"message": "nothing yet"}


@router.get(
    "/transformations/{transformation_scoped_key}", response_class=GufeJSONResponse
)
def get_transformation(
    transformation_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    transformation = n4js.get_gufe(scoped_key=sk)
    return gufe_to_json(transformation)


@router.get("/chemicalsystems")
def query_chemicalsystems():
    return {"message": "nothing yet"}


@router.get(
    "/chemicalsystems/{chemicalsystem_scoped_key}", response_class=GufeJSONResponse
)
def get_chemicalsystem(
    chemicalsystem_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(chemicalsystem_scoped_key)
    validate_scopes(sk.scope, token)

    chemicalsystem = n4js.get_gufe(scoped_key=sk)
    return gufe_to_json(chemicalsystem)


### compute


@router.post("/networks/{scoped_key}/strategy")
def set_strategy(scoped_key: str, *, strategy: Dict = Body(...), scope: Scope):
    ...


@router.post("/transformations/{transformation_scoped_key}/tasks")
def create_tasks(
    transformation_scoped_key,
    *,
    extends: Optional[ScopedKey] = None,
    count: int = Body(...),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[str]:
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    task_sks = []
    for i in range(count):
        task_sks.append(
            n4js.create_task(transformation=sk, extends=extends, creator=token.entity)
        )

    return [str(sk) for sk in task_sks]


@router.get("/transformations/{transformation_scoped_key}/tasks")
def get_tasks(
    transformation_scoped_key,
    *,
    extends: str = None,
    return_as: str = "list",
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    task_sks = n4js.get_tasks(sk, extends=extends, return_as=return_as)

    if return_as == "list":
        return [str(sk) for sk in task_sks]
    elif return_as == "graph":
        return {
            str(sk): str(extends) if extends is not None else None
            for sk, extends in task_sks.items()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"`return_as` takes 'list' or 'graph', not '{return_as}'",
        )


@router.post("/networks/{network_scoped_key}/tasks/action")
def action_tasks(
    network_scoped_key,
    *,
    tasks: List[ScopedKey] = Body(embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[Union[str, None]]:
    sk = ScopedKey.from_str(network_scoped_key)
    validate_scopes(sk.scope, token)

    taskhub_sk = n4js.get_taskhub(sk)
    actioned_sks = n4js.action_tasks(tasks, taskhub_sk)

    return [str(sk) if sk is not None else None for sk in actioned_sks]


@router.post("/networks/{network_scoped_key}/tasks/cancel")
def cancel_tasks(
    network_scoped_key,
    *,
    tasks: List[ScopedKey] = Body(embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[Union[str, None]]:
    sk = ScopedKey.from_str(network_scoped_key)
    validate_scopes(sk.scope, token)

    taskhub_sk = n4js.get_taskhub(sk)
    canceled_sks = n4js.cancel_tasks(tasks, taskhub_sk)

    return [str(sk) if sk is not None else None for sk in canceled_sks]


@router.post("/tasks/{task_scoped_key}/status")
def set_task_status(
    task_scoped_key,
    status: str = Body(),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    status = TaskStatusEnum(status)
    if status not in (
        TaskStatusEnum.waiting,
        TaskStatusEnum.invalid,
        TaskStatusEnum.deleted,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot set status to '{status}', must be one of 'waiting', 'invalid', 'deleted'",
        )
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)
    tasks_statused = n4js.set_task_status([task_sk], status)
    return [str(t) if t is not None else None for t in tasks_statused][0]


@router.get("/tasks/{task_scoped_key}/status")
def get_task_status(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    status = n4js.get_task_status([task_sk])

    return status[0].value


### results


@router.get(
    "/transformations/{transformation_scoped_key}/results",
    response_class=GufeJSONResponse,
)
def get_transformation_results(
    transformation_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    # get all ProtocolDAGResultRefs for the given transformation's results
    refs: List[ProtocolDAGResultRef] = n4js.get_transformation_results(sk)

    return [i.to_dict() for i in refs]


@router.get(
    "/transformations/{transformation_scoped_key}/failures",
    response_class=GufeJSONResponse,
)
def get_transformation_failures(
    transformation_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    # get all ProtocolDAGResultRefs for the given transformation's results
    refs: List[ProtocolDAGResultRef] = n4js.get_transformation_failures(sk)

    return [i.to_dict() for i in refs]


@router.get(
    "/transformations/{transformation_scoped_key}/results/{protocoldagresult_scoped_key}",
    response_class=GufeJSONResponse,
)
def get_protocoldagresult(
    protocoldagresult_scoped_key,
    transformation_scoped_key,
    *,
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[str]:
    sk = ScopedKey.from_str(protocoldagresult_scoped_key)
    tf_sk = ScopedKey.from_str(transformation_scoped_key)

    validate_scopes(sk.scope, token)
    validate_scopes(tf_sk.scope, token)

    # we leave each ProtocolDAGResult in string form to avoid
    # deserializing/reserializing here; just passing through to client
    pdr: str = s3os.pull_protocoldagresult(sk, tf_sk, return_as="json", ok=True)

    return [pdr]


@router.get(
    "/transformations/{transformation_scoped_key}/failures/{protocoldagresult_scoped_key}",
    response_class=GufeJSONResponse,
)
def get_protocoldagresult_failure(
    protocoldagresult_scoped_key,
    transformation_scoped_key,
    *,
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> List[str]:
    sk = ScopedKey.from_str(protocoldagresult_scoped_key)
    tf_sk = ScopedKey.from_str(transformation_scoped_key)

    validate_scopes(sk.scope, token)
    validate_scopes(tf_sk.scope, token)

    # we leave each ProtocolDAGResult in string form to avoid
    # deserializing/reserializing here; just passing through to client
    pdr: str = s3os.pull_protocoldagresult(sk, tf_sk, return_as="json", ok=False)

    return [pdr]


@router.get("/tasks/{task_scoped_key}/transformation", response_class=GufeJSONResponse)
def get_task_transformation(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    transformation: ScopedKey

    transformation, protocoldagresultref = n4js.get_task_transformation(
        task=task_scoped_key,
        return_gufe=False,
    )

    return str(transformation)


@router.get(
    "/tasks/{task_scoped_key}/results",
    response_class=GufeJSONResponse,
)
def get_task_results(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    # get all ProtocolDAGResultRefs for the given transformation's results
    refs: List[ProtocolDAGResultRef] = n4js.get_task_results(sk)

    return [i.to_dict() for i in refs]


@router.get(
    "/tasks/{task_scoped_key}/failures",
    response_class=GufeJSONResponse,
)
def get_task_failures(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    # get all ProtocolDAGResultRefs for the given transformation's results
    refs: List[ProtocolDAGResultRef] = n4js.get_task_failures(sk)

    return [i.to_dict() for i in refs]


### add router

app.include_router(router)
