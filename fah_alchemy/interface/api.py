"""FahAlchemyClientAPI

"""


from typing import Any, Dict, List
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
)
from ..settings import get_api_settings
from ..settings import get_base_api_settings, get_api_settings
from ..storage.statestore import Neo4jStore
from ..storage.objectstore import S3ObjectStore
from ..storage.models import ObjectStoreRef
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedUserIdentity


app = FastAPI(title="FahAlchemyAPI")
app.dependency_overrides[get_base_api_settings] = get_api_settings
app.include_router(base_router)


def get_cred_user():
    return CredentialedUserIdentity


app.dependency_overrides[get_cred_entity] = get_cred_user

router = APIRouter(
    dependencies=[Depends(get_token_data_depends)],
)


@app.get("/ping")
async def ping():
    return {"api": "FahAlchemyAPI"}


@router.get("/info")
async def info():
    return {"message": "nothing yet"}


@router.get("/check")
async def check(
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
):
    # check connectivity of storage components
    return _check_store_connectivity(n4js, s3os)


### inputs


@router.get("/networks", response_class=GufeJSONResponse)
async def query_networks(
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
    return network.to_dict()


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
    return n4js.create_network(network=an, scope=scope)


@router.get("/transformations")
async def query_transformations():
    return {"message": "nothing yet"}


@router.get(
    "/transformations/{transformation_scoped_key}", response_class=GufeJSONResponse
)
async def get_transformation(
    transformation_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    transformation = n4js.get_gufe(scoped_key=sk)
    return transformation.to_dict()


@router.get("/chemicalsystems")
async def query_chemicalsystems():
    return {"message": "nothing yet"}


@router.get(
    "/chemicalsystems/{chemicalsystem_scoped_key}", response_class=GufeJSONResponse
)
async def get_chemicalsystem(
    chemicalsystem_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(chemicalsystem_scoped_key)
    validate_scopes(sk.scope, token)

    chemicalsystem = n4js.get_gufe(scoped_key=sk)
    return chemicalsystem.to_dict()


### compute


@router.put("/networks/{scoped_key}/strategy")
def set_strategy(scoped_key: str, *, strategy: Dict = Body(...), scope: Scope):
    ...


### results


@router.get(
    "/transformations/{transformation_scoped_key}/result",
    response_class=GufeJSONResponse,
)
def get_transformation_result(
    transformation_scoped_key,
    *,
    limit: int = 10,
    skip: int = 0,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(transformation_scoped_key)
    validate_scopes(sk.scope, token)

    # get all ObjectStoreRefs for the given transformation's results in a nested list
    # each list corresponds to a single chain of extension results
    refs: List[ObjectStoreRef] = n4js.get_transformation_results(sk)

    # walk through the list, getting the actual ProtocolDAGResult object
    # for each ObjectStoreRef, starting from `skip` and up to `limit`
    pdrs: List[List[str]] = []
    for reflist in refs[skip : skip + limit]:
        pdrs_i = []
        for ref in reflist:
            # we leave each ProtocolDAGResult in string form to avoid
            # deserializing/reserializing here; just passing through to clinet
            pdr: str = s3os.pull_protocoldagresult(ref, return_as="json")
            pdrs_i.append(pdr)
        pdrs.append(pdrs_i)

    return pdrs


### add router

app.include_router(router)
