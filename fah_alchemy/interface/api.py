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
    scope_params,
    get_token_data_depends,
    get_n4js_depends,
    get_s3os_depends,
    base_router,
    get_cred_entity,
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


### inputs


@router.get("/networks", response_class=GufeJSONResponse)
async def query_networks(
    *,
    name: str = None,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js_depends),
):

    networks = n4js.query_networks(name=name, scope=scope, return_gufe=return_gufe)

    if return_gufe:
        return {str(sk): tq.to_dict() for sk, tq in networks.items()}
    else:
        return [str(sk) for sk in networks]


@router.get("/networks/{network}", response_class=GufeJSONResponse)
def get_network(
    network,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    network = n4js.get_gufe(scoped_key=network)
    return network.to_dict()


@router.post("/networks", response_model=ScopedKey)
def create_network(
    *,
    network: Dict = Body(...),
    scope: Scope,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    an = AlchemicalNetwork.from_dict(network)
    return n4js.create_network(network=an, scope=scope)


@router.get("/transformations")
async def query_transformations():
    return {"message": "nothing yet"}


@router.get("/transformations/{transformation}", response_class=GufeJSONResponse)
async def get_transformation(
    transformation,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    transformation = n4js.get_gufe(scoped_key=transformation)
    return transformation.to_dict()


@router.get("/chemicalsystems")
async def query_chemicalsystems():
    return {"message": "nothing yet"}


@router.get("/chemicalsystems/{chemicalsystem}", response_class=GufeJSONResponse)
async def get_chemicalsystem(
    chemicalsystem,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    chemicalsystem = n4js.get_gufe(scoped_key=chemicalsystem)
    return chemicalsystem.to_dict()


### compute


@router.put("/networks/{scoped_key}/strategy")
def set_strategy(scoped_key: str, *, strategy: Dict = Body(...), scope: Scope):
    ...


### results


@router.get("/transformations/{transformation}/result", response_class=GufeJSONResponse)
def get_transformation_result(
    transformation,
    *,
    limit: int = 10,
    skip: int = 0,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
):
    # get all ObjectStoreRefs for the given transformation's results in a nested list
    # each list corresponds to a single chain of extension results
    refs: List[List[ObjectStoreRef]] = n4js.get_transformation_results(transformation)

    # walk through the nested list, getting the actual ProtocolDAGResult object
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
