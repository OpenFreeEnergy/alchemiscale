"""FahAlchemyClientAPI

"""


from typing import Any, Dict, List
import os
import json

from fastapi import FastAPI, APIRouter, Body, Depends, HTTPException, status
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from ..base.api import (
    PermissiveJSONResponse,
    scope_params,
    get_token_data_depends,
    base_router,
)
from ..settings import ComputeAPISettings, get_api_settings, get_jwt_settings
from ..storage.statestore import Neo4jStore, get_n4js
from ..models import Scope, ScopedKey
from ..security.auth import get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedComputeIdentity


app = FastAPI(title="FahAlchemyAPI")
app.dependency_overrides[get_jwt_settings] = get_api_settings
app.include_router(base_router)

router = APIRouter(
    dependencies=[Depends(get_token_data_depends)],
)


@app.get("/info")
async def info():
    return {"message": "nothing yet"}


@app.get("/networks/", response_class=PermissiveJSONResponse)
async def query_networks(
    *,
    name: str = None,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js),
):

    networks = n4js.query_networks(name=name, scope=scope, return_gufe=return_gufe)

    if return_gufe:
        return {str(sk): tq.to_dict() for sk, tq in networks.items()}
    else:
        return [str(sk) for sk in networks]


@app.get("/networks/{scoped_key}", response_class=PermissiveJSONResponse)
def get_network(
    scoped_key: str,
    n4js: Neo4jStore = Depends(get_n4js),
):
    network = n4js.get_network(scoped_key=scoped_key)
    return network.to_dict()


@app.post("/networks", response_class=PermissiveJSONResponse)
def create_network(
    *,
    network: Dict = Body(...),
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js),
):

    an = AlchemicalNetwork.from_dict(network)
    scoped_key = n4js.create_network(an, scope.org, scope.campaign, scope.project)
    return scoped_key


@app.get("/transformations")
async def transformations():
    return {"message": "nothing yet"}


@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}


### compute


@app.put("networks/{scoped_key}/strategy")
def set_strategy(scoped_key: str, *, strategy: Dict = Body(...), scope: Scope):
    ...
