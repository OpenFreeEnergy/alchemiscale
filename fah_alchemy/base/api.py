"""Reusable components for API services.

"""


from functools import lru_cache
from typing import Any, Dict, List
import os
import json

from starlette.responses import JSONResponse
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from py2neo import Graph
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.tokenization import JSON_HANDLER

from ..settings import (
    JWTSettings,
    Neo4jStoreSettings,
    S3ObjectStoreSettings,
    get_base_api_settings
)
from ..storage.statestore import Neo4jStore, get_n4js
from ..storage.objectstore import S3ObjectStore, get_s3os
from ..models import Scope, ScopedKey
from ..security.auth import (
    authenticate,
    create_access_token,
    get_token_data,
    oauth2_scheme,
)
from ..security.models import Token, TokenData, CredentialedEntity


class GufeJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=JSON_HANDLER.encoder
        ).encode("utf-8")


def scope_params(org: str = None, campaign: str = None, project: str = None):
    return Scope(org=org, campaign=campaign, project=project)


async def get_token_data_depends(
    token: str = Depends(oauth2_scheme),
    settings: JWTSettings = Depends(get_base_api_settings),
) -> TokenData:
    return get_token_data(
        secret_key=settings.JWT_SECRET_KEY,
        token=token,
        jwt_algorithm=settings.JWT_ALGORITHM,
    )


@lru_cache
def get_n4js_depends(
    settings: Neo4jStoreSettings = Depends(get_base_api_settings),
) -> Neo4jStore:
    return get_n4js(settings)


@lru_cache
def get_s3os_depends(
    settings: S3ObjectStoreSettings = Depends(get_base_api_settings),
) -> S3ObjectStore:
    return get_s3os(settings)


async def get_cred_entity():
    return CredentialedEntity


base_router = APIRouter()


@base_router.post("/token", response_model=Token)
async def get_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    settings: JWTSettings = Depends(get_base_api_settings),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    cred_cls: CredentialedEntity = Depends(get_cred_entity),
):

    entity = authenticate(n4js, cred_cls, form_data.username, form_data.password)

    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect identity or key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": entity.identifier, "scopes": entity.scopes},
        secret_key=settings.JWT_SECRET_KEY,
        expires_seconds=settings.JWT_EXPIRE_SECONDS,
        jwt_algorithm=settings.JWT_ALGORITHM,
    )

    return {"access_token": access_token, "token_type": "bearer"}
