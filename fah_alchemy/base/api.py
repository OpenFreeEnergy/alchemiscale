"""Reusable components for API services.

"""


from typing import Any, Dict, List
import os
import json

from starlette.responses import JSONResponse
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from py2neo import Graph
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from ..settings import JWTSettings, get_jwt_settings
from ..storage.statestore import Neo4jStore, get_n4js
from ..models import Scope, ScopedKey
from ..security.auth import authenticate, create_access_token, get_token_data, oauth2_scheme
from ..security.models import Token, TokenData, CredentialedComputeIdentity


class PermissiveJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


def scope_params(org: str = None, campaign: str = None, project: str = None):
    return Scope(org=org, campaign=campaign, project=project)


async def get_token_data_depends(
        token: str = Depends(oauth2_scheme),
        settings: JWTSettings = Depends(get_jwt_settings),
        ) -> TokenData:
    return get_token_data(
            secret_key=settings.JWT_SECRET_KEY,
            token=token,
            jwt_algorithm=settings.JWT_ALGORITHM)


base_router = APIRouter()

@base_router.post("/token", response_model=Token)
async def get_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           settings: JWTSettings = Depends(get_jwt_settings),
                           n4js: Neo4jStore = Depends(get_n4js)):

    entity = authenticate(n4js, CredentialedComputeIdentity, form_data.username, form_data.password)

    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect identity or key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": entity.identifier,
              "scopes": entity.scopes}, 
        secret_key=settings.JWT_SECRET_KEY,
        expires_seconds=settings.JWT_EXPIRE_SECONDS,
        jwt_algorithm=settings.JWT_ALGORITHM
    )

    return {"access_token": access_token, "token_type": "bearer"}
