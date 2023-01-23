"""
Reusable components for API services. --- :mod:`fah-alchemy.base.api`
=====================================================================

"""


from functools import lru_cache
from typing import Any, Union, Dict, List
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
    get_base_api_settings,
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


def validate_scopes(scope: Scope, token: TokenData) -> None:
    """Verify that token data has specified Scope encoded directly or is accessible via
    scope hierarchy."""

    if not isinstance(scope, Scope):
        raise ValueError("`scope` must be a `Scope` object to ensure validity")

    scope_in_token = any([Scope.from_str(ts).is_superset(scope) for ts in token.scopes])

    if not scope_in_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Targeted scope '{scope}' not accessible via scopes for this identity: {token.scopes}."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_scopes_query(
    query_scope: Scope, token: TokenData, as_str: bool = False
) -> Union[list[Scope], list[str]]:
    """
    Create the intersection of queried scopes and token, where query scopes may include 'all' / wildcard (`None`).
    No scopes outside of those included in token will be included in scopes returned.

    If as_str is True, returns a list of str rather than list of Scopes.

    """

    token_scopes = [Scope.from_str(ts) for ts in token.scopes]

    # we want to return all (and only) authorized token scopes that fall within
    # the query_scope
    scope_space = {ts for ts in token_scopes if query_scope.is_superset(ts)}

    # we also want to return the query_scope if it is a subset of any of the
    # authorized token scopes
    if any([Scope.from_str(ts).is_superset(query_scope) for ts in token.scopes]):
        scope_space.add(query_scope)

    scope_space = list(scope_space)

    if as_str:
        scope_space = [str(s) for s in scope_space]
    return scope_space


class QueryGUFEHandler:
    """
    Helper class to provide a single-dispatch like handling of the query
    operations since they can return list or dict.
    """

    def __init__(self, return_gufe: bool):
        self._return_gufe = return_gufe
        self._results = self.clear_data()

    def clear_data(self):
        return {} if self.return_gufe else []

    @property
    def results(self):
        return self._results

    @property
    def return_gufe(self):
        return self._return_gufe

    def update_results(self, data: Union[list, dict]):
        if self.return_gufe:
            # handle dict
            self._results.update(data)
        else:
            # handle list
            self._results.extend(data)

    def format_return(self):
        if self.return_gufe:
            return {str(sk): tq.to_dict() for sk, tq in self._results.items()}
        else:
            return [str(sk) for sk in self._results]


class GufeJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=JSON_HANDLER.encoder).encode("utf-8")


def scope_params(org: str = None, campaign: str = None, project: str = None):
    try:
        return Scope(org=org, campaign=campaign, project=project)
    except (AttributeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested Scope cannot be processed as a 3-object tuple of form"
                f'"X-Y-Z" and cast to string. Alpha numerical values (a-z A-Z 0-9) and "*" are accepted for '
                f'parameter "scope"'
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )


def _check_store_connectivity(n4js: Neo4jStore, s3os: S3ObjectStore) -> dict:
    """Check if neo4j and s3 object store are reachable"""
    # check if neo4j database is reachable
    neo4jreachable = n4js._store_check()
    # check if s3 object store is reachable
    s3reachable = s3os._store_check()

    if not neo4jreachable or not s3reachable:
        detail = f"Attempt to reach services failed, Neo4j reachable: {neo4jreachable}, S3 reachable: {s3reachable}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
        )
    else:
        return True


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
