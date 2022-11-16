"""FahAlchemyComputeAPI

"""


from typing import Any, Dict, List
import os
import json
from datetime import timedelta
from functools import lru_cache

from starlette.responses import JSONResponse
from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from py2neo import Graph
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from ..settings import Settings, get_settings
from ..storage.statestore import Neo4jStore
from ..models import Scope, ScopedKey
from ..security.auth import authenticate, create_access_token, get_token_data
from ..security.models import Token, TokenData, CredentialedComputeService


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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


# TODO:
# - add periodic removal of task claims from compute services that are no longer alive
#   - can be done with an asyncio.sleeping task added to event loop: https://stackoverflow.com/questions/67154839/fastapi-best-way-to-run-continuous-get-requests-in-the-background
# - on startup, 

app = FastAPI(
        title="FahAlchemyComputeAPI"
        )


@lru_cache()
def get_n4js(settings: Settings = Depends(get_settings)):

    graph = Graph(settings.NEO4J_URL, 
                  auth=(settings.NEO4J_USER,
                        settings.NEO4J_PASS),
              name=settings.NEO4J_DBNAME)
    return Neo4jStore(graph)


def scope_params(org: str = None, campaign: str = None, project: str = None):
    return Scope(org=org, campaign=campaign, project=project)


async def get_token_data_depends(
        token: str = Depends(oauth2_scheme),
        settings: Settings = Depends(get_settings),
        ) -> TokenData:
    return get_token_data(
            secret_key=settings.JWT_SECRET_KEY,
            token=token,
            jwt_algorithm=settings.JWT_ALGORITHM)


@app.post("/token", response_model=Token)
async def get_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           settings: Settings = Depends(get_settings),
                           n4js: Neo4jStore = Depends(get_n4js)):

    entity = authenticate(n4js, CredentialedComputeService, form_data.username, form_data.password)

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


@app.get("/info")
async def info():
    return {"message": "nothing yet"}


@app.get("/taskqueues")
async def query_taskqueues(*, 
                           return_gufe: bool = False,
                           scope: Scope = Depends(scope_params), 
                           n4js: Neo4jStore = Depends(get_n4js)):
    taskqueues = n4js.query_taskqueues(scope=scope, return_gufe=return_gufe)

    if return_gufe:
        return {str(sk): tq.to_dict() for sk, tq in taskqueues.items()}
    else:
        return [str(sk) for sk in taskqueues]


#@app.get("/taskqueues/{scoped_key}")
#async def get_taskqueue(scoped_key: str, 
#                        *,
#                        n4js: Neo4jStore = Depends(get_n4js)):
#    return 


@app.get("/taskqueues/{taskqueue}/tasks")
async def get_taskqueue_tasks():
    return {"message": "nothing yet"}


@app.post("/taskqueues/{taskqueue}/claim")
async def claim_taskqueue_tasks(taskqueue,
                                *,
                                claimant: str = Body(),
                                count: int = Body(),
                                n4js: Neo4jStore = Depends(get_n4js),
                                tokendata: TokenData = Depends(get_token_data_depends)
                                ):
    tasks = n4js.claim_taskqueue_tasks(taskqueue=taskqueue,
                                       claimant=claimant,
                                       count=count)

    return [str(t) if t is not None else None for t in tasks]


@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}
