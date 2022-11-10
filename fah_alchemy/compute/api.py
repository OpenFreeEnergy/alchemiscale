"""FahAlchemyComputeAPI

"""


from typing import Any, Dict, List
import os
import json
from functools import lru_cache

from pydantic import BaseSettings
from starlette.responses import JSONResponse
from fastapi import FastAPI, Body, Depends
from py2neo import Graph

from fah_alchemy.storage.statestore import Neo4jStore
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from ..models import Scope


class Settings(BaseSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """
    NEO4J_URL: str
    NEO4J_DBNAME: str = 'neo4j'
    NEO4J_USER: str
    NEO4J_PASS: str
    FA_COMPUTE_API_HOST: str
    FA_COMPUTE_API_PORT: int
    FA_COMPUTE_API_LOGLEVEL: str = 'info'



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
def get_settings():
    return Settings()


@lru_cache()
def get_n4js(settings: Settings = Depends(get_settings)):

    graph = Graph(settings.NEO4J_URL, 
                  auth=(settings.NEO4J_USER,
                        settings.NEO4J_PASS),
              name=settings.NEO4J_DBNAME)
    return Neo4jStore(graph)


def scope_params(org: str = None, campaign: str = None, project: str = None):
    return Scope(org=org, campaign=campaign, project=project)


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


@app.get("/taskqueues/{scoped_key}/tasks")
async def get_taskqueue_tasks():
    return {"message": "nothing yet"}


@app.patch("/taskqueues/{scoped_key}/tasks")
async def claim_taskqueue_tasks():
    return {"message": "nothing yet"}


@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}
