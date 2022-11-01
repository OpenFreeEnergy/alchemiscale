"""FahAlchemyComputeAPI

"""


from typing import Any, Dict, List
import os
import json

from starlette.responses import JSONResponse
from fastapi import APIRouter, FastAPI, Body
from py2neo import Graph

from fah_alchemy.storage.metadatastore import Neo4jStore
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from ..models import Scope


graph = Graph("bolt://localhost:7687", 
              auth=(os.environ.get('NEO4J_USER'),
                    os.environ.get('NEO4J_PASS')),
              name='neo4j')


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
# - 

n4js = Neo4jStore(graph)

app = FastAPI(
        title="FahAlchemyComputeAPI"
        )


@app.get("/info")
async def info():
    return {"message": "nothing yet"}


@app.get("/taskqueues")
async def query_taskqueues(*, scope: Scope):
    return {"message": "nothing yet"}

@app.get("/taskqueues/{scoped_key}")
async def get_taskqueue_tasks():
    return {"message": "nothing yet"}

@app.put("/taskqueues/{scoped_key}")
async def claim_taskqueue_tasks():
    return {"message": "nothing yet"}

@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}
