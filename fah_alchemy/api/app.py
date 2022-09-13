from typing import Any
import os
import json

from starlette.responses import JSONResponse
from fastapi import APIRouter, FastAPI
from py2neo import Graph

from fah_alchemy.storage.metadatastore import Neo4jStore


graph = Graph("bolt://localhost:7687", 
              auth=(os.environ.get('NEO4J_USER'),
                    os.environ.get('NEO4J_PASS')),
              name='neo4j')

#class PermissiveJSONResponse(Response):
#    media_type = "application/json"
#    def render(self, content: Any) -> bytes:
#        return json.dumps(content).encode('utf-8')

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

n4js = Neo4jStore(graph)

app = FastAPI(
        title="FahAlchemyAPIServer"
        )


@app.get("/info")
async def info():
    return {"message": "Hello World"}


@app.get("/users")
async def users():
    return {"message": "nothing yet"}


@app.get("/networks", response_class=PermissiveJSONResponse)
def networks(name: str = None):
    networks = n4js.query_networks(name=name)
    return [n.to_dict() for n in networks]


@app.get("/transformations")
async def transformations():
    return {"message": "nothing yet"}


@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}
