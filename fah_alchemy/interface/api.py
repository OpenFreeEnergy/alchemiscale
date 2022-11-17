"""FahAlchemyClientAPI

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


n4js = Neo4jStore(graph)

app = FastAPI(
        title="FahAlchemyClientAPI"
        )


@app.get("/info")
async def info():
    return {"message": "nothing yet"}


@app.get("/users")
async def users():
    return {"message": "nothing yet"}

@app.get("/networks/", response_class=PermissiveJSONResponse)
def query_networks(*, name: str = None, scope: Scope):
    networks = n4js.query_networks(name=name)
    return [n.to_dict() for n in networks]


@app.get("/networks/{scoped_key}", response_class=PermissiveJSONResponse)
def get_network(scoped_key: str):
    network = n4js.get_network(scoped_key=scoped_key)
    return network.to_dict()


@app.post("/networks", response_class=PermissiveJSONResponse)
def create_network(*, network: Dict = Body(...), scope: Scope):
    an = AlchemicalNetwork.from_dict(network)
    scoped_key = n4js.create_network(an, scope.org, scope.campaign, scope.project)
    return scoped_key


@app.put("/networks", response_class=PermissiveJSONResponse)
def update_network(*, network: Dict = Body(...), scope: Scope):
    an = AlchemicalNetwork.from_dict(network)
    scoped_key = n4js.update_network(an, scope.org, scope.campaign, scope.project)
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

