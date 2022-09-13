import os

from fastapi import APIRouter, FastAPI
from py2neo import Graph

from fah_alchemy.storage.metadatastore import Neo4jStore


graph = Graph("bolt://localhost:7687", 
              auth=(os.environ.get('NEO4J_USER'),
                    os.environ.get('NEO4J_PASS')),
              name='neo4j')

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


@app.get("/networks")
async def networks(name: str = None):
    networks = n4js.query_networks()
    return [n.to_dict() for n in networks]


@app.get("/transformations")
async def transformations():
    return {"message": "nothing yet"}


@app.get("/chemicalsystems")
async def chemicalsystems():
    return {"message": "nothing yet"}
