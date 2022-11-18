from functools import lru_cache

from pydantic import BaseSettings


class Neo4jStoreSettings(BaseSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """
    NEO4J_URL: str
    NEO4J_DBNAME: str = 'neo4j'
    NEO4J_USER: str
    NEO4J_PASS: str

    class Config:
        frozen = True

class APISettings(Neo4jStoreSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """
    FA_API_HOST: str = '127.0.0.1'
    FA_API_PORT: int = 80
    FA_API_LOGLEVEL: str = 'info'
    JWT_SECRET_KEY: str
    JWT_EXPIRE_SECONDS: int = 1800
    JWT_ALGORITHM: str = 'HS256'


class ComputeAPISettings(Neo4jStoreSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """
    FA_COMPUTE_API_HOST: str = '127.0.0.1'
    FA_COMPUTE_API_PORT: int = 80
    FA_COMPUTE_API_LOGLEVEL: str = 'info'
    JWT_SECRET_KEY: str
    JWT_EXPIRE_SECONDS: int = 1800
    JWT_ALGORITHM: str = 'HS256'


@lru_cache()
def get_neo4jstore_settings():
    return Neo4jStoreSettings()


@lru_cache()
def get_api_settings():
    return APISettings()


@lru_cache()
def get_compute_api_settings():
    return ComputeAPISettings()

