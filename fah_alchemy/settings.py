from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """
    NEO4J_URL: str
    NEO4J_DBNAME: str = 'neo4j'
    NEO4J_USER: str
    NEO4J_PASS: str
    FA_COMPUTE_API_HOST: str = '127.0.0.1'
    FA_COMPUTE_API_PORT: int = 80
    FA_COMPUTE_API_LOGLEVEL: str = 'info'
    JWT_SECRET_KEY: str
    JWT_EXPIRE_SECONDS: int = 1800
    JWT_ALGORITHM: str = 'HS256'

    def __hash__(self):
        return hash(tuple(self.dict().items()))


@lru_cache()
def get_settings():
    return Settings()

