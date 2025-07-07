from pydantic import BaseModel, Field, PositiveInt

from ..models import Scope
from ..settings import Neo4jStoreSettings, S3ObjectStoreSettings


class StrategistSettings(BaseModel):
    """Settings for the Strategist service."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    sleep_interval: PositiveInt = Field(
        300, 
        description="Time in seconds between strategist iterations."
    )
    max_workers: PositiveInt = Field(
        4,
        description="Maximum number of worker processes for parallel strategy execution."
    )
    scopes: list[Scope] | None = Field(
        None,
        description="Scopes to limit strategy execution to; defaults to all accessible scopes."
    )
    cache_size: PositiveInt = Field(
        1000,
        description="LRU cache size for ProtocolDAGResults to reduce object store hits."
    )
    neo4j_settings: Neo4jStoreSettings | None = Field(
        None,
        description="Neo4j database settings; if None, uses defaults from environment."
    )
    s3_settings: S3ObjectStoreSettings | None = Field(
        None,
        description="S3 object store settings; if None, uses defaults from environment."
    )