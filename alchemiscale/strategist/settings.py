from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt

from ..models import Scope
from ..settings import Neo4jStoreSettings, S3ObjectStoreSettings


class StrategistSettings(BaseModel):
    """Settings for the Strategist service."""

    model_config = {"arbitrary_types_allowed": True}

    sleep_interval: PositiveInt = Field(
        300, description="Time in seconds between strategist iterations."
    )
    max_workers: PositiveInt = Field(
        4,
        description="Maximum number of worker processes for parallel strategy execution.",
    )
    scopes: list[Scope] | None = Field(
        None,
        description="Scopes to limit strategy execution to; defaults to all accessible scopes.",
    )
    cache_directory: Path | str | None = Field(
        None,
        description="Location of the cache directory; defaults to ${HOME}/.cache/alchemiscale-strategist",
    )
    cache_size_limit: int = Field(
        1073741824, description="Maximum size of the strategist cache in bytes"  # 1 GiB
    )
    use_local_cache: bool = Field(
        True, description="Whether to use persistent disk-based caching"
    )
    neo4j_settings: Neo4jStoreSettings | None = Field(
        None,
        description="Neo4j database settings; if None, uses defaults from environment.",
    )
    s3_settings: S3ObjectStoreSettings | None = Field(
        None,
        description="S3 object store settings; if None, uses defaults from environment.",
    )
