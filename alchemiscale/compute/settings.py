from pathlib import Path
from pydantic import BaseModel, Field

from ..models import Scope


class ComputeServiceSettings(BaseModel):
    """Core settings schema for a compute service."""

    class Config:
        arbitrary_types_allowed = True

    api_url: str = Field(
        ..., description="URL of the compute API to execute Tasks for."
    )
    identifier: str = Field(
        ..., description="Identifier for the compute identity used for authentication."
    )
    key: str = Field(
        ..., description="Credential for the compute identity used for authentication."
    )
    name: str = Field(
        ...,
        description=(
            "The name to give this compute service; used for Task provenance, so "
            "typically set to a distinct value to distinguish different compute "
            "resources, e.g. different hosts or HPC clusters."
        ),
    )
    compute_manager_id: str | None = Field(
        None,
        description=(
            "The ID of the autoscaling compute manager responsible for the creation "
            "of this compute service. Default is None."
        ),
    )
    shared_basedir: Path = Field(
        ..., description="Filesystem path to use for `ProtocolDAG` `shared` space."
    )
    scratch_basedir: Path = Field(
        ..., description="Filesystem path to use for `ProtocolUnit` `scratch` space."
    )
    keep_shared: bool = Field(
        False,
        description="If True, don't remove shared directories for `ProtocolDAG`s after completion.",
    )
    keep_scratch: bool = Field(
        False,
        description="If True, don't remove scratch directories for `ProtocolUnit`s after completion.",
    )
    n_retries: int = Field(
        3,
        description="Number of times to attempt a given Task on failure.",
    )
    sleep_interval: int = Field(
        30, description="Time in seconds to sleep if no Tasks claimed from compute API."
    )
    deep_sleep_interval: int = Field(
        300,
        description="Time in seconds to sleep if a Task claim request is denied by the compute API.",
    )
    heartbeat_interval: int = Field(
        300, description="Frequency at which to send heartbeats to compute API."
    )
    scopes: list[Scope] | None = Field(
        None,
        description="Scopes to limit Task claiming to; defaults to all Scopes accessible by compute identity.",
    )
    protocols: list[str] | None = Field(
        None,
        description="Names of Protocols to run with this service; `None` means no restriction.",
    )
    claim_limit: int = Field(
        1, description="Maximum number of Tasks to claim at a time from a TaskHub."
    )
    loglevel: str = Field(
        "WARN",
        description="The loglevel at which to report; see the :mod:`logging` docs for available levels.",
    )
    logfile: Path | None = Field(
        None,
        description="Path to file for logging output; if not set, logging will only go to STDOUT.",
    )
    client_cache_directory: Path | str | None = Field(
        None,
        description=(
            "Location of the cache directory as either a `pathlib.Path` or `str`. "
            "If ``None`` is provided then the directory will be determined via "
            "the ``XDG_CACHE_HOME`` environment variable or default to "
            "``${HOME}/.cache/alchemiscale``. Default ``None``."
        ),
    )
    client_cache_size_limit: int = Field(
        1073741824,
        description="Maximum size of the client cache in bytes. Default 1 GiB.",
    )
    client_use_local_cache: bool = Field(
        False, description="Whether or not to use the local cache on disk."
    )
    client_max_retries: int = Field(
        5,
        description=(
            "Maximum number of times to retry a request. "
            "In the case the API service is unresponsive an expoenential backoff "
            "is applied with retries until this number is reached. "
            "If set to -1, retries will continue indefinitely until success."
        ),
    )
    client_retry_base_seconds: float = Field(
        2.0,
        description="The base number of seconds to use for exponential backoff. Must be greater than 1.0.",
    )
    client_retry_max_seconds: float = Field(
        60.0,
        description=(
            "Maximum number of seconds to sleep between retries; "
            "avoids runaway exponential backoff while allowing for many retries."
        ),
    )
    client_verify: bool = Field(
        True,
        description="Whether to verify SSL certificate presented by the API server.",
    )


class ComputeManagerSettings(BaseModel):
    name: str = Field(
        ...,
        description=(
            "The name to give this compute manager. This value should be distinct from all"
            "other compute managers."
        ),
    )
    logfile: Path | None = Field(..., description="File path to write logs to.")
    max_compute_services: int = Field(
        ...,
        description="Maximum number of compute services the manager is allowed to have running at a time.",
    )
    sleep_interval: int = Field(
        1800,
        description="Time in seconds to sleep before requesting another instruction.",
    )
    loglevel: str = Field(
        "WARN",
        description="The loglevel at which to report; see the :mod:`logging` docs for available levels.",
    )
