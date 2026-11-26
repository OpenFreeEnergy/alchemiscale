"""
:mod:`alchemiscale.settings` --- settings
=========================================

"""

from functools import lru_cache
import logging

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrozenSettings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
    )


class Neo4jStoreSettings(FrozenSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """

    NEO4J_URL: str
    NEO4J_DBNAME: str = "neo4j"
    NEO4J_USER: str
    NEO4J_PASS: str


class S3ObjectStoreSettings(FrozenSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    If deploying APIs that use the S3ObjectStore on an EC2 host or other
    role-based resource (e.g. an ECS container), then don't set these.
    Instead rely on the IAM role of that resource for granting access to S3.

    """

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_SESSION_TOKEN: str | None = None
    AWS_S3_BUCKET: str
    AWS_S3_PREFIX: str
    AWS_DEFAULT_REGION: str
    AWS_ENDPOINT_URL: str | None = None


class JWTSettings(FrozenSettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """

    JWT_SECRET_KEY: str
    JWT_EXPIRE_SECONDS: int = 1800
    JWT_ALGORITHM: str = "HS256"


class BaseAPISettings(Neo4jStoreSettings, S3ObjectStoreSettings, JWTSettings): ...


class APISettings(BaseAPISettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """

    ALCHEMISCALE_API_HOST: str = "127.0.0.1"
    ALCHEMISCALE_API_PORT: int = 80
    ALCHEMISCALE_API_LOGLEVEL: str = "info"


class ComputeAPISettings(BaseAPISettings):
    """Automatically populates settings from environment variables where they
    match; case-insensitive.

    """

    ALCHEMISCALE_COMPUTE_API_HOST: str = "127.0.0.1"
    ALCHEMISCALE_COMPUTE_API_PORT: int = 80
    ALCHEMISCALE_COMPUTE_API_LOGLEVEL: str = "info"
    ALCHEMISCALE_COMPUTE_API_REGISTRATION_EXPIRE_SECONDS: int = 1800
    ALCHEMISCALE_COMPUTE_API_FORGIVE_TIME_SECONDS: int = 600
    ALCHEMISCALE_COMPUTE_API_MAX_FAILURES: int = 3
    # a compute manager registration not updated within this many seconds is
    # stale, whatever its status; set it to a small multiple of the managers'
    # ``sleep_interval``, since it is how long a replacement for a killed
    # manager starts from scratch rather than joining its state
    ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS: int = 3600
    # deprecated and unused: ERROR registrations expire like any other
    ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS_ERROR: int = 86400

    @model_validator(mode="after")
    def _warn_deprecated(self):
        if "ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS_ERROR" in (
            self.model_fields_set
        ):
            # a DeprecationWarning would be hidden by the default filters
            logging.getLogger(__name__).warning(
                "ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS_ERROR is "
                "deprecated and has no effect; compute manager registrations "
                "of any status expire after "
                "ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS"
            )
        return self


@lru_cache
def get_neo4jstore_settings():
    return Neo4jStoreSettings()


@lru_cache
def get_s3objectstore_settings():
    return S3ObjectStoreSettings()


@lru_cache
def get_jwt_settings():
    return JWTSettings()


@lru_cache
def get_base_api_settings():
    return BaseAPISettings()


@lru_cache
def get_api_settings():
    return APISettings()


@lru_cache
def get_compute_api_settings():
    return ComputeAPISettings()
