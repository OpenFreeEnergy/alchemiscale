"""
:mod:`alchemiscale.security.models` --- data models for security components
===========================================================================

"""

from datetime import datetime, timedelta

from pydantic import BaseModel, validator

from ..models import Scope


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    entity: str | None = None
    scopes: list[str] = None


class CredentialedEntity(BaseModel):
    hashed_key: str
    expires: datetime | None = None


class ScopedIdentity(BaseModel):
    identifier: str
    disabled: bool = False
    scopes: list[str] = []

    @validator("scopes", pre=True, each_item=True)
    def cast_scopes_to_str(cls, scope):
        """Ensure that each scope object is correctly cast to its str representation"""
        if isinstance(scope, Scope):
            scope = str(scope)
        elif isinstance(scope, str):
            try:
                Scope.from_str(scope)
            except:
                raise ValueError(f"Invalid scope `{scope}` set for `{cls}`")
        else:
            raise ValueError(f"Invalid scope `{scope}` set for `{cls}`")

        return scope


class UserIdentity(ScopedIdentity):
    email: str | None = None
    full_name: str | None = None


class CredentialedUserIdentity(UserIdentity, CredentialedEntity): ...


class ComputeIdentity(ScopedIdentity):
    email: str | None = None


class CredentialedComputeIdentity(ComputeIdentity, CredentialedEntity): ...
