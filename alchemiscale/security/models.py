"""
:mod:`alchemiscale.security.models` --- data models for security components
===========================================================================

"""

import datetime

from pydantic import BaseModel, field_validator

from ..models import Scope


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    entity: str | None = None
    scopes: list[str] = None


class CredentialedEntity(BaseModel):
    hashed_key: str
    expires: datetime.datetime | None = None

    def to_dict(self):
        return self.model_dump()


class ScopedIdentity(BaseModel):
    identifier: str
    disabled: bool = False
    scopes: list[str] = []

    @field_validator("scopes", mode="before")
    @classmethod
    def cast_scopes_to_str(cls, scopes):
        """Ensure that each scope object is correctly cast to its str representation.

        :meta private:
        """
        scopes_ = []
        for scope in scopes:
            if isinstance(scope, Scope):
                scopes_.append(str(scope))
            elif isinstance(scope, str):
                try:
                    scopes_.append(str(Scope.from_str(scope)))
                except Exception:
                    raise ValueError(f"Invalid scope `{scope}` set for `{cls}`")
            else:
                raise ValueError(f"Invalid scope `{scope}` set for `{cls}`")

        return scopes_


class UserIdentity(ScopedIdentity):
    email: str | None = None
    full_name: str | None = None


class CredentialedUserIdentity(UserIdentity, CredentialedEntity): ...


class ComputeIdentity(ScopedIdentity):
    email: str | None = None


class CredentialedComputeIdentity(ComputeIdentity, CredentialedEntity): ...
