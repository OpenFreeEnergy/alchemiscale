from datetime import datetime, timedelta
from typing import List, Union, Optional

from pydantic import BaseModel, validator

from ..models import Scope, ScopedKey


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    entity: Optional[str] = None
    scopes: List[str] = None


class CredentialedEntity(BaseModel):
    hashed_key: str
    expires: Optional[datetime] = None


class UserIdentity(BaseModel):
    identifier: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: Optional[List[str]] = None

    @validator("scopes", pre=True, each_item=True)
    def cast_scopes_to_str(cls, scope):
        """Ensure that each scope object is correctly cast to its str representation"""
        if isinstance(scope, ScopedKey):
            scope = scope.scope
        if isinstance(scope, Scope):
            scope = str(scope)
        return scope


class CredentialedUserIdentity(UserIdentity, CredentialedEntity):
    ...


class ComputeIdentity(BaseModel):
    identifier: str
    disabled: bool = False
    scopes: Optional[List[str]] = None


class CredentialedComputeIdentity(ComputeIdentity, CredentialedEntity):
    ...
