from datetime import datetime, timedelta
from typing import List, Union, Optional

from pydantic import BaseModel


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


class CredentialedUser(UserIdentity, CredentialedEntity):
    ...


class ComputeIdentity(BaseModel):
    identifier: str
    disabled: bool = False
    scopes: Optional[List[str]] = None


class CredentialedComputeIdentity(ComputeIdentity, CredentialedEntity):
    ...
