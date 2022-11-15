"""Security components for APIs.

"""

from datetime import datetime, timedelta
from typing import Union, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .models import Token, TokenData, User, CredentialedUser, CredentialedEntity


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def authenticate(db, identifier: str, key: str) -> CredentialedEntity:
    entity: CredentialedEntity = db.get_credentialed_entity(identifier)
    if entity is None:
        return False
    if not pwd_context.verify(key, entity.hashed_key):
        return False
    return entity


def create_access_token(
        data: dict, 
        secret_key: str,
        expires_seconds: Optional[int] = 900,
        jwt_algorithm: Optional[str] = "HS256"
        ) -> str:

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(seconds=expires_seconds)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=jwt_algorithm)
    return encoded_jwt


def get_token_data(
        secret_key: str,
        token: str,
        jwt_algorithm: Optional[str] = "HS256"
        ) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[jwt_algorithm])

        token_data = TokenData(entity=payload.get('sub'),
                               scopes=payload.get('scopes'))
    except JWTError:
        raise credentials_exception

    return token_data
