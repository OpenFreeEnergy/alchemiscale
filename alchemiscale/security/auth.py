"""
:mod:`alchemiscale.security.auth` --- security components for API services
==========================================================================

"""

import secrets
import datetime
from datetime import timedelta

import bcrypt
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .models import CredentialedEntity, TokenData


# we set a max size to avoid denial-of-service attacks
# since an extremely large secret attempted by an attacker can take
# increasing amounts of time or memory to validate;
# this is deliberately higher than any reasonable key length
# this is the same max size that `passlib` defaults to
MAX_SECRET_SIZE = 4096
# Bcrypt truncates the secret at first NULL it encounters. For this reason,
# passlib forbids NULL bytes in the secret. This is not necessary for backwards
# compatibility, but we follow passlib's lead.
_BNULL = b"\x00"
# Bcrypt raises a ValueError when passed more than 72 bytes
BCRYPT_MAX_KEY_SIZE = 72


class BcryptPasswordHandler:

    def __init__(self, rounds: int = 12, ident: str = "2b"):
        self.rounds = rounds
        self.ident = ident

    def hash(self, key: str) -> str:
        validate_secret(key)

        # generate a salt unique to this key
        salt = bcrypt.gensalt(rounds=self.rounds, prefix=self.ident.encode("ascii"))
        hashed_salted = bcrypt.hashpw(key.encode("utf-8"), salt)

        return hashed_salted.decode("utf-8")

    def verify(self, key: str, hashed_salted: str) -> bool:
        validate_secret(key)
        truncated_encoded_key = key.encode("utf-8")[:BCRYPT_MAX_KEY_SIZE]
        return bcrypt.checkpw(truncated_encoded_key, hashed_salted.encode("utf-8"))


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = BcryptPasswordHandler()


def validate_secret(secret: str):
    """ensure secret has correct type & size"""
    if not isinstance(secret, str):
        raise TypeError("secret must be a string")
    if len(secret) > MAX_SECRET_SIZE:
        raise ValueError(
            f"secret is too long, maximum length is {MAX_SECRET_SIZE} characters"
        )
    if _BNULL in secret.encode("utf-8"):
        raise ValueError("secret contains NULL byte")


def generate_secret_key():
    return secrets.token_hex(32)


def authenticate(db, cls, identifier: str, key: str) -> CredentialedEntity | None:
    """Authenticate the given identity+key against the db instance.

    Parameters
    ----------
    db
        State store instance featuring a `get_credentialed_entity` method.
    cls
        The `CredentialedEntity` subclass the identity corresponds to.
    identity
        String identifier for the the identity.
    key
        Secret key string for the identity.

    Returns
    -------
    If successfully authenticated, returns the `CredentialedEntity` subclass instance.
    If not, returns `None`.

    """
    entity: CredentialedEntity = db.get_credentialed_entity(identifier, cls)
    if entity is None:
        return None
    if not pwd_context.verify(key, entity.hashed_key):
        return None
    return entity


class AuthenticationError(Exception): ...


def hash_key(key):
    return pwd_context.hash(key)


def create_access_token(
    *,
    data: dict,
    secret_key: str,
    expires_seconds: int | None = 900,
    jwt_algorithm: str | None = "HS256",
) -> str:
    to_encode = data.copy()

    expire = datetime.datetime.now(tz=datetime.UTC) + timedelta(seconds=expires_seconds)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=jwt_algorithm)
    return encoded_jwt


def get_token_data(
    *, token: str, secret_key: str, jwt_algorithm: str | None = "HS256"
) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[jwt_algorithm])

        token_data = TokenData(entity=payload.get("sub"), scopes=payload.get("scopes"))
    except JWTError:
        raise credentials_exception

    return token_data
