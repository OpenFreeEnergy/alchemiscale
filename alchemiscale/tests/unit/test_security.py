import pytest

from alchemiscale.security import auth


@pytest.fixture
def secret_key():
    return auth.generate_secret_key()


def test_create_token(secret_key):
    token = auth.create_access_token(
        data={
            "sub": "nothing",
            "scopes": ["*-*-*"],
        },
        secret_key=secret_key,
    )


def test_token_data(secret_key):
    token = auth.create_access_token(
        data={
            "sub": "nothing",
            "scopes": ["*-*-*"],
        },
        secret_key=secret_key,
    )

    token_data = auth.get_token_data(token=token, secret_key=secret_key)

    assert token_data.scopes == ["*-*-*"]


def test_bcrypt_password_handler():
    handler = auth.BcryptPasswordHandler()
    hash_ = handler.hash("test")
    assert handler.verify("test", hash_)
    assert not handler.verify("deadbeef", hash_)


def test_bcrypt_against_passlib():
    """Test the our bcrypt handler has the same behavior as passlib did"""

    # pre-generated hash from
    # `passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto").hash()`
    test_password = "the quick brown fox jumps over the lazy dog"
    test_hash = "$2b$12$QZTnjdx/sJS7jnEnCqhM3uS8mZ4mhLV5dDfM7ZBUT6LwDiNZ2p7S."

    # test that we get the same thing back from our bcrypt handler
    handler = auth.BcryptPasswordHandler()
    assert handler.verify(test_password, test_hash)


def test_bcrypt_against_passlib_long():
    """Test the our bcrypt handler has the same behavior as passlib did for passwords longer than 72 characters, in which bcrypt truncates"""

    # pre-generated hash from
    # `passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto").hash()`
    test_password = "this password is so long, it's longer than 72 characters; this should get truncated by bcrypt, so we can ensure we get the same verification behavior as passlib gives"
    test_hash = "$2b$12$DQd5IPjlc8z4FZjBIdaquOlVc9whAqnkpZRsnuUUWvfHvwWy.dZ16"

    # test that we get the same thing back from our bcrypt handler
    handler = auth.BcryptPasswordHandler()
    assert handler.verify(test_password, test_hash)
