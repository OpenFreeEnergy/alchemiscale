import pytest

from fah_alchemy.security import auth


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
