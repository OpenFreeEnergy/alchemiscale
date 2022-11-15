
from fah_alchemy.security import auth

SECRET_KEY = "c1662025f059affce30d1c7cc4d3d43b4f956286d85220c8f011f52073612db9"

def test_create_token():
    token = auth.create_access_token(
            data={'sub': 'nothing',
                  'scopes': ['*-*-*'],
                  },
            secret_key=SECRET_KEY
            )

def test_token_data():
    token = auth.create_access_token(
            data={'sub': 'nothing',
                  'scopes': ['*-*-*'],
                  },
            secret_key=SECRET_KEY
            )

    token_data = auth.get_token_data(token=token, secret_key=SECRET_KEY)

    assert token_data.scopes == ['*-*-*']


