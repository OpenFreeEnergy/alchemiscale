
import pytest

from fastapi import HTTPException

from fah_alchemy.base.api import validate_scopes, validate_scopes_query
from fah_alchemy.models import Scope
from fah_alchemy.security.models import Token, TokenData, CredentialedEntity


@pytest.fixture
def tokendata():
    return TokenData(
            entity='not-a-real-user',
            scopes=['org1-campaignA-projectI',
                    'org1-campaignB-projectI',
                    'org1-campaignB-projectII']
        )


def test_validate_scopes(tokendata):
    assert validate_scopes(Scope('org1', 'campaignB', 'projectI'), tokendata) is None
    assert validate_scopes(Scope('org1', 'campaignA', 'projectI'), tokendata) is None
    assert validate_scopes(Scope('org1', 'campaignB', 'projectII'), tokendata) is None

    with pytest.raises(HTTPException):
        validate_scopes(Scope('org2', 'campaignB', 'projectII'), tokendata) is None


def test_validate_scopes_query(tokendata):
    matches = validate_scopes_query(Scope('org1', 'campaignB', 'projectI'), tokendata)

    assert matches == [Scope.from_str(tokendata.scopes[1])]

    matches = validate_scopes_query(Scope('org1', 'campaignB', None), tokendata, as_str=True)

    assert len(matches) == 2
    assert matches == tokendata.scopes[1:]

    matches = validate_scopes_query(Scope('org1', None, None), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes

    matches = validate_scopes_query(Scope(), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes
