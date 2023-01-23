import pytest

from fastapi import HTTPException
from pydantic import ValidationError

from fah_alchemy.base.api import validate_scopes, validate_scopes_query
from fah_alchemy.models import Scope, ScopedKey
from fah_alchemy.security.models import Token, TokenData, CredentialedEntity


@pytest.mark.parametrize(
    "scope", (("a-", "a", "a"), ("a", "a-", "a"), ("a", "a", "a-"))
)
@pytest.mark.parametrize("scope_cls", [Scope, ScopedKey])
def test_create_invalid_scope_with_dash(scope_cls, scope):
    org, campaign, project = scope
    with pytest.raises(ValidationError):
        scope_cls(org=org, campaign=campaign, project=project)


@pytest.fixture
def tokendata():
    return TokenData(
        entity="not-a-real-user",
        scopes=[
            "org1-campaignA-projectI",
            "org1-campaignB-projectI",
            "org1-campaignB-projectII",
            "org2-*-*",
            "org3-campaignA-*",
        ],
    )


@pytest.mark.parametrize(
    "scope_str",
    [
        "org1-campaignA-projectI",
        "org1-campaignB-projectI",
        "org1-campaignB-projectII",
        "org2-campaignB-projectII",
        "org3-campaignA-projectII",
    ],
)
def test_validate_scopes_valid(tokendata, scope_str):
    assert validate_scopes(Scope.from_str(scope_str), tokendata) is None


@pytest.mark.parametrize(
    "scope_str",
    [
        "org4-campaignB-projectII",
        "org3-campaignB-projectII",
        "org1-campaignB-projectIII",
    ],
)
def test_validate_scopes_invalid(tokendata, scope_str):
    with pytest.raises(HTTPException):
        validate_scopes(Scope.from_str(scope_str), tokendata)


def test_validate_scopes_query(tokendata):
    matches = validate_scopes_query(Scope("org1", "campaignB", "projectI"), tokendata)

    assert matches == [Scope.from_str(tokendata.scopes[1])]

    matches = validate_scopes_query(
        Scope("org1", "campaignB", None), tokendata, as_str=True
    )

    assert len(matches) == 2
    assert matches == tokendata.scopes[1:]

    matches = validate_scopes_query(
        Scope("org1", "campaignB", "*"), tokendata, as_str=True
    )

    assert len(matches) == 2
    assert matches == tokendata.scopes[1:]

    matches = validate_scopes_query(Scope("org1", None, None), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes

    matches = validate_scopes_query(Scope("org1", "*", "*"), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes

    matches = validate_scopes_query(Scope(), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes

    matches = validate_scopes_query(Scope("*", "*", "*"), tokendata, as_str=True)

    assert len(matches) == 3
    assert matches == tokendata.scopes
