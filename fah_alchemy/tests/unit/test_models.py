import pytest

from pydantic import ValidationError

from fah_alchemy.models import Scope


@pytest.mark.parametrize(
    "scope_string", ["*-*-*", "org1-*-*", "org1-campaignA-*", "org1-campaignA-projectI"]
)
def test_wildcard_scopes_valid(scope_string):
    scope = Scope.from_str(scope_string)


@pytest.mark.parametrize(
    "scope_string",
    ["*-*-projectI", "*-campaignA-*", "*-campaignA-projectI", "org1-*-projectI"],
)
def test_wildcard_scopes_invalid(scope_string):
    with pytest.raises(ValidationError):
        scope = Scope.from_str(scope_string)


@pytest.mark.parametrize(
    "scope_string1, scope_string2",
    [
        ("*-*-*", "org1-*-*"),
        ("org1-*-*", "org1-campaignA-*"),
        ("org1-campaignA-*", "org1-campaignA-projectI"),
        ("*-*-*", "org1-campaignA-projectI"),
    ],
)
def test_scope_superset_true(scope_string1, scope_string2):
    scope1 = Scope.from_str(scope_string1)
    scope2 = Scope.from_str(scope_string2)
    assert scope1.is_superset(scope2)
