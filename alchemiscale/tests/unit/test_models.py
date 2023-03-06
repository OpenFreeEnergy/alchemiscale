import pytest

from pydantic import ValidationError

from alchemiscale.models import Scope


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
    "super_scope_str, sub_scope_str",
    [
        ("*-*-*", "*-*-*"),
        ("*-*-*", "org1-*-*"),
        ("*-*-*", "org1-campaignA-*"),
        ("*-*-*", "org1-campaignA-projectI"),
        ("org1-*-*", "org1-*-*"),
        ("org1-*-*", "org1-campaignA-*"),
        ("org1-*-*", "org1-campaignA-projectI"),
        ("org1-campaignA-*", "org1-campaignA-*"),
        ("org1-campaignA-*", "org1-campaignA-projectI"),
        ("org1-campaignA-projectI", "org1-campaignA-projectI"),
    ],
)
def test_scope_superset_true(super_scope_str, sub_scope_str):
    super_scope = Scope.from_str(super_scope_str)
    sub_scope = Scope.from_str(sub_scope_str)
    assert super_scope.is_superset(sub_scope)


@pytest.mark.parametrize(
    # test the inverse of test_scope_superset_true() and also cases where
    # the scopes have the same level but not the same value
    "sub_scope_str, super_scope_str",
    [
        ("*-*-*", "org1-*-*"),
        ("*-*-*", "org1-campaignA-*"),
        ("*-*-*", "org1-campaignA-projectI"),
        ("org1-*-*", "org1-campaignA-*"),
        ("org1-*-*", "org1-campaignA-projectI"),
        ("org1-campaignA-projectI", "org1-campaignA-projectII"),
        ("org1-campaignB-projectI", "org1-campaignA-projectI"),
        ("org2-campaignB-projectI", "org1-campaignA-projectI"),
        ("org1-campaignB-projectI", "org1-campaignA-*"),
        ("org1-campaignB-*", "org1-campaignA-*"),
        ("org1-campaignA-*", "org2-*-*"),
        ("org1-*-*", "org2-*-*"),
        ("org1-campaignA-projectI", "org2-*-*"),
    ],
)
def test_scope_superset_false(super_scope_str, sub_scope_str):
    super_scope = Scope.from_str(super_scope_str)
    sub_scope = Scope.from_str(sub_scope_str)
    assert not super_scope.is_superset(sub_scope)


@pytest.mark.parametrize(
    "scope_string",
    [
        "*foo-*-*",
        "**-*-*",
        "*_foo-*-*",
        "!@#$%^&-bar-baz",
        "☺☻♥♦♣-bar-baz",
        ",.<>/?|-bar-baz",
        "𝄞𝄢𝄪𝄫𝅘𝅥𝅮𝅥𝅲𝅳𝆺𝅥𝆹𝅥𝅮𝆺𝅥𝅮𝆹𝅥𝅯𝆺𝅥𝅯𝇁𝇂𝇃-bar-baz",
    ],
)
def test_scope_non_alphanumeric_invalid(scope_string):
    with pytest.raises(ValidationError, match="must be alphanumeric or underscore"):
        scope = Scope.from_str(scope_string)


@pytest.mark.parametrize(
    "scope_string",
    [
        "a_b-*-*",
        "a_b-c_d-*",
        "a_b-c_d-e_f",
        "org_1-campaign_A-project_I",
    ],
)
def test_underscore_scopes_valid(scope_string):
    scope = Scope.from_str(scope_string)
