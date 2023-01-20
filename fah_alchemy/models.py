"""
Data models --- :mod:`fah-alchemy.models`
=========================================

"""
from typing import Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from gufe.tokenization import GufeKey


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None

    def __init__(self, org=None, campaign=None, project=None):
        # we add this to allow for arg-based creation, not just keyword-based
        super().__init__(org=org, campaign=campaign, project=project)

    def __str__(self):
        triple = (
            i if i is not None else "*" for i in (self.org, self.campaign, self.project)
        )
        return "-".join(triple)

    class Config:
        frozen = True

    @staticmethod
    def _validate_component(v, component):
        if v is not None and "-" in v:
            raise ValueError(f"'{component}' must not contain dashes ('-')")
        elif v == "*":
            return None
        return v

    @validator("org")
    def valid_org(cls, v):
        return cls._validate_component(v, "org")

    @validator("campaign")
    def valid_campaign(cls, v):
        return cls._validate_component(v, "campaign")

    @validator("project")
    def valid_project(cls, v):
        return cls._validate_component(v, "project")

    @root_validator
    def check_scope_hierarchy(cls, values):
        if not _hierarchy_valid(values):
            raise InvalidScopeError(
                f"Invalid scope hierarchy: {values}, cannot specify wildcard ('*')"
                " in a scope component if a less specific scope component is not"
                " given, unless all components are wildcards (*-*-*)."
            )
        return values

    def to_tuple(self):
        return (self.org, self.campaign, self.project)

    @classmethod
    def from_str(cls, string):
        org, campaign, project = (i if i != "*" else None for i in string.split("-"))
        return cls(org=org, campaign=campaign, project=project)

    def superset(self, other):
        """Return True if this Scope is a superset of another."""
        return NotImplementedError

    def __repr__(self):  # pragma: no cover
        return f"<Scope('{str(self)}')>"

    def specific(self):
        """Return `True` if this Scope has no unspecified elements."""
        return all(self.to_tuple())


class ScopedKey(BaseModel):
    """Unique identifier for GufeTokenizables in state store.

    For this object, `org`, `campaign`, and `project` cannot contain wildcards.
    In other words, the Scope of a ScopedKey must be *specific*.

    """

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    class Config:
        frozen = True

    @validator("gufe_key")
    def cast_gufe_key(cls, v):
        return GufeKey(v)

    @staticmethod
    def _validate_component(v, component):
        if v is not None and "-" in v:
            raise ValueError(f"'{component}' must not contain dashes ('-')")
        return v

    def __repr__(self):  # pragma: no cover
        return f"<ScopedKey('{str(self)}')>"

    def __str__(self):
        return "-".join([self.gufe_key, self.org, self.campaign, self.project])

    @classmethod
    def from_str(cls, string):
        prefix, token, org, campaign, project = string.split("-")
        gufe_key = GufeKey(f"{prefix}-{token}")

        return cls(gufe_key=gufe_key, org=org, campaign=campaign, project=project)

    @property
    def scope(self):
        return Scope(org=self.org, campaign=self.campaign, project=self.project)

    @property
    def qualname(self):
        return self.gufe_key.split("-")[0]

    def to_dict(self):
        return self.dict()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class InvalidScopeError(ValueError):
    ...


def _is_wildcard(char: Union[str, None]) -> bool:
    return char is None


def _find_wildcard(scope_list: list) -> Union[int, None]:
    """Finds the index of the first wildcard in a scope list."""
    for i, scope in enumerate(scope_list):
        if _is_wildcard(scope):
            return i
    return None


def _hierarchy_valid(scope_dict: dict[str : Union[str, None]]) -> bool:
    """Checks that the scope hierarchy is valid from a dictionary of scope components."""

    org = scope_dict.get("org")
    campaign = scope_dict.get("campaign")
    project = scope_dict.get("project")
    scope_list = [org, campaign, project]

    first_wildcard_ix = _find_wildcard(scope_list)
    if first_wildcard_ix is None:  # no wildcards, so we're good
        return True

    sublevels = scope_list[first_wildcard_ix:]
    # now check if any of the sublevels are not wildcards
    if any([not _is_wildcard(i) for i in sublevels]):
        return False
    return True
