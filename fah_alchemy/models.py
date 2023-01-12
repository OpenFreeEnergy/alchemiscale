"""
Data models --- :mod:`fah-alchemy.models`
=========================================

"""
from typing import Optional
from pydantic import BaseModel, Field, validator
from gufe.tokenization import GufeKey


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None

    def __init__(self, org=None, campaign=None, project=None):
        # we add this to allow for arg-based creation, not just keyword-based
        super().__init__(org=org, campaign=campaign, project=project)

    @staticmethod
    def _validate_component(v, component):
        if v is not None and "-" in v:
            raise ValueError(f"'{component}' must not contain dashes ('-')")
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

    class Config:
        frozen = True

    def __str__(self):
        triple = (
            i if i is not None else "*" for i in (self.org, self.campaign, self.project)
        )
        return "-".join(triple)

    def to_tuple(self):
        return (self.org, self.campaign, self.project)

    @classmethod
    def from_str(cls, string):
        org, campaign, project = (i if i != "*" else None for i in string.split("-"))
        return cls(org=org, campaign=campaign, project=project)

    def overlap(self, other):
        """Return True if this Scope overlaps with another"""
        return NotImplementedError

    def __repr__(self):  # pragma: no cover
        return f"<Scope('{str(self)}')>"


class ScopedKey(BaseModel):
    """Unique identifier for GufeTokenizables in state store."""

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    @validator("gufe_key")
    def cast_gufe_key(cls, v):
        return GufeKey(v)

    class Config:
        frozen = True

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
