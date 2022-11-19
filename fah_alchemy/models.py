from typing import Optional
from pydantic import BaseModel, Field, validator
from gufe.tokenization import GufeKey


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None

    class Config:
        frozen = True

    def overlap(self, other):
        """Return True if this Scope overlaps with another"""
        return NotImplementedError


class ScopedKey(BaseModel):
    """Unique identifier for GufeTokenizables in state store."""

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    @validator('gufe_key')
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
