from typing import Optional
from pydantic import BaseModel, Field
from gufe.tokenization import GufeKey


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None


class ScopedKey(BaseModel):
    """Unique identifier for GufeTokenizables in state store."""

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    def __repr__(self):   # pragma: no cover
        return f"<ScopedKey('{str(self)}')>"

    def __str__(self):
        return "-".join([self.gufe_key, self.org, self.campaign, self.project])

