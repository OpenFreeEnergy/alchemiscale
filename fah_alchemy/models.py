from typing import Optional
from pydantic import BaseModel, Field


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None
