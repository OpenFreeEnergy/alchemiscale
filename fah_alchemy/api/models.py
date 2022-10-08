from pydantic import BaseModel, Field


class Scope(BaseModel):
    org: str
    campaign: str
    project: str
