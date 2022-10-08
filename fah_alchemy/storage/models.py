from pydantic import BaseModel, Field
from gufe.tokenization import GufeKey, GufeTokenizable


class ScopedKey(BaseModel):
    """

    """

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    def __repr__(self):   # pragma: no cover
        return f"<ScopedKey('{str(self)}')>"

    def __str__(self):
        "-".join([self.gufe_key, self.org, self.campaign, self.project])


class Task(GufeTokenizable):
    ...

    def _to_dict(self):
        ...

    def _from_dict(self):
        ...

    @property
    def _defaults(self):
        ...


class TaskQueue(GufeTokenizable):
    ...

    def _to_dict(self):
        ...

    def _from_dict(self):
        ...

    @property
    def _defaults(self):
        ...

class TaskArchive(GufeTokenizable):
    ...

    def _to_dict(self):
        ...

    def _from_dict(self):
        ...

    @property
    def _defaults(self):
        ...
