from enum import Enum
from typing import Union
from pydantic import BaseModel, Field
from gufe.tokenization import GufeTokenizable



class ComputeKey(BaseModel):
    """Unique identifier for FahAlchemyComputeService instances."""

    identifier: str

    def __repr__(self):   # pragma: no cover
        return f"<ComputeKey('{str(self)}')>"

    def __str__(self):
        return "-".join([self.identifier])

class TaskStatusEnum(Enum):
    complete = "complete"
    waiting = "waiting"
    running = "running"
    error = "error"
    cancelled = "cancelled"
    invalid = "invalid"
    deleted = "deleted"


class Task(GufeTokenizable):
    """A Task that can be used to generate a `ProtocolDAG` on a compute node"""

    status: TaskStatusEnum
    priority: int

    def __init__(self, status: Union[str, TaskStatusEnum] = TaskStatusEnum.waiting):
        self.status: TaskStatusEnum = TaskStatusEnum(status)

    def _to_dict(self):
        return {'status': self.status.value}

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    def _defaults(self):
        return super()._defaults()


class TaskQueue(GufeTokenizable):
    ...

    weight: float

    def __init__(self, weight: int = .5):
        self.weight = weight

    def _to_dict(self):
        return {}

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    def _defaults(self):
        return super()._defaults()


class TaskArchive(GufeTokenizable):
    ...

    def _to_dict(self):
        return {}

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    def _defaults(self):
        return super()._defaults()

