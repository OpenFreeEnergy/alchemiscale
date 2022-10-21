from enum import Enum
from typing import Union
from uuid import uuid4


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
    """A Task that can be used to generate a `ProtocolDAG` on a compute node

    Attributes
    ----------

    """

    status: TaskStatusEnum
    priority: int
    uuid: str

    def __init__(
            self, 
            status: Union[str, TaskStatusEnum] = TaskStatusEnum.waiting,
            priority: int = 1,
            uuid = None
        ):
        self.status: TaskStatusEnum = TaskStatusEnum(status)
        self.priority = priority

        if uuid is None:
            self.uuid = str(uuid4())

    def _to_dict(self):
        return {'status': self.status.value,
                'priority': self.priority,
                'uuid': self.uuid}

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    def _defaults(self):
        return super()._defaults()


class TaskQueue(GufeTokenizable):
    """

    Attributes
    ----------
    weight : float
        Value between 0.0 and 1.0 giving the weight of this TaskQueue. This
        number is used to allocate attention to this TaskQueue relative to
        others by a ComputeService. TaskQueues with equal weight will be given
        equal attention; a TaskQueue with greater weight than another will
        receive more attention.

        Setting the weight to 0.0 will give the TaskQueue no attention,
        effectively disabling it.

    """

    weight: float
    uuid: str

    def __init__(
            self, 
            weight: int = .5,
            uuid = None
        ):
        self.weight = weight

        if uuid is None:
            self.uuid = str(uuid4())

    def _to_dict(self):
        return {'weight': self.weight,
                'uuid': self.uuid}

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

