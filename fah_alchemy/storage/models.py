from enum import Enum
from typing import Union, Dict
from uuid import uuid4


from pydantic import BaseModel, Field
from gufe.tokenization import GufeTokenizable, GufeKey

from ..models import ScopedKey


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

    def __init__(
            self, 
            status: Union[str, TaskStatusEnum] = TaskStatusEnum.waiting,
            priority: int = 1,
            _key: str = None
        ):
        if _key is not None:
            self._key = GufeKey(_key)

        self.status: TaskStatusEnum = TaskStatusEnum(status)
        self.priority = priority

    def _gufe_tokenize(self):
        # tokenize with uuid
        return uuid4()

    def _to_dict(self):
        return {'status': self.status.value,
                'priority': self.priority,
                '_key': str(self.key),
               }

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)



    def _defaults(self):
        return super()._defaults()


class TaskQueue(GufeTokenizable):
    """

    Attributes
    ----------
    network : str
        ScopedKey of the AlchemicalNetwork this TaskQueue corresponds to.
        Used to ensure that there is only one TaskQueue for a given
        AlchemicalNetwork using neo4j constraints.
    weight : float
        Value between 0.0 and 1.0 giving the weight of this TaskQueue. This
        number is used to allocate attention to this TaskQueue relative to
        others by a ComputeService. TaskQueues with equal weight will be given
        equal attention; a TaskQueue with greater weight than another will
        receive more attention.

        Setting the weight to 0.0 will give the TaskQueue no attention,
        effectively disabling it.

    """

    network: str
    weight: float

    def __init__(
            self, 
            network: ScopedKey,
            weight: int = .5,
            _key: str = None
        ):
        if _key is not None:
            self._key = GufeKey(_key)

        self.network = network
        self.weight = weight

    def _gufe_tokenize(self):
        # tokenize with uuid
        return self.network

    def _to_dict(self):
        return {
                'network': self.network,
                'weight': self.weight,
                '_key': str(self.key),
               }

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

