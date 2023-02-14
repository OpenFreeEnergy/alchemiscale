"""
Data models for storage components --- :mod:`alchemiscale.storage.models`
========================================================================

"""

from copy import copy
from datetime import datetime
from enum import Enum
from typing import Union, Dict, Optional
from uuid import uuid4
import hashlib


from pydantic import BaseModel, Field
from gufe.tokenization import GufeTokenizable, GufeKey

from ..models import ScopedKey, Scope


class ComputeKey(BaseModel):
    """Unique identifier for AlchemiscaleComputeService instances."""

    identifier: str

    def __repr__(self):  # pragma: no cover
        return f"<ComputeKey('{str(self)}')>"

    def __str__(self):
        return "-".join([self.identifier])


class TaskProvenance(BaseModel):
    computekey: ComputeKey
    datetime_start: datetime
    datetime_end: datetime

    # this should include versions of various libraries


class TaskStatusEnum(Enum):
    complete = "complete"
    waiting = "waiting"
    running = "running"
    error = "error"
    cancelled = "cancelled"
    invalid = "invalid"
    deleted = "deleted"


class Task(GufeTokenizable):
    """A Task that can be used to generate a `ProtocolDAG` on a
    compute node.

    Attributes
    ----------
    status
        Status of the task.
    priority
        Priority of the task; 1 is highest, larger values indicate lower priority.
    claim
        Identifier of the compute service that has a claim on this task.
    datetime_created

    """

    status: TaskStatusEnum
    priority: int
    claim: Optional[str]
    datetime_created: Optional[datetime]
    creator: Optional[str]
    extends: Optional[str]

    def __init__(
        self,
        *,
        status: Union[str, TaskStatusEnum] = TaskStatusEnum.waiting,
        priority: int = 10,
        datetime_created: Optional[datetime] = None,
        creator: Optional[str] = None,
        extends: Optional[str] = None,
        claim: Optional[str] = None,
        _key: str = None,
    ):
        if _key is not None:
            self._key = GufeKey(_key)

        self.status: TaskStatusEnum = TaskStatusEnum(status)
        self.priority = priority

        self.datetime_created = (
            datetime_created if datetime_created is not None else datetime.utcnow()
        )

        self.creator = creator
        self.extends = extends
        self.claim = claim

    def _gufe_tokenize(self):
        # tokenize with uuid
        return uuid4().hex

    def _to_dict(self):
        return {
            "status": self.status.value,
            "priority": self.priority,
            "datetime_created": self.datetime_created,
            "creator": self.creator,
            "extends": self.extends,
            "claim": self.claim,
            "_key": str(self.key),
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


class TaskHub(GufeTokenizable):
    """

    Attributes
    ----------
    network : str
        ScopedKey of the AlchemicalNetwork this TaskHub corresponds to.
        Used to ensure that there is only one TaskHub for a given
        AlchemicalNetwork using neo4j constraints.
    weight : float
        Value between 0.0 and 1.0 giving the weight of this TaskHub. This
        number is used to allocate attention to this TaskHub relative to
        others by a ComputeService. TaskHub with equal weight will be given
        equal attention; a TaskHub with greater weight than another will
        receive more attention.

        Setting the weight to 0.0 will give the TaskHub no attention,
        effectively disabling it.

    """

    network: str
    weight: float

    def __init__(self, network: ScopedKey, weight: int = 0.5):
        self.network = network
        self.weight = weight

    def _gufe_tokenize(self):
        return hashlib.md5(
            str(self.network).encode(), usedforsecurity=False
        ).hexdigest()

    def _to_dict(self):
        return {
            "network": self.network,
            "weight": self.weight,
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


class TaskArchive(GufeTokenizable):
    ...

    def _to_dict(self):
        return {}

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


class ObjectStoreRef(GufeTokenizable):
    location: Optional[str]
    obj_key: Optional[GufeKey]
    scope: Scope

    def __init__(self, *, location: str = None, obj_key: GufeKey = None, scope: Scope):
        self.location = location
        self.obj_key = GufeKey(obj_key) if obj_key is not None else None
        self.scope = scope

    def _to_dict(self):
        return {
            "location": self.location,
            "obj_key": str(self.obj_key),
            "scope": str(self.scope),
        }

    @classmethod
    def _from_dict(cls, d):
        d_ = copy(d)
        d_["scope"] = Scope.from_str(d["scope"])
        return cls(**d_)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


class ProtocolDAGResultRef(ObjectStoreRef):
    success: bool

    def __init__(
        self,
        *,
        location: str = None,
        obj_key: GufeKey,
        scope: Scope,
        success: bool,
    ):
        self.location = location
        self.obj_key = GufeKey(obj_key)
        self.scope = scope
        self.success = success

    def _to_dict(self):
        return {
            "location": self.location,
            "obj_key": str(self.obj_key),
            "scope": str(self.scope),
            "success": self.success,
        }


class TaskArchive(GufeTokenizable):
    ...
