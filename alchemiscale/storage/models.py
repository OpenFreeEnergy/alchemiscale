"""
:mod:`alchemiscale.storage.models` --- data models for storage components
=========================================================================

"""

from abc import abstractmethod
from copy import copy
import datetime
from enum import Enum, StrEnum
from uuid import uuid4, UUID
import re
import hashlib


from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator
from gufe.tokenization import GufeTokenizable, GufeKey

from ..models import ScopedKey, Scope


class ComputeIDBase(str):

    _allowed_name_pattern = r"^[a-zA-Z][a-zA-Z0-9_\.\:]*$"

    def __init__(self, _value):
        # don't need to process _value, handled by str.__new__
        parts = self.split("-")

        if len(parts) != 2:
            raise ValueError(
                f"{self.__class__.__name__} must have the form `{{name}}-{{uuid}}` with uuid in hex form"
            )

        self._name = parts[0]
        self._uuid = parts[1]

        if not re.fullmatch(self._allowed_name_pattern, self.name):
            raise ValueError(
                f"{self.__class__.__name__} must either start with an alphabetical and contain "
                "only alphanumeric, underscores ('_'), periods ('.'), or colons (':') thereafter"
            )

        try:
            UUID(self.uuid)
        except ValueError:
            raise ValueError("Could not interpret the provided UUID.")

    def to_dict(self):
        return {"name": self.name, "uuid": self.uuid}

    @classmethod
    def from_dict(cls, dct):
        name = dct["name"]
        uuid = dct["uuid"]
        return cls(name + "-" + uuid)

    @property
    def name(self) -> str:
        return self._name

    @property
    def uuid(self) -> str:
        return self._uuid

    @classmethod
    def new_from_name(cls, name: str):
        return cls(f"{name}-{uuid4().hex}")


class ComputeServiceID(ComputeIDBase): ...


class ComputeManagerID(ComputeIDBase): ...


class ComputeServiceRegistration(BaseModel):
    """Registration for AlchemiscaleComputeService instances."""

    identifier: ComputeServiceID
    registered: datetime.datetime
    heartbeat: datetime.datetime
    failure_times: list[datetime.datetime] = []
    manager_name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self):  # pragma: no cover
        return f"<ComputeServiceRegistration('{str(self)}')>"

    def __str__(self):
        return "-".join([self.identifier])

    @classmethod
    def from_now(cls, identifier: ComputeServiceID):
        now = datetime.datetime.now(tz=datetime.UTC)
        return cls(
            identifier=identifier, registered=now, heartbeat=now, failure_times=[]
        )

    def to_dict(self):
        dct = self.model_dump()
        dct["identifier"] = str(self.identifier)

        return dct

    @classmethod
    def from_dict(cls, dct):
        dct_ = copy(dct)
        dct_["identifier"] = ComputeServiceID(dct_["identifier"])

        return cls(**dct_)


class ComputeManagerInstruction(StrEnum):
    OK = "OK"
    SKIP = "SKIP"
    SHUTDOWN = "SHUTDOWN"


class ComputeManagerStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class ComputeManagerRegistration(BaseModel):
    """Registration shared by all instances of a compute manager with the
    same name.

    Besides its status, the registration holds the pause state that
    instances share. Fields listed in :attr:`TRANSITION_FIELDS` change only
    through a compare-and-set on :attr:`version`.

    """

    name: str
    uuid: str
    last_status_update: datetime.datetime
    status: str
    detail: str | None = None
    saturation: float | None = None
    registered: datetime.datetime

    #: incremented whenever a state transition is applied
    version: int = 0
    #: time of the most recent sign of health
    last_healthy_at: datetime.datetime | None = None
    #: start of the current backend outage, if any
    outage_started_at: datetime.datetime | None = None
    #: why the manager is DEGRADED: "workload", "permanent" or "outage"
    pause_cause: str | None = None
    #: current probe backoff, in seconds
    backoff_seconds: float | None = None
    #: no probe is sent before this time
    backoff_until: datetime.datetime | None = None
    #: name of the outstanding probe job, if any
    probe_job: str | None = None
    #: start of probation after a passing probe, if on probation
    probation_started_at: datetime.datetime | None = None
    #: compute identities that created or joined the registration
    identities: list[str] = []

    TRANSITION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "status",
            "detail",
            "pause_cause",
            "backoff_seconds",
            "backoff_until",
            "probe_job",
            "probation_started_at",
        }
    )

    def __repr__(self):  # pragma: no cover
        return f"<ComputeManagerRegistration('{str(self)}')>"

    def __str__(self):
        return "-".join([self.name, self.uuid])

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_compute_manager_id(self):
        return ComputeManagerID("-".join([self.name, self.uuid]))


class ComputeManagerError(BaseModel):
    """An entry in a compute manager's error history, kept by name
    independently of its registration.

    Besides errors, the history records events in the manager's pause and
    recovery, such as a probe passing, with the status reported alongside.

    """

    name: str
    time: datetime.datetime
    status: str
    detail: str

    def to_dict(self):
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, dct):
        return cls.model_validate(dct)


class JobRecordKind(StrEnum):
    """Kinds of :class:`ComputeManagerJobRecord`."""

    #: the job was seen pending or running
    ACTIVE = "active"
    #: the job's compute service was seen registered with the server
    REGISTERED = "registered"
    #: the job was found running without a registered compute service
    UNREGISTERED = "unregistered"
    #: the job failed
    FAILURE = "failure"
    #: the job's terminal state has been handled, or it vanished
    HANDLED = "handled"


class ComputeManagerJobRecord(BaseModel):
    """A fact about one job of a compute manager, stored on its registration.

    Records are keyed by ``(job_name, kind)`` and merge by union: writing a
    record that already exists leaves the stored one unchanged.

    """

    job_name: str
    kind: JobRecordKind
    #: when the record was first written
    at: datetime.datetime
    submitted_at: datetime.datetime | None = None
    ended_at: datetime.datetime | None = None
    #: for FAILURE records, a :class:`alchemiscale.compute.backend.FailureKind` value
    failure_kind: str | None = None
    reason: str | None = None

    def to_dict(self):
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, dct):
        return cls.model_validate(dct)


class ComputeManagerState(BaseModel):
    """A compute manager's registration, together with its job records."""

    registration: ComputeManagerRegistration
    records: list[ComputeManagerJobRecord] = []

    def to_dict(self):
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, dct):
        return cls.model_validate(dct)

    def records_of(self, kind: JobRecordKind) -> dict[str, ComputeManagerJobRecord]:
        """Return the records of one kind, keyed by job name."""
        return {r.job_name: r for r in self.records if r.kind == kind}


class ComputeManagerStateUpdate(BaseModel):
    """Changes to a compute manager's shared state, sent with a status update.

    Everything except ``transition`` merges without conflict and is always
    applied. ``transition`` is applied only if ``expected_version`` matches
    the registration's ``version``, and the update is for the registration's
    uuid.

    """

    #: records to add; existing records are left unchanged
    records: list[ComputeManagerJobRecord] = []
    #: ``(job_name, kind)`` pairs of records to remove
    forget: list[tuple[str, JobRecordKind]] = []
    #: merged into the registration's ``last_healthy_at`` by maximum
    last_healthy_at: datetime.datetime | None = None
    #: ``"start"`` records the start of a backend outage if none is under
    #: way; ``"clear"`` ends it
    outage: Literal["start", "clear"] | None = None
    #: values for :attr:`ComputeManagerRegistration.TRANSITION_FIELDS`,
    #: other than ``status`` and ``detail``
    transition: dict[str, Any] = {}
    #: the registration ``version`` the transition was decided from;
    #: ``None`` applies the status unconditionally
    expected_version: int | None = None
    #: descriptions of events in the manager's pause and recovery, recorded
    #: in its error history if the transition is applied
    events: list[str] = []

    def to_dict(self):
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, dct):
        return cls.model_validate(dct)


class TaskProvenance(BaseModel):
    computeserviceid: ComputeServiceID
    datetime_start: datetime.datetime
    datetime_end: datetime.datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # this should include versions of various libraries


class TaskStatusEnum(Enum):
    complete = "complete"
    waiting = "waiting"
    running = "running"
    error = "error"
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
    claim: str | None
    datetime_created: datetime.datetime | None
    creator: str | None
    extends: str | None

    def __init__(
        self,
        *,
        status: str | TaskStatusEnum = TaskStatusEnum.waiting,
        priority: int = 10,
        datetime_created: datetime.datetime | None = None,
        creator: str | None = None,
        extends: str | None = None,
        claim: str | None = None,
        _key: str = None,
    ):
        if _key is not None:
            self._key = GufeKey(_key)

        self.status: TaskStatusEnum = TaskStatusEnum(status)
        self.priority = priority

        self.datetime_created = (
            datetime_created
            if datetime_created is not None
            else datetime.datetime.now(tz=datetime.UTC)
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


class TaskRestartPattern(GufeTokenizable):
    """A pattern to compare returned Task tracebacks to.

    Attributes
    ----------
    pattern: str
        A regular expression pattern that can match to returned tracebacks of errored Tasks.
    max_retries: int
        The number of times the pattern can trigger a restart for a Task.
    taskhub_sk: str
        The TaskHub the pattern is bound to. This is needed to properly set a unique Gufe key.
    """

    pattern: str
    max_retries: int
    taskhub_sk: str

    def __init__(
        self, pattern: str, max_retries: int, taskhub_scoped_key: str | ScopedKey
    ):

        if not isinstance(pattern, str) or pattern == "":
            raise ValueError("`pattern` must be a non-empty string")

        self.pattern = pattern

        if not isinstance(max_retries, int) or max_retries <= 0:
            raise ValueError("`max_retries` must have a positive integer value.")
        self.max_retries = max_retries

        self.taskhub_scoped_key = str(taskhub_scoped_key)

    def _gufe_tokenize(self):
        key_string = self.pattern + self.taskhub_scoped_key
        return hashlib.md5(key_string.encode()).hexdigest()

    @classmethod
    def _defaults(cls):
        return super()._defaults()

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)

    def _to_dict(self):
        return {
            "pattern": self.pattern,
            "max_retries": self.max_retries,
            "taskhub_scoped_key": self.taskhub_scoped_key,
        }


class Tracebacks(GufeTokenizable):
    """
    Attributes
    ----------
    tracebacks: list[str]
        The tracebacks returned with the ProtocolUnitFailures.
    source_keys: list[GufeKey]
        The GufeKeys of the ProtocolUnits that failed.
    failure_keys: list[GufeKey]
        The GufeKeys of the ProtocolUnitFailures.
    """

    def __init__(
        self,
        tracebacks: list[str],
        source_keys: list[GufeKey],
        failure_keys: list[GufeKey],
    ):
        value_error = ValueError(
            "`tracebacks` must be a non-empty list of non-empty string values"
        )
        if not isinstance(tracebacks, list) or tracebacks == []:
            raise value_error

        all_string_values = all([isinstance(value, str) for value in tracebacks])
        if not all_string_values or "" in tracebacks:
            raise value_error

        # TODO: validate
        self.tracebacks = tracebacks
        self.source_keys = source_keys
        self.failure_keys = failure_keys

    @classmethod
    def _defaults(cls):
        return super()._defaults()

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)

    def _to_dict(self):
        return {
            "tracebacks": self.tracebacks,
            "source_keys": self.source_keys,
            "failure_keys": self.failure_keys,
        }


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


class Mark(GufeTokenizable):

    def __init__(self, target: ScopedKey):
        self.target = str(target)

    @abstractmethod
    def _to_dict(self):
        raise NotImplementedError

    def _gufe_tokenize(self):
        hash_string = str(self.target)
        return hashlib.md5(hash_string.encode(), usedforsecurity=False).hexdigest()

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


class NetworkStateEnum(Enum):
    active = "active"
    inactive = "inactive"
    invalid = "invalid"
    deleted = "deleted"


class NetworkMark(Mark):
    """Mark object for AlchemicalNetworks.

    Attributes
    ----------
    network : str
        ScopedKey of the AlchemicalNetwork this NetworkMark corresponds to.
        Used to ensure that there is only one NetworkMark for a given
        AlchemicalNetwork using neo4j constraints.

    state : NetworkStateEnum
        State of the AlchemicalNetwork, stored on this NetworkMark.
    """

    network: str
    state: NetworkStateEnum

    def __init__(
        self,
        target: ScopedKey,
        state: str | NetworkStateEnum = NetworkStateEnum.active,
    ):
        self.state = state
        super().__init__(target)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state_value):
        try:
            self._state = NetworkStateEnum(state_value)
        except ValueError:
            valid_states_string = ", ".join(sorted([i.value for i in NetworkStateEnum]))
            msg = f"`state` = {state_value} must be one of the following: {valid_states_string}"
            raise ValueError(msg)

    def _to_dict(self):
        return {"target": self.target, "state": self._state.value}


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
    location: str | None
    obj_key: GufeKey | None
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
    ok: bool

    def __init__(
        self,
        *,
        location: str | None = None,
        obj_key: GufeKey,
        scope: Scope,
        ok: bool,
        datetime_created: datetime.datetime | None = None,
        creator: str | None = None,
    ):
        self.location = location
        self.obj_key = GufeKey(obj_key)
        self.scope = scope
        self.ok = ok
        self.datetime_created = datetime_created
        self.creator = creator

    def _to_dict(self):
        return {
            "location": self.location,
            "obj_key": str(self.obj_key),
            "scope": str(self.scope),
            "ok": self.ok,
            "datetime_created": (
                self.datetime_created.isoformat()
                if self.datetime_created is not None
                else None
            ),
            "creator": self.creator,
        }

    @classmethod
    def _from_dict(cls, d):
        d_ = copy(d)
        d_["datetime_created"] = (
            datetime.datetime.fromisoformat(d["datetime_created"])
            if d.get("received") is not None
            else None
        )

        return super()._from_dict(d_)


class StrategyModeEnum(StrEnum):
    full = "full"
    partial = "partial"
    disabled = "disabled"


class StrategyStatusEnum(StrEnum):
    awake = "awake"
    dormant = "dormant"
    error = "error"


class StrategyTaskScalingEnum(StrEnum):
    linear = "linear"
    exponential = "exponential"


class StrategyState(BaseModel):
    """State information for a Strategy on an AlchemicalNetwork."""

    mode: StrategyModeEnum = StrategyModeEnum.partial
    status: StrategyStatusEnum = StrategyStatusEnum.awake
    iterations: int = 0
    sleep_interval: PositiveInt = 3600  # seconds
    last_iteration: datetime.datetime | None = None
    last_iteration_result_count: int = 0
    max_tasks_per_transformation: PositiveInt = 3
    task_scaling: StrategyTaskScalingEnum = StrategyTaskScalingEnum.exponential
    exception: tuple[str, str] | None = None  # (exception_type, exception_message)
    traceback: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("last_iteration", mode="before")
    @classmethod
    def _convert_neo4j_datetime(cls, v):
        """Convert neo4j DateTime objects to Python datetime objects."""
        if v is not None and hasattr(v, "to_native"):
            # This is a neo4j DateTime object
            return v.to_native()
        return v

    def to_dict(self):
        dct = self.dict()

        dct["mode"] = dct["mode"].value
        dct["status"] = dct["status"].value
        dct["task_scaling"] = dct["task_scaling"].value

        return dct

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
