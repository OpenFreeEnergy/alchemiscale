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


from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator
from gufe.tokenization import GufeTokenizable, GufeKey

from ..models import ScopedKey, Scope


def _coerce_datetime(v) -> datetime.datetime | None:
    """Coerce a neo4j ``DateTime``, ISO string, or ``datetime`` to ``datetime``."""
    if v is None:
        return None
    if hasattr(v, "to_native"):
        return v.to_native()
    if isinstance(v, str):
        return datetime.datetime.fromisoformat(v)
    return v


def _iso(v: datetime.datetime | None) -> str | None:
    return v.isoformat() if v is not None else None


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
    hostname: str | None = None

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
    ERROR = "ERROR"


class ComputeManagerRegistration(BaseModel):

    name: str
    uuid: str
    last_status_update: datetime.datetime
    status: str
    detail: str
    saturation: float
    registered: datetime.datetime

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


class TaskOutcomeEnum(Enum):
    """Terminal outcome of a single execution attempt of a `Task`.

    Attributes
    ----------
    complete
        The attempt produced a successful `ProtocolDAGResult`.
    error
        The attempt produced a failed `ProtocolDAGResult`, or errored during
        `ProtocolDAG` creation.
    expired
        The attempt's compute service lost its registration (expiry or
        deregistration) before the attempt produced a result.
    released
        A user forced the claimed `Task` to another status (e.g. `waiting`,
        `invalid`, `deleted`) before the attempt produced a result.
    """

    complete = "complete"
    error = "error"
    expired = "expired"
    released = "released"


class TaskProvenance(GufeTokenizable):
    """A record of a single execution attempt of a `Task`.

    A `TaskProvenance` node is created at claim time and finalized when the
    attempt ends. It survives claim teardown and registration expiry, so that
    the history of *who ran what, when* is preserved. The identifying
    information (compute service id, hostname, manager name) is copied onto the
    record rather than held as a relationship to the (potentially deleted)
    `ComputeServiceRegistration`.

    It is a `GufeTokenizable` so it carries a `ScopedKey`: provenance is a
    scoped entity, authorized through the same ``validate_scopes(sk.scope,
    token)`` path as every other scoped object, rather than relying on an
    anchoring `Task`. Like `Task`, it tokenizes on a uuid (see
    `_gufe_tokenize`), *not* its contents, so its `GufeKey`/`ScopedKey` is fixed
    at creation and unaffected by the in-place mutations (`outcome`,
    `datetime_end`, and the progress counts) applied over the attempt's life.
    Because of that, a `TaskProvenance` must never be re-tokenized after
    creation (never round-tripped object -> node a second time); mutations go
    straight to the node via Cypher.

    Attributes
    ----------
    compute_service_id
        The identifier of the compute service that claimed the `Task`.
    hostname
        The hostname of the compute service, copied from its registration.
    manager_name
        The name of the compute manager responsible for the compute service,
        if any.
    datetime_claimed
        When the `Task` was claimed for this attempt.
    datetime_end
        When the attempt was finalized; `None` while the attempt is open.
    outcome
        The terminal outcome of the attempt; `None` while the attempt is open.
    units_completed
        The number of distinct `ProtocolUnit` objects successfully completed in this
        attempt, as of the last progress update.
    units_total
        The total number of `ProtocolUnit` objects in the attempt's `ProtocolDAG`.
    """

    compute_service_id: ComputeServiceID
    hostname: str | None
    manager_name: str | None
    datetime_claimed: datetime.datetime | None
    datetime_end: datetime.datetime | None
    outcome: TaskOutcomeEnum | None
    units_completed: int | None
    units_total: int | None

    def __init__(
        self,
        *,
        compute_service_id: ComputeServiceID | str,
        datetime_claimed: datetime.datetime | None = None,
        hostname: str | None = None,
        manager_name: str | None = None,
        datetime_end: datetime.datetime | None = None,
        outcome: str | TaskOutcomeEnum | None = None,
        units_completed: int | None = None,
        units_total: int | None = None,
        _key: str = None,
    ):
        if _key is not None:
            self._key = GufeKey(_key)

        self.compute_service_id = ComputeServiceID(compute_service_id)
        self.hostname = hostname
        self.manager_name = manager_name
        self.datetime_claimed = _coerce_datetime(datetime_claimed)
        self.datetime_end = _coerce_datetime(datetime_end)
        self.outcome = TaskOutcomeEnum(outcome) if outcome is not None else None
        self.units_completed = units_completed
        self.units_total = units_total

    def _gufe_tokenize(self):
        # tokenize with a uuid, not content: identity is per-attempt, and the
        # record is mutated in place (outcome/datetime_end/progress) after
        # creation, so a content hash would neither be stable nor unique.
        return uuid4().hex

    def _to_dict(self):
        return {
            "compute_service_id": str(self.compute_service_id),
            "hostname": self.hostname,
            "manager_name": self.manager_name,
            "datetime_claimed": self.datetime_claimed,
            "datetime_end": self.datetime_end,
            "outcome": self.outcome.value if self.outcome is not None else None,
            "units_completed": self.units_completed,
            "units_total": self.units_total,
            "_key": str(self.key),
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


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
        Identifier of the compute service that has a claim on this task, if any.
    datetime_created
        When the `Task` was created.
    datetime_status_changed
        When the `Task`'s `status` was last changed; refreshed at every
        status-mutation site (claim, `set_task_*`, expiry, deregistration,
        restart-renew), but not on an idempotent no-op re-set of the same status.
    reason
        Human-readable reason for the current `status`, if any --- e.g. a
        `ProtocolDAG` creation-failure traceback, or a user-supplied reason for
        an `invalid`/`deleted` transition. Cleared when the status changes to
        one that carries no reason.
    creator
        Identifier of the identity that created the `Task`, if recorded.
    extends
        `ScopedKey` (as a string) of the `Task` this one extends (continues
        from), if any.

    """

    status: TaskStatusEnum
    priority: int
    claim: str | None
    datetime_created: datetime.datetime | None
    datetime_status_changed: datetime.datetime | None
    reason: str | None
    creator: str | None
    extends: str | None

    def __init__(
        self,
        *,
        status: str | TaskStatusEnum = TaskStatusEnum.waiting,
        priority: int = 10,
        datetime_created: datetime.datetime | None = None,
        datetime_status_changed: datetime.datetime | None = None,
        reason: str | None = None,
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

        self.datetime_status_changed = datetime_status_changed
        self.reason = reason
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
            "datetime_status_changed": self.datetime_status_changed,
            "reason": self.reason,
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


class ProtocolUnitResultRef(ObjectStoreRef):
    """A reference to the artifacts of a single `ProtocolUnitResult` or
    `ProtocolUnitFailure` within a `ProtocolDAGResult`.

    One `ProtocolUnitResultRef` node is derived per unit result when a
    `ProtocolDAGResultRef` is stored, keyed by the unit result's gufe key. The
    `has_*` flags record which per-unit artifacts (logs, stdout, stderr) are
    present in the object store under `location`.

    Attributes
    ----------
    obj_key
        The gufe key of the `ProtocolUnitResult`/`ProtocolUnitFailure`.
    source_key
        The gufe key of the originating `ProtocolUnit`.
    name
        The name of the unit result, if any.
    ok
        Whether the unit result is a success (`True`) or failure (`False`).
    start_time, end_time
        When execution of the unit attempt began and ended.
    location
        The object store prefix under which this unit result's artifacts live.
    scope
        The `Scope` (org/campaign/project) this reference lives in.
    has_logs, has_stdout, has_stderr
        Whether captured log/stdout/stderr artifacts exist for this unit result.

    Note
    ----
    The ``has_logs``, ``has_stdout``, and ``has_stderr`` flags (and nothing
    else) are mutated in place via Cypher after
    the node is created, as artifacts arrive. The node's `_scoped_key` and
    `_gufe_key`
    are computed once at creation and never recomputed, so lookups stay stable
    even though these tokenizable-contributing fields change. This is safe only
    because `ProtocolUnitResultRef` nodes are an internal state-store detail,
    never re-tokenized after creation; keep it that way.
    """

    ok: bool
    source_key: GufeKey
    name: str | None
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None
    has_logs: bool
    has_stdout: bool
    has_stderr: bool

    def __init__(
        self,
        *,
        location: str | None = None,
        obj_key: GufeKey,
        source_key: GufeKey,
        scope: Scope,
        ok: bool,
        name: str | None = None,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
        has_logs: bool = False,
        has_stdout: bool = False,
        has_stderr: bool = False,
    ):
        self.location = location
        self.obj_key = GufeKey(obj_key)
        self.source_key = GufeKey(source_key)
        self.scope = scope
        self.ok = ok
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.has_logs = has_logs
        self.has_stdout = has_stdout
        self.has_stderr = has_stderr

    def _to_dict(self):
        return {
            "location": self.location,
            "obj_key": str(self.obj_key),
            "source_key": str(self.source_key),
            "scope": str(self.scope),
            "ok": self.ok,
            "name": self.name,
            "start_time": (
                self.start_time.isoformat() if self.start_time is not None else None
            ),
            "end_time": (
                self.end_time.isoformat() if self.end_time is not None else None
            ),
            "has_logs": self.has_logs,
            "has_stdout": self.has_stdout,
            "has_stderr": self.has_stderr,
        }

    @classmethod
    def _from_dict(cls, d):
        d_ = copy(d)
        d_["scope"] = Scope.from_str(d["scope"])
        d_["source_key"] = GufeKey(d["source_key"])
        d_["start_time"] = (
            datetime.datetime.fromisoformat(d["start_time"])
            if d.get("start_time") is not None
            else None
        )
        d_["end_time"] = (
            datetime.datetime.fromisoformat(d["end_time"])
            if d.get("end_time") is not None
            else None
        )
        return cls(**d_)

    @classmethod
    def _defaults(cls):
        return super()._defaults()


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


# --- client-facing API record models --------------------------------------
#
# These models are the user-facing surface for Task introspection. They are
# deliberately decoupled from the state-store node types (`TaskProvenance`,
# `ProtocolDAGResultRef`, `ProtocolUnitResultRef`): the two families evolve
# independently, joined only by `ScopedKey`s. User-level names are used
# throughout; no storage jargon (`Ref` suffixes, `pdrr`/`purr`) leaks in.


class TaskAttempt(BaseModel):
    """A single execution attempt of a `Task`, as reported by `get_task_history`.

    Bundles a `TaskProvenance` record's properties with the `ScopedKey` of the
    `ProtocolDAGResultRef` the attempt produced (via `PROVENANCE_OF`), where one
    exists; `expired`/`released` attempts have none.
    """

    compute_service_id: str
    hostname: str | None = None
    manager_name: str | None = None
    datetime_claimed: datetime.datetime
    datetime_end: datetime.datetime | None = None
    outcome: TaskOutcomeEnum | None = None
    units_completed: int | None = None
    units_total: int | None = None
    protocoldagresultref: ScopedKey | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "compute_service_id": self.compute_service_id,
            "hostname": self.hostname,
            "manager_name": self.manager_name,
            "datetime_claimed": _iso(self.datetime_claimed),
            "datetime_end": _iso(self.datetime_end),
            "outcome": self.outcome.value if self.outcome is not None else None,
            "units_completed": self.units_completed,
            "units_total": self.units_total,
            "protocoldagresultref": (
                str(self.protocoldagresultref)
                if self.protocoldagresultref is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            compute_service_id=d["compute_service_id"],
            hostname=d.get("hostname"),
            manager_name=d.get("manager_name"),
            datetime_claimed=_coerce_datetime(d["datetime_claimed"]),
            datetime_end=_coerce_datetime(d.get("datetime_end")),
            outcome=(
                TaskOutcomeEnum(d["outcome"]) if d.get("outcome") is not None else None
            ),
            units_completed=d.get("units_completed"),
            units_total=d.get("units_total"),
            protocoldagresultref=(
                ScopedKey.from_str(d["protocoldagresultref"])
                if d.get("protocoldagresultref") is not None
                else None
            ),
        )


class TaskClaim(BaseModel):
    """The live claim currently held on a `running` `Task`."""

    compute_service_id: str
    hostname: str | None = None
    datetime_claimed: datetime.datetime | None = None
    units_completed: int | None = None
    units_total: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "compute_service_id": self.compute_service_id,
            "hostname": self.hostname,
            "datetime_claimed": _iso(self.datetime_claimed),
            "units_completed": self.units_completed,
            "units_total": self.units_total,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            compute_service_id=d["compute_service_id"],
            hostname=d.get("hostname"),
            datetime_claimed=_coerce_datetime(d.get("datetime_claimed")),
            units_completed=d.get("units_completed"),
            units_total=d.get("units_total"),
        )


class TaskDetails(BaseModel):
    """Bulk indicator summary for a `Task`, as returned by `get_tasks_details`."""

    task: ScopedKey
    status: TaskStatusEnum
    datetime_status_changed: datetime.datetime | None = None
    reason: str | None = None
    num_claims: int = 0
    current_claim: TaskClaim | None = None
    most_recent_attempt: TaskAttempt | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "task": str(self.task),
            "status": self.status.value,
            "datetime_status_changed": _iso(self.datetime_status_changed),
            "reason": self.reason,
            "num_claims": self.num_claims,
            "current_claim": (
                self.current_claim.to_dict() if self.current_claim is not None else None
            ),
            "most_recent_attempt": (
                self.most_recent_attempt.to_dict()
                if self.most_recent_attempt is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            task=ScopedKey.from_str(d["task"]),
            status=TaskStatusEnum(d["status"]),
            datetime_status_changed=_coerce_datetime(d.get("datetime_status_changed")),
            reason=d.get("reason"),
            num_claims=d.get("num_claims", 0),
            current_claim=(
                TaskClaim.from_dict(d["current_claim"])
                if d.get("current_claim") is not None
                else None
            ),
            most_recent_attempt=(
                TaskAttempt.from_dict(d["most_recent_attempt"])
                if d.get("most_recent_attempt") is not None
                else None
            ),
        )


class TaskUnitTraceback(BaseModel):
    """A single `ProtocolUnitFailure` traceback within a `TaskTracebacks`."""

    failure_key: GufeKey
    source_key: GufeKey
    traceback: str
    protocolunitresultref: ScopedKey | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "failure_key": str(self.failure_key),
            "source_key": str(self.source_key),
            "traceback": self.traceback,
            "protocolunitresultref": (
                str(self.protocolunitresultref)
                if self.protocolunitresultref is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            failure_key=GufeKey(d["failure_key"]),
            source_key=GufeKey(d["source_key"]),
            traceback=d["traceback"],
            protocolunitresultref=(
                ScopedKey.from_str(d["protocolunitresultref"])
                if d.get("protocolunitresultref") is not None
                else None
            ),
        )


class TaskTracebacks(BaseModel):
    """Tracebacks for one failed `ProtocolDAGResult` of a `Task`.

    Returned by `get_task_tracebacks`, one record per failed
    `ProtocolDAGResultRef`, most recent first.
    """

    protocoldagresultref: ScopedKey
    datetime_created: datetime.datetime | None = None
    creator: str | None = None
    tracebacks: list[TaskUnitTraceback] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "protocoldagresultref": str(self.protocoldagresultref),
            "datetime_created": _iso(self.datetime_created),
            "creator": self.creator,
            "tracebacks": [tb.to_dict() for tb in self.tracebacks],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            protocoldagresultref=ScopedKey.from_str(d["protocoldagresultref"]),
            datetime_created=_coerce_datetime(d.get("datetime_created")),
            creator=d.get("creator"),
            tracebacks=[TaskUnitTraceback.from_dict(tb) for tb in d["tracebacks"]],
        )


class ProtocolDAGResultRec(BaseModel):
    """A record describing one `ProtocolDAGResult` of a `Task`.

    Returned by `get_task_result_recs`. Carries the `ScopedKey` of the
    underlying `ProtocolDAGResultRef` as `scoped_key`, which every drill-down
    method accepts directly.
    """

    scoped_key: ScopedKey
    ok: bool
    datetime_created: datetime.datetime | None = None
    creator: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "scoped_key": str(self.scoped_key),
            "ok": self.ok,
            "datetime_created": _iso(self.datetime_created),
            "creator": self.creator,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            scoped_key=ScopedKey.from_str(d["scoped_key"]),
            ok=d["ok"],
            datetime_created=_coerce_datetime(d.get("datetime_created")),
            creator=d.get("creator"),
        )


class ProtocolUnitResultRec(BaseModel):
    """A record describing one `ProtocolUnitResult` of a `ProtocolDAGResult`.

    Returned by `get_result_unit_recs`. Carries the `ScopedKey` of the
    underlying `ProtocolUnitResultRef` as `scoped_key`, plus the `obj_key`/
    `source_key` that let a user correlate it against a deserialized
    `ProtocolDAGResult`.
    """

    scoped_key: ScopedKey
    obj_key: GufeKey
    source_key: GufeKey
    name: str | None = None
    ok: bool
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    has_logs: bool = False
    has_stdout: bool = False
    has_stderr: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        return {
            "scoped_key": str(self.scoped_key),
            "obj_key": str(self.obj_key),
            "source_key": str(self.source_key),
            "name": self.name,
            "ok": self.ok,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "has_logs": self.has_logs,
            "has_stdout": self.has_stdout,
            "has_stderr": self.has_stderr,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            scoped_key=ScopedKey.from_str(d["scoped_key"]),
            obj_key=GufeKey(d["obj_key"]),
            source_key=GufeKey(d["source_key"]),
            name=d.get("name"),
            ok=d["ok"],
            start_time=_coerce_datetime(d.get("start_time")),
            end_time=_coerce_datetime(d.get("end_time")),
            has_logs=d.get("has_logs", False),
            has_stdout=d.get("has_stdout", False),
            has_stderr=d.get("has_stderr", False),
        )
