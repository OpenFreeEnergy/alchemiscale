"""
:mod:`alchemiscale.compute.backend` --- interface between compute managers and batch systems
============================================================================================

A :class:`ComputeBackend` reports facts about a compute manager's jobs on a
batch system, and submits and removes them.
:class:`~alchemiscale.compute.manager.ComputeManager` makes every decision
from those facts.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, StrEnum
import re
from uuid import uuid4


class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED)


class FailureKind(StrEnum):
    """Why a job failed, as far as the batch system can tell."""

    #: the platform failed the job, e.g. preemption, a lost node, or a
    #: rejection at admission; never counts against the manager
    INFRASTRUCTURE = "INFRASTRUCTURE"
    #: the job itself failed, e.g. a non-zero exit or running out of memory
    WORKLOAD = "WORKLOAD"


@dataclass
class Job:
    """A job of a compute manager, as reported by its backend."""

    name: str
    state: JobState
    submitted_at: datetime
    #: ``None`` until the job starts running
    started_at: datetime | None = None
    #: set once the job is SUCCEEDED or FAILED
    ended_at: datetime | None = None
    #: set if the job is FAILED; a failure the backend can't classify is
    #: left ``None``, and treated as WORKLOAD, so that it counts against the
    #: manager rather than being retried forever (log the raw reason, so the
    #: classification can be extended)
    failure: FailureKind | None = None
    #: a human-readable explanation of the job's state, for logs
    reason: str | None = None


class BackendError(Exception):
    """Base class for errors raised by a :class:`ComputeBackend`.

    Raise one of its subclasses, chained from the original exception.

    """


class TransientBackendError(BackendError):
    """An error expected to clear without a change, such as a batch system
    restarting, an API timeout, or a submission queue that is full.

    The backend should retry briefly before raising it.

    """


class PermanentBackendError(BackendError):
    """An error that won't clear without a change, such as an invalid
    partition, account or QoS, or missing permissions."""


class ComputeBackend(ABC):
    """Driver for the batch system a compute manager submits jobs to.

    A backend reports facts about its jobs and doesn't make policy decisions:
    :class:`~alchemiscale.compute.manager.ComputeManager` decides when to
    submit and remove jobs, and how to treat failures.

    Each job starts a compute service that registers with the alchemiscale
    server under the job's name, which the manager uses to match jobs to
    compute services.

    Methods may raise :class:`TransientBackendError` or
    :class:`PermanentBackendError`. Methods are called synchronously from the
    manager's cycle, so each must return, or raise, well within
    ``ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS``: bound any retries and
    timeouts.

    """

    #: How long a terminal job keeps appearing in :meth:`list_jobs` after it
    #: ends, or ``None`` if :meth:`remove` deletes it (e.g. Kubernetes).
    finished_job_lookback: timedelta | None = None

    #: The ID of the manager's registration, set by the manager before each
    #: cycle; pass it to the compute services a job starts, as their
    #: ``compute_manager_id``.
    compute_manager_id: str | None = None

    @abstractmethod
    def validate(self) -> None:
        """Check settings and templates at startup.

        Raise :class:`PermanentBackendError` if they're invalid; the manager
        then exits.

        """

    @abstractmethod
    def list_jobs(self) -> list[Job]:
        """Return this manager's jobs, read from the batch system.

        Use :func:`is_own_job_name` to pick this manager's jobs out of the
        batch system's; never keep the list in memory.

        """

    @abstractmethod
    def submit(self, name: str) -> None:
        """Submit a job named ``name`` that starts a compute service
        registering under ``name``.

        Submitting a name that is already submitted does nothing.

        """

    @abstractmethod
    def remove(self, job: Job) -> None:
        """Remove ``job`` from the batch system, cancelling it if it hasn't
        ended.

        Removing a job that is already gone does nothing, and a backend that
        can't delete finished jobs (e.g. SLURM accounting records) may do
        nothing for them.

        """


def make_job_name(manager_name: str) -> str:
    """Return a new, unique job name for the compute manager ``manager_name``.

    Job names have the form ``{manager_name}.{uuid_hex}``, with the 32
    character lowercase hex form of a :func:`uuid.uuid4`. A compute service
    started by the job registers under this name, so its ``ComputeServiceID``
    is ``{job_name}-{uuid_hex}``.

    """
    return f"{manager_name}.{uuid4().hex}"


def is_own_job_name(manager_name: str, job_name: str) -> bool:
    """Return whether ``job_name`` was made by :func:`make_job_name` for the
    compute manager ``manager_name``.

    This is an exact match, not a prefix check: manager names may contain
    ``.``, and manager ``foo`` must not claim the jobs of manager ``foo.bar``
    (named ``foo.bar.{uuid_hex}``) or ``foobar``.

    """
    return (
        re.fullmatch(rf"{re.escape(manager_name)}\.[0-9a-f]{{32}}", job_name)
        is not None
    )
