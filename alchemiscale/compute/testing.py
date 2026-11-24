"""
:mod:`alchemiscale.compute.testing` --- test helpers for compute managers
=========================================================================

"""

import datetime
from datetime import timedelta

from .backend import (
    ComputeBackend,
    FailureKind,
    Job,
    JobState,
    PermanentBackendError,
    TransientBackendError,
)


class FakeComputeBackend(ComputeBackend):
    """An in-memory :class:`~alchemiscale.compute.backend.ComputeBackend`, for
    testing :class:`~alchemiscale.compute.manager.ComputeManager` without a
    cluster.

    Jobs are submitted PENDING; tests move them through their lifecycle with
    :meth:`start`, :meth:`succeed`, :meth:`fail` and :meth:`vanish`. Several
    managers may share one backend, as instances of one manager share a batch
    system.

    Set :attr:`list_error`, :attr:`submit_error` or :attr:`remove_error` to an
    exception to have the corresponding method raise it.

    """

    def __init__(
        self,
        finished_job_lookback: timedelta | None = None,
        validate_error: Exception | None = None,
    ):
        self.finished_job_lookback = finished_job_lookback
        self.validate_error = validate_error
        self.jobs: dict[str, Job] = {}
        self.removed: list[str] = []
        self.list_error: Exception | None = None
        self.submit_error: Exception | None = None
        self.remove_error: Exception | None = None

    @staticmethod
    def now() -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.UTC)

    def validate(self) -> None:
        if self.validate_error is not None:
            raise self.validate_error

    def list_jobs(self) -> list[Job]:
        if self.list_error is not None:
            raise self.list_error
        return list(self.jobs.values())

    def submit(self, name: str) -> None:
        if self.submit_error is not None:
            raise self.submit_error
        if name not in self.jobs:
            self.jobs[name] = Job(
                name=name, state=JobState.PENDING, submitted_at=self.now()
            )

    def remove(self, job: Job) -> None:
        if self.remove_error is not None:
            raise self.remove_error
        if self.finished_job_lookback is None or not job.state.terminal:
            self.jobs.pop(job.name, None)
        self.removed.append(job.name)

    # lifecycle, driven by tests

    def names(self, state: JobState | None = None) -> list[str]:
        return [
            name
            for name, job in self.jobs.items()
            if state is None or job.state == state
        ]

    def start(self, name: str, started_at: datetime.datetime | None = None):
        job = self.jobs[name]
        job.state = JobState.RUNNING
        job.started_at = started_at or self.now()

    def succeed(self, name: str):
        job = self.jobs[name]
        job.state = JobState.SUCCEEDED
        job.started_at = job.started_at or self.now()
        job.ended_at = self.now()

    def fail(
        self,
        name: str,
        kind: FailureKind = FailureKind.WORKLOAD,
        reason: str | None = None,
    ):
        job = self.jobs[name]
        job.state = JobState.FAILED
        job.started_at = job.started_at or self.now()
        job.ended_at = self.now()
        job.failure = kind
        job.reason = reason

    def vanish(self, name: str):
        del self.jobs[name]


__all__ = [
    "FakeComputeBackend",
    "PermanentBackendError",
    "TransientBackendError",
]
