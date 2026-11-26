**Added:**

* Compute managers keep a history of their recent errors by name, independent of their registration. An error is recorded whenever a manager's status changes to ``DEGRADED`` or ``ERROR``, or its detail changes. ``ALCHEMISCALE_COMPUTE_API_MANAGER_ERROR_HISTORY`` sets how many are kept (default 20).
* ``GET /computemanager/{name}`` on the compute API, ``AlchemiscaleComputeManagerClient.get_status``, and the ``alchemiscale compute manager status`` command, which return a compute manager's registration, shared state, and recent errors.
* The error history also records events sent with an applied state transition, through ``ComputeManagerStateUpdate.events``, and notes when a paused manager's registration expires, resetting its pause state.
* ``GET /computemanager/{name}`` and ``get_status`` also return ``failure_count``, the number of jobs that failed with a WORKLOAD failure since the manager's last sign of health.
* ``alchemiscale database migrate v08-to-v09`` copies the detail of each ``ERROR`` registration into the error history.
