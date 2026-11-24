**Added:**

* ``alchemiscale.compute.backend``, with the ``ComputeBackend`` interface that compute managers run on: ``validate()``, ``list_jobs()``, ``submit(name)`` and ``remove(job)``, the ``Job``, ``JobState`` and ``FailureKind`` types it reports, the ``TransientBackendError`` and ``PermanentBackendError`` exceptions it raises, and ``make_job_name`` and ``is_own_job_name``.
* ``alchemiscale.compute.testing.FakeComputeBackend``, an in-memory backend for testing compute managers without a cluster.
* ``ComputeManagerSettings.registration_grace_period``, the time a running job's compute service may take to register, counted from when the job started running.

**Changed:**

* ``ComputeManager`` takes a ``ComputeBackend`` and runs the whole autoscaling cycle. Every cycle, whatever the instruction, it lists jobs, handles each terminal job once, checks that running jobs have registered, and submits if there is work and capacity. Facts about jobs are kept on the shared registration, so instances under one name behave like one manager.
* Capacity counts distinct names across pending and running jobs and registered compute services.
* A job failure's recorded reason says whether it failed before or after its compute service registered. A failure the backend can't classify counts as ``WORKLOAD``.

**Removed:**

* ``ComputeManager.create_compute_services``; implement a ``ComputeBackend`` instead.
