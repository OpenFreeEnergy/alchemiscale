.. _compute:

#######
Compute
#######

In order to actually execute :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s to obtain free energy estimates, you must deploy compute services to resources suitable for executing these types of calculations.
This document details how to do this on several different types of compute resources.

There currently exists a single implementation of an ``alchemiscale`` compute service: the :py:class:`~alchemiscale.compute.service.SynchronousComputeService`.
Other variants will likely be created in the future, optimized for different use cases.
This documentation will expand over time as these variants become available; for now, it assumes use of this variant.

In all cases, you will need to define a configuration file for your compute services to consume on startup.
A template for this file can be found here; replace ``$ALCHEMISCALE_VERSION`` with the version tag, e.g. ``v0.1.4``, you have deployed for your server::

    https://raw.githubusercontent.com/OpenFreeEnergy/alchemiscale/$ALCHEMISCALE_VERSION/devtools/configs/synchronous-compute-settings.yaml

.. note::
   Compute services must run the same ``alchemiscale`` version as the compute API they claim
   :py:class:`~alchemiscale.storage.models.Task`\s from.
   The two exchange ``gufe`` objects in a form that is not guaranteed to be stable across versions,
   so upgrade the compute API and its compute services together rather than one at a time.


***********
Single-host
***********

To deploy a compute service (or multiple services) to a single host, we recommend one of two routes:

* installing all dependencies in a ``conda``/``mamba`` environment
* running the services as Docker containers, with all dependencies baked in


.. _compute_conda:

Deploying with conda/mamba
==========================

To deploy via ``conda``/``mamba``, first create an environment (we recommend ``mamba`` for its performance)::

    mamba env create -n alchemiscale-compute-$ALCHEMISCALE_VERSION \
                     -f https://raw.githubusercontent.com/OpenFreeEnergy/alchemiscale/$ALCHEMISCALE_VERSION/devtools/conda-envs/alchemiscale-compute.yml

Once created, activate the environment in your current shell::

    conda activate alchemiscale-compute-$ALCHEMISCALE_VERSION

Then start a compute service, assuming your configuration file is in the current working directory, with::

    alchemiscale compute synchronous -c synchronous-compute-settings.yaml


.. _compute_docker:

Deploying with Docker
=====================

Assuming your configuration file is in the current working directory, to deploy with Docker, you might use::

    docker run --gpus all \
               --rm \
               -v $(pwd):/mnt ghcr.io/OpenFreeEnergy/alchemiscale-compute:$ALCHEMISCALE_VERSION \
               compute synchronous -c /mnt/synchronous-compute-settings.yaml


See the `official Docker documentation on GPU use`_ for details on how to specify individual GPUs for each container you launch.
It may also make sense to apply constraints to the number of CPUs available to each container to avoid oversubscription.


.. _official Docker documentation on GPU use: https://docs.docker.com/config/containers/resource_constraints/#gpu

***********
HPC cluster
***********

To deploy compute services to an HPC cluster, we recommend submitting them as individual jobs to the HPC cluster's scheduler.
Different clusters feature different schedulers (e.g. SLURM, LSF, TORQUE/PBS, etc.), and vary widely in their hardware and queue configurations.
You will need to tailor your specific approach to the constraints of the cluster you are targeting.

The following is an example of the *content* of a script submitted to an HPC cluster. 
We have omitted queuing system-specific options and flags, and certain environment variables (e.g. ``JOBID``, ``JOBINDEX``) should be tailored to those presented by the queuing system.
Note that for this case we've made use of a ``conda``/``mamba``-based deployment, detailed above in :ref:`compute_conda`::

    # don't limit stack size
    ulimit -s unlimited
    
    # make scratch space (path will be HPC system dependent)
    ALCHEMISCALE_SCRATCH=/scratch/${USER}/${JOBID}-${JOBINDEX}
    mkdir -p $ALCHEMISCALE_SCRATCH
    
    # activate environment
    conda activate alchemiscale-compute-$ALCHEMISCALE_VERSION
    
    # create a YAML file with specific substitutions
    # each service in this job can share the same config
    envsubst < settings.yaml > configs/settings.${JOBID}-${JOBINDEX}.yaml
    
    # start up a single service
    alchemiscale compute synchronous -c configs/settings.${JOBID}-${JOBINDEX}.yaml
    
    # remove scratch space
    rm -r $ALCHEMISCALE_SCRATCH


The ``envsubst`` line in particular will make a config specific to this job, with environment variable substitutions.
A subset of options used in the config file are given below::

    ---
    # Filesystem path to use for `ProtocolDAG` `shared` space.
    shared_basedir: "/scratch/${USER}/${JOBID}-${JOBINDEX}/shared"

    # Filesystem path to use for `ProtocolUnit` `scratch` space.
    scratch_basedir: "/scratch/${USER}/${JOBID}-${JOBINDEX}/scratch"

    # Path to file for logging output; if not set, logging will only go to
    # STDOUT.
    logfile: /home/${USER}/logs/service.${JOBID}.log

    # Max number of Tasks to execute before exiting. If `null`, the service will
    # have no task limit.
    max_tasks: 1

    # Max number of seconds to run before exiting. If `null`, the service will
    # have no time limit.
    max_time: 300


For HPC job-based execution, we recommend limiting the number of :py:class:`~alchemiscale.storage.models.Task`\s the compute service executes to a small number, preferrably 1, and setting a time limit beyond which the compute service will shut down.
With this configuration, when a compute service comes up and claims a :py:class:`~alchemiscale.storage.models.Task`, it will have nearly the full walltime of its job to execute it.
Any compute service that fails to claim a :py:class:`~alchemiscale.storage.models.Task` will shut itself down, and the job will exit, avoiding waste and a scenario where a :py:class:`~alchemiscale.storage.models.Task` is claimed without enough walltime left on the job to complete it.


******************
Kubernetes cluster
******************

To deploy compute services to a Kubernetes ("k8s") cluster, we make use of a similar approach to deployment with Docker detailed above in :ref:`compute_docker`.
We define a k8s `Deployment`_ featuring a single container spec as the file ``compute-services.yaml``::

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: alchemiscale-synchronouscompute
      labels:
        app: alchemiscale-synchronouscompute
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: alchemiscale-synchronouscompute
      template:
        metadata:
          labels:
            app: alchemiscale-synchronouscompute
        spec:
          containers:
          - name: alchemiscale-synchronous-container
            image: ghcr.io/OpenFreeEnergy/alchemiscale-compute:$ALCHEMISCALE_VERSION
            args: ["compute", "synchronous", "-c", "/mnt/settings/synchronous-compute-settings.yaml"]
            resources:
              limits:
                cpu: 2
                memory: 12Gi
                ephemeral-storage: 48Gi
                nvidia.com/gpu: 1
              requests:
                cpu: 2
                memory: 12Gi
                ephemeral-storage: 48Gi
            volumeMounts:
              - name: alchemiscale-compute-settings-yaml
                mountPath: "/mnt/settings"
                readOnly: true
            env:
              - name: OPENMM_CPU_THREADS
                value: "2"
          volumes:
            - name: alchemiscale-compute-settings-yaml
              secret:
                secretName: alchemiscale-compute-settings-yaml


This assumes our configuration file has been defined as a *secret* in the cluster.
Assuming the file is in the current working directory, we can add it as a secret with::

    kubectl create secret generic alchemiscale-compute-settings-yaml \
                                  --from-file=synchronous-compute-settings.yaml


Then we can then deploy the compute services with::

    kubectl apply -f compute-services.yaml

To scale up the number of compute services on the cluster, increase ``replicas`` to the number desired, and re-run the ``kubectl apply`` command above.

A more complete example of this type of deployment can be found in `alchemiscale-k8s`_.

****************
Compute managers
****************

Compute manager are dedicated services responsible for creating new compute services in response to the workload present on the statestore.
These services are designed to loosely manage compute services and are assumed to be unable to communicate with their created services by default.
This requires that created compute services are short-lived and deregister themselves appropriately.
Information of existing services and current workload is reported to the manager through an instruction request through the compute API. Managers expect one of two instructions from alchemiscale:

1. ``OK``: the manager may allocate compute resources
2. ``SKIP``: the manager should take no action

After either instruction, a manager is expected to provide a status update to the compute API.
The status can be:

1. ``OK``: the manager is working as expected, requires a saturation value (ratio of active services to the maximum number of services allowed by the manager)
2. ``DEGRADED``: the manager has paused creating compute services, but keeps running; requires a saturation value and a detail string explaining why
3. ``ERROR``: deprecated; compute managers before v0.9 reported it when they shut down after an error, with a detail string explaining why

Currently, these supporting data are only used by administrators for introspection, though this might change in the future.

Registrations
=============

All instances of a compute manager running under the same name share one registration.
The first instance to register creates it, and later instances join it, so a rolling update, ``replicas: 2`` or an accidental second launch doesn't conflict with a running instance.
Compute services belong to the registration rather than to an instance, so every instance counts the services any of them created.

Each status update from any instance renews the registration.
A registration that hasn't been updated within ``ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS`` is stale, whatever its status.
The compute API treats a stale registration as gone: the next instance to register replaces it, and an instance requesting an instruction or updating its status gets a 404, re-registers, and carries on.
Set ``ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS`` to a small multiple of the managers' ``sleep_interval``: it's how long a registration outlives a manager that was killed.

The compute API finds a registration by the manager's name alone: the UUID in a compute manager ID identifies the instance or registration that created it, but an instance can keep using its own ID after joining, or after its registration was replaced.
The identities that created or joined a registration are recorded in its ``identities``.

Everything stored on a registration, including its job records and pause state, goes when it expires.
A compute service may start while its manager's registration is missing, for instance while it is being recreated; the service registers anyway, and belongs to the registration once there is one, since services belong to the registration under their manager's name.

A manager that exits deregisters, but the shared registration is left to expire, since other instances may still be renewing it.
Deregistering therefore always succeeds, whether or not the registration exists.
The ``ERROR`` status never prevents a manager from registering, so ``clear_error`` and ``steal`` are no longer needed; both are deprecated, and will be removed in a future release, as will ``ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS_ERROR``, which no longer has any effect.
``steal`` is ignored: registering with it joins a live registration like any other.
The compute API logs a warning whenever either is used, or the setting is set.

The compute manager endpoints return a 404 when the manager isn't registered, and a 422 when the request conflicts with the registration's state.

Inspecting a compute manager
============================

Each time a compute manager's status changes to ``DEGRADED`` or ``ERROR``, or its detail changes, the error is recorded under the manager's name.
The history also records events in the manager's pause and recovery, such as a probe passing or failing, and a registration expiring while paused, which resets its pause state.
The most recent entries are kept, ``ALCHEMISCALE_COMPUTE_API_MANAGER_ERROR_HISTORY`` of them (20 by default), independently of the registration, so they remain readable after it expires or is replaced; older entries are deleted as new ones are recorded.

To see a compute manager's registration, its shared pause state, the identities that registered it, the number of jobs that failed since the last sign of health, and its recent errors and events::

    $ alchemiscale compute manager status <name> --api-url <compute API URL> \
          --identifier <compute identity> --key <key>

``--api-url``, ``--identifier`` and ``--key`` can also be set with the ``ALCHEMISCALE_URL``, ``ALCHEMISCALE_ID`` and ``ALCHEMISCALE_KEY`` environment variables, and ``--json`` prints the result as JSON.
The same is available from ``AlchemiscaleComputeManagerClient.get_status``, and from ``GET /computemanager/{name}`` on the compute API.

Before registering, a compute manager checks the alchemiscale version the server reports from its ``/info`` endpoint.
If the server is older than the oldest version the manager supports, the manager exits with a message naming both versions.
Upgrade the server before upgrading compute managers.

The cycle
=========

Every cycle, whatever the instruction, a ``ComputeManager``:

1. lists its jobs on the batch system;
2. handles each terminal job it hasn't handled before: records a failure, or a sign of health, and removes the job;
3. checks that running jobs' compute services have registered with the server;
4. if the instruction is ``OK``, no job is pending, and there are tasks and spare capacity, submits new jobs;
5. posts its status.

Maintenance therefore runs even when there are no tasks, no capacity, or the instruction is ``SKIP``.

Jobs are linked to compute services by name.
The manager names each job ``{manager name}.{uuid hex}``, and the job's compute service registers under that name.
Capacity is the number of distinct names across pending and running jobs and registered compute services, so a running job whose service has registered counts once.

A batch system such as SLURM keeps reporting finished jobs for a while.
The names of handled jobs are kept on the shared registration, so each terminal job is handled once, by whichever instance sees it first.
A job that disappears without being seen in a terminal state has an unknown outcome: it is logged, and counts as neither a failure nor a sign of health.
If it is listed again still pending or running, as when the batch system's accounting lags behind, it is treated as active again rather than removed.
A handled job's name is kept until the job is no longer listed, or, with a ``finished_job_lookback``, until the lookback has passed since it ended.

A sign of health is a job that succeeded after its compute service was seen registered.
The server records a compute service registering on its manager's shared registration, so a service that registers and finishes between two cycles is still seen.
A job that fails with a ``WORKLOAD`` failure (non-zero exit, out of memory) is recorded as a failure; an ``INFRASTRUCTURE`` failure (preemption, a lost node) is recorded, but never counts against the manager.
A failure the backend can't classify is treated as ``WORKLOAD``.
Each failure's reason says whether the job failed before or after its compute service registered, which tells a broken template from, say, a GPU lost mid-run.

A running job whose compute service hasn't registered within ``registration_grace_period`` seconds of the job starting is checked again on the next cycle.
Once two checks at least ``registration_grace_period`` apart find it unregistered, it is removed and recorded as a ``WORKLOAD`` failure.
The grace period is counted from when the job started running, not from submission, since a job may wait in a queue for hours.

Creating a compute manager
==========================

A compute manager runs on a batch system through a :class:`~alchemiscale.compute.backend.ComputeBackend`, a small driver that reports facts about its jobs and makes no policy decisions.
A backend implements:

``validate()``
    Check settings and templates at startup. Raising :class:`~alchemiscale.compute.backend.PermanentBackendError` exits the manager.

``list_jobs()``
    Return this manager's jobs, read from the batch system, as :class:`~alchemiscale.compute.backend.Job` s with their state, ``submitted_at``, ``started_at`` and ``ended_at`` (timezone-aware), and for failures a :class:`~alchemiscale.compute.backend.FailureKind`.
    Pick out the manager's jobs with :func:`~alchemiscale.compute.backend.is_own_job_name`, rather than keeping them in memory.

``submit(name)``
    Submit a job named ``name`` that starts a compute service with that name, and with ``compute_manager_id`` set to the backend's ``compute_manager_id`` attribute.
    A batch system that can't use the name as given, such as Kubernetes, can submit under a derived name, keep ``name`` in the job's metadata, and report it as the job's name.

``remove(job)``
    Remove a job from the batch system, cancelling it if it is still running.

and sets ``finished_job_lookback``: how long a finished job keeps appearing in ``list_jobs()``, or ``None`` if ``remove()`` deletes it.

Backend methods run synchronously in the manager's cycle, so each must return within a bounded time; bound any retries and timeouts well within ``ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS``.
Backend methods raise :class:`~alchemiscale.compute.backend.TransientBackendError` for errors expected to clear on their own, such as a full submission queue or an API timeout, and :class:`~alchemiscale.compute.backend.PermanentBackendError` for errors that need a change, such as an invalid partition or missing permissions, chained from the original exception.

To run a manager, pass the backend to :class:`~alchemiscale.compute.manager.ComputeManager`::

    manager = ComputeManager(manager_settings, service_settings, MyBackend(...))
    manager.start()

:class:`~alchemiscale.compute.testing.FakeComputeBackend` is an in-memory backend for testing a manager without a cluster.

.. _Deployment: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
.. _alchemiscale-k8s: https://github.com/datryllic/alchemiscale-k8s/tree/main/compute
