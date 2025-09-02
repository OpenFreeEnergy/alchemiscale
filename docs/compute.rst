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
    # options for service initialization
    init:
    
      # Filesystem path to use for `ProtocolDAG` `shared` space.
      shared_basedir: "/scratch/${USER}/${JOBID}-${JOBINDEX}/shared"
    
      # Filesystem path to use for `ProtocolUnit` `scratch` space.
      scratch_basedir: "/scratch/${USER}/${JOBID}-${JOBINDEX}/scratch"
    
      # Path to file for logging output; if not set, logging will only go to
      # STDOUT.
      logfile: /home/${USER}/logs/service.${JOBID}.log
    
    # options for service execution
    start:
    
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
Information of existing services and current workload is reported to the manager through an instruction request through the compute API. Managers expect one of three instructions from the alchemiscale:

1. ``OK``: the manager may allocate compute resources
2. ``SKIP``: the manager should take no action
3. ``SHUTDOWN``: the manager should shut down

In the case of the OK or the SKIP instruction, a manager is expected to provide a status update to the compute API.
The status can either be:

1. ``OK``: the manager is working as expected, requires a saturation value (ratio of active services to the maximum number of services allowed by the manager)
2. ``ERROR``: the manager experienced an error and will shut down locally, requires a detail string explaining why the ``ERROR`` state was entered

Currently, these supporting data are only used by administrators for introspection, though this might change in the future.

Creating a compute manager
==========================

``alchemiscale`` provides a base class for creating a custom compute manager and, in the simplest case, only require defining how a compute service is created on a target compute platform.

.. _Deployment: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
.. _alchemiscale-k8s: https://github.com/datryllic/alchemiscale-k8s/tree/main/compute
