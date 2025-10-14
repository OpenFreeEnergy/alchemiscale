.. _getting-started:

###############
Getting Started
###############

This document details the basic usage of the :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` for evaluating :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.
It assumes that you already have a user identity on the target ``alchemiscale`` instance, with access to :py:class:`~alchemiscale.models.Scope`\s to submit :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s to.


************
Installation
************

Create a conda environment using, e.g. `micromamba`_::

    $ micromamba create -n alchemiscale-client -c conda-forge alchemiscale-client

Once installed, activate the environment::

    $ micromamba activate alchemiscale-client

You may wish to install other packages into this environment, such as ``feflow`` or ``jupyterlab``.

.. _micromamba: https://github.com/mamba-org/micromamba-releases


*****************************
Creating an AlchemicalNetwork
*****************************

To create an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, review this notebook and apply the same approach to your systems of interest: `Preparing AlchemicalNetworks.ipynb`_

Note that there are several Protocols you can use, including at least:

* :py:class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol`
* :py:class:`feflow.protocols.nonequilibrium_cycling.NonEquilibriumCyclingProtocol`

Try each one out with default options for a start.
Below are notes on settings you may find more optimal for each, however.

Note that for the ``feflow`` protocol, you will need to install ``feflow`` into your environment with::

    $ micromamba install -n alchemiscale-client -c conda-forge feflow

.. _Preparing AlchemicalNetworks.ipynb: https://github.com/OpenFreeEnergy/ExampleNotebooks/blob/main/networks/Preparing%20AlchemicalNetworks.ipynb


RelativeHybridTopologyProtocol usage notes
==========================================

For production use of this protocol, we recommend the default settings, with these changes to reduce execution times per :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` :py:class:`~alchemiscale.storage.models.Task`::

    >>> from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    >>> from openff.units import unit

    >>> settings = RelativeHybridTopologyProtocol.default_settings()
    >>> settings.protocol_repeats = 1
    >>> settings.engine_settings.compute_platform = "CUDA"
    >>> settings.simulation_settings.equilibration_length = 1 * unit.nanosecond
    >>> settings.simulation_settings.production_length = 5 * unit.nanosecond
    >>> settings.simulation_settings.time_per_iteration = 2.5 * unit.picosecond
    >>> settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    >>> settings.solvation_settings.box_shape = 'dodecahedron'
    >>> settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer


NonEquilibriumCyclingProtocol usage notes
=========================================

For production use of this protocol, we recommend the default settings::

    >>> from feflow.protocols.nonequilibrium_cycling import NonEquilibriumCyclingProtocol

    >>> settings = NonEquilibriumCyclingProtocol.default_settings()


.. _user-guide-submit-network:

*************************************************
Submitting your AlchemicalNetwork to alchemiscale
*************************************************

Once you’ve defined an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, you can submit it to an ``alchemiscale`` instance.
This assumes the instance has been deployed and is network-accessible from your workstation.
See :ref:`deployment` for deployment options if you do not already have an instance available for your use.

Instantiating an AlchemiscaleClient
===================================

Create an :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` instance with your ``api_url``, user ``identifier``, and ``key``::

    >>> from alchemiscale import AlchemiscaleClient, Scope, ScopedKey
    >>> asc = AlchemiscaleClient('https://api.<alchemiscale-uri>', user_identifier, user_key)

Additionally, the :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` can automatically use the following environment variables:

``ALCHEMISCALE_URL``
    The URL of the API to interact with.
``ALCHEMISCALE_ID``
    The identifier for the identity used for authentication.
``ALCHEMISCALE_KEY``
    Credential for the identity used for authentication.

For example, this will work if all aforementioned environment variables are set::

    >>> from alchemiscale import AlchemiscaleClient, Scope, ScopedKey
    >>> asc = AlchemiscaleClient()

.. warning ::
   Direct arguments take precedence over environment variables.
   If both are set with different values, the client will use the Python arguments and issue a warning about the mismatch.

After creating the :py:class:`~alchemiscale.interface.client.AlchemiscaleClient`, you can interact with it as described in the following sections.

Choosing a Scope
================

Choose a :py:class:`~alchemiscale.models.Scope` to submit your :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` to. 
A :py:class:`~alchemiscale.models.Scope` is an org-campaign-project triple, and your user will have permissions to work within some of these.
You can list your accessible ``Scopes`` with::

    >>> asc.list_scopes()
    [<Scope('org1-*-*')>,
     <Scope('org2-*-*')>
     ...]

If you are a user, you will likely see the :py:class:`~alchemiscale.models.Scope` ``<Scope('openff-*-*')>`` among this list. 
This means that you can submit your :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` to any Scope matching that pattern, such as ``'openff-my_special_campaign-tyk2_testing_1'``.
A :py:class:`~alchemiscale.models.Scope` without any wildcards (``'*'``) is considered a *specific* :py:class:`~alchemiscale.models.Scope`; an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` can only be submitted to a *specific* :py:class:`~alchemiscale.models.Scope`.

You can create one with, e.g.::

    >>> scope = Scope('my_org', 'my_campaign', 'my_project')

Within a :py:class:`~alchemiscale.models.Scope`, components of an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` are deduplicated against other components already present, allowing you to e.g. submit new :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s sharing :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s with previous ones and benefit from existing results.
If you prefer to have an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` not share any components with previously-submitted :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s, then submit it into a different :py:class:`~alchemiscale.models.Scope`.


Submitting and retrieving an AlchemicalNetwork
==============================================

Submit your :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`::

    >>> an_sk = asc.create_network(network, scope)

This will return a :py:class:`~alchemiscale.models.ScopedKey` uniquely identifying your :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`. A :py:class:`~alchemiscale.models.ScopedKey` is a combination of ``network.key`` and the :py:class:`~alchemiscale.models.Scope` we submitted it to, e.g.::

    >>> an_sk
    <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>

You can pull the full :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` back down (even in another Python session) with::

    >>> network_again = asc.get_network(an_sk)
    >>> network_again
    <AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c>

You can always produce a :py:class:`~alchemiscale.models.ScopedKey` from its string representation with ``ScopedKey.from-str(<scoped-key-str>)``, allowing for copy-paste from one session to another.

You can list all your accessible ``AlchemicalNetworks`` on the ``alchemiscale`` instance with::

    >>> asc.query_networks()
    [<ScopedKey('AlchemicalNetwork-4617c8d8d6599124af3b4561b8d910a0-my_org-my_campaign-my_project')>,
     <ScopedKey('AlchemicalNetwork-d90bd97079cd965b887b373307ea7bab-my_org-my_campaign-my_project')>,
     ...]

and you can use these with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_network` above to pull them down as desired.


.. _user-guide-create-tasks:

****************************
Creating and actioning Tasks
****************************

Submitting an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` defines it on the ``alchemiscale`` server, but it does not define where to allocate effort in evaluating the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s in the network.
To do this, we need to create and action :py:class:`~alchemiscale.storage.models.Task`\s on the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s we are most interested in.

For this example, we’ll loop through every :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` in our :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, creating and actioning 3 :py:class:`~alchemiscale.storage.models.Task`\s for each::

    >>> tasks = []
    >>> for tf_sk in asc.get_network_transformations(an_sk):
            tasks.extend(asc.create_tasks(tf_sk, count=3))
    
    >>> asc.action_tasks(tasks, an_sk)
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]

A :py:class:`~alchemiscale.storage.models.Task` is associated with a :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` on creation, and actioning the :py:class:`~alchemiscale.storage.models.Task` marks it for execution for our :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` we submitted earlier.
If we submit another :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` including some of the same :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s later to the same :py:class:`~alchemiscale.models.Scope`, we could get the :py:class:`~alchemiscale.storage.models.Task`\s for each :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` and only create new :py:class:`~alchemiscale.storage.models.Task`\s if necessary, actioning the existing ones to that :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` as well::

    >>> tasks = []
    >>> for tf_sk in asc.get_network_transformations(other_an_sk):
    >>>     existing_tasks = asc.get_transformation_tasks(tf_sk)
    >>>     tasks.extend(asc.create_tasks(transformation_sk, count=max(3 - len(existing_tasks), 0)) 
                         + existing_tasks)

    >>> asc.action_tasks(tasks, other_an_sk)
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     None,
     ...]

The more :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s a :py:class:`~alchemiscale.storage.models.Task` is actioned to, the higher its chances of being picked up by a compute service.
In this way, actioning is an indicator of demand for a given :py:class:`~alchemiscale.storage.models.Task` and its corresponding :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`.

.. note:: 
   Alchemiscale :py:class:`~alchemiscale.storage.models.Task`\s can be considered a single independent “repeat” of an alchemical transformation, or a :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAG` as defined in :py:mod:`gufe`.
   What this exactly means will be subtly different depending on the type of alchemical :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` employed.

   In the case of the :py:class:`~openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol` (i.e. for HREX, and SAMS), this effectively means that each :py:class:`~alchemiscale.storage.models.Task` carries out all the computation required to obtain a single estimate of the free energy (in practice one would want to do several repeats to get an idea of the sampling error).

   In the case of the :py:class:`~feflow.protocols.nonequilibrium_cycling.NonEquilibriumCyclingProtocol`, a :py:class:`~alchemiscale.storage.models.Task` instead encompasses a non-equilibrium cycle and will return a single work estimate.
   The work values of multiple :py:class:`~alchemiscale.storage.models.Task`\s can then be gathered to obtain a free energy estimate, and more :py:class:`~alchemiscale.storage.models.Task`\s will improve the convergence of the estimate.


To get all :py:class:`~alchemiscale.storage.models.Task`\s actioned on an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, you can use::

    >>> asc.get_network_actioned_tasks(an_sk)
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]

On the other hand, to get all :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s a given :py:class:`~alchemiscale.storage.models.Task` is actioned on, you can use::

    >>> asc.get_task_actioned_networks(task)
    [<ScopedKey('AlchemicalNetwork-4617c8d8d6599124af3b4561b8d910a0-my_org-my_campaign-my_project')>,
     <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>,
     ...]


Setting the weight of an AlchemicalNetwork
==========================================

When a compute service claims a :py:class:`~alchemiscale.storage.models.Task`, it first performs a weighted, random selection of :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s in the :py:class:`~alchemiscale.models.Scope`\s visible to it.
Upon choosing an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, it performs a weighted, random selection of :py:class:`~alchemiscale.storage.models.Task`\s actioned on that :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.

You can set the ``weight`` of an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` to influence the likelihood that the :py:class:`~alchemiscale.storage.models.Task`\s actioned on it are picked up for compute, increasing or decreasing the rate at which results become available relative to other :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.
To get and set the ``weight`` of an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, use::

    >>> asc.get_network_weight(an_sk)
    0.5
    >>> asc.set_network_weight(an_sk, 0.9)
    >>> asc.get_network_weight(an_sk)
    0.9


Setting the weight of actioned Tasks
====================================

As mentioned above, upon choosing an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, a compute service performs a weighted, random selection of :py:class:`~alchemiscale.storage.models.Task`\s actioned on that :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.
You can set the ``weight`` of an actioned :py:class:`~alchemiscale.storage.models.Task` to influence the likelihood that it will be picked up for compute relative to the other :py:class:`~alchemiscale.storage.models.Task`\s actioned on the given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.
To set the ``weight`` of an actioned :py:class:`~alchemiscale.storage.models.Task` on an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, use :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.action_tasks` with the ``weight`` keyword argument::

    >>> # get all networks that the given Task is actioned on, with weights as dict values
    >>> asc.get_task_actioned_networks(task, task_weights=True)
    {<ScopedKey('AlchemicalNetwork-4617c8d8d6599124af3b4561b8d910a0-my_org-my_campaign-my_project')>: 0.5,
     <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>: 0.5}

    >>> asc.action_tasks([task], an_sk, weight=0.7)
    >>> asc.get_task_actioned_networks(task, task_weights=True)
    {<ScopedKey('AlchemicalNetwork-4617c8d8d6599124af3b4561b8d910a0-my_org-my_campaign-my_project')>: 0.5,
     <ScopedKey('AlchemicalNetwork-66d7676b10a1fd9cb3f75e6e2e7f6e9c-my_org-my_campaign-my_project')>: 0.7}

Because this ``weight`` is a property of the actions relationship between the :py:class:`~alchemiscale.storage.models.Task` and the :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, there is a distinct ``weight`` associated with each actions relationship between a :py:class:`~alchemiscale.storage.models.Task` and the :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s it is actioned on.
These ``weight``\s can be set independently.
Also, the :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.action_tasks` method is idempotent, so repeated calls will serve to set the ``weight`` to the value specified, even for already-actioned :py:class:`~alchemiscale.storage.models.Task`\s.


Setting the priority of Tasks
=============================

The ``weight`` of an actioned :py:class:`~alchemiscale.storage.models.Task` influences how likely it is to be chosen among the other :py:class:`~alchemiscale.storage.models.Task`\s actioned on the given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.
A complementary mechanism to ``weight`` is :py:class:`~alchemiscale.storage.models.Task` ``priority``, which is a property of the :py:class:`~alchemiscale.storage.models.Task` itself and introduces some determinism to when the :py:class:`~alchemiscale.storage.models.Task` is executed relative to other :py:class:`~alchemiscale.storage.models.Task`\s actioned on the same :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.
When a compute service has selected an :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` to draw :py:class:`~alchemiscale.storage.models.Task`\s from, it first partitions the :py:class:`~alchemiscale.storage.models.Task`\s by ``priority``;
the weighted selection is then performed *only* on those :py:class:`~alchemiscale.storage.models.Task`\s of the same, highest priority.
In this way, a :py:class:`~alchemiscale.storage.models.Task` with ``priority`` 1 will always be chosen before a :py:class:`~alchemiscale.storage.models.Task` with ``priority`` 2, and so on, if they are both actioned on the same :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`\s.

You can get and set the ``priority`` for a number of :py:class:`~alchemiscale.storage.models.Task`\s at a time with::

    >>> asc.get_tasks_priority(tasks)
    [5,
     1,
     3,
     ...]
    >>> asc.set_tasks_priority(tasks, [2, 3, 599, ...])
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]


.. note::
   Unlike the ``weight`` of an actioned :py:class:`~alchemiscale.storage.models.Task`, the ``priority`` of a :py:class:`~alchemiscale.storage.models.Task` is a property of a :py:class:`~alchemiscale.storage.models.Task` itself: it influences selection order of the :py:class:`~alchemiscale.storage.models.Task` for *every* :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` it is actioned on.

*************************
Cancelling actioned Tasks
*************************

Only *actioned* :py:class:`~alchemiscale.storage.models.Task`\s are available for execution to compute services, and if you decide later that you would prefer a given :py:class:`~alchemiscale.storage.models.Task` not be actioned for a given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` you can *cancel* it.
To *cancel* a :py:class:`~alchemiscale.storage.models.Task` is the opposite of *actioning* it::

    >>> asc.cancel_tasks(tasks, an_sk)
    [<ScopedKey('Task-06cb9804356f4af1b472cc0ab689036a-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-129a9e1a893f4c24a6dd3bdcc25957d6-my_org-my_campaign-my_project')>,
     <ScopedKey('Task-157232d7ff794a0985ebce5055e0f336-my_org-my_campaign-my_project')>,
     ...]


********************************
Getting the status of your Tasks
********************************

As you await results for your actioned :py:class:`~alchemiscale.storage.models.Task`\s, it’s often desirable to check their status to ensure they are running or completing at the rate you expect.
You can quickly obtain statuses for all Tasks associated with various levels, including:

* :py:class:`~alchemiscale.models.Scope`
* :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`
* :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`


Scope
=====

For example, to get the status counts for all :py:class:`~alchemiscale.storage.models.Task`\s within a particular :py:class:`~alchemiscale.models.Scope`, you could do::

    >>> # corresponds to the scope 'my_org-my_campaign-*'
    >>> asc.get_scope_status(Scope('my_org', 'my_campaign'))
    {'complete': 324,
     'error': 37,
     'invalid': 6,
     'deleted': 13,
     'waiting': 372,
     'running': 66}

For a *specific* :py:class:`~alchemiscale.models.Scope`, this will give status counts of all :py:class:`~alchemiscale.storage.models.Task`\s within that exact :py:class:`~alchemiscale.models.Scope`, assuming your user has permissions on it (see :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.list_scopes` for your accessible :py:class:`~alchemiscale.models.Scope` space).
For a *non-specific* :py:class:`~alchemiscale.models.Scope` (like ``my_org-my_campaign-*`` above), this will give the aggregate status counts across the :py:class:`~alchemiscale.models.Scope` space visible to your user under the given :py:class:`~alchemiscale.models.Scope`.

Calling :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_scope_status` without arguments will default to the highest non-specific :py:class:`~alchemiscale.models.Scope` of ``*-*-*``.

To get the individual statuses of all :py:class:`~alchemiscale.storage.models.Task`\s in a given :py:class:`~alchemiscale.models.Scope`, use the :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.query_tasks` method in combination with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_status`::

    >>> tasks = asc.query_tasks(scope=Scope('my_org', 'my_campaign'))
    >>> asc.get_tasks_status(tasks)
    ['complete',
     'complete',
     'complete',
     'waiting',
     'complete',
     'error',
     'invalid',
     'running',
     'deleted',
     'complete'
     ...]


AlchemicalNetwork
=================

You can get the status counts of all :py:class:`~alchemiscale.storage.models.Task`\s associated with :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s within a given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork` with::

    >>> asc.get_network_status(an_sk)
    {'complete': 138,
     'error': 14,
     'invalid': 2,
     'deleted': 9,
     'waiting': 57,
     'running': 33}

Note that this will show status counts for all such :py:class:`~alchemiscale.storage.models.Task`\s, whether or not they have been actioned on the given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`.

To get the specific statuses of all :py:class:`~alchemiscale.storage.models.Task`\s for a given :external+gufe:py:class:`~gufe.network.AlchemicalNetwork`, use the :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_network_tasks` method in combination with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_status`::

    >>> tasks = asc.get_network_tasks(an_sk)
    >>> asc.get_tasks_status(tasks)
    ['complete',
     'error',
     'waiting',
     'complete',
     'running',
     'running',
     'deleted',
     'invalid',
     ...]


Transformation
==============

To get the status counts of all :py:class:`~alchemiscale.storage.models.Task`\s associated with only a given :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, use::

    >>> asc.get_transformation_status(tf_sk)
    {'complete': 2,
     'error': 1,
     'running': 3}

To get the specific statuses of all :py:class:`~alchemiscale.storage.models.Task`\s for a given :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, use the :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_transformation_tasks` method in combination with :py:meth:`~alchemiscale.interface.client.AlchemiscaleClient.get_tasks_status`::

    >>> tasks = asc.get_transformation_tasks(tf_sk)
    >>> asc.get_tasks_status(tasks)
    ['complete',
     'error',
     'complete',
     'running',
     'running',
     'running']



******************************
Pulling and assembling results
******************************

A :py:class:`~gufe.protocols.protocol.Protocol` is attached to each :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`, and that :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` defines how each :py:class:`~alchemiscale.storage.models.Task` is computed.
It also defines how the results of each :py:class:`~alchemiscale.storage.models.Task` (called a :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`) are combined to give an estimate of the free energy difference for that :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`.

We can check the status of a :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` with::

    >>> asc.get_transformation_status(tf_sk)
    {'complete': 2,
     'error': 1,
     'running': 3}

If there are complete :py:class:`~alchemiscale.storage.models.Task`\s, we can pull in all successful :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s for the :external+gufe:py:class:`~gufe.transformations.transformation.Transformation` and combine them into a :external+gufe:py:class:`~gufe.protocols.protocol.ProtocolResult` corresponding to that :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`/'s :external+gufe:py:class:`~gufe.protocols.protocol.Protocol` with::

    >>> protocol_result = asc.get_transformation_results(tf_sk)
    >>> protocol_result
    <RelativeHybridTopologyProtocolResult-44b0f588f5f3073aa58d86e1017ef623>

This object features a :external+gufe:py:meth:`~gufe.protocols.protocol.ProtocolResult.get_estimate` and :external+gufe:py:meth:`~gufe.protocols.protocol.ProtocolResult.get_uncertainty` method, giving the best available estimate of the free energy difference and its uncertainty. 

To pull the :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s and not combine them into a :external+gufe:py:class:`~gufe.protocols.protocol.ProtocolResult` object, you can give ``return_protocoldagresults=True`` to this method.
Any number of :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s can then be manually combined into a single :external+gufe:py:class:`~gufe.protocols.protocol.ProtocolResult` with::

    >>> # protocol_dag_results: List[ProtocolDAGResult]
    >>> protocol_dag_results = asc.get_transformation_results(tf_sk, return_protocoldagresults=True)
    >>> protocol_result = transformation.gather(protocol_dag_results)
    >>> protocol_result
    <RelativeHybridTopologyProtocolResult-44b0f588f5f3073aa58d86e1017ef623>

This can be useful for subsampling the available :external+gufe:py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s and building estimates from these subsamples, such as for an analysis of convergence for the :py:class:`~feflow.protocols.nonequilibrium_cycling.NonEquilibriumCyclingProtocol`.

If you wish to pull results for only a single :py:class:`~alchemiscale.storage.models.Task`, you can do so with::

    >>> task: ScopedKey
    >>> protocol_dag_results = asc.get_task_results(task)
    >>> protocol_dag_results
    [<ProtocolDAGResult-54a3ed32cbd3e3d60d87b2a17519e399>]

You can then iteratively create and action new :py:class:`~alchemiscale.storage.models.Task`\s on your desired :external+gufe:py:class:`~gufe.transformations.transformation.Transformation`\s based on their current estimate and uncertainty, allocating effort where it will be most beneficial.
