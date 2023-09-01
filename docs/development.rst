.. _developers:

##############
For Developers
##############

``alchemiscale`` is an open-source project, and we invite developers to contribute to its advancement and to extend its functionality.
This document provides some guidance on:

* the system architecture, guiding philosophy for design choices
* components and their technology stacks
* the overall layout of the library
* how best to engage with the project


.. _system-architecture:

*****************************************
System architecture and design philosophy
*****************************************

``alchemiscale`` is an example of a `service-oriented architecture`_, with the following components:

* the system state represented in a graph database (`Neo4j`_), often referred to as the *state store*
* result objects (e.g. files) stored in an *object store*, such as `AWS S3`_
* user access to the *state store* and *object store* via a `RESTful`_ API (``AlchemiscaleAPI``), often using the included Python client (:py:class:`~alchemiscale.interface.client.AlchemiscaleClient`)
* compute services deployed to resources suitable for performing free energy calculations, typically with `GPUs`_ for simulation acceleration
* another `RESTful`_ API (``AlchemiscaleComputeAPI``) used by the compute services for interaction with the *state store* and *object store*

These components function together to create a complete ``alchemiscale`` deployment.
They are shown together visually in :numref:`system-architecture-figure`.

.. _system-architecture-figure:
.. figure:: assets/system-architecture.png
   :alt: alchemiscale system architecture

   Diagram of the system architecture for ``alchemiscale``.
   Colored arrows on the diagram correspond to descriptions on the right.


.. _service-oriented architecture: https://en.wikipedia.org/wiki/Service-oriented_architecture
.. _Neo4j: https://neo4j.com/
.. _AWS S3: https://aws.amazon.com/s3/
.. _GPUs: https://en.wikipedia.org/wiki/Graphics_processing_unit
.. _RESTful: https://en.wikipedia.org/wiki/Representational_state_transfer


********************************
Components and technology stacks
********************************

Each component of ``alchemiscale`` makes use of an underlying technology stack specifically oriented to that component's purpose and needs.
We detail these components in this section.


.. _component-state-store:

State store
===========

The *state store* for ``alchemiscale`` represents the current state of the system at all times, without regard to the state of any other component.
It represents the single source of truth for what exists and what does not in the deployment, the status of running calculations, available results, etc.
Other components can experience failures and faults, but the content of the *state store* is the only content that really matters at any given moment.

We use a `graph database`_, `Neo4j`_, as the *state store*.
The choice of a graph database (over e.g. a `relational database`_ or a `document database`_) was natural given the graph structure of :py:class:`~gufe.AlchemicalNetwork`\s,
which constitute the core data model ``alchemiscale`` operates on.
With Neo4j, it wasn't necessary to contort these networks into relational tables or into loosely-related document records, and we can take advantage of deduplication of network nodes where appropriate for database performance and efficient use of compute resources.

The :py:class:`~alchemiscale.storage.statestore.Neo4jStore` class is ``alchemiscale``'s interface to its Neo4j instance, and broadly defines the interaction points for the *state store*.
The methods on this class interact with Neo4j via `py2neo`_, a client library that features a flexible object model for defining complex networks in Python and representing them in Neo4j.
Neo4j itself uses `Cypher`_ as its query language, and this is used throughout the :py:class:`~alchemiscale.storage.statestore.Neo4jStore` for modifying the state of the nodes and edges in the database.
It is worth reviewing the `Cypher manual`_ if you wish to make contributions to ``alchemiscale`` that require new interactions with the *state store*.


.. _graph database: https://en.wikipedia.org/wiki/Graph_database
.. _relational database: https://en.wikipedia.org/wiki/Relational_database
.. _document database: https://en.wikipedia.org/wiki/Document-oriented_database

.. _py2neo: https://github.com/py2neo-org/py2neo
.. _Cypher: https://en.wikipedia.org/wiki/Cypher_(query_language)
.. _Cypher manual: https://neo4j.com/docs/cypher-manual/current/introduction/


.. _component-object-store:

Object store
============

The *object store* is used for result storage.
Execution of :py:class:`~alchemiscale.storage.models.Task``\s by :ref:`component-compute-services` yields :py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult` objects, and these are stored
in a directory-like structure within the *object store* for later retrieval.
References to these objects are created in the *state store*, allowing the *state store* to function as a fast index for finding individual results on request.
When a user makes use of the :py:class:`~alchemiscale.interface.client.AlchemiscaleClient` to request results for a given :py:class:`~gufe.transformations.Transformation`, the ``AlchemiscaleAPI`` queries the *state store* for these references, then pulls the corresponding results from the *object store* and returns them as responses to the request.

The choice of *object store* corresponds to the platform ``alchemiscale`` is being deployed to.
Currently, there is only one implementation, using `AWS S3`_ as the *object store*, but there are plans to create implementations appropriate for other cloud providers, as well as to provide a "local" *object store* for single-host deployments.

For the `AWS S3`_ *object store*, ``alchemiscale`` makes use of :py:class:`alchemiscale.storage.S3ObjectStore` as its interface.
This object provides methods for storing and retrieving :py:class:`~gufe.protocols.protocoldag.ProtocolDAGResult`\s, and over time will support methods for storage of arbitrary files as required by certain :py:class:`~gufe.protocols.protocol.Protocol`\s.


.. _component-apis:

RESTful APIs
============

A complete ``alchemiscale`` deployment (currently) features two `RESTful`_ APIs, which handle `HTTP`_ client requests:

* ``AlchemiscaleAPI``: handles requests from *user* identities; includes submitting :py:class:`~gufe.network.AlchemicalNetwork`\s, actioning ``Task``\s, and retrieving results
* ``AlchemiscaleComputeAPI``: handles requests from *compute* identities; includes claiming ``Task``\s, submitting results on completion or failure

All API services in ``alchemiscale`` are implemented via `FastAPI`_, and deployed as `Gunicorn`_ applications with `Uvicorn`_ workers.
These services are "stateless": they modify the state of the *state store* and *object store*, but the state of the service workers themselves is ephemeral and relatively disposable.
Workers can be scaled up or scaled down to handle more or fewer requests from clients, but this has no bearing on the overall state of the ``alchemiscale`` deployment.

By construction, these API services can be horizontally scaled across many physical servers, and need not be co-located with the *state store*.
This is the approach taken, for example, when deploying to Kubernetes via `alchemiscale-k8s`_.


.. _HTTP: https://en.wikipedia.org/wiki/HTTP
.. _FastAPI: https://en.wikipedia.org/wiki/HTTP
.. _Gunicorn: https://docs.gunicorn.org/en/latest/custom.html
.. _Uvicorn: https://www.uvicorn.org/

.. _alchemiscale-k8s: https://github.com/datryllic/alchemiscale-k8s


.. _component-user-client:

User-facing Python client
=========================

Users interact


.. _component-compute-services:

Compute services
================

Compute services make use of the :py:class:`~alchemiscale.compute.client.AlchemiscaleComputeClient`.


**************
Library layout
**************




*****************
How to contribute
*****************


