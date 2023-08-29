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


.. _ system-architecture:

*****************************************
System architecture and design philosophy
*****************************************

``alchemiscale`` is an example of a `service-oriented architecture`_, with the following components:

* the system state represented in a graph database (`Neo4j`_), referred to as the *state store*
* result objects (e.g. files) stored in an *object store*, such as `AWS S3`_
* user access via a `RESTful`_ API (``AlchemiscaleAPI``), often using the included Python client (:py:class:`~alchemiscale.AlchemiscaleClient`)
* compute services deployed to resources suitable for performing free energy calculations, typically with `GPUs` for simulation acceleration
* another `RESTful`_ API (``AlchemiscaleComputeAPI``) used by the compute services for interaction with *state store* and *object store*

These components function together to create a complete ``alchemiscale`` deployment.
They are shown together visually in

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


State store
===========

The *state store* for ``alchemiscale`` represents the current state of the system at all times, without regard to the state of any other component.
It represents the single source of truth for what exists and what does not in the deployment, the status of running calculations, available results, etc.
Other components can experience failures and faults, but the content of the *state store* is the only content that really matters at any given moment.

We use a `graph database`_, `Neo4j`_, as the *state store*.
The choice of a graph database (over e.g. a `relational database`_ or a `document database`_) was natural given the graph structure of :ref:`~gufe.AlchemicalNetwork`\s,
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


Object store
============


RESTful APIs
============


User-facing Python client
=========================


Compute services
================


**************
Library layout
**************




*****************
How to contribute
*****************


