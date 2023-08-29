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


*****************************************
System architecture and design philosophy
*****************************************

``alchemiscale`` is an example of a `service-oriented architecture`_, with the following components:

* the system state represented in a database (`Neo4j`_), referred to as the *state store*
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


**************
Library layout
**************




*****************
How to contribute
*****************
