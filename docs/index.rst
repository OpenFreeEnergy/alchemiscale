.. alchemiscale documentation master file, created by
   sphinx-quickstart on Wed Nov 23 20:51:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to alchemiscale's documentation!
========================================

**alchemiscale** is a service-oriented execution system for ``AlchemicalNetworks``,
suitable for utilizing multiple compute resources,
such as HPC clusters, individual hosts, Kubernetes clusters, Folding@Home work servers, etc., 
to support large campaigns requiring high-throughput.

It is designed for maximum interoperability with the `Open Molecular Software Foundation`_ stack,
in particular the `OpenForceField`_ and `OpenFreeEnergy`_ ecosystems. 

.. note::
   This software is pre-alpha and under active development. It is not yet ready
   for production use and the API is liable to change rapidly at any time. 

.. _Open Molecular Software Foundation: https://omsf.io
.. _OpenForceField: https://openforcefield.org
.. _OpenFreeEnergy: https://openfree.energy/

.. toctree::
   :maxdepth: 1
   :caption: Contents:


   ./overview
   ./user_guide
   ./deployment
   ./compute
   ./operations
   ./development
   ./api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
