##########
Deployment
##########

``alchemiscale`` consists of multiple services that can be independently scaled.
This document details on how to deploy and configure the server service.


********
Overview
********

The alchemiscale server deployment consists of a ``neo4j`` database, a client API endpoint, a reverse proxy (``traefik``), and a compute API endpoint.
The client and compute API endpoints can be scaled by adjusting the number of workers.
A single ``docker-compose.yml`` file defines all of these services.
Because our deployment process is containerized, the only requirements for the host is to be able to run ``docker compose`` in a ``x86_64`` environment.
Installation of ``alchemiscale`` software dependencies is  unnecessary on the host.


Host Configuration
==================

Install `docker compose <https://docs.docker.com/compose/install/#scenario-two-install-the-compose-plugin>`_
We recommend using "Scenario two: Install the Compose plugin" since Docker Desktop may require a paid subscription.
First install the `docker engine <https://docs.docker.com/engine/install/#server>`_ and then install the `plugin <https://docs.docker.com/compose/install/linux/>`_.

AWS
---


###########
Maintenance
###########

*********
Add Users
*********

*******
Backups
*******

