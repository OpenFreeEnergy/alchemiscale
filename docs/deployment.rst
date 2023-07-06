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

Now clone the repository and then navigate to the ``alchemiscale/docker/alchemiscale-server`` folder::
    
    $ git clone https://github.com/openforcefield/alchemiscale.git
    $ cd alchemiscale/docker/alchemiscale-server

.. note ::
   It is not strictly necessary to clone the repository. 
   For host deployment, only ``alchemiscale/docker/alchemiscale-server/docker-compose.yml`` and either a ``.env`` file and/or environment variables set are needed.
   By cloning the repository, a ``git pull`` can be used to retrieve an updated ``.env.template`` and ``docker-compose.yml`` which may be useful.

Now make a copy of ``.env.template``:

.. code-block:: bash
   
   $ cp .env.template .env

and modify ``.env`` with your favorite text editor.

.. warning::
   The ``.env`` file will contain sensitive information and should not be checked into version control programs or shared publicly.

See ``.env.testing`` for an example. 

The ``neo4j`` database requires the directory for the data store to exist before it starts.
This location should be on a storage medium that can handle the IOPS demand of a ``neo4j`` database.
For example, using the location set in ``.env.testing``::

    $ mkdir -p data/server

Now start the service with::

    $ USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up -d

We set ``USER_ID`` and ``GROUP_ID`` to be the same as the user running the ``docker-compose up -d`` command.

AWS
---

.. Note:: This is a guide on how to setup a fresh EC2 x86_64 instance running a Amazon Linux 2023 AMI.
   These steps should generally work for other linux distributions, but may require some modifcation e.g. the package manger may be ``apt`` instead of ``dnf``.


Once connected to the instance, run the following commands::

    $ sudo dnf check-release-update  # Check for updates
    $ sudo dnf --releasever=version update  # Update if new version is available NOTE: This guide used Amazon Linux Version 2023.1.20230705
    $ sudo dnf -y install docker git
    $ sudo service docker start  # Start docker service
    $ sudo systemctl enable docker.service  # Start docker service on boot
    $ sudo usermod -a -G docker ec2-user  # Add ec2-user to docker group
    $ newgrp docker  # Trick so we don't have to reboot (or login and logout) after adding ec2-user to docker group
    # Now we have to manually install the docker compose plugin until this issue gets resolved https://github.com/amazonlinux/amazon-linux-2023/issues/186
    $ DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}  # Set location to install plugin 
    $ mkdir -p $DOCKER_CONFIG/cli-plugins  # Create the directory to install the plugin
    $ curl -SL https://github.com/docker/compose/releases/download/v2.19.1/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose  # Download plugin
    $ chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose  # Set executable permissions to the plugin 
    $ docker info  # Test if everything works
    $ docker compose version  # Test if plugin was installed correctly

Now the instance has all of the dependences required to deploy an alchemical server.


###########
Maintenance
###########

*********
Add Users
*********

*******
Backups
*******

