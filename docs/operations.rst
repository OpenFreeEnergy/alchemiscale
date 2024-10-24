##########
Operations
##########
After deploying an ``alchemiscale`` instance, it is necessary to manage the instance, especially the state maintained in its state store.
This document details common operations an administrator may need to perform over the life of the instance to keep it in good working order.

************
Adding users
************

To add a new user identity, you will generally use the ``alchemiscale`` CLI::


    export NEO4J_URL=bolt://<NEO4J_HOSTNAME>:7687
    export NEO4J_USER=<NEO4J_USERNAME>
    export NEO4J_PASS=<NEO4J_PASSWORD>

    # add a user identity, with key
    alchemiscale identity add -t user -i <user identity> -k <user key>

    # add one or more scopes the user should have access to
    alchemiscale identity add-scope -t user -i <user identity> -s <org-campaign-project> -s ...

To add a new compute identity, perform the same operation as for user identities given above, **but replace ``-t user`` with ``-t compute``**.
Compute identities are needed by compute services to authenticate with and use the compute API.


``docker-compose`` deployment
=============================

For a ``docker-compose``-based deployment, it is easiest to do the above using the same ``alchemiscale-server`` image the API services are deployed with::

    docker run --rm -it --network alchemiscale-server_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> \
               <ALCHEMISCALE_DOCKER_IMAGE> \
               identity add -t user \
                            -i <user identity> \
                            -k <user key>
    docker run --rm -it --network alchemiscale-server_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> \
               <ALCHEMISCALE_DOCKER_IMAGE> \
               identity add-scope -t user \
                                  -i <user identity> \
                                  -s <org-campaign-project> -s ...

The important bits here are:

``--network alchemiscale-server_db``
    We need to make sure the docker container we are using can talk to the database container.

``-e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD>``
    We need to pass in these environment variables so that the container can talk to the database.
    These should match the values set in ``.env``.


*************************************
Performing and restoring from backups
*************************************

Performing regular backups of the state store is an important operational component for any production deployment of ``alchemiscale``.
To do this, **first shut down the Neo4j service so that no database processes are currently running**.

The instructions below assume a Docker-based deployment, perhaps via ``docker-compose`` as in :ref:`deploy-docker-compose`.
The same general principles apply to any deployment type, however.


.. _database-dump:

Creating a database dump
========================

**With the Neo4j service shut down**, navigate to the directory containing your database data, set ``$BACKUPS_DIR`` to the absolute path of your choice and ``$NEO4J_VERSION`` to the version of Neo4j you are using, then run::

    # create the dump `neo4j.dump`
    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               -c "neo4j-admin database dump --to-path /tmp neo4j"

    # create a copy of the dump with a timestamp
    cp ${BACKUPS_DIR}/neo4j.dump ${BACKUPS_DIR}/neo4j-$(date -I).dump

This will create a new database dump in the ``$BACKUPS_DIR`` directory.
Note that this command will fail if ``neo4j.dump`` already exists in this directory.
It is recommended to copy this file to one with a timestamp (e.g. ``neo4j-$(date -I).dump``), as above.

Restoring from a database dump
==============================

To later restore from a database dump, navigate to the directory containing your current database data, and clear or move the current data to another location (Neo4j will not restore to a database that is already initialized).

**With the Neo4j service shut down**, choose ``$DUMP_DATE`` and set ``$NEO4J_VERSION`` to the version of Neo4j you are using, then run::

    # create a copy of the timestamped dump to `neo4j.dump`
    cp ${BACKUPS_DIR}/neo4j-${DUMP_DATE}.dump ${BACKUPS_DIR}/neo4j.dump

    # load the dump `neo4j.dump`
    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               -c "neo4j-admin database load --from-path=/tmp neo4j"

You may need to perform a ``chown -R`` following this operation to set correct ownership of the newly-loaded database contents.

Automating the backup process to perform regular backups without human intervention for your deployment is ideal, but out of scope for this document.


**********************************
Performing upgrades and migrations
**********************************
In most cases, upgrading an ``alchemiscale`` instance to a new ``alchemiscale`` release only requires re-deployment of the API and compute services with a Docker image corresponding to that new release, and informing your users to also upgrade their client environments with the latest release as well.
In other cases, a migration may need to be performed on the state and/or object store to reflect schema changes, or to upgrade the state store itself to a newer version of ``neo4j``.

This section gives specific guidance for ``alchemiscale`` release upgrades, in particular migration steps.

v0.3 to v0.4
============
``alchemiscale`` v0.4 introduced a ``NetworkMark`` node and relationship for each ``AlchemicalNetwork``, supporting the concept of network state.
This change requires a migration on the state store.
In addition, ``alchemiscale`` v0.4 is the first release to use ``neo4j`` 5.x, requiring a migration of existing database data from ``neo4j`` 4.x.

The instructions below assume a ``docker-compose``-based deployment; follow them in-order to complete the data migration.

Migrate data from ``neo4j`` 4.4 to 5.18
---------------------------------------
1. Shut down your ``alchemiscale`` instance, including ``neo4j``. Perform a database dump as detailed above in :ref:`database-dump`.

2. Rename this dump to ``neo4j.dump``.

3. Delete the contents of the directory containing your database data; this directory contains a file called ``server_id``.

4. Load the dump using ``neo4j`` 5.18; ``$BACKUPS_DIR`` should be set from the database dump performed in step 1::

    export NEO4J_VERSION=5.18
    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               -c "neo4j-admin database load --from-path=/tmp neo4j"

5. Migrate the loaded database from ``neo4j`` 4.x to 5.x::

    export NEO4J_VERSION=5.18
    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               -c "neo4j-admin database migrate --force-btree-indexes-to-range neo4j"

6. If necessary, perform a ``chown -R`` following this operation on the database data directory to set correct ownership of the newly-loaded database contents.


Migrate schema from ``alchemiscale`` 0.3 to 0.4
-----------------------------------------------
1. Set the env variable ``NEO4J_DOCKER_IMAGE=neo4j:5.18`` in your ``.env`` file for your ``docker-compose`` deployment.

2. Start up the ``neo4j`` service only::

    USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up neo4j

3. In another shell on the same host, perform the `alchemiscale` schema migration::

    docker run --rm -it --network alchemiscale-server_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> \
               ghcr.io/openforcefield/alchemiscale-server:v0.4.0 \
               database migrate v03-to-v04

4. Shut down the ``neo4j`` service (``Ctrl+C`` of running instance in step 2), then bring up the full set of services::

    USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up -d
