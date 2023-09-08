##########
Operations
##########

*********
Add users
*********

To add a new user identity, you will generally use the ``alchemiscale`` CLI::


    $ export NEO4J_URL=bolt://<NEO4J_HOSTNAME>:7687
    $ export NEO4J_USER=<NEO4J_USERNAME>
    $ export NEO4J_PASS=<NEO4J_PASSWORD>
    $
    $ # add a user identity, with key
    $ alchemiscale identity add -t user -i <user identity> -k <user key>
    $
    $ # add one or more scopes the user should have access to
    $ alchemiscale identity add-scope -t user -i <user identity> -s <org-campaign-project> -s ...

To add a new compute identity, perform the same operation as for user identities given above, **but replace ``-t user`` with ``-t compute``**.
Compute identities are needed by compute services to authenticate with and use the compute API.


``docker-compose`` deployment
=============================

For a ``docker-compose``-based deployment, it is easiest to do the above using the same ``alchemiscale-server`` image the API services are deployed with::

    $ docker run --rm -it --network docker_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> \
      <ALCHEMISCALE_DOCKER_IMAGE> identity add -t user \
                                               -i <user identity> \
                                               -k <user key>
    $ docker run --rm -it --network docker_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> \
      <ALCHEMISCALE_DOCKER_IMAGE> identity add-scope -t user \
                                                     -i <user identity> \
                                                     -s <org-campaign-project> -s ...

The important bits here are:

``--network docker_db``
    We need to make sure the docker container we are using can talk to the database container.

``-e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD>``
    We need to pass in these environment variables so that the container can talk to the database.
    These should match the values set in ``.env``.


*******
Backups
*******

Performing regular backups of the state store is an important operational component for any production deployment of ``alchemiscale``.
To do this, **first shut down the Neo4j service so that no database processes are currently running**.

The instructions below assume a Docker-based deployment, perhaps via ``docker-compose`` as in :ref:`deploy-docker-compose`.
The same general principles apply to any deployment type, however.

Creating a database dump
========================

**With the Neo4j service shut down**, navigate to the directory containing your database data, set ``$BACKUPS_DIR`` to the absolute path of your choice and ``$NEO4J_VERSION`` to the version of Neo4j you are using, then run::

    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               neo4j-admin dump --to /tmp/neo4j-$(date -I).dump

This will create a new database dump in the ``$BACKUPS_DIR`` directory.


Restoring from a database dump
==============================

To later restore from a database dump, navigate to the directory containing your current database data, and clear or move the current data to another location (Neo4j will not restore to a database that is already initialized).

**With the Neo4j service shut down**, choose ``$DUMP_DATE`` and set ``$NEO4J_VERSION`` to the version of Neo4j you are using, then run::

    docker run --rm \
               -v $(pwd):/var/lib/neo4j/data \
               -v ${BACKUPS_DIR}:/tmp \
               --entrypoint /bin/bash \
               neo4j:${NEO4J_VERSION} \
               neo4j-admin load --from /tmp/neo4j-${DUMP_DATE}.dump

Automating the backup process to perform regular backups without human intervention for your deployment is ideal, but out of scope for this document.
