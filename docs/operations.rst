##########
Operations
##########

*********
Add Users
*********

To add a new user identity, you will generally use the ``alchemiscale`` CLI::


    $ export NEO4J_URL=bolt://<NEO4J_HOSTNAME>7687
    $ export NEO4J_USER=<NEO4J_USERNAME>
    $ export NEO4J_PASS=<NEO4J_PASSWORD>
    $
    $ # add a user identity, with key
    $ alchemiscale identity add -t user -i <user identity> -k <user key>
    $
    $ add one or more scopes the user should have access to
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

