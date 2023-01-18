# Deployment via `docker-compose`

Note: this assumes a running Docker daemon and the `docker-compose` plugin.

To deploy (using bash + EC2 host):

1. Make a copy of `.env.template`
```bash
cp .env.template .env
```
and fill in the environmental variables.
See `.env.testing` for an example.

2. Comment out this line:
`- "--certificatesresolvers.myresolver.acme.caserver=https://acme-staging-v02.api.letsencrypt.org/directory"`
in the docker-compose.yml file if you want Let's Encrypt (LE) to issue a real SSL certificate.
The API for LE is rate limited, so I recommend first keeping this line in to see if a certificate is issued.

3. Make the data directory for `neo4j`

The neo4j container requires the directory for the data store to exist before it starts.
Create the location according to what you set for `NEO4J_DATA_STORE`.
For example:

```bash
mkdir -p data/server
```

4. Start the service
 
```bash 
USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up -d
```
