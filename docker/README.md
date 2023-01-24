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

# AWS EC Setup

```bash
# Install docker
sudo amazon-linux-extras install docker
# Start the docker daemon
sudo service docker start
# Add ec2-user to docker group
sudo usermod -a -G docker ec2-user
# Add ec2-user to docker group (no login/reboot required)
newgrp docker
# Have docker daemon start on reboot
sudo chkconfig docker on
# Install handy tools
sudo yum install -y git tmux
# Install docker compose
sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
# Fix permissions
sudo chmod +x /usr/local/bin/docker-compose
# check docker compose install
docker-compose version
# check docker install
docker info
```

# AWS EC2 Admin

## Example Admin Tasks

* Adding a new user to the database:
`docker run --rm -it --network docker_db -e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD> ghcr.io/openforcefield/fah-alchemy:feat-add_docker_compose identity add -t user -i testuser -k some-password`

The import bits here are:
1. `--network docker_db`
We need to make sure the docker container we are using can talk to the database container
2. `-e NEO4J_URL=bolt://neo4j:7687 -e NEO4J_USER=<USER> -e NEO4J_PASS=<PASSWORD>`
We need to pass in these environmental so that the container can talk to the database. These should match the values set in `.env`

