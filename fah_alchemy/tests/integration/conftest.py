"""Components for standing up services for integration tests, including databases.

"""

## storage
### below from `py2neo.test.integration.conftest.py`

from os import getenv
from time import sleep

from grolt import Neo4jService, Neo4jDirectorySpec, docker
from grolt.security import install_self_signed_certificate
from pytest import fixture
from moto import mock_s3

from py2neo import ServiceProfile, Graph
from py2neo.client import Connector

from gufe import ChemicalSystem, Transformation, AlchemicalNetwork
from gufe.tests.test_protocol import DummyProtocol
from openfe_benchmarks import tyk2

from fah_alchemy.models import Scope
from fah_alchemy.settings import Neo4jStoreSettings, S3ObjectStoreSettings
from fah_alchemy.storage import Neo4jStore, S3ObjectStore, get_s3os
from fah_alchemy.protocols import FAHOpenmmNonEquilibriumCyclingProtocol


NEO4J_PROCESS = {}
NEO4J_VERSION = getenv("NEO4J_VERSION", "")


class DeploymentProfile(object):
    def __init__(self, release=None, topology=None, cert=None, schemes=None):
        self.release = release
        self.topology = topology  # "CE|EE-SI|EE-C3|EE-C3-R2"
        self.cert = cert
        self.schemes = schemes

    def __str__(self):
        server = "%s.%s %s" % (self.release[0], self.release[1], self.topology)
        if self.cert:
            server += " %s" % (self.cert,)
        schemes = " ".join(self.schemes)
        return "[%s]-[%s]" % (server, schemes)


class TestProfile:
    def __init__(self, deployment_profile=None, scheme=None):
        self.deployment_profile = deployment_profile
        self.scheme = scheme
        assert self.topology == "CE"

    def __str__(self):
        extra = "%s" % (self.topology,)
        if self.cert:
            extra += "; %s" % (self.cert,)
        bits = [
            "Neo4j/%s.%s (%s)" % (self.release[0], self.release[1], extra),
            "over",
            "'%s'" % self.scheme,
        ]
        return " ".join(bits)

    @property
    def release(self):
        return self.deployment_profile.release

    @property
    def topology(self):
        return self.deployment_profile.topology

    @property
    def cert(self):
        return self.deployment_profile.cert

    @property
    def release_str(self):
        return ".".join(map(str, self.release))

    def generate_uri(self, service_name=None):
        if self.cert == "full":
            raise NotImplementedError("Full certificates are not yet supported")
        elif self.cert == "ssc":
            certificates_dir = install_self_signed_certificate(self.release_str)
            dir_spec = Neo4jDirectorySpec(certificates_dir=certificates_dir)
        else:
            dir_spec = None
        with Neo4jService(
            name=service_name,
            image=self.release_str,
            auth=("neo4j", "password"),
            dir_spec=dir_spec,
        ) as service:
            uris = [router.uri(self.scheme) for router in service.routers()]
            yield service, uris[0]


# TODO: test with full certificates
neo4j_deployment_profiles = [
    DeploymentProfile(release=(4, 4), topology="CE", schemes=["bolt"]),
]

if NEO4J_VERSION == "LATEST":
    neo4j_deployment_profiles = neo4j_deployment_profiles[:1]
elif NEO4J_VERSION == "4.x":
    neo4j_deployment_profiles = [
        profile for profile in neo4j_deployment_profiles if profile.release[0] == 4
    ]
elif NEO4J_VERSION == "4.4":
    neo4j_deployment_profiles = [
        profile for profile in neo4j_deployment_profiles if profile.release == (4, 4)
    ]


neo4j_test_profiles = [
    TestProfile(deployment_profile, scheme=scheme)
    for deployment_profile in neo4j_deployment_profiles
    for scheme in deployment_profile.schemes
]


@fixture(
    scope="session", params=neo4j_test_profiles, ids=list(map(str, neo4j_test_profiles))
)
def test_profile(request):
    test_profile = request.param
    yield test_profile


@fixture(scope="session")
def neo4j_service_and_uri(test_profile):
    for service, uri in test_profile.generate_uri("py2neo"):
        yield service, uri

    # prune all docker volumes left behind
    docker.volumes.prune()
    return


@fixture(scope="session")
def uri(neo4j_service_and_uri):
    _, uri = neo4j_service_and_uri
    return uri


@fixture(scope="session")
def graph(uri):
    return Graph(uri)


## data
### below specific to fah-alchemy


@fixture(scope="module")
def n4js(graph):
    return Neo4jStore(graph)


@fixture
def n4js_fresh(graph):
    n4js = Neo4jStore(graph)

    n4js.reset()
    n4js.initialize()

    return n4js


@fixture(scope="module")
def s3objectstore_settings():
    return S3ObjectStoreSettings(
        AWS_ACCESS_KEY_ID="test-key-id",
        AWS_SECRET_ACCESS_KEY="test-key",
        AWS_SESSION_TOKEN="test-session-token",
        AWS_S3_BUCKET="test-bucket",
        AWS_S3_PREFIX="test-prefix",
        AWS_DEFAULT_REGION="us-east-1",
    )


@fixture
def s3os(s3objectstore_settings):
    with mock_s3():
        s3os = get_s3os(s3objectstore_settings)
        s3os.initialize()

        yield s3os


# test alchemical networks

# TODO: add in atom mapping once `gufe`#35 is settled


@fixture(scope="session")
def network_tyk2():

    tyk2s = tyk2.get_system()

    solvated = {
        l.name: ChemicalSystem(
            components={"ligand": l, "solvent": tyk2s.solvent_component},
            name=f"{l.name}_water",
        )
        for l in tyk2s.ligand_components
    }
    complexes = {
        l.name: ChemicalSystem(
            components={
                "ligand": l,
                "solvent": tyk2s.solvent_component,
                "protein": tyk2s.protein_component,
            },
            name=f"{l.name}_complex",
        )
        for l in tyk2s.ligand_components
    }

    complex_network = [
        Transformation(
            stateA=complexes[edge[0]],
            stateB=complexes[edge[1]],
            protocol=DummyProtocol(settings=None),
        )
        for edge in tyk2s.connections
    ]
    solvent_network = [
        Transformation(
            stateA=solvated[edge[0]],
            stateB=solvated[edge[1]],
            protocol=DummyProtocol(settings=None),
        )
        for edge in tyk2s.connections
    ]

    return AlchemicalNetwork(
        edges=(solvent_network + complex_network), name="tyk2_relative_benchmark"
    )


@fixture(scope="session")
def scope_test():
    return Scope(org="test_org", campaign="test_campaign", project="test_project")
