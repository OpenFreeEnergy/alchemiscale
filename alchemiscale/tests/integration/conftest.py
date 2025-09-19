"""Components for standing up services for integration tests, including databases."""

## storage
### below from `py2neo.test.integration.conftest.py`

import os
from pathlib import Path
import logging

from grolt import Neo4jService, Neo4jDirectorySpec, docker
from grolt.security import install_self_signed_certificate
from pytest import fixture
from moto import mock_aws
from moto.server import ThreadedMotoServer

from neo4j import GraphDatabase

from gufe import ChemicalSystem, NonTransformation, Transformation, AlchemicalNetwork
from gufe.protocols import ProtocolResult
from gufe.protocols.protocoldag import execute_DAG
from gufe.tests.test_protocol import DummyProtocol, BrokenProtocol
from openfe_benchmarks import tyk2

from alchemiscale.models import Scope
from alchemiscale.settings import S3ObjectStoreSettings, Neo4jStoreSettings
from alchemiscale.storage.statestore import Neo4jStore
from alchemiscale.storage.objectstore import get_s3os
from alchemiscale.storage.models import ComputeServiceID
from stratocaster.base import Strategy, StrategyResult, StrategySettings


NEO4J_PROCESS = {}
NEO4J_VERSION = os.getenv("NEO4J_VERSION", "")


# suppress warnings from neo4j python driver
neo4j_logger = logging.getLogger("neo4j")
neo4j_logger.setLevel(logging.ERROR)


class DeploymentProfile:
    def __init__(self, release=None, topology=None, cert=None, schemes=None):
        self.release = release
        self.topology = topology  # "CE|EE-SI|EE-C3|EE-C3-R2"
        self.cert = cert
        self.schemes = schemes

    def __str__(self):
        server = f"{self.release[0]}.{self.release[1]} {self.topology}"
        if self.cert:
            server += f" {self.cert}"
        schemes = " ".join(self.schemes)
        return f"[{server}]-[{schemes}]"


class TestProfile:
    def __init__(self, deployment_profile=None, scheme=None):
        self.deployment_profile = deployment_profile
        self.scheme = scheme
        assert self.topology == "CE"

    def __str__(self):
        extra = f"{self.topology}"
        if self.cert:
            extra += f"; {self.cert}"
        bits = [
            f"Neo4j/{self.release[0]}.{self.release[1]} ({extra})",
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
            config={},
        ) as service:
            uris = [router.uri(self.scheme) for router in service.routers()]
            yield service, uris[0]


# TODO: test with full certificates
neo4j_deployment_profiles = [
    DeploymentProfile(release=(5, 25), topology="CE", schemes=["bolt"]),
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
    yield from test_profile.generate_uri("py2neo")

    # prune all docker volumes left behind
    docker.volumes.prune()
    return


@fixture(scope="session")
def uri(neo4j_service_and_uri):
    _, uri = neo4j_service_and_uri
    return uri


## data
### below specific to alchemiscale


@fixture(scope="module")
def n4jstore_settings(uri):

    os.environ["NEO4J_URL"] = uri
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASS"] = "password"
    os.environ["NEO4J_DBNAME"] = "neo4j"

    return Neo4jStoreSettings()


@fixture(scope="module")
def n4js(n4jstore_settings):

    n4js = Neo4jStore(n4jstore_settings)

    yield n4js

    n4js.close()


@fixture
def n4js_fresh(n4jstore_settings):
    n4js = Neo4jStore(n4jstore_settings)

    n4js.reset()
    n4js.initialize()

    yield n4js

    n4js.close()


@fixture
def n4js_task_restart_policy(
    n4js_fresh: Neo4jStore, network_tyk2: AlchemicalNetwork, scope_test
):

    n4js = n4js_fresh

    _, taskhub_scoped_key_with_policy, _ = n4js.assemble_network(
        network_tyk2, scope_test
    )

    _, taskhub_scoped_key_no_policy, _ = n4js.assemble_network(
        network_tyk2.copy_with_replacements(name=network_tyk2.name + "_no_policy"),
        scope_test,
    )

    transformation_1_scoped_key, transformation_2_scoped_key = map(
        lambda transformation: n4js.get_scoped_key(transformation, scope_test),
        list(network_tyk2.edges)[:2],
    )

    # create 4 tasks for each of the 2 selected transformations
    task_scoped_keys = n4js.create_tasks(
        [transformation_1_scoped_key] * 4 + [transformation_2_scoped_key] * 4
    )

    # action the tasks for transformation 1 on the taskhub with no policy
    # action the tasks for both transformations on the taskhub with a policy
    assert all(n4js.action_tasks(task_scoped_keys[:4], taskhub_scoped_key_no_policy))
    assert all(n4js.action_tasks(task_scoped_keys, taskhub_scoped_key_with_policy))

    patterns = [
        r"Error message \d, round \d",
        "This is an example pattern that will be used as a restart string.",
    ]

    n4js.add_task_restart_patterns(
        taskhub_scoped_key_with_policy, patterns=patterns, number_of_retries=2
    )

    return n4js


@fixture(scope="module")
def s3objectstore_settings():
    os.environ["AWS_ACCESS_KEY_ID"] = "test-key-id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-key"
    os.environ["AWS_SESSION_TOKEN"] = "test-session-token"
    os.environ["AWS_S3_BUCKET"] = "test-bucket"
    os.environ["AWS_S3_PREFIX"] = "test-prefix"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    return S3ObjectStoreSettings()


@fixture(scope="module")
def s3objectstore_settings_endpoint(s3objectstore_settings):

    settings = s3objectstore_settings.model_dump()
    settings["AWS_ENDPOINT_URL"] = "http://127.0.0.1:5000"

    return S3ObjectStoreSettings(**settings)


@fixture(scope="module")
def s3os_server(s3objectstore_settings_endpoint):

    server = ThreadedMotoServer()
    server.start()

    # Create settings with endpoint URL for testing
    s3os = get_s3os(s3objectstore_settings_endpoint)
    s3os.initialize()

    yield s3os

    server.stop()


@fixture
def s3os_server_fresh(s3os_server):
    s3os_server.reset()
    s3os_server.initialize()

    return s3os_server


@fixture(scope="module")
def s3os(s3objectstore_settings):

    with mock_aws():
        s3os = get_s3os(s3objectstore_settings)
        s3os.initialize()

        yield s3os


# test alchemical networks


## define varying protocols to simulate protocol variety
class DummyProtocolA(DummyProtocol):
    pass


class DummyProtocolB(DummyProtocol):
    pass


class DummyProtocolC(DummyProtocol):
    pass


class DummyStrategySettings(StrategySettings):
    """Settings for DummyStrategy."""

    ...


class DummyStrategy(Strategy):
    """Test strategy for integration tests."""

    _settings_cls = DummyStrategySettings

    def __init__(self, settings=None):
        if settings is None:
            settings = self._default_settings()
        super().__init__(settings)

    @classmethod
    def _default_settings(cls):
        return DummyStrategySettings()

    def _propose(self, alchemical_network, protocol_results):
        """Simple strategy that returns equal weights for any transformations
        if no results exist, and ``None`` for each transformation if *any*
        results exist.

        """

        weights = {}
        for transformation in alchemical_network.edges:
            if isinstance(protocol_results[transformation.key], ProtocolResult):
                weights[transformation.key] = None
            else:
                weights[transformation.key] = 0.5

        return StrategyResult(weights)


# TODO: add in atom mapping once `gufe`#35 is settled


@fixture(scope="module")
def network_tyk2():
    tyk2s = tyk2.get_system()

    solvated = {
        ligand.name: ChemicalSystem(
            components={"ligand": ligand, "solvent": tyk2s.solvent_component},
            name=f"{ligand.name}_water",
        )
        for ligand in tyk2s.ligand_components
    }
    complexes = {
        ligand.name: ChemicalSystem(
            components={
                "ligand": ligand,
                "solvent": tyk2s.solvent_component,
                "protein": tyk2s.protein_component,
            },
            name=f"{ligand.name}_complex",
        )
        for ligand in tyk2s.ligand_components
    }

    complex_network = [
        Transformation(
            stateA=complexes[edge[0]],
            stateB=complexes[edge[1]],
            protocol=DummyProtocolA(settings=DummyProtocolA.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_complex",
        )
        for edge in tyk2s.connections
    ]
    solvent_network = [
        Transformation(
            stateA=solvated[edge[0]],
            stateB=solvated[edge[1]],
            protocol=DummyProtocolB(settings=DummyProtocolB.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_solvent",
        )
        for edge in tyk2s.connections
    ]

    nontransformations = []
    for cs in list(solvated.values()) + list(complexes.values()):
        nt = NonTransformation(
            system=cs,
            protocol=DummyProtocolC(DummyProtocolC.default_settings()),
            name=f"f{cs.name}_nt",
        )
        nontransformations.append(nt)

    return AlchemicalNetwork(
        edges=(solvent_network + complex_network + nontransformations),
        name="tyk2_relative_benchmark",
    )


def get_edge_type(
    network: AlchemicalNetwork, edge_class
) -> Transformation | NonTransformation:
    for tf in sorted(network.edges):
        if type(tf) is edge_class:
            return tf
    raise RuntimeError("Network does not contain a `{edge_class.__qualname__}`")


@fixture(scope="module")
def transformation(network_tyk2):
    return get_edge_type(network_tyk2, Transformation)


@fixture(scope="module")
def nontransformation(network_tyk2):
    return get_edge_type(network_tyk2, NonTransformation)


@fixture(scope="module")
def chemicalsystem(network_tyk2):
    return list(network_tyk2.nodes)[0]


@fixture(scope="module")
def protocoldagresults(tmpdir_factory, transformation):
    pdrs = []
    for i in range(3):
        # Use tempdir_factory instead of tempdir to handle session level scope correctly
        protocoldag = transformation.create()

        # execute the task
        with tmpdir_factory.mktemp("protocol_dag").as_cwd():
            shared_basedir = Path("shared").absolute()
            shared_basedir.mkdir()
            scratch_basedir = Path("scratch").absolute()
            scratch_basedir.mkdir()

            protocoldagresult = execute_DAG(
                protocoldag,
                shared_basedir=shared_basedir,
                scratch_basedir=scratch_basedir,
            )

        pdrs.append(protocoldagresult)
    return pdrs


@fixture
def network_tyk2_failure(network_tyk2):
    transformation = get_edge_type(network_tyk2, Transformation)

    broken_transformation = Transformation(
        stateA=transformation.stateA,
        stateB=transformation.stateB,
        protocol=BrokenProtocol(settings=BrokenProtocol.default_settings()),
        name="broken",
    )

    return AlchemicalNetwork(
        edges=[broken_transformation] + list(network_tyk2.edges), name="tyk2_broken"
    )


@fixture
def transformation_failure(network_tyk2_failure):
    return [t for t in network_tyk2_failure.edges if t.name == "broken"][0]


@fixture
def protocoldagresults_failure(tmpdir_factory, transformation_failure):
    pdrs = []
    for i in range(3):
        # Use tempdir_factory instead of tempdir to handle session level scope correctly
        protocoldag = transformation_failure.create()

        # execute the task
        with tmpdir_factory.mktemp("protocol_dag").as_cwd():
            shared_basedir = Path("shared").absolute()
            shared_basedir.mkdir()
            scratch_basedir = Path("scratch").absolute()
            scratch_basedir.mkdir()

            protocoldagresult = execute_DAG(
                protocoldag,
                shared_basedir=shared_basedir,
                scratch_basedir=scratch_basedir,
                raise_error=False,
            )

        pdrs.append(protocoldagresult)
    return pdrs


@fixture(scope="module")
def scope_test():
    """Primary scope for individual tests"""
    return Scope(org="test_org", campaign="test_campaign", project="test_project")


@fixture(scope="module")
def multiple_scopes(scope_test):
    scopes = [scope_test]  # Append initial test
    # Augment
    scopes.extend(
        [
            Scope(
                org=f"test_org_{x}",
                campaign=f"test_campaign_{x}",
                project=f"test_project_{x}",
            )
            for x in range(1, 3)
        ]
    )
    return scopes


@fixture(scope="module")
def compute_service_id():
    return ComputeServiceID("compute-service-123")
