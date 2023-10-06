import pytest

from openfe_benchmarks import tyk2
from gufe import ChemicalSystem, Transformation, AlchemicalNetwork
from gufe.tests.test_protocol import DummyProtocol, BrokenProtocol

from alchemiscale import validators


@pytest.fixture(scope="session")
def network_self_transformation():
    tyk2s = tyk2.get_system()
    ligand = tyk2s.ligand_components[0]

    cs = ChemicalSystem(
        components={"ligand": ligand, "solvent": tyk2s.solvent_component},
        name=f"{ligand.name}_water",
    )

    tf = Transformation(
        stateA=cs,
        stateB=cs,
        protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
        name=f"{ligand.name}->{ligand.name}_water",
    )

    return AlchemicalNetwork(edges=[tf], name="self_transformation")


@pytest.fixture(scope="session")
def network_nonself_transformation():
    tyk2s = tyk2.get_system()
    ligand = tyk2s.ligand_components[0]
    ligand2 = tyk2s.ligand_components[1]

    cs = ChemicalSystem(
        components={"ligand": ligand, "solvent": tyk2s.solvent_component},
        name=f"{ligand.name}_water",
    )

    cs2 = ChemicalSystem(
        components={"ligand": ligand2, "solvent": tyk2s.solvent_component},
        name=f"{ligand2.name}_water",
    )

    tf = Transformation(
        stateA=cs,
        stateB=cs2,
        protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
        name=f"{ligand.name}->{ligand2.name}_water",
    )

    return AlchemicalNetwork(edges=[tf], name="nonself_transformation")


def test_validate_network_nonself(
    network_self_transformation, network_nonself_transformation
):
    with pytest.raises(ValueError, match="uses the same `ChemicalSystem`"):
        validators.validate_network_nonself(network_self_transformation)

    out = validators.validate_network_nonself(network_nonself_transformation)

    assert out is None
