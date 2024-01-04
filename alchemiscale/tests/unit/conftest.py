from pytest import fixture
from openfe_benchmarks import tyk2
from gufe import ChemicalSystem, Transformation, AlchemicalNetwork
from gufe.tests.test_protocol import DummyProtocol


@fixture(scope="module")
def network():
    tyk2s = tyk2.get_system()

    solvated = {
        lig.name: ChemicalSystem(
            components={"ligand": lig, "solvent": tyk2s.solvent_component},
            name=f"{lig.name}_water",
        )
        for lig in tyk2s.ligand_components
    }
    complexes = {
        lig.name: ChemicalSystem(
            components={
                "ligand": lig,
                "solvent": tyk2s.solvent_component,
                "protein": tyk2s.protein_component,
            },
            name=f"{lig.name}_complex",
        )
        for lig in tyk2s.ligand_components
    }

    complex_network = [
        Transformation(
            stateA=complexes[edge[0]],
            stateB=complexes[edge[1]],
            protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_complex",
        )
        for edge in tyk2s.connections
    ]
    solvent_network = [
        Transformation(
            stateA=solvated[edge[0]],
            stateB=solvated[edge[1]],
            protocol=DummyProtocol(settings=DummyProtocol.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_solvent",
        )
        for edge in tyk2s.connections
    ]

    return AlchemicalNetwork(
        edges=(solvent_network + complex_network), name="tyk2_relative_unit"
    )


@fixture(scope="module")
def chemicalsystem_lig_emj_50_complex(network):
    system_name = "lig_ejm_50_complex"
    for n in network.nodes:
        if n.name == system_name:
            cs = n
            break
    else:
        raise ValueError(
            f"Could not find the target chemical system {system_name}. Has the"
            " benchmark system changed?"
        )

    return cs
