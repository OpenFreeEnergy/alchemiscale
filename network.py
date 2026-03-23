from openfe_benchmarks import tyk2

from gufe import ChemicalSystem, Transformation, NonTransformation, AlchemicalNetwork
from gufe.tests.test_protocol import DummyProtocol, BrokenProtocol

class DummyProtocolA(DummyProtocol):
    ...

class DummyProtocolB(DummyProtocol):
    ...

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
            protocol=BrokenProtocol(settings=BrokenProtocol.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_solvent",
        )
        for edge in tyk2s.connections
    ]

    return AlchemicalNetwork(
        edges=(solvent_network + complex_network),
        name="tyk2_relative_benchmark",
    )
