import time

from openfe_benchmarks import tyk2

from gufe import ChemicalSystem, Transformation, NonTransformation, AlchemicalNetwork
from gufe.tests.test_protocol import DummyProtocol, BrokenProtocol, FinishUnit, SimulationUnit, InitializeUnit, ProtocolUnit
from gufe.protocols import ProtocolUnit

class WeightedInitializeUnit(InitializeUnit):
    value = 2

class WeightedSimulationUnit(SimulationUnit):
    value = 2

    @staticmethod
    def _execute(ctx, *, initialization, **inputs):
        time.sleep(WeightedSimulationUnit.value * 2)
        return SimulationUnit._execute(ctx, initialization=initialization, **inputs)

class WeightedFinishUnit(FinishUnit):
    value = 1

    @staticmethod
    def _execute(ctx, *, simulations, **inputs):
        time.sleep(WeightedFinishUnit.value * 2)
        return FinishUnit._execute(ctx, simulations=simulations, **inputs)


class WeightedDummyProtocol(DummyProtocol):

    def _create(
            self,
            stateA,
            stateB,
            mapping = None,
            extends = None,
    ):
        if extends is not None:
            # this is an example; wouldn't want to pass in whole ProtocolDAGResult into
            # any ProtocolUnits below, since this could create dependency hell;
            # instead, extract what's needed from it for starting point here
            starting_point = extends.protocol_unit_results[-1].outputs["key_results"]
        else:
            starting_point = None

        # convert protocol inputs into starting points for independent simulations
        alpha = WeightedInitializeUnit(
            name="the beginning",
            settings=self.settings,
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            start=starting_point,
            some_dict={"a": 2, "b": 12},
        )

        # create several units that would each run an independent simulation
        simulations: list[ProtocolUnit] = [
            WeightedSimulationUnit(settings=self.settings, name=f"sim {i}", window=i, initialization=alpha)
            for i in range(self.settings.n_repeats)  # type: ignore
        ]

        # gather results from simulations, finalize outputs
        omega = WeightedFinishUnit(settings=self.settings, name="the end", simulations=simulations)

        # return all `ProtocolUnit`s we created
        return [alpha, *simulations, omega]

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
            protocol=WeightedDummyProtocol(settings=WeightedDummyProtocol.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_complex",
        )
        for edge in tyk2s.connections
    ]
    solvent_network = [
        Transformation(
            stateA=solvated[edge[0]],
            stateB=solvated[edge[1]],
            protocol=WeightedDummyProtocol(settings=WeightedDummyProtocol.default_settings()),
            name=f"{edge[0]}_to_{edge[1]}_solvent",
        )
        for edge in tyk2s.connections
    ]
    # breakpoint()
    # pu = solvent_network[0].create().protocol_units[0]
    # pu.execute(context=None, initialization=None)

    return AlchemicalNetwork(
        edges=(solvent_network + complex_network),
        name="tyk2_relative_benchmark",
    )
