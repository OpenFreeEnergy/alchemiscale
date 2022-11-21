from typing import Optional, Iterable, List, Dict, Any

from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping

from gufe.protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
    ProtocolUnitResult,
    Context,
    execute,
)


class FAHOpenmmNonEquilibriumCyclingResult(ProtocolResult):
    def get_estimate(self):
        ...

    def get_uncertainty(self):
        ...

    def get_rate_of_convergence(self):
        ...


class FAHOpenmmNonEquilibriumCyclingProtocol(Protocol):
    """Perform a nonequilibrium cycling transformation.

    This is based on `perses.protocols.NonEquilibriumCyclingProtocol`.

    """

    _results_cls = FAHOpenmmNonEquilibriumCyclingResult
    _supported_engines = ["openmm"]

    @classmethod
    def _default_settings(cls):
        return {}

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[ComponentMapping] = None,
        extend_from: Optional[ProtocolDAGResult] = None,
    ) -> List[ProtocolUnit]:

        # we generate a linear DAG here, since OpenMM performs nonequilibrium
        # cycling in a single simulation
        genhtop = GenerateHybridTopology(
            name="the beginning",
            settings=self.settings,
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            start=extend_from,
            some_dict={"a": 2, "b": 12},
        )

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        sim = SimulationUnit(self.settings, initialization=genhtop)

        end = GatherUnit(self.settings, name="gather", simulations=[sim])

        return [genhtop, sim, end]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> Dict[str, Any]:

        outputs = []
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "gather":
                    outputs.append(pur.data)

        return dict(data=outputs)
