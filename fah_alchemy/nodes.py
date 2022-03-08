from typing import FrozenSet, Iterable

from openff.toolkit.topology import Topology

class Microstate:
    """A node of an alchemical network.

    Attributes
    ----------
    identifier : str
        Unique identifier for the microstate; used as the graph node itself
        when the microstate is added to an `AlchemicalNetwork`
    topology : `openff.toolkit.Topology`
        

    """

    def __init__(
            self,
            identifier: str,
            topology: Topology,
            components,
            ):
        ...
