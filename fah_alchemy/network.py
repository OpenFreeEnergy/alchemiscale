"""

"""

from typing import FrozenSet, Iterable

import networkx as nx
from openfe.setup import Network, AtomMapping

from .nodes import Microstate
from .edges import Transformation

class AlchemicalNetwork(Network):
    """A network of microstates as nodes, alchemical transformations as edges.

    Attributes
    ----------

    """
    
    def __init__(
            self,
            microstates: Iterable[Microstate] = None,
            transformations: Iterable[Transformation] = None
            ):

        
        
        self._graph = None


    def dg(estimator=None):
        """Free energy differences for all transformations based on given estimator,
        using all transformation data in the network.

        """
        ...

    def ddg(estimator=None):
        ...
