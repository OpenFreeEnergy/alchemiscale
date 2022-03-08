
from .protocols import Protocol


class Transformation:
    """An edge of an alchemical network.

    Connects two microstates, with directionality.

    Attributes
    ----------
    protocol : Protocol
        The protocol used to perform the transformation.
        Includes all details needed to perform calculation and encodes the
        alchemical pathway used.

    """

    def dg(self, estimator=None):
        """Free energy difference of transformation based on given estimator,
        using only data available for this edge.

        """
        ...

    def ddg(self, estimator=None):
        ...

    def microstate_start(self):
        ...

    def microstate_end(self):
        ...

