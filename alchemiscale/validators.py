"""
:mod:`alchemiscale.validators` --- validation guardrails for user input
=======================================================================

"""

from gufe import AlchemicalNetwork, Transformation


def validate_network_nonself(network: AlchemicalNetwork):
    """Check that the given AlchemicalNetwork features no Transformations with
    the same ChemicalSystem for its two states.

    A ``ValueError`` is raised if a `Transformation` is detected.

    """
    for transformation in network.edges:
        if transformation.stateA == transformation.stateB:
            raise ValueError(
                f"`Transformation` '{transformation.key}' uses the same `ChemicalSystem` '{transformation.stateA.key}' for both states; "
                "this is currently not supported in `alchemiscale`"
            )
