"""
:mod:`alchemiscale.strategist` --- strategist service
====================================================

The strategist service coordinates the execution of strategies on alchemical networks.
"""

from .service import StrategistService
from .settings import StrategistSettings

__all__ = ["StrategistService", "StrategistSettings"]
