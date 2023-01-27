from . import _version

__version__ = _version.get_versions()["version"]

from .interface import FahAlchemyClient
from .models import Scope, ScopedKey
