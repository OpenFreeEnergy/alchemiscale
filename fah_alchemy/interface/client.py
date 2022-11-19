"""Client for interacting with user-facing API.

"""

import requests

from gufe import AlchemicalNetwork

from ..base.client import FahAlchemyBaseClient, FahAlchemyBaseClientError
from ..models import Scope, ScopedKey
from ..strategies import Strategy


class FahAlchemyClientError(FahAlchemyBaseClientError):
    ...


class FahAlchemyClient(FahAlchemyBaseClient):
    """Client for user interaction with API service."""

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
        """Submit an AlchemicalNetwork along with a compute Strategy."""
        ...
        data = dict(network=network.to_dict(), scope=scope.dict())
        scoped_key = self._post_resource("/networks", data)
        return ScopedKey.from_str(scoped_key)

    def query_networks(self):
        ...

    def get_network(self, scoped_key: str):
        ...

    def set_strategy(self, network: ScopedKey, strategy: Strategy):
        ...
