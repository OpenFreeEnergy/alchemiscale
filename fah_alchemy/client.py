"""fah-alchemy client for interacting with API server.

"""

import requests

from gufe import AlchemicalNetwork

from .strategies import Strategy


class FahAlchemyClient:


    def submit_network(self, network: AlchemicalNetwork, strategy: Strategy):
        """Submit an AlchemicalNetwork along with a compute Strategy.

        """

        

    def query_networks(self):
        ...


    def get_network(self, scoped_key: str):
        ...
