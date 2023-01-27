"""
Client for interacting with user-facing API. --- :mod:`alchemiscale.interface.client`
====================================================================================


"""

from typing import Union, List
import requests
import json

from gufe import AlchemicalNetwork, Transformation, ChemicalSystem
from gufe.tokenization import GufeTokenizable, JSON_HANDLER
from gufe.protocols import ProtocolResult, ProtocolDAGResult

from ..base.client import AlchemiscaleBaseClient, AlchemiscaleBaseClientError
from ..models import Scope, ScopedKey
from ..strategies import Strategy


class AlchemiscaleClientError(AlchemiscaleBaseClientError):
    ...


class AlchemiscaleClient(AlchemiscaleBaseClient):
    """Client for user interaction with API service."""

    _exception = AlchemiscaleClientError

    ### inputs

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
        """Submit an AlchemicalNetwork along with a compute Strategy."""
        ...
        data = dict(network=network.to_dict(), scope=scope.dict())
        scoped_key = self._post_resource("/networks", data)
        return ScopedKey.from_str(scoped_key)

    def query_networks(self) -> List[ScopedKey]:
        raise NotImplementedError

    def get_network(self, network: Union[ScopedKey, str]) -> AlchemicalNetwork:
        return self._get_resource(f"/networks/{network}", {}, return_gufe=True)

    def get_transformation(
        self, transformation: Union[ScopedKey, str]
    ) -> Transformation:
        return self._get_resource(
            f"/transformations/{transformation}", {}, return_gufe=True
        )

    def get_chemicalsystem(
        self, chemicalsystem: Union[ScopedKey, str]
    ) -> ChemicalSystem:
        return self._get_resource(
            f"/chemicalsystems/{chemicalsystem}", {}, return_gufe=True
        )

    ### compute

    def set_strategy(self, network: ScopedKey, strategy: Strategy):
        ...

    ### results

    def get_transformation_result(
        self,
        transformation: ScopedKey,
        return_protocoldagresults: bool = False,
    ) -> Union[ProtocolResult, List[List[ProtocolDAGResult]]]:
        """Get `ProtocolResult` for the given `Transformation`.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve results for.
        return_protocoldagresults
            If `True`, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`.

        """

        pdrs_json = []
        limit = 10
        skip = 0

        # first, get the transformation; also confirms it exists
        tf: Transformation = self.get_transformation(transformation)

        while True:
            # iterate through all results with paginated API calls
            params = {"limit": limit, "skip": skip}
            pdrs_i = self._get_resource(
                f"/transformations/{transformation}/result",
                params=params,
                return_gufe=False,
            )

            # we break if we get nothing back; means we're at the end of the line
            if len(pdrs_i) == 0:
                break

            pdrs_json.extend(pdrs_i)
            skip += limit

        # walk through data structure, and turn each structure into a `ProtocolDAGResult`
        # TODO: [OPTIMIZATION] do this as we make our requests above; use async/await

        pdrs = []
        for pdrlist in pdrs_json:
            pdrs_i = []
            for pdr_json in pdrlist:
                pdr = GufeTokenizable.from_dict(
                    json.loads(pdr_json, cls=JSON_HANDLER.decoder)
                )
                pdrs_i.append(pdr)

            pdrs.append(pdrs_i)

        if return_protocoldagresults:
            return pdrs
        else:
            return tf.protocol.gather(pdrs)
