"""Client for interacting with user-facing API.

"""

from typing import Union, List, Dict, Optional
import requests
import json

import networkx as nx
from gufe import AlchemicalNetwork, Transformation, ChemicalSystem
from gufe.tokenization import GufeTokenizable, JSON_HANDLER
from gufe.protocols import ProtocolResult, ProtocolDAGResult

from ..base.client import FahAlchemyBaseClient, FahAlchemyBaseClientError
from ..models import Scope, ScopedKey
from ..strategies import Strategy


class FahAlchemyClientError(FahAlchemyBaseClientError):
    ...


class FahAlchemyClient(FahAlchemyBaseClient):
    """Client for user interaction with API service."""

    _exception = FahAlchemyClientError

    ### inputs

    def get_scoped_key(self, obj: GufeTokenizable, scope: Scope):
        """Given any gufe object and a fully-specified Scope, return corresponding ScopedKey.

        This method does not check that this ScopedKey is represented in the database.
        It is only a convenience for properly constructing a ScopedKey from a
        gufe object and a Scope.

        """
        if scope.specific():
            return ScopedKey(gufe_key=obj.key, **scope.dict())

    def create_network(self, network: AlchemicalNetwork, scope: Scope):
        """Submit an AlchemicalNetwork along with a compute Strategy."""
        ...
        data = dict(network=network.to_dict(), scope=scope.dict())
        scoped_key = self._post_resource("/networks", data)
        return ScopedKey.from_dict(scoped_key)

    def query_networks(
            self, 
            name: Optional[str] = None, 
            scope: Optional[Scope] = None, 
            return_gufe=False, 
            limit=None, 
            skip=None
    ) -> Union[List[ScopedKey], Dict[ScopedKey, AlchemicalNetwork]]:
        """Query for AlchemicalNetworks, optionally by name or Scope.

        Calling this method with no query arguments will return ScopedKeys for
        all AlchemicalNetworks that are within the Scopes this user has access
        to.

        """
        if return_gufe:
            networks = {}
        else:
            networks = []

        if scope is None:
            scope = Scope()

        params = dict(
            name=name, return_gufe=return_gufe, limit=limit, skip=skip, **scope.dict()
        )
        if return_gufe:
            networks.update(self._query_resource("/networks", params=params))
        else:
            networks.extend(self._query_resource("/networks", params=params))

        return networks

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
        """Set the Strategy for evaluating the given AlchemicalNetwork.

        The Strategy will be applied to create and action tasks for the
        Transformations in the AlchemicalNetwork without user interaction.

        """
        raise NotImplementedError

    def create_transformation_task(
            transformation: ScopedKey,
            extend_from: ScopedKey):
        ...

    def get_transformation_tasks() -> nx.DiGraph:
        """Return a net

        """
        ...

    def action_transformation_tasks(
            tasks: List[ScopedKey],
            network: ScopedKey
            ):
        ...

    ### results

    def get_transformation_result(
        self, 
        transformation: ScopedKey, 
        return_protocoldagresults: bool = False,
    ) -> Union[ProtocolResult, List[ProtocolDAGResult]]:
        """Get `ProtocolResult` for the given `Transformation`.

        Parameters
        ----------
        transformation
            The `ScopedKey` of the `Transformation` to retrieve results for.
        return_protocoldagresults
            If `True`, return the raw `ProtocolDAGResult`s instead of returning
            a processed `ProtocolResult`.

        """

        # first, get the transformation; also confirms it exists
        tf: Transformation = self.get_transformation(transformation)

        # get all objectstorerefs for the given transformation
        objectstorerefs = self._get_resource(
            f"/transformations/{transformation}/results",
            return_gufe=False,
        )

        # get each protocoldagresult; could optimize by parallelizing these
        # calls to some extent, or at least using async/await
        pdrs = []
        for objectstoreref in objectstorerefs:
            pdr_key = objectstoreref['obj_key']

            pdr_json = self._get_resource(
                f"/protocoldagresults/{pdr_key}",
                return_gufe=False,
            )[0]

            pdr = GufeTokenizable.from_dict(
                json.loads(pdr_json, cls=JSON_HANDLER.decoder)
            )
            pdrs.append(pdr)

        if return_protocoldagresults:
            return pdrs
        else:
            return tf.protocol.gather(pdrs)
