from gufe.tokenization import GufeTokenizable, key_decode_dependencies
import networkx as nx
from alchemiscale.utils import gufe_to_digraph

from typing import List, Tuple, Dict, Generator


class KeyedChain(object):
    """Keyed chain representation of a GufeTokenizable.

    The keyed chain representation of a GufeTokenizable provides a
    topologically sorted list of gufe keys and GufeTokenizable keyed dicts
    that can be used to fully recreate a GufeTokenizable without the need for a
    populated TOKENIZATION_REGISTRY.

    The class wraps around a list of tuples containing the gufe key and the
    keyed dict form of the GufeTokenizable.

    """

    def __init__(self, keyed_chain):
        self._keyed_chain = keyed_chain

    @classmethod
    def from_gufe(cls, gufe_object: GufeTokenizable) -> super:
        """Initialize a KeyedChain from a GufeTokenizable."""
        return cls(cls.gufe_to_keyed_chain_rep(gufe_object))

    def to_gufe(self) -> GufeTokenizable:
        """Initialize a GufeTokenizable."""
        gts = {}
        for gufe_key, keyed_dict in self:
            gt = key_decode_dependencies(keyed_dict, registry=gts)
            gts[gufe_key] = gt
        return gt

    @staticmethod
    def gufe_to_keyed_chain_rep(
        gufe_object: GufeTokenizable,
    ) -> List[Tuple[str, Dict]]:
        """Create the keyed chain represenation of a GufeTokenizable.

        This represents the GufeTokenizable as a list of two-element tuples
        containing, as their first and second elements, the gufe key and keyed
        dict form of the GufeTokenizable, respectively, and provides the
        underlying structure used in the KeyedChain class.

        Parameters
        ----------
        gufe_object
            The GufeTokenizable for which the KeyedChain is generated.

        Returns
        -------
        key_and_keyed_dicts
            The keyed chain represenation of a GufeTokenizable.

        """
        key_and_keyed_dicts = [
            (str(gt.key), gt.to_keyed_dict())
            for gt in nx.topological_sort(gufe_to_digraph(gufe_object))
        ][::-1]
        return key_and_keyed_dicts

    def gufe_keys(self) -> Generator[str, None, None]:
        """Create a generator that iterates over the gufe keys in the KeyedChain."""
        for key, _ in self:
            yield key

    def keyed_dicts(self) -> Generator[Dict, None, None]:
        """Create a generator that iterates over the keyed dicts in the KeyedChain."""
        for _, _dict in self:
            yield _dict

    def __len__(self):
        return len(self._keyed_chain)

    def __iter__(self):
        return self._keyed_chain.__iter__()

    def __getitem__(self, index):
        return self._keyed_chain[index]
