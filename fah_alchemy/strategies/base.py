"""
Alchemical strategy base class. --- :mod:`fah-alchemy.strategies.base`
======================================================================

"""

from gufe.tokenization import GufeTokenizable


class Strategy(GufeTokenizable):
    ...

    def _to_dict(self):
        ...

    def _from_dict(self):
        ...

    @property
    def _defaults(self):
        ...
