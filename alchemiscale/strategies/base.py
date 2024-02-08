"""
:mod:`alchemiscale.strategies.base` --- alchemical strategy base class
======================================================================

"""

from gufe.tokenization import GufeTokenizable


class Strategy(GufeTokenizable):
    ...

    def _to_dict(self): ...

    def _from_dict(self): ...

    @property
    def _defaults(self): ...
