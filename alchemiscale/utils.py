from gufe.tokenization import (
    GufeTokenizable,
    TOKENIZABLE_REGISTRY,
    get_all_gufe_objs,
    is_gufe_obj,
    modify_dependencies,
)
from itertools import chain
from typing import Union, Dict, List
import networkx as nx


class RegistryBackup(object):
    def __init__(self, gufe_object=None, keep_changes=False):
        self.gufe_object = gufe_object
        self.keep_changes = keep_changes

    def __enter__(self):
        self.registry_backup = TOKENIZABLE_REGISTRY.copy()
        self.clear_gufe_deps()
        return self.registry_backup

    def __exit__(self, exception, value, traceback):
        if not self.keep_changes or exception is not None:
            TOKENIZABLE_REGISTRY.clear()
        TOKENIZABLE_REGISTRY.update(self.registry_backup)

    def clear_gufe_deps(self):
        if self.gufe_object is not None:
            objects = get_all_gufe_objs(self.gufe_object)

            for o in objects:
                gufe_key = o.key
                TOKENIZABLE_REGISTRY.pop(gufe_key)
        else:
            TOKENIZABLE_REGISTRY.clear()


def gufe_objects_from_shallow_dict(
    obj: Union[List, Dict, GufeTokenizable]
) -> List[GufeTokenizable]:
    """Find GufeTokenizables within a shallow dict.

    This function recursively looks through the list/dict structures encoding
    GufeTokenizables and returns list of all GufeTokenizables found
    within those structures, which may be potentially nested.

    Parameters
    ----------
    obj
        The input data structure to recursively traverse. For the initial call
        of this function, this should be the shallow dict of a GufeTokenizable.
        Input of a GufeTokenizable will immediately return a base case.

    Returns
    -------
    List[GufeTokenizable]
        All GufeTokenizables found in the shallow dict representation of a
        GufeTokenizable.

    """
    if is_gufe_obj(obj):
        return [obj]

    elif isinstance(obj, list):
        return list(
            chain.from_iterable([gufe_objects_from_shallow_dict(item) for item in obj])
        )

    elif isinstance(obj, dict):
        return list(
            chain.from_iterable(
                [gufe_objects_from_shallow_dict(item) for item in obj.values()]
            )
        )

    return []


def gufe_to_digraph(gufe_obj):
    """Recursively construct a DiGraph from a GufeTokenizable.

    The DiGraph encodes the dependency structure of the GufeTokenizable on
    other GufeTokenizables.
    """
    graph = nx.DiGraph()
    shallow_dicts = {}

    def add_edges(o):
        # if we've made a shallow dict before, we've already added this one
        # and all its dependencies; return `None` to avoid going down the tree
        # again
        sd = shallow_dicts.get(o.key)
        if sd is not None:
            return None

        # if not, then we make the shallow dict only once, add it to our index,
        # add edges to dependencies, and return it so we continue down the tree
        sd = o.to_shallow_dict()

        shallow_dicts[o.key] = sd

        # add the object node in case there aren't any connections
        graph.add_node(o)
        connections = gufe_objects_from_shallow_dict(sd)

        for c in connections:
            graph.add_edge(o, c)

        return sd

    sd = add_edges(gufe_obj)
    _ = modify_dependencies(sd, add_edges, is_gufe_obj, mode="encode")

    return graph
