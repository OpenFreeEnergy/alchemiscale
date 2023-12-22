from gufe.tokenization import modify_dependencies, is_gufe_obj, GufeTokenizable
from itertools import chain
from typing import Union, Dict, List
import networkx as nx


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
    """Recursively construct a DiGraph from a GufeTokenizable."""
    graph = nx.DiGraph()

    def add_edges(o):
        # graph.add_node(o)
        connections = gufe_objects_from_shallow_dict(o.to_shallow_dict())

        for c in connections:
            # graph.add_node(c)
            graph.add_edge(o, c)

    add_edges(gufe_obj)

    def modifier(o):
        add_edges(o)
        return o.to_shallow_dict()

    _ = modify_dependencies(
        gufe_obj.to_shallow_dict(), modifier, is_gufe_obj, mode="encode"
    )

    return graph
