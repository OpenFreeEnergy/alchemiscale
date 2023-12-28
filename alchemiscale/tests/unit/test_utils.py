from openfe_benchmarks import tyk2
from alchemiscale.utils import (
    gufe_objects_from_shallow_dict,
    gufe_to_digraph,
    gufe_to_keyed_dicts,
    keyed_dicts_to_gufe,
    RegistryBackup,
)
from gufe.tokenization import GufeTokenizable


def test_gufe_objects_from_shallow_dict(chemical_system):
    shallow_dict = chemical_system.to_shallow_dict()
    gufe_objects = gufe_objects_from_shallow_dict(shallow_dict)

    for gufe_object in gufe_objects:
        assert gufe_object in chemical_system.components.values()


def test_gufe_to_digraph(chemical_system):
    graph = gufe_to_digraph(chemical_system)

    connected_objects = gufe_objects_from_shallow_dict(
        chemical_system.to_shallow_dict()
    )

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3

    for node_a, node_b in graph.edges:
        assert node_b in connected_objects
        assert node_a is chemical_system


def test_gufe_to_keyed_dicts(network_tyk2):
    with RegistryBackup(gufe_object=network_tyk2) as original:
        keyed_dicts = gufe_to_keyed_dicts(network_tyk2)

        assert network_tyk2.to_keyed_dict() == keyed_dicts[-1]

        elems = []
        for dct in keyed_dicts:
            elems.append(GufeTokenizable.from_keyed_dict(dct))

        assert network_tyk2 == elems[-1]


def test_keyed_dicts_to_gufe(network_tyk2):
    with RegistryBackup(network_tyk2) as original:
        keyed_dicts = gufe_to_keyed_dicts(network_tyk2)
        assert keyed_dicts_to_gufe(keyed_dicts) == network_tyk2
