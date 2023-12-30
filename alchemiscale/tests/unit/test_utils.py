from openfe_benchmarks import tyk2
from alchemiscale.utils import (
    gufe_objects_from_shallow_dict,
    gufe_to_digraph,
    gufe_to_keyed_dicts,
    keyed_dicts_to_gufe,
    RegistryBackup,
)
from gufe.tokenization import GufeTokenizable, get_all_gufe_objs, TOKENIZABLE_REGISTRY


def test_keyed_dicts_full_network(network):
    objects = get_all_gufe_objs(network)

    for o in objects:
        with RegistryBackup():
            keyed_dicts = gufe_to_keyed_dicts(o)
            _o = keyed_dicts_to_gufe(keyed_dicts)
            assert o == _o


def test_gufe_objects_from_shallow_dict(chemicalsystem_lig_emj_50_complex):
    cs = chemicalsystem_lig_emj_50_complex
    shallow_dict = cs.to_shallow_dict()
    gufe_objects = gufe_objects_from_shallow_dict(shallow_dict)

    for gufe_object in gufe_objects:
        assert gufe_object in cs.components.values()


def test_gufe_to_digraph(chemicalsystem_lig_emj_50_complex):
    cs = chemicalsystem_lig_emj_50_complex
    graph = gufe_to_digraph(cs)

    connected_objects = gufe_objects_from_shallow_dict(cs.to_shallow_dict())

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3

    for node_a, node_b in graph.edges:
        assert node_b in connected_objects
        assert node_a is cs


def test_gufe_to_keyed_dicts(network):
    with RegistryBackup(gufe_object=network):
        keyed_dicts = gufe_to_keyed_dicts(network)

        assert network.to_keyed_dict() == keyed_dicts[-1]

        elems = []
        for dct in keyed_dicts:
            elems.append(GufeTokenizable.from_keyed_dict(dct))

        assert network == elems[-1]


def test_keyed_dicts_to_gufe(network):
    with RegistryBackup(network):
        keyed_dicts = gufe_to_keyed_dicts(network)
        assert keyed_dicts_to_gufe(keyed_dicts) == network


def test_registry_backup_enter_return(network):
    direct_copy = TOKENIZABLE_REGISTRY.copy()

    with RegistryBackup() as original:
        assert direct_copy == original
        assert len(direct_copy) != 0


def test_registry_backup_full_clear(network):
    original_length = len(TOKENIZABLE_REGISTRY)

    assert original_length != 0

    with RegistryBackup():
        assert len(TOKENIZABLE_REGISTRY) == 0

    assert len(TOKENIZABLE_REGISTRY) != 0


def test_registry_backup_partial_clear(network, chemicalsystem_lig_emj_50_complex):
    original_len = len(TOKENIZABLE_REGISTRY)
    removed_objects = get_all_gufe_objs(chemicalsystem_lig_emj_50_complex)

    expected_len = original_len - len(removed_objects)

    with RegistryBackup(gufe_object=chemicalsystem_lig_emj_50_complex):
        assert len(TOKENIZABLE_REGISTRY) == expected_len != 0


def test_registry_keep_changes(network, chemicalsystem_lig_emj_50_complex):
    pass
