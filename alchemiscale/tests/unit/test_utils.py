from alchemiscale.utils import (
    gufe_objects_from_shallow_dict,
    gufe_to_digraph,
    RegistryBackup,
)
from gufe.tokenization import get_all_gufe_objs, TOKENIZABLE_REGISTRY
import pytest


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


@pytest.mark.parametrize(
    "keep_changes",
    [
        (True),
        (False),
    ],
)
def test_registry_keep_changes(
    keep_changes, network, chemicalsystem_lig_emj_50_complex
):
    with RegistryBackup(keep_changes=keep_changes):
        an = network.copy_with_replacements(name="new_network")

    if keep_changes:
        assert an in TOKENIZABLE_REGISTRY.values()
    else:
        assert an not in TOKENIZABLE_REGISTRY.values()
