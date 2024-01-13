from alchemiscale.keyedchain import KeyedChain
from alchemiscale.utils import RegistryBackup

from gufe.tokenization import get_all_gufe_objs, GufeTokenizable, is_gufe_key_dict


def test_keyedchain_full_network(network):
    objects = get_all_gufe_objs(network)

    for o in objects:
        with RegistryBackup():
            kc = KeyedChain.from_gufe(o)
            _o = kc.to_gufe()
            assert o == _o


def test_keyedchain_len(network):
    objects = get_all_gufe_objs(network)
    expect_len = len(objects)

    keyedchain = KeyedChain.from_gufe(network)

    assert len(keyedchain) == expect_len


def test_keyedchain_get_keys(network):
    keyedchain = KeyedChain.from_gufe(network)
    keys = list(map(lambda x: x.key, get_all_gufe_objs(network)))

    for key in keyedchain.gufe_keys():
        assert key in keys


def test_keyedchain_get_keyed_dicts(network):
    keyedchain = KeyedChain.from_gufe(network)

    for keyed_dict in keyedchain.keyed_dicts():
        assert isinstance(keyed_dict, dict)


def test_keyedchain_iteration(network):
    keyedchain = KeyedChain.from_gufe(network)

    for key, keyed_dict in keyedchain:
        gt = GufeTokenizable.from_keyed_dict(keyed_dict)
        assert gt.key == key
