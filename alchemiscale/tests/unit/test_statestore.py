"""Unit tests for :mod:`alchemiscale.storage.statestore` subgraph construction."""

import datetime
import json

import numpy as np
import pytest
from gufe.tokenization import JSON_HANDLER, KeyedChain
from openff.units import unit

from alchemiscale.models import Scope
from alchemiscale.storage.statestore import Neo4jStore

# Property types neo4j stores natively, and which `_keyed_chain_to_subgraph`
# must therefore leave untouched. This set is closed: it is fixed by the Cypher
# type system, not by anything in `gufe`. Storing any of these as JSON instead
# would silently change their representation in the graph -- and would *not* be
# caught by a round-trip test, since `JSON_HANDLER` decodes them back to equal
# Python objects.
NEO4J_NATIVE_PROPERTY_TYPES = {
    "none": None,
    "bool": True,
    "int": 42,
    "float": 1.5,
    "str": "spam",
    "bytes": b"eggs",
    "bytearray": bytearray(b"ham"),
    "date": datetime.date(2026, 8, 27),
    "time": datetime.time(13, 30),
    "datetime": datetime.datetime(2026, 8, 27, 13, 30, tzinfo=datetime.UTC),
    "timedelta": datetime.timedelta(hours=3),
}

# Types neo4j cannot store as properties, which must be JSON-serialized on the
# way in and recorded in `_json_props` so they are decoded on the way out
NON_NATIVE_PROPERTY_TYPES = {
    "quantity": 5.0 * unit.nanometer,
    "quantity_array": np.eye(3) * unit.nanometer,
    "ndarray": np.arange(4),
    "np_int": np.int64(7),
}


def keyed_chain_from_props(props):
    """Build a single-node `KeyedChain` carrying `props` as its attributes."""
    keyed_dict = {
        "__qualname__": "DummyTokenizable",
        "__module__": "alchemiscale.tests.unit.test_statestore",
        ":version:": 1,
        **props,
    }
    return KeyedChain([("DummyTokenizable-abc123", keyed_dict)])


class TestKeyedChainToSubgraph:
    @pytest.fixture
    def n4js(self):
        # `_keyed_chain_to_subgraph` is pure; it needs no driver connection
        return Neo4jStore.__new__(Neo4jStore)

    @pytest.fixture
    def scope(self):
        return Scope("test_org", "test_campaign", "test_project")

    @pytest.mark.parametrize("key, value", sorted(NEO4J_NATIVE_PROPERTY_TYPES.items()))
    def test_native_types_stored_natively(self, n4js, scope, key, value):
        """Types neo4j supports must be passed through untouched."""
        _, node, _ = n4js._keyed_chain_to_subgraph(
            keyed_chain_from_props({key: value}), scope
        )

        assert key not in node["_json_props"]

        if value is None:
            # `Node` drops `None`-valued properties entirely
            assert node[key] is None
        else:
            assert node[key] is value

    @pytest.mark.parametrize("key, value", sorted(NON_NATIVE_PROPERTY_TYPES.items()))
    def test_non_native_types_json_serialized(self, n4js, scope, key, value):
        """Everything else must be JSON-serialized and marked for decoding."""
        _, node, _ = n4js._keyed_chain_to_subgraph(
            keyed_chain_from_props({key: value}), scope
        )

        assert key in node["_json_props"]
        assert isinstance(node[key], str)

        # and it must decode back to the original value
        decoded = json.loads(node[key], cls=JSON_HANDLER.decoder)
        assert np.all(decoded == value)

    def test_mixed_props(self, n4js, scope):
        """Native and non-native attributes coexist on a single node."""
        props = {**NEO4J_NATIVE_PROPERTY_TYPES, **NON_NATIVE_PROPERTY_TYPES}
        _, node, _ = n4js._keyed_chain_to_subgraph(keyed_chain_from_props(props), scope)

        assert set(node["_json_props"]) == set(NON_NATIVE_PROPERTY_TYPES)
