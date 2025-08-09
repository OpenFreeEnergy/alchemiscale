from gufe.tokenization import GufeTokenizable, JSON_HANDLER
import json
import zstandard as zstd


def json_to_gufe(jsondata):
    return GufeTokenizable.from_dict(json.loads(jsondata, cls=JSON_HANDLER.decoder))


def compress_keyed_chain_zstd(keyed_chain: list[tuple[str, dict]]) -> bytes:
    """Compress a keyed chain using zstandard compression.

    Parameters
    ----------
    keyed_chain: list[tuple[str, dict]]
        The keyed chain to be compressed.

    Returns
    -------
    bytes
        The compressed byte form of the keyed chain.
    """
    stringified_json = json.dumps(keyed_chain, cls=JSON_HANDLER.encoder)
    uncompressed_bytes = stringified_json.encode("utf-8")

    compressor = zstd.ZstdCompressor()
    compressed_keyed_chain = compressor.compress(uncompressed_bytes)

    return compressed_keyed_chain


def decompress_keyed_chain_zstd(compressed_bytes: bytes) -> list[tuple[str, dict]]:
    """Decompress a zstandard compressed keyed chain.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed byte form of a keyed chain.

    Returns
    -------
    list[tuple[str, dict]]
        The keyed chain representation of a GufeTokenizable.
    """
    decompressor = zstd.ZstdDecompressor()
    keyed_chain_bytes: bytes = decompressor.decompress(compressed_bytes)

    keyed_chain = json.loads(
        keyed_chain_bytes.decode("utf-8"), cls=JSON_HANDLER.decoder
    )
    return keyed_chain


def compress_gufe_zstd(gufe_object: GufeTokenizable) -> bytes:
    """Compress a GufeTokenizable using zstandard compression.

    After the GufeTokenizable is converted to a KeyedChain, it's
    serialized into JSON using the gufe provided
    JSON_HANDLER.encoder. The resulting string is utf-8 encoded and
    compressed with the zstandard compressor. These bytes are returned
    by the function.

    Parameters
    ----------
    gufe_object: GufeTokenizable
        The GufeTokenizable to compress.

    Returns
    -------
    bytes
        Compressed byte form of the GufeTokenizable.
    """
    keyed_chain = gufe_object.to_keyed_chain()
    return compress_keyed_chain_zstd(keyed_chain)


def decompress_gufe_zstd(compressed_bytes: bytes) -> GufeTokenizable:
    """Decompress a zstandard compressed GufeTokenizable.

    The bytes encoding a zstandard compressed GufeTokenizable are
    decompressed and decoded using the gufe provided
    JSON_HANDLER.decoder. It is assumed that the decompressed bytes
    are utf-8 encoded.

    This is the inverse operation of `compress_gufe_zstd`.

    Parameters
    ----------
    compressed_bytes: bytes
        The compressed byte form of a GufeTokenizable.

    Returns
    -------
    GufeTokenizable
        The decompressed GufeTokenizable.
    """

    keyed_chain = decompress_keyed_chain_zstd(compressed_bytes)
    gufe_object = GufeTokenizable.from_keyed_chain(keyed_chain)
    return gufe_object
