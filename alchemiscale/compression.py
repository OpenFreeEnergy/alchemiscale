from gufe.tokenization import GufeTokenizable, JSON_HANDLER
import json
import zstandard as zstd


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
    keyed_chain_rep = gufe_object.to_keyed_chain()
    json_rep = json.dumps(keyed_chain_rep, cls=JSON_HANDLER.encoder)
    json_bytes = json_rep.encode("utf-8")

    compressor = zstd.ZstdCompressor()
    compressed_gufe = compressor.compress(json_bytes)

    return compressed_gufe


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
    decompressor = zstd.ZstdDecompressor()
    decompressed_gufe: bytes = decompressor.decompress(compressed_bytes)

    keyed_chain_rep = json.loads(
        decompressed_gufe.decode("utf-8"), cls=JSON_HANDLER.decoder
    )
    gufe_object = GufeTokenizable.from_keyed_chain(keyed_chain_rep)
    return gufe_object
