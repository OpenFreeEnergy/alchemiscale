"""
:mod:`alchemiscale.storage.objectstore` --- object store interface
==================================================================

"""

import os
import datetime
from boto3.session import Session
from functools import lru_cache

import zstandard as zstd

from gufe.tokenization import GufeKey

from ..models import ScopedKey
from .models import ProtocolDAGResultRef
from ..settings import S3ObjectStoreSettings

# default filename for object store files
OBJECT_FILENAME = "obj.json.zst"

# per-unit-result artifact layout, relative to a ProtocolUnitResultRef location
# (``.../{pdr_key}/units/{unit_result_key}``)
LOGS_FILENAME = "logs.txt.zst"
STDOUT_DIRNAME = "stdout"
STDERR_DIRNAME = "stderr"


def get_s3os(settings: S3ObjectStoreSettings) -> "S3ObjectStore":
    """Convenience function for getting an S3ObjectStore directly from settings."""
    return S3ObjectStore(settings)


class S3ObjectStoreError(Exception): ...


class S3ObjectStore:
    """Object storage for use with AWS S3."""

    def __init__(self, settings: S3ObjectStoreSettings):
        """Initialize S3ObjectStore from settings.

        Parameters
        ----------
        settings : S3ObjectStoreSettings
            Configuration settings for S3 object store.
        """
        self.settings = settings

        # Create a boto3 Session from settings
        self.session = Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            aws_session_token=settings.AWS_SESSION_TOKEN,
            region_name=settings.AWS_DEFAULT_REGION,
        )
        self.resource = self.session.resource(
            "s3", endpoint_url=settings.AWS_ENDPOINT_URL
        )

        self.bucket = settings.AWS_S3_BUCKET
        self.prefix = settings.AWS_S3_PREFIX

    def initialize(self):
        """Initialize object store.

        Creates bucket if it does not exist.

        """
        bucket = self.resource.Bucket(self.bucket)
        bucket.create()
        bucket.wait_until_exists()

    def check(self):
        """Check consistency of object store."""
        raise NotImplementedError

    def _store_check(self):
        """Check that the ObjectStore is in a state that can be used by the API."""
        try:
            # read check
            self.resource.meta.client.list_buckets()

            # write check
            self._store_bytes("_check_test", b"test_check")
            self._delete("_check_test")
        except Exception:
            return False
        return True

    def reset(self):
        """Remove all data from object store.

        Deletes all objects, including the bucket itself.

        """
        bucket = self.resource.Bucket(self.bucket)

        # delete all objects, then the bucket
        bucket.objects.delete()
        bucket.delete()
        bucket.wait_until_not_exists()

    def iter_contents(self, prefix=""):
        """Iterate over the labels in this storage.

        Parameters
        ----------
        prefix : str
            Only iterate over paths that start with the given prefix.

        Returns
        -------
        Iterator[str] :
            Contents of this storage, which may include items without
            metadata.
        """

        filter_prefix = os.path.join(self.prefix, prefix)

        return self.resource.Bucket(self.bucket).objects.filter(Prefix=filter_prefix)

    def _store_bytes(self, location, byte_data):
        """
        For implementers: This should be blocking, even if the storage
        backend allows asynchronous storage.
        """
        key = os.path.join(self.prefix, location)

        response = self.resource.Object(self.bucket, key).put(Body=byte_data)

        if not response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            raise S3ObjectStoreError(f"Could not store given object at key {key}")

        return response

    def _get_bytes(self, location):
        key = os.path.join(self.prefix, location)

        _ = self.resource.Bucket(self.bucket)

        return self.resource.Object(self.bucket, key).get()["Body"].read()

    def _store_path(self, location, path):
        """
        For implementers: This should be blocking, even if the storage
        backend allows asynchronous storage.
        """
        """
        For implementers: This should be blocking, even if the storage
        backend allows asynchronous storage.
        """
        key = os.path.join(self.prefix, location)

        with open(path, "rb") as f:
            self.resource.Bucket(self.bucket).upload_fileobj(f, key)

        _ = self.resource.Bucket(self.bucket)

    def _exists(self, location) -> bool:
        from botocore.exceptions import ClientError

        key = os.path.join(self.prefix, location)

        # we do a metadata load as our existence check
        # appears to be most recommended approach
        try:
            self.resource.Object(self.bucket, key).load()
            return True
        except ClientError:
            return False

    def _delete(self, location):
        key = os.path.join(self.prefix, location)

        if self._exists(location):
            self.resource.Object(self.bucket, key).delete()
        else:
            raise S3ObjectStoreError(
                f"Unable to delete '{str(key)}': Object does not exist"
            )

    def _get_filename(self, location):
        key = os.path.join(self.prefix, location)

        object = self.bucket.Object(key)

        url = object.meta.client.generate_presigned_url(
            "get_object",
            ExpiresIn=0,
            Params={"Bucket": self.bucket.name, "Key": object.key},
        )

        # drop query params from url
        url = url.split("?")[0]

        return url

    def push_protocoldagresult(
        self,
        protocoldagresult: bytes,
        protocoldagresult_ok: bool,
        protocoldagresult_gufekey: GufeKey,
        transformation: ScopedKey,
        creator: str | None = None,
    ) -> ProtocolDAGResultRef:
        """Push given `ProtocolDAGResult` to this `ObjectStore`.

        Parameters
        ----------
        protocoldagresult
            ProtocolDAGResult to store, in some bytes representation.
        protocoldagresult_ok
            ``True`` if ProtocolDAGResult completed successfully; ``False`` if failed.
        protocoldagresult_gufekey
            The GufeKey of the ProtocolDAGResult.
        transformation
            The ScopedKey of the Transformation this ProtocolDAGResult
            corresponds to.

        Returns
        -------
        ProtocolDAGResultRef
            Reference to the serialized `ProtocolDAGResult` in the object store.

        """

        ok = protocoldagresult_ok
        route = "results" if ok else "failures"

        # build `location` based on gufe key
        location = os.path.join(
            "protocoldagresult",
            *transformation.scope.to_tuple(),
            transformation.gufe_key,
            route,
            protocoldagresult_gufekey,
            OBJECT_FILENAME,
        )

        self._store_bytes(location, protocoldagresult)

        return ProtocolDAGResultRef(
            location=location,
            obj_key=protocoldagresult_gufekey,
            scope=transformation.scope,
            ok=ok,
            datetime_created=datetime.datetime.now(tz=datetime.UTC),
            creator=creator,
        )

    def pull_protocoldagresult(
        self,
        protocoldagresult: ScopedKey | None = None,
        transformation: ScopedKey | None = None,
        location: str | None = None,
        ok=True,
    ) -> bytes:
        """Pull the `ProtocolDAGResult` corresponding to the given `ProtocolDAGResultRef`.

        Parameters
        ----------
        protocoldagresult
            ScopedKey for ProtocolDAGResult in the object store.
            Must be provided if `location` is ``None``.
        transformation
            The ScopedKey of the Transformation this ProtocolDAGResult
            corresponds to.
            Must be provided if `location` is ``None``.
        location
            The full path in the object store to the ProtocolDAGResult. If
            provided, this will be used to retrieve it.

        Returns
        -------
        ProtocolDAGResult
            The ProtocolDAGResult corresponding to the given `ProtocolDAGResultRef`, in a bytes representation.

        """
        route = "results" if ok else "failures"

        # build `location` based on provided ScopedKey if not provided
        if location is None:
            if None in (transformation, protocoldagresult):
                raise ValueError(
                    "`transformation` and `protocoldagresult` must both be given if `location` is ``None``"
                )
            if transformation.scope != protocoldagresult.scope:
                raise ValueError(
                    f"transformation scope '{transformation.scope}' differs from protocoldagresult scope '{protocoldagresult.scope}'"
                )

            location = os.path.join(
                "protocoldagresult",
                *protocoldagresult.scope.to_tuple(),
                transformation.gufe_key,
                route,
                protocoldagresult.gufe_key,
                OBJECT_FILENAME,
            )

        ## TODO: want organization alongside `obj.json` of `ProtocolUnit` gufe_keys
        ## for any file objects stored in the same space
        pdr_bytes = self._get_bytes(location)

        return pdr_bytes

    # --- per-unit-result artifacts (logs, stdout, stderr) -----------------
    #
    # These live under the unit result's location prefix
    # (`ProtocolUnitResultRef.location`), a retrieval-optimized copy so that log
    # retrieval never requires pulling and deserializing the whole
    # `ProtocolDAGResult`. Bytes are zstd-compressed, consistent with
    # `alchemiscale.compression`.

    def push_protocol_unit_result_logs(self, unit_location: str, logtext: str) -> str:
        """Store captured log text for a unit result. Returns the artifact key.

        `logtext` is already truncated to the per-unit cap by the compute
        service; here it is simply zstd-compressed and stored.
        """
        location = os.path.join(unit_location, LOGS_FILENAME)
        compressed = zstd.ZstdCompressor().compress(logtext.encode("utf-8"))
        self._store_bytes(location, compressed)
        return location

    def pull_protocol_unit_result_logs(self, unit_location: str) -> str:
        """Return decompressed log text for a unit result."""
        location = os.path.join(unit_location, LOGS_FILENAME)
        compressed = self._get_bytes(location)
        return (
            zstd.ZstdDecompressor()
            .decompress(compressed)
            .decode("utf-8", errors="replace")
        )

    def push_protocol_unit_result_streams(
        self, unit_location: str, stream: str, files: dict[str, bytes]
    ) -> list[str]:
        """Store a unit result's captured stream files (stdout or stderr).

        `files` maps filename -> raw bytes (the `dict[filename, bytes]` gufe
        embeds on the `ProtocolUnitResult`). Each is zstd-compressed and stored
        under the stream subdirectory, keeping the protocol-given filename.
        """
        if stream not in (STDOUT_DIRNAME, STDERR_DIRNAME):
            raise ValueError("`stream` must be 'stdout' or 'stderr'")
        locations = []
        compressor = zstd.ZstdCompressor()
        for filename, data in files.items():
            location = os.path.join(unit_location, stream, f"{filename}.zst")
            self._store_bytes(location, compressor.compress(data))
            locations.append(location)
        return locations

    def pull_protocol_unit_result_streams(
        self, unit_location: str, stream: str
    ) -> dict[str, str]:
        """Return a unit result's captured stream files, filename -> decoded text.

        Bytes are decoded as UTF-8 with ``errors="replace"``; protocols
        overwhelmingly archive text (binary outputs are #180 `ResultFile`
        territory).
        """
        if stream not in (STDOUT_DIRNAME, STDERR_DIRNAME):
            raise ValueError("`stream` must be 'stdout' or 'stderr'")
        prefix = os.path.join(unit_location, stream) + "/"
        decompressor = zstd.ZstdDecompressor()
        out = {}
        for obj in self._get_filename_prefix_contents(prefix):
            # key includes self.prefix and the full location; recover the
            # filename relative to the stream directory, dropping the .zst suffix
            key = obj.key
            filename = key.rsplit("/", 1)[-1]
            if filename.endswith(".zst"):
                filename = filename[: -len(".zst")]
            data = obj.get()["Body"].read()
            out[filename] = decompressor.decompress(data).decode(
                "utf-8", errors="replace"
            )
        return out

    def _get_filename_prefix_contents(self, prefix: str):
        """Iterate S3 objects under a location prefix (excluding ``self.prefix``)."""
        filter_prefix = os.path.join(self.prefix, prefix)
        return self.resource.Bucket(self.bucket).objects.filter(Prefix=filter_prefix)
