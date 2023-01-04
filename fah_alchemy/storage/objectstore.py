import os
import io
import json
from boto3.session import Session
from functools import lru_cache

from gufe.protocols import ProtocolDAGResult
from gufe.tokenization import JSON_HANDLER, GufeTokenizable

from ..models import ScopedKey, Scope
from .models import ObjectStoreRef
from ..settings import S3ObjectStoreSettings, get_s3objectstore_settings


@lru_cache()
def get_s3os(settings: S3ObjectStoreSettings, endpoint_url=None):
    """Convenience function for getting an S3ObjectStore directly from settings."""

    # create a boto3 Session and parameterize with keys
    session = Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        aws_session_token=settings.AWS_SESSION_TOKEN,
        region_name=settings.AWS_DEFAULT_REGION,
    )

    return S3ObjectStore(
        session=session,
        bucket=settings.AWS_S3_BUCKET,
        prefix=settings.AWS_S3_PREFIX,
        endpoint_url=endpoint_url,
    )


class S3ObjectStoreError(Exception):
    ...


class S3ObjectStore:
    """Object storage for use with AWS S3."""

    def __init__(
        self, session: "boto3.Session", bucket: str, prefix: str, endpoint_url=None
    ):
        """ """
        self.session = session
        self.resource = self.session.resource("s3", endpoint_url=endpoint_url)

        self.bucket = bucket
        self.prefix = prefix

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

        print(f"Key: {key}")

        print("Download")
        b = self.resource.Bucket(self.bucket)
        print(f"Bucket name: {self.bucket}")
        print(f"Bucket content: {list(b.objects.all())}")

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

        print("Upload")
        b = self.resource.Bucket(self.bucket)
        print(f"Bucket name: {self.bucket}")
        print(f"Bucket content: {list(b.objects.all())}")

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
            self.resouce.Object(self.bucket, key).delete()
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
        self, protocoldagresult: ProtocolDAGResult, scope: Scope
    ):
        """Push given `ProtocolDAGResult` to this `ObjectStore`.

        Parameters
        ----------
        protocoldagresult
            ProtocolDAGResult to store.
        scope
            Scope to store ProtocolDAGResult under.

        Returns
        -------
        ObjectStoreRef
            Reference to the serialized `ProtocolDAGResult` in the object store.

        """

        # build `location` based on gufe key
        location = os.path.join(
            "protocoldagresult", *scope.to_tuple(), protocoldagresult.key, "obj.json"
        )

        # TODO: add support for compute client-side compressed protocoldagresults
        pdr_jb = json.dumps(
            protocoldagresult.to_dict(), cls=JSON_HANDLER.encoder
        ).encode("utf-8")
        response = self._store_bytes(location, pdr_jb)

        return ObjectStoreRef(
            location=location, obj_key=protocoldagresult.key, scope=scope
        )

    def pull_protocoldagresult(self, protocoldagresult: ScopedKey, return_as="gufe"):
        """Pull the `ProtocolDAGResult` corresponding to the given `ObjectStoreRef`.

        Parameters
        ----------
        protocoldagresult
            ScopedKey for ProtocolDAGResult in the object store.
        return_as : ['gufe', 'dict', 'json']
            Form in which to return result; this is provided to avoid
            unnecessary deserializations where desired.

        Returns
        -------
        ProtocolDAGResult
            The ProtocolDAGResult corresponding to the given `ObjectStoreRef`.

        """
        # build `location` based on gufe key
        location = os.path.join(
            "protocoldagresult",
            *protocoldagresult.scope.to_tuple(),
            protocoldagresult.gufe_key,
            "obj.json",
        )

        pdr_j = self._get_bytes(location).decode("utf-8")

        # TODO: add support for interface client-side decompression
        if return_as == "gufe":
            pdr = GufeTokenizable.from_dict(json.loads(pdr_j, cls=JSON_HANDLER.decoder))
        elif return_as == "dict":
            pdr = json.loads(pdr_j, cls=JSON_HANDLER.decoder)
        elif return_as == "json":
            pdr = pdr_j

        return pdr
