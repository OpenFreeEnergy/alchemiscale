from pathlib import Path
import os

import pytest
from gufe.protocols.protocoldag import execute_DAG

from alchemiscale.storage import S3ObjectStore
from alchemiscale.storage.models import ObjectStoreRef


class TestS3ObjectStore:
    def test_delete(self, s3os: S3ObjectStore):
        # write check
        s3os._store_bytes("_check_test", b"test_check")
        s3os._delete("_check_test")

    def test_push_protocolresult(self, s3os: S3ObjectStore, protocoldagresult):

        # try to push the result
        objstoreref: ObjectStoreRef = s3os.push_protocoldagresult(protocoldagresult)

        assert objstoreref.location == os.path.join(
            "protocoldagresult", protocoldagresult.key
        )

        # examine object metadata
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())

        assert len(objs) == 1
        assert objs[0].key == os.path.join(s3os.prefix, objstoreref.location)

    def test_pull_protocolresult(self, s3os: S3ObjectStore, protocoldagresult):

        objstoreref: ObjectStoreRef = s3os.push_protocoldagresult(protocoldagresult)

        # round trip it
        pdr = s3os.pull_protocoldagresult(objstoreref)

        assert pdr.key == protocoldagresult.key
        assert pdr.protocol_unit_results == pdr.protocol_unit_results
