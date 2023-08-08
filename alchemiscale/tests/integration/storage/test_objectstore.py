from pathlib import Path
import os

import pytest

from alchemiscale.models import ScopedKey
from alchemiscale.storage.objectstore import S3ObjectStore
from alchemiscale.storage.models import ProtocolDAGResultRef


class TestS3ObjectStore:
    def test_delete(self, s3os: S3ObjectStore):
        # write check
        s3os._store_bytes("_check_test", b"test_check")
        s3os._delete("_check_test")

    def test_push_protocolresult(
        self, s3os: S3ObjectStore, protocoldagresults, transformation, scope_test
    ):
        transformation_sk = ScopedKey(gufe_key=transformation.key, **scope_test.dict())

        # try to push the result
        objstoreref: ProtocolDAGResultRef = s3os.push_protocoldagresult(
            protocoldagresults[0], transformation=transformation_sk
        )

        assert objstoreref.obj_key == protocoldagresults[0].key

        # examine object metadata
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())

        assert len(objs) == 1
        assert objs[0].key == os.path.join(s3os.prefix, objstoreref.location)

    def test_pull_protocolresult(
        self, s3os: S3ObjectStore, protocoldagresults, transformation, scope_test
    ):
        transformation_sk = ScopedKey(gufe_key=transformation.key, **scope_test.dict())

        objstoreref: ProtocolDAGResultRef = s3os.push_protocoldagresult(
            protocoldagresults[0], transformation=transformation_sk
        )

        # round trip it
        sk = ScopedKey(gufe_key=objstoreref.obj_key, **scope_test.dict())
        tf_sk = ScopedKey(
            gufe_key=protocoldagresults[0].transformation_key, **scope_test.dict()
        )
        pdr = s3os.pull_protocoldagresult(sk, tf_sk)

        assert pdr.key == protocoldagresults[0].key
        assert pdr.protocol_unit_results == pdr.protocol_unit_results

        # test location-based pull
        pdr = s3os.pull_protocoldagresult(location=objstoreref.location)

        assert pdr.key == protocoldagresults[0].key
        assert pdr.protocol_unit_results == pdr.protocol_unit_results
