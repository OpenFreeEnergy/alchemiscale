from pathlib import Path
import os

import pytest
from gufe.protocols.protocoldag import execute_DAG

from fah_alchemy.storage import S3ObjectStore
from fah_alchemy.storage.models import ObjectStoreRef


class TestS3ObjectStore:
    def test_push_protocolresult(self, s3os: S3ObjectStore, network_tyk2, tmpdir):
        an = network_tyk2

        transformation = list(an.edges)[0]
        protocoldag = transformation.protocol.create(
            stateA=transformation.stateA,
            stateB=transformation.stateB,
            mapping=transformation.mapping,
            name="silly rabbit",
        )

        # execute the task
        with tmpdir.as_cwd():
            protocoldagresult = execute_DAG(protocoldag, shared=Path(".").absolute())

        # try to push the result
        objstoreref: ObjectStoreRef = s3os.push_protocoldagresult(protocoldagresult)

        assert objstoreref.location == os.path.join(
            "protocoldagresult", protocoldagresult.key
        )

        # examine object metadata
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())

        assert len(objs) == 1
        assert objs[0].key == os.path.join(s3os.prefix, objstoreref.location)

    def test_pull_protocolresult(self, s3os: S3ObjectStore, network_tyk2, tmpdir):
        an = network_tyk2

        transformation = list(an.edges)[0]
        protocoldag = transformation.protocol.create(
            stateA=transformation.stateA,
            stateB=transformation.stateB,
            mapping=transformation.mapping,
            name="silly rabbit",
        )

        # execute the task
        with tmpdir.as_cwd():
            protocoldagresult = execute_DAG(protocoldag, shared=Path(".").absolute())

        objstoreref: ObjectStoreRef = s3os.push_protocoldagresult(protocoldagresult)

        # round trip it
        pdr = s3os.pull_protocoldagresult(objstoreref)

        assert pdr.key == protocoldagresult.key
        assert pdr.protocol_unit_results == pdr.protocol_unit_results
