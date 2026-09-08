"""Integration tests for the v0.8.0 per-unit-result artifact methods on
``S3ObjectStore`` (log/stdout/stderr push and pull), exercised against the
moto-mocked ``s3os`` fixture.
"""

import os

import pytest

from alchemiscale.storage.objectstore import (
    S3ObjectStore,
    LOGS_FILENAME,
    STDOUT_DIRNAME,
    STDERR_DIRNAME,
)


class TestS3ObjectStoreArtifacts:

    UNIT_LOCATION = "protocoldagresult/o/c/p/T/results/PDR/units/PUR"

    def test_push_pull_logs_roundtrip(self, s3os: S3ObjectStore):
        logtext = "[2026-07-10 00:00:00] [gufekey-x] [INFO] line one\nline two\n"

        location = s3os.push_protocol_unit_result_logs(self.UNIT_LOCATION, logtext)
        assert location == os.path.join(self.UNIT_LOCATION, LOGS_FILENAME)

        # the object exists under the store prefix
        objs = list(s3os.resource.Bucket(s3os.bucket).objects.all())
        assert any(o.key == os.path.join(s3os.prefix, location) for o in objs)

        # round-trip decompresses to the original text
        assert s3os.pull_protocol_unit_result_logs(self.UNIT_LOCATION) == logtext

    def test_push_pull_stdout_roundtrip(self, s3os: S3ObjectStore):
        files = {
            "out.txt": b"hello stdout\n",
            "sub.log": b"more stdout bytes",
        }
        locations = s3os.push_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDOUT_DIRNAME, files
        )
        assert len(locations) == 2
        for name in files:
            assert (
                os.path.join(self.UNIT_LOCATION, STDOUT_DIRNAME, f"{name}.zst")
                in locations
            )

        pulled = s3os.pull_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDOUT_DIRNAME
        )
        assert pulled == {
            "out.txt": "hello stdout\n",
            "sub.log": "more stdout bytes",
        }

    def test_push_pull_stderr_roundtrip(self, s3os: S3ObjectStore):
        files = {"err.txt": b"a traceback here"}
        s3os.push_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDERR_DIRNAME, files
        )
        pulled = s3os.pull_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDERR_DIRNAME
        )
        assert pulled == {"err.txt": "a traceback here"}

    def test_streams_decode_errors_replace(self, s3os: S3ObjectStore):
        # invalid UTF-8 bytes are decoded with errors="replace", not raised
        files = {"binary.dat": b"\xff\xfe valid tail"}
        s3os.push_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDOUT_DIRNAME, files
        )
        pulled = s3os.pull_protocol_unit_result_streams(
            self.UNIT_LOCATION, STDOUT_DIRNAME
        )
        assert "valid tail" in pulled["binary.dat"]
        # replacement character present for the invalid bytes
        assert "�" in pulled["binary.dat"]

    def test_pull_streams_empty_when_absent(self, s3os: S3ObjectStore):
        # a location with no stream artifacts yields an empty mapping
        assert (
            s3os.pull_protocol_unit_result_streams(
                "protocoldagresult/o/c/p/T/results/PDR/units/NOPE", STDOUT_DIRNAME
            )
            == {}
        )

    @pytest.mark.parametrize("method", ["push", "pull"])
    def test_invalid_stream_name_raises(self, s3os: S3ObjectStore, method):
        with pytest.raises(ValueError, match="stream"):
            if method == "push":
                s3os.push_protocol_unit_result_streams(
                    self.UNIT_LOCATION, "notastream", {"x": b"y"}
                )
            else:
                s3os.pull_protocol_unit_result_streams(self.UNIT_LOCATION, "notastream")
