import pytest

from pathlib import Path

from fah_alchemy.compute.service import SynchronousComputeService


class TestSynchronousComputeService:
    ...

    
    @pytest.fixture
    def service(self, n4js_clear, compute_client, tmpdir):
        with tmpdir.as_cwd():
            return SynchronousComputeService(
                    compute_api_uri=compute_client.compute_api_url,
                    compute_api_key=compute_client.compute_api_key,
                    name='test_compute_service',
                    shared_path=Path('.').absolute())


    def test_get_task(self, service):

        service.get_tasks()
        ...
