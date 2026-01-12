"""Unit tests for AsynchronousComputeService components."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from alchemiscale.compute.service import ResourceMetrics, ResourceMonitor, TaskExecution
from alchemiscale.models import ScopedKey


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""

    def test_init_without_gpu(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0)
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.gpu_percent is None
        assert isinstance(metrics.timestamp, datetime)

    def test_init_with_gpu(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0, gpu_percent=70.0)
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.gpu_percent == 70.0

    def test_is_oversaturated_cpu(self):
        metrics = ResourceMetrics(cpu_percent=95.0, memory_percent=60.0)
        assert metrics.is_oversaturated(90.0, 85.0, 90.0) is True

    def test_is_oversaturated_memory(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=90.0)
        assert metrics.is_oversaturated(90.0, 85.0, 90.0) is True

    def test_is_oversaturated_gpu(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0, gpu_percent=95.0)
        assert metrics.is_oversaturated(90.0, 85.0, 90.0) is True

    def test_is_not_oversaturated(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0, gpu_percent=70.0)
        assert metrics.is_oversaturated(90.0, 85.0, 90.0) is False

    def test_is_underutilized_all_low(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=50.0, gpu_percent=50.0)
        # Default margin is 20.0, so thresholds are 90-20=70, 85-20=65, 90-20=70
        assert metrics.is_underutilized(90.0, 85.0, 90.0) is True

    def test_is_underutilized_cpu_high(self):
        metrics = ResourceMetrics(cpu_percent=75.0, memory_percent=50.0, gpu_percent=50.0)
        # CPU is above 90-20=70, so not underutilized
        assert metrics.is_underutilized(90.0, 85.0, 90.0) is False

    def test_is_underutilized_memory_high(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=70.0, gpu_percent=50.0)
        # Memory is above 85-20=65, so not underutilized
        assert metrics.is_underutilized(90.0, 85.0, 90.0) is False

    def test_is_underutilized_gpu_high(self):
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=50.0, gpu_percent=75.0)
        # GPU is above 90-20=70, so not underutilized
        assert metrics.is_underutilized(90.0, 85.0, 90.0) is False

    def test_is_underutilized_custom_margin(self):
        metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=50.0)
        # With margin=10, threshold is 90-10=80, so CPU at 85 is not underutilized
        assert metrics.is_underutilized(90.0, 85.0, 90.0, margin=10.0) is False
        # With margin=30, threshold is 90-30=60, so CPU at 85 is not underutilized
        assert metrics.is_underutilized(90.0, 85.0, 90.0, margin=30.0) is False


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    def test_init_without_gpu(self):
        monitor = ResourceMonitor(enable_gpu=False)
        assert monitor.enable_gpu is False
        assert monitor._gpu_available is False

    @patch("alchemiscale.compute.service.psutil")
    def test_get_metrics_without_gpu(self, mock_psutil):
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 67.3
        mock_psutil.virtual_memory.return_value = mock_memory

        monitor = ResourceMonitor(enable_gpu=False)
        metrics = monitor.get_metrics()

        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.3
        assert metrics.gpu_percent is None
        mock_psutil.cpu_percent.assert_called_once_with(interval=0.1)
        mock_psutil.virtual_memory.assert_called_once()

    @patch("alchemiscale.compute.service.psutil")
    def test_get_metrics_with_gpu_unavailable(self, mock_psutil):
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 67.3
        mock_psutil.virtual_memory.return_value = mock_memory

        # Even if enable_gpu=True, if pynvml is not available, it should work without GPU
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = ResourceMonitor(enable_gpu=True)
            assert monitor._gpu_available is False

            metrics = monitor.get_metrics()
            assert metrics.cpu_percent == 45.5
            assert metrics.memory_percent == 67.3
            assert metrics.gpu_percent is None

    @patch("alchemiscale.compute.service.psutil")
    def test_get_metrics_with_gpu_available(self, mock_psutil):
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 67.3
        mock_psutil.virtual_memory.return_value = mock_memory

        # Mock pynvml
        mock_pynvml = MagicMock()
        mock_handle = Mock()
        mock_utilization = Mock()
        mock_utilization.gpu = 85.0
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor = ResourceMonitor(enable_gpu=True)
            monitor._gpu_available = True
            monitor._pynvml = mock_pynvml

            metrics = monitor.get_metrics()
            assert metrics.cpu_percent == 45.5
            assert metrics.memory_percent == 67.3
            assert metrics.gpu_percent == 85.0

    def test_cleanup_without_gpu(self):
        monitor = ResourceMonitor(enable_gpu=False)
        # Should not raise any errors
        monitor.cleanup()

    def test_cleanup_with_gpu(self):
        mock_pynvml = MagicMock()
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor = ResourceMonitor(enable_gpu=True)
            monitor._gpu_available = True
            monitor._pynvml = mock_pynvml

            monitor.cleanup()
            mock_pynvml.nvmlShutdown.assert_called_once()


class TestTaskExecution:
    """Test TaskExecution dataclass."""

    def test_init_basic(self):
        task = ScopedKey(gufe_key="test-key", scope=Mock())
        started = datetime.now()
        task_exec = TaskExecution(task=task, started_at=started)

        assert task_exec.task == task
        assert task_exec.started_at == started
        assert task_exec.process is None
        assert task_exec.terminated is False
        assert task_exec.retry_after is None

    def test_init_with_process(self):
        task = ScopedKey(gufe_key="test-key", scope=Mock())
        started = datetime.now()
        mock_process = Mock()
        task_exec = TaskExecution(task=task, started_at=started, process=mock_process)

        assert task_exec.task == task
        assert task_exec.started_at == started
        assert task_exec.process == mock_process
        assert task_exec.terminated is False

    def test_termination_state(self):
        task = ScopedKey(gufe_key="test-key", scope=Mock())
        started = datetime.now()
        retry_time = started + timedelta(seconds=300)
        task_exec = TaskExecution(
            task=task, started_at=started, terminated=True, retry_after=retry_time
        )

        assert task_exec.terminated is True
        assert task_exec.retry_after == retry_time
