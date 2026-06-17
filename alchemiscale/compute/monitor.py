"""
:mod:`alchemiscale.compute.monitor` --- resource monitoring for compute services
================================================================================

"""

from abc import abstractmethod
from enum import auto, IntEnum
import os
import subprocess
import time
from threading import Lock

from alchemiscale.compute.settings import AsynchronousComputeServiceSettings


# Signal to be issued by a resource manager to a compute
# service. These signals are suggestions and are meant to inform the
# service but will not force a particular behavior.
#
#    1. TERMINATE: tool/resource failure, shut stop all calculations
#    2. SHRINK: resource is oversubscribed, scale down
#    3. MAINTAIN: keep at current subscription
#    4. GROW: resource is under-subscribed
class ResourceSignal(IntEnum):
    # order matters, higher priority signals should appear at top
    TERMINATE = auto()
    SHRINK = auto()
    MAINTAIN = auto()
    GROW = auto()


class Monitor:
    """Base class for a resource monitor.

    This class handles provides three abstract methods for developers to define.

    1. _setup: uses ComputeServiceSettings to set up datastructures
               needed for monitoring. Note that this method must
               define the ``sample_time`` attribute.
    2. _monitor_cycle: method that updates the above data structures
               to later be analyzed.
    3. _signal: from the data collected by _monitor_cycle, return a
               ResourceSignal.

    Thread locking is handled automatically by the wrapping ``signal``
    and ``monitor_cycle`` methods to settle race conditions and data
    corruption.

    """

    def __init__(self, settings: AsynchronousComputeServiceSettings):
        self._setup(settings)
        self._lock = Lock()
        self._terminate = False

        if not hasattr(self, "sample_time"):
            raise AttributeError(
                f"{self.__class__.__name__} implementation requires definition of the `sample_time` attribute"
            )

    def signal(self) -> ResourceSignal:
        """Issue signal based on collected measurements."""
        with self._lock:
            return self._signal()

    def monitor_cycle(self):
        """Continuously cycle, collecting results as defined by the
        derived class _monitor_cycle implementation.
        """
        while not self._terminate:
            with self._lock:
                self._monitor_cycle()
            time.sleep(self.sample_time)

    @abstractmethod
    def _setup(self, settings):
        """Abstract method for setting up data structures needed for
        storing resource measurements. These structures should be
        mutated by ``_monitor_cycle`` and read ``_signal`` for issuing
        resource signals.
        """
        raise NotImplementedError

    @abstractmethod
    def _monitor_cycle(self):
        """Abstract method for collecting and updating internal data
        to later be analyzed by ``_signal``.

        """
        raise NotImplementedError

    @abstractmethod
    def _signal(self) -> ResourceSignal:
        """Abstract method for analyzing data collected by
        ``_monitor_cycle``.

        """
        raise NotImplementedError


class GPUMonitor(Monitor):

    def _setup(self, settings):
        self.history = []
        self.history_size = settings.gpu_monitor_sample_history_size
        self.gpu_index = settings.gpu_monitor_gpu_index
        self.grow_limit = settings.gpu_monitor_grow_limit
        self.maintain_limit = settings.gpu_monitor_maintain_limit
        self.sample_time = settings.gpu_monitor_sample_time

    @staticmethod
    def _nvidia_smi() -> int:
        """Collect utilization of the GPU."""
        fields = ["index", "utilization.gpu"]
        cmd = [
            "nvidia-smi",
            f"--query-gpu={''.join(fields)}",
            "--format=csv",
        ]
        completed_process = subprocess.run(cmd, capture_output=True)
        output = completed_process.stdout.decode()
        num_fields = len(fields)
        # discard header
        gpu_entries = output.split("\n")[1:]
        for gpu in gpu_entries:
            idx, util = gpu.split(", ")
            if idx == self.gpu_index:
                util, _ = util.split(" ")
                util = int(util)
                return util

    def _monitor_cycle(self):
        util = self._nvidia_smi()
        self.history.append(util)
        self.history = self.history[-self.history_size :]

    def _signal(self) -> ResourceSignal:
        utilization = sum(self.history) / len(self.history)
        if utilization < self.grow_limit:
            return ResourceSignal.GROW
        elif utilization < self.maintain_limit:
            return ResourceSignal.MAINTAIN
        return ResourceSignal.SHRINK


class CPUMonitor(Monitor):

    def _setup(self, settings):
        self.history = []
        self.history_size = settings.cpu_monitor_sample_history_size
        self.sample_time = settings.cpu_monitor_sample_time
        self.grow_limit = settings.cpu_monitor_grow_limit
        self.maintain_limit = settings.cpu_monitor_maintain_limit

    def _signal(self) -> ResourceSignal:
        total_load = sum(self.history) / len(self.history)
        if total_load < self.grow_limit:
            return ResourceSignal.GROW
        elif total_load < self.maintain_limit:
            return ResourceSignal.MAINTAIN
        return ResourceSignal.SHRINK

    def _monitor_cycle(self):
        load, _, _ = os.getloadavg()
        cpu_count = os.cpu_count()
        total_load = load / cpu_count
        self.history.append(total_load)
        self.history = self.history[-self.history_size :]


class MemInfoParseError(Exception):
    pass


class MemoryMonitor(Monitor):

    def _setup(self, settings):
        self.history = []
        self.history_size = settings.memory_monitor_sample_history_size
        self.sample_time = settings.memory_monitor_sample_time
        self.grow_limit = settings.memory_monitor_grow_limit
        self.maintain_limit = settings.memory_monitor_maintain_limit

    @staticmethod
    def _get_memory() -> tuple[int, int]:
        """Parse /proc/meminfo for memory info."""
        total = None
        available = None
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal") or line.startswith("MemAvailable"):
                    field, size_kb, _ = line.split()
                    field = field.rstrip(":")
                    size_kb = int(size_kb)
                    match field:
                        case "MemTotal":
                            total = size_kb
                        case "MemAvailable":
                            available = size_kb
                # break from for loop and avoid error-raising else block
                if total is not None and available is not None:
                    break
            else:
                # unable to determine the MemTotal or MemAvailable
                raise MemInfoParseError
        return total, available

    def _monitor_cycle(self):
        try:
            total, avail = self._get_memory()
            fraction_used = (total - avail) / total
            self.history.append(fraction_used)
            # roughly the last minute of entries
            self.history = self.history[-self.history_size :]
        except Exception:
            self._terminate = True

    def _signal(self) -> ResourceSignal:
        fraction_used = sum(self.history) / len(self.history)
        if fraction_used < self.grow_limit:
            return ResourceSignal.GROW
        elif fraction_used < self.maintain_limit:
            return ResourceSignal.MAINTAIN
        return ResourceSignal.SHRINK
