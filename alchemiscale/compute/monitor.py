from abc import abstractmethod
from enum import auto, IntEnum
import os
import subprocess
import time
from threading import Lock


class ResourceSignal(IntEnum):
    # order matters, higher priority signals should appear at top
    TERMINATE = auto()
    SHRINK = auto()
    MAINTAIN = auto()
    GROW = auto()


class Monitor:

    def __init__(self, settings):
        self._setup(settings)
        self._lock = Lock()
        self._terminate = False

    def signal(self) -> ResourceSignal:
        with self._lock:
            return self._signal()

    def monitor_cycle(self):
        """Method to be run in a thread, changing state of the monitor
        such that the signal method can process the results.
        """
        while not self._terminate:
            with self._lock:
                self._monitor_cycle()
            # TODO: make configurable
            time.sleep(1)

    @abstractmethod
    def _setup(self, settings):
        raise NotImplementedError

    @abstractmethod
    def _monitor_cycle(self):
        """Mutating method"""
        raise NotImplementedError

    @abstractmethod
    def _signal(self) -> ResourceSignal:
        raise NotImplementedError


class GPUMonitor(Monitor):

    def _setup(self, settings):
        self.history = []
        self.gpu_index = settings.gpu_monitor_gpu_id

    @staticmethod
    def _nvidia_smi() -> int:
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
        self.history = self.history[-60:]

    def _signal(self) -> ResourceSignal:
        utilization = sum(self.history) / len(self.history)
        if utilization < self.grow_limit:
            return ResourceSignal.GROW
        elif utilization < self.maintain_limit:
            return ResourceSignal.MAINTAIN
        return ResourceSignal.SHRINK


class CPUMonitor(Monitor):

    # TODO: make configurable
    grow_limit = 0.50
    maintain_limit = 0.75

    def _setup(self, settings):
        self.history = []

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
        self.history = self.history[-60:]


class MemInfoParseError(Exception):
    pass


class MemoryMonitor(Monitor):

    # TODO: make configurable
    grow_limit = 0.65
    maintain_limit = 0.85

    def _setup(self, settings):
        self.history = []

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
            self.history = self.history[-60:]
        except Exception:
            self._terminate = True

    def _signal(self) -> ResourceSignal:
        fraction_used = sum(self.history) / len(self.history)
        if fraction_used < self.grow_limit:
            return ResourceSignal.GROW
        elif fraction_used < self.maintain_limit:
            return ResourceSignal.MAINTAIN
        return ResourceSignal.SHRINK
