"""Sample this process tree, not unrelated applications on the machine."""

import platform
from threading import Event, Thread
from time import perf_counter, time

import psutil
import torch


def runtime_metadata(device) -> dict:
    device = torch.device(device)
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "psutil": psutil.__version__,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "system_ram_bytes": psutil.virtual_memory().total,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": gpu.total_memory if gpu is not None else None,
        "gpu_compute_capability": [gpu.major, gpu.minor] if gpu is not None else None,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if gpu is not None else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "resource_protocol": "process_tree_sampled_v1",
        "resource_sample_interval_seconds": 0.2,
    }


class ResourceMonitor:
    """One measured region. CUDA peaks cover only the current process allocator.

    RSS is summed across parent/descendants and can double-count shared pages.
    CPU uses user+system deltas (100% = one CPU core); short-lived processes or
    peaks between samples may be missed. Permission failures are reported.
    """

    def __init__(self, device="cpu", sample_interval=0.2):
        if sample_interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.device = torch.device(device)
        self.sample_interval = sample_interval
        self._stop = Event()
        self._thread = None
        self._previous = {}
        self._last_sample = None
        self._result = None
        self.cpu_seconds = 0.0
        self.peak_cpu_percent = None
        self.peak_rss_bytes = None
        self.samples = 0
        self.errors = 0

    def _snapshot(self):
        records = {}
        root = psutil.Process()
        try:
            children = root.children(recursive=True)
        except psutil.Error:
            children = []
            self.errors += 1
        for process in [root, *children]:
            try:
                with process.oneshot():
                    created = process.create_time()
                    cpu = process.cpu_times()
                    records[(process.pid, created)] = (
                        cpu.user + cpu.system, process.memory_info().rss,
                    )
            except psutil.NoSuchProcess:
                continue
            except psutil.Error:
                self.errors += 1
        return records

    def _sample(self):
        records = self._snapshot()
        now = perf_counter()
        if records:
            rss = sum(record[1] for record in records.values())
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            self.samples += 1
        delta = 0.0
        for identity, (cpu, _) in records.items():
            if identity in self._previous:
                delta += max(0.0, cpu - self._previous[identity])
            elif self._last_sample is not None and identity[1] >= self._started_wall:
                # Include CPU since birth for workers spawned inside this region.
                delta += cpu
            self._previous[identity] = cpu
        if self._last_sample is not None:
            elapsed = now - self._last_sample
            self.cpu_seconds += delta
            if records and elapsed > 0:
                self.peak_cpu_percent = max(self.peak_cpu_percent or 0, 100 * delta / elapsed)
        self._last_sample = now

    def _poll(self):
        while not self._stop.wait(self.sample_interval):
            self._sample()

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("ResourceMonitor instances cannot be reused")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._started_wall = time()
        self._started = perf_counter()
        self._sample()
        self._thread = Thread(target=self._poll, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            self._stop.set()
            self._thread.join()
            self._sample()
        elapsed = perf_counter() - self._started
        cuda = self.device.type == "cuda"
        self._result = {
            "wall_seconds": elapsed,
            "cpu_seconds_sampled": self.cpu_seconds if self.samples else None,
            "cpu_mean_percent": 100 * self.cpu_seconds / elapsed if self.samples and elapsed else None,
            "cpu_peak_percent_sampled": self.peak_cpu_percent,
            "peak_process_tree_rss_bytes_sampled": self.peak_rss_bytes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else None,
            "resource_samples": self.samples,
            "resource_read_errors": self.errors,
            "resource_sample_interval_seconds": self.sample_interval,
        }

    def summary(self):
        if self._result is None:
            raise RuntimeError("The measured region has not finished")
        return dict(self._result)
