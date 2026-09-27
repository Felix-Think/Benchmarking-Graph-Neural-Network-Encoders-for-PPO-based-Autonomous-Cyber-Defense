import json
import random
import numpy as np
import pytest
import torch

from acd_benchmark.metrics import resources
from acd_benchmark.metrics.compute import ComputeMetrics
from acd_benchmark.metrics.resources import ResourceMonitor


def test_cpu_sampler_counts_parent_workers_and_new_worker(monkeypatch):
    monitor = ResourceMonitor()
    monitor._started_wall = 1000
    snapshots = iter([
        {(1, 10): (100, 10), (2, 20): (8, 20)},
        {(1, 10): (101, 15), (2, 20): (10, 20), (3, 1001): (1, 30)},
        {(1, 10): (102, 15)},
    ])
    clock = iter([0, 2, 5])
    monkeypatch.setattr(monitor, "_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(resources, "perf_counter", lambda: next(clock))
    monitor._sample()
    monitor._sample()
    monitor._sample()
    assert monitor.cpu_seconds == 5
    assert monitor.peak_cpu_percent == 200
    assert monitor.peak_rss_bytes == 65
    assert monitor.samples == 3


def test_monitor_closes_thread_on_error_and_preserves_rng():
    random.seed(6)
    np.random.seed(6)
    torch.manual_seed(6)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    monitor = ResourceMonitor(sample_interval=0.01)
    with pytest.raises(ValueError, match="intentional"):
        with monitor:
            raise ValueError("intentional")
    assert not monitor._thread.is_alive()
    record = monitor.summary()
    assert record["wall_seconds"] > 0
    assert record["cuda_peak_allocated_bytes"] is None
    assert record["cuda_peak_reserved_bytes"] is None
    assert record["resource_samples"] >= 2
    assert record["peak_process_tree_rss_bytes_sampled"] > 0
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    json.dumps(record, allow_nan=False)


def test_no_resource_access_is_missing_not_zero(monkeypatch):
    monitor = ResourceMonitor(sample_interval=100)
    monkeypatch.setattr(monitor, "_snapshot", lambda: {})
    with monitor:
        pass
    record = monitor.summary()
    assert record["cpu_seconds_sampled"] is None
    assert record["cpu_mean_percent"] is None
    assert record["peak_process_tree_rss_bytes_sampled"] is None


def test_cuda_peak_api_is_device_specific_and_synchronized(monkeypatch):
    calls = []
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda device, name=name: calls.append((name, str(device))))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1234)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 5678)
    monitor = ResourceMonitor("cuda:1", sample_interval=100)
    monkeypatch.setattr(monitor, "_snapshot", lambda: {})
    with monitor:
        pass
    assert calls == [("synchronize", "cuda:1"), ("reset_peak_memory_stats", "cuda:1"), ("synchronize", "cuda:1")]
    assert monitor.summary()["cuda_peak_allocated_bytes"] == 1234
    assert monitor.summary()["cuda_peak_reserved_bytes"] == 5678


def test_compute_sums_stage_costs_but_takes_maximum_of_peaks(monkeypatch):
    from acd_benchmark.metrics import compute as module

    class FakeMonitor:
        index = 0

        def __init__(self, device):
            self.index = FakeMonitor.index
            FakeMonitor.index += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def summary(self):
            return {
                "wall_seconds": 2,
                "cpu_seconds_sampled": 3,
                "peak_process_tree_rss_bytes_sampled": [100, 200, 150][self.index],
                "cuda_peak_allocated_bytes": None,
                "resource_samples": 2,
                "resource_read_errors": 0,
            }

    monkeypatch.setattr(module, "ResourceMonitor", FakeMonitor)
    metrics = ComputeMetrics()
    for stage in ("rollout", "rollout", "ppo_update"):
        with metrics.measure(stage):
            pass
    metrics.training_seconds = 6
    metrics.simulator_ticks = 600
    result = metrics.summary()
    assert result["rollout/wall_seconds"] == 4
    assert result["ppo_update/wall_seconds"] == 2
    assert result["cpu_seconds_sampled"] == 9
    assert result["cpu_mean_percent"] == 150
    assert result["peak_process_tree_rss_bytes_sampled"] == 200
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["simulator_ticks_per_second"] == 100
