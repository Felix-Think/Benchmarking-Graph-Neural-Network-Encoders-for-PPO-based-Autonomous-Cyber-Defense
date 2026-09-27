from contextlib import contextmanager
from time import perf_counter

from acd_benchmark.metrics.resources import ResourceMonitor

class ComputeMetrics:
    def __init__(self, device="cpu"):
        self.device = device
        self.reset()

    def reset(self):
        self.training_seconds = 0
        self.simulator_ticks = 0
        self._started_at = None 
        self.stages = {}

    def load_summary(self, summary):
        """Restore completed work only; downtime/replayed work is not included."""
        self.training_seconds = summary.get("training_seconds", 0)
        self.simulator_ticks = summary.get("simulator_ticks", 0)
        self.stages = {}
        for key, value in summary.items():
            if "/" in key:
                stage, name = key.split("/", 1)
                self.stages.setdefault(stage, {})[name] = value

    @contextmanager
    def measure(self, stage):
        with ResourceMonitor(self.device) as monitor:
            yield
        sample = monitor.summary()
        totals = self.stages.setdefault(stage, {})
        for key, value in sample.items():
            if key in ("cpu_mean_percent", "resource_sample_interval_seconds"):
                continue
            previous = totals.get(key)
            if value is None:
                totals.setdefault(key, None)
            elif "peak" in key:
                totals[key] = max(previous or 0, value)
            else:
                totals[key] = (previous or 0) + value
        seconds = totals["wall_seconds"]
        cpu = totals.get("cpu_seconds_sampled")
        totals["cpu_mean_percent"] = 100 * cpu / seconds if cpu is not None and seconds else None

    def start_training(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("A traning segment is already running.")

        self._started_at = perf_counter() 

    def end_training(self, simulator_ticks: int) -> None:
        if self._started_at is None:
            raise RuntimeError("No training segment is running.")

        if simulator_ticks < 0:
            raise ValueError("Simulator ticks must be no-negative.")

        elapsed = perf_counter() - self._started_at 
        self.training_seconds += elapsed 
        self.simulator_ticks += simulator_ticks
        self._started_at = None 

    def summary(self) -> dict: 
        seconds = self.training_seconds
        resources = {}
        if self.stages:
            for key in next(iter(self.stages.values())):
                if key in ("wall_seconds", "cpu_mean_percent"):
                    continue
                values = [stage[key] for stage in self.stages.values() if stage.get(key) is not None]
                resources[key] = (max(values) if "peak" in key else sum(values)) if values else None
            cpu = resources.get("cpu_seconds_sampled")
            resources["cpu_mean_percent"] = 100 * cpu / seconds if cpu is not None and seconds else None
        return {
            "training_seconds": seconds,
            "simulator_ticks" : self.simulator_ticks,
            "simulator_ticks_per_second": ( self.simulator_ticks / seconds if seconds > 0 else None),
            **resources,
            **{
                f"{stage}/{key}": value
                for stage, values in self.stages.items()
                for key, value in values.items()
            },
        }
