from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMetrics:
    """Minimal metrics available before deeper state-based evaluators."""

    total_reward: float
    steps: int
    done: bool
