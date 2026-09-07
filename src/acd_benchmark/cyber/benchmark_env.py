"""Single assembly point for training and evaluation environments."""

from collections.abc import Mapping

from acd_benchmark.cyber.canonical_observation_condition import (
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv


def make_benchmark_env(config: Mapping, *, seed: int | None = None):
    canonical = CC4CanonicalFlatEnv(
        episode_steps=int(config.get("episode_steps", 500)),
        seed=seed,
        canonical_variant=str(config.get("canonical_variant", "action_aware")),
        canonical_schema=str(config.get("canonical_schema", "v2")),
        environment_distribution=str(
            config.get("environment_distribution", "random_host")
        ),
        passive_memory_ttl=int(config.get("passive_memory_ttl", 5)),
        analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
        semantic_action_mask=bool(config.get("semantic_action_mask", False)),
    )
    structured = CC4StructuredActionEnv(canonical)
    return CanonicalObservationCondition(
        structured, condition=str(config.get("observation_condition", "full"))
    )
