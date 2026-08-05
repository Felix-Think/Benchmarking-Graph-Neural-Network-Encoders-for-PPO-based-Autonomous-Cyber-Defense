"""Construct matched Stage 1 environments and policies from frozen config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from acd_benchmark.rl.policies import (
    ActorCriticPolicy,
    FlatActorCritic,
    GCNActorCritic,
    GraphSAGEActorCritic,
    infer_action_head_spec,
)
from acd_benchmark.wrappers.stage1.env import Stage1FlatEnv, Stage1GraphEnv


SUPPORTED_STAGE1_MODELS = ("flat", "gcn", "graphsage")


def resolve_model_hidden_dim(model: str, model_config: dict[str, Any]) -> int:
    """Resolve an optional per-model width used for parameter matching."""
    hidden_by_model = model_config.get("hidden_dim_by_model", {})
    hidden_dim = int(hidden_by_model.get(model, model_config["hidden_dim"]))
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive for {model}.")
    return hidden_dim


def make_stage1_env(
    model: str,
    *,
    scenario_path: Path,
    node_count: int,
    observation_probability: float,
    max_steps: int,
):
    """Build the model-specific view over one shared Stage 1 environment."""
    common = {
        "scenario_path": scenario_path,
        "node_count": node_count,
        "observation_probability": observation_probability,
        "max_steps": max_steps,
    }
    if model == "flat":
        return Stage1FlatEnv(**common)
    if model in {"gcn", "graphsage"}:
        return Stage1GraphEnv(**common)
    raise ValueError(f"Unsupported Stage 1 model: {model}")


def build_stage1_policy(
    model: str,
    observation: Any,
    env: Any,
    *,
    hidden_dim: int,
) -> ActorCriticPolicy:
    """Build policies that share PPO and actor/critic output dimensions."""
    if model == "flat":
        return FlatActorCritic(
            input_dim=int(observation.shape[0]),
            hidden_dim=hidden_dim,
            action_count=env.action_count,
        )
    if model in {"gcn", "graphsage"}:
        action_head = infer_action_head_spec(env.action_specs, observation.hostnames)
        graph_policy_cls = GCNActorCritic if model == "gcn" else GraphSAGEActorCritic
        return graph_policy_cls(
            node_feature_dim=int(observation.node_features.shape[1]), # iput_dim
            hidden_dim=hidden_dim,# hidden_dim
            action_head=action_head, # output_dim
        )
    raise ValueError(f"Unsupported Stage 1 model: {model}")
