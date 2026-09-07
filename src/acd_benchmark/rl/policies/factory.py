"""Rebuild exactly the structured architecture recorded beside a checkpoint."""

from collections.abc import Mapping

from acd_benchmark.cyber.cc4_structured_action_env import NUM_STRUCTURED_ACTIONS
from acd_benchmark.rl.policies.gcn import StructuredGCNActorCritic
from acd_benchmark.rl.policies.nodemlp import StructuredNodeMLPActorCritic


def build_policy(config: Mapping):
    if int(config["num_actions"]) != NUM_STRUCTURED_ACTIONS:
        raise ValueError("Expected the 242-action structured catalog")
    policy_type = config.get("policy_type")
    common = dict(
        observation_dim=int(config["observation_dim"]),
        node_embedding_dim=int(config["node_embedding_dim"]),
        context_dim=int(config["context_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
        canonical_schema=str(config["canonical_schema"]),
    )
    if policy_type == "structured_nodemlp":
        return StructuredNodeMLPActorCritic(
            **common, node_hidden_dim=int(config["node_hidden_dim"])
        )
    if policy_type == "structured_gcn":
        return StructuredGCNActorCritic(
            **common,
            gcn_hidden_dim=int(config["gcn_hidden_dim"]),
            gcn_layers=int(config["gcn_layers"]),
        )
    raise ValueError(
        f"Unsupported policy {policy_type!r}; use the archive for legacy policies"
    )
