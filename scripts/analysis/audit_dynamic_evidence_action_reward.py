from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch.distributions import Categorical

from CybORG.Shared.Enums import TernaryEnum

from acd_benchmark.cyber.canonical_observation import (
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    PASSIVE_MEMORY_FEATURES,
    PASSIVE_NOW_FEATURES,
    RELATIONAL_EVENT_MEMORY_FEATURES,
    RELATIONAL_EVENT_NOW_FEATURES,
    V2_NODE_FEATURES,
)
from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import build_gcn_policy_from_config
from acd_benchmark.rl.policies.graphsage import (
    build_graphsage_policy_from_config,
)
from acd_benchmark.rl.policies.structured_action import (
    build_structured_gcn_policy_from_config,
    build_structured_nodemlp_policy_from_config,
)
from acd_benchmark.rl.ppo.evaluation import load_policy_checkpoint
from scripts.evaluation.evaluate_canonical_observation_gate import (
    DYNAMIC_NODE_FEATURES,
)


CURRENT_EVIDENCE_FEATURES = (
    "process_event_now",
    "connection_event_now",
    "analysis_result_now",
    "max_file_density_now",
)
MEMORY_EVIDENCE_FEATURES = (
    "process_event_memory",
    "connection_event_memory",
    "analysis_memory",
    "max_file_density_memory",
)
HOST_RESPONSE_ACTIONS = {"Analyse", "DeployDecoy", "Remove", "Restore"}
RESPONSE_WINDOWS = (0, 1, 5, 10)
REWARD_WINDOWS = (1, 5, 10)
V1_FEATURE_GROUPS = {
    "passive_event_now": ("process_event_now", "connection_event_now"),
    "analysis_now": ("analysis_result_now", "max_file_density_now"),
    "memory": MEMORY_EVIDENCE_FEATURES,
    "passive_event_memory": (
        "process_event_memory",
        "connection_event_memory",
    ),
    "analysis_memory": ("analysis_memory", "max_file_density_memory"),
    "temporal_markers": ("observed_now", "evidence_age_normalized"),
    "observed_now": ("observed_now",),
    "evidence_age": ("evidence_age_normalized",),
    "decoy": ("decoy_active",),
}
V2_FEATURE_GROUPS = {
    "passive_event_now": PASSIVE_NOW_FEATURES,
    "relational_event_now": RELATIONAL_EVENT_NOW_FEATURES,
    "analysis_now": ("analyse_completed_now", "max_file_density_now"),
    "memory": (
        *PASSIVE_MEMORY_FEATURES,
        "passive_memory_active",
        "passive_memory_age_normalized",
        "analyse_memory_active",
        "max_file_density_memory",
        "analyse_memory_age_normalized",
    ),
    "passive_event_memory": (*PASSIVE_MEMORY_FEATURES, "passive_memory_active"),
    "relational_event_memory": RELATIONAL_EVENT_MEMORY_FEATURES,
    "analysis_memory": (
        "analyse_memory_active",
        "max_file_density_memory",
    ),
    "temporal_markers": (
        "observed_now",
        "passive_memory_age_normalized",
        "analyse_memory_age_normalized",
    ),
    "observed_now": ("observed_now",),
    "evidence_age": (
        "passive_memory_age_normalized",
        "analyse_memory_age_normalized",
    ),
    "decoy": ("decoy_active",),
}
STATIC_NODE_FEATURES = {
    "exists",
    "is_subnet",
    "is_host",
    "is_user",
    "is_server",
    "controlled",
}


def _feature_groups(node_feature_names: tuple[str, ...]):
    if tuple(node_feature_names) == tuple(V2_NODE_FEATURES):
        return V2_FEATURE_GROUPS
    return V1_FEATURE_GROUPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace dynamic evidence, policy decisions, action completion and "
            "reward for a trained canonical CAGE 4 policy."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--action-seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def _build_policy(config: dict) -> torch.nn.Module:
    policy_type = config.get("policy_type", "flat")
    if policy_type == "structured_nodemlp":
        return build_structured_nodemlp_policy_from_config(config)
    if policy_type == "structured_gcn":
        return build_structured_gcn_policy_from_config(config)
    if policy_type == "gcn":
        return build_gcn_policy_from_config(config)
    if policy_type == "graphsage":
        return build_graphsage_policy_from_config(config)
    return FlatActorCritic(
        observation_dim=int(config["observation_dim"]),
        num_actions=int(config["num_actions"]),
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
    )


def _zero_dynamic_features(
    observation: np.ndarray,
    node_feature_names: tuple[str, ...] = NODE_FEATURES,
) -> np.ndarray:
    static = observation.copy()
    nodes = static[: NUM_NODES * len(node_feature_names)].reshape(
        NUM_NODES, len(node_feature_names)
    )
    feature_index = {
        name: index for index, name in enumerate(node_feature_names)
    }
    dynamic_features = (
        DYNAMIC_NODE_FEATURES
        if tuple(node_feature_names) == tuple(NODE_FEATURES)
        else tuple(
            name for name in node_feature_names if name not in STATIC_NODE_FEATURES
        )
    )
    nodes[:, [feature_index[name] for name in dynamic_features]] = 0.0
    return static


def _zero_node_feature_group(
    observation: np.ndarray,
    features: tuple[str, ...],
    node_feature_names: tuple[str, ...] = NODE_FEATURES,
) -> np.ndarray:
    transformed = observation.copy()
    nodes = transformed[: NUM_NODES * len(node_feature_names)].reshape(
        NUM_NODES, len(node_feature_names)
    )
    feature_index = {
        name: index for index, name in enumerate(node_feature_names)
    }
    nodes[:, [feature_index[name] for name in features]] = 0.0
    return transformed


def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.shape != action_mask.shape:
        raise ValueError(
            f"logit/mask mismatch: logits={tuple(logits.shape)}, "
            f"mask={tuple(action_mask.shape)}"
        )
    return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def _hostname_node_map(env: CC4CanonicalFlatEnv, agent: str) -> dict[str, int]:
    builder = env._builders[agent]
    mapping = {}
    for hostname in env.env.hosts(agent):
        node_index = builder._node_index(hostname)
        if node_index is not None and node_index >= NUM_SUBNET_NODES:
            mapping[hostname] = node_index
    return mapping


def _evidence_snapshot(
    env: CC4CanonicalFlatEnv,
    agent: str,
) -> dict[str, dict[str, object]]:
    canonical = env.last_canonical[agent]
    state = env.env.env.environment_controller.state
    node_feature_names = env.node_feature_names
    feature_index = {
        name: index for index, name in enumerate(node_feature_names)
    }
    is_v2 = tuple(node_feature_names) == tuple(V2_NODE_FEATURES)

    def maximum(node: np.ndarray, features: tuple[str, ...]) -> float:
        return max(float(node[feature_index[name]]) for name in features)

    snapshot = {}
    for hostname, node_index in _hostname_node_map(env, agent).items():
        if hostname not in state.hosts:
            continue
        node = canonical.node_features[node_index]
        if is_v2:
            current = {
                "process_event_now": maximum(
                    node,
                    (
                        "process_event_count_now",
                        "process_has_pid_now",
                        "process_has_connection_now",
                    ),
                ),
                "connection_event_now": maximum(
                    node,
                    (
                        "connection_event_count_now",
                        "connection_has_remote_address_now",
                        "connection_has_remote_port_now",
                        "connection_has_pid_now",
                    ),
                ),
                "analysis_result_now": float(
                    node[feature_index["analyse_completed_now"]]
                ),
                "max_file_density_now": float(
                    node[feature_index["max_file_density_now"]]
                ),
            }
            memory = {
                "process_event_memory": maximum(
                    node,
                    (
                        "process_event_count_memory",
                        "process_has_pid_memory",
                        "process_has_connection_memory",
                    ),
                ),
                "connection_event_memory": maximum(
                    node,
                    (
                        "connection_event_count_memory",
                        "connection_has_remote_address_memory",
                        "connection_has_remote_port_memory",
                        "connection_has_pid_memory",
                    ),
                ),
                "analysis_memory": float(
                    node[feature_index["analyse_memory_active"]]
                ),
                "max_file_density_memory": float(
                    node[feature_index["max_file_density_memory"]]
                ),
            }
        else:
            current = {
                name: float(node[feature_index[name]])
                for name in CURRENT_EVIDENCE_FEATURES
            }
            memory = {
                name: float(node[feature_index[name]])
                for name in MEMORY_EVIDENCE_FEATURES
            }
        host_events = state.hosts[hostname].events
        snapshot[hostname] = {
            "current": current,
            "memory": memory,
            "current_any": any(value > 0.0 for value in current.values()),
            "memory_any": any(value > 0.0 for value in memory.values()),
            "process_old_count": len(host_events.old_process_creation),
            "process_current_count": len(host_events.process_creation),
            "connection_old_count": len(host_events.old_network_connections),
            "connection_current_count": len(host_events.network_connections),
        }
    return snapshot


def _action_metadata(env, agent: str, action_id: int) -> dict[str, object]:
    if isinstance(env, CC4StructuredActionEnv):
        native_action_id = int(env._structured_to_native[agent][action_id])
        action_object = env.env.env.actions(agent)[native_action_id]
    else:
        action_object = env.env.actions(agent)[action_id]
    return {
        "action_id": action_id,
        "action_label": env.action_labels(agent)[action_id],
        "action_type": type(action_object).__name__,
        "target_host": getattr(action_object, "hostname", None),
        "from_subnet": getattr(action_object, "from_subnet", None),
        "to_subnet": getattr(action_object, "to_subnet", None),
        "declared_duration": getattr(action_object, "duration", None),
        "declared_cost": float(getattr(action_object, "cost", 0.0)),
    }


def _ternary_name(value: object) -> str:
    if isinstance(value, TernaryEnum):
        return value.name
    return str(value)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _action_proportions(records: list[dict]) -> dict[str, float]:
    counts = Counter(record["action_type"] for record in records)
    total = sum(counts.values())
    return {
        action: count / total
        for action, count in sorted(counts.items())
    } if total else {}


def _reward_group(records: list[dict]) -> dict[str, object]:
    output: dict[str, object] = {"count": len(records)}
    for window in REWARD_WINDOWS:
        output[f"mean_reward_next_{window}"] = _mean(
            [float(record[f"reward_next_{window}"]) for record in records]
        )
    output["mean_blue_reward_machine_next_5"] = _mean(
        [float(record["blue_reward_machine_next_5"]) for record in records]
    )
    output["mean_action_cost_next_5"] = _mean(
        [float(record["action_cost_next_5"]) for record in records]
    )
    return output


def _enrich_episode(
    decisions: list[dict],
    events: list[dict],
    ticks: list[dict],
) -> None:
    reward = np.asarray([tick["reward"] for tick in ticks], dtype=np.float64)
    blue_reward = np.asarray(
        [tick["blue_reward_machine"] for tick in ticks], dtype=np.float64
    )
    action_cost = np.asarray(
        [tick["action_cost"] for tick in ticks], dtype=np.float64
    )

    for decision in decisions:
        tick = int(decision["tick"])
        for window in REWARD_WINDOWS:
            stop = min(tick + window, len(ticks))
            decision[f"reward_next_{window}"] = float(reward[tick:stop].sum())
        stop = min(tick + 5, len(ticks))
        decision["blue_reward_machine_next_5"] = float(
            blue_reward[tick:stop].sum()
        )
        decision["action_cost_next_5"] = float(action_cost[tick:stop].sum())

    by_agent = defaultdict(list)
    for decision in decisions:
        by_agent[decision["agent"]].append(decision)

    for event in events:
        tick = int(event["tick"])
        candidates = [
            decision
            for decision in by_agent[event["agent"]]
            if int(decision["tick"]) >= tick
        ]
        event["first_decision_latency"] = (
            int(candidates[0]["tick"]) - tick if candidates else None
        )
        same_host = [
            decision
            for decision in candidates
            if decision["target_host"] == event["hostname"]
            and decision["action_type"] in HOST_RESPONSE_ACTIONS
        ]
        event["first_same_host_action_latency"] = (
            int(same_host[0]["tick"]) - tick if same_host else None
        )
        event["first_same_host_action_type"] = (
            same_host[0]["action_type"] if same_host else None
        )
        for window in RESPONSE_WINDOWS:
            event[f"same_host_response_within_{window}"] = any(
                int(decision["tick"]) - tick <= window
                for decision in same_host
            )
        for window in REWARD_WINDOWS:
            stop = min(tick + window, len(ticks))
            event[f"reward_after_evidence_{window}"] = float(
                reward[tick:stop].sum()
            )


def _write_csv_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with gzip.open(path, "wt", encoding="utf-8", newline=""):
            return
    fieldnames = sorted({key for record in records for key in record})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _summarize(
    config: dict,
    episode_returns: list[float],
    decisions: list[dict],
    events: list[dict],
    ticks: list[dict],
    feature_groups: dict[str, tuple[str, ...]],
) -> dict:
    current_decisions = [record for record in decisions if record["current_any"]]
    no_current_decisions = [
        record for record in decisions if not record["current_any"]
    ]
    memory_decisions = [record for record in decisions if record["memory_any"]]
    host_decisions = [record for record in decisions if record["target_host"]]
    host_current_context = [
        record for record in host_decisions if record["current_any"]
    ]
    host_memory_context = [
        record for record in host_decisions if record["memory_any"]
    ]
    current_aligned = [
        record for record in host_decisions if record["target_current_evidence"]
    ]
    memory_aligned = [
        record for record in host_decisions if record["target_memory_evidence"]
    ]
    unaligned_current_context = [
        record
        for record in host_current_context
        if not record["target_current_evidence"]
    ]
    no_evidence_context = [
        record
        for record in host_decisions
        if not record["current_any"] and not record["memory_any"]
    ]

    event_type_counts = Counter()
    for event in events:
        if event["process_event_now"]:
            event_type_counts["process_event_now"] += 1
        if event["connection_event_now"]:
            event_type_counts["connection_event_now"] += 1
        if event["analysis_result_now"]:
            event_type_counts["analysis_result_now"] += 1
        if event["max_file_density_now"] > 0:
            event_type_counts["max_file_density_now"] += 1

    sensitivity = {}
    for name, records in (
        ("all", decisions),
        ("current_evidence_present", current_decisions),
        ("current_evidence_absent", no_current_decisions),
        ("memory_evidence_present", memory_decisions),
    ):
        sensitivity[name] = {
            "count": len(records),
            "mean_total_variation": _mean(
                [float(record["full_static_total_variation"]) for record in records]
            ),
            "mean_kl_full_static": _mean(
                [float(record["kl_full_static"]) for record in records]
            ),
            "argmax_change_rate": _rate(
                sum(bool(record["argmax_changed"]) for record in records),
                len(records),
            ),
            "mean_absolute_value_change": _mean(
                [float(record["absolute_value_change"]) for record in records]
            ),
        }

    feature_group_sensitivity = {}
    for group in feature_groups:
        records_with_group = [
            record for record in decisions if record[f"{group}_present"]
        ]
        feature_group_sensitivity[group] = {
            "all_decisions": {
                "count": len(decisions),
                "mean_total_variation": _mean(
                    [
                        float(record[f"tv_zero_{group}"])
                        for record in decisions
                    ]
                ),
                "argmax_change_rate": _rate(
                    sum(bool(record[f"argmax_changed_zero_{group}"]) for record in decisions),
                    len(decisions),
                ),
            },
            "group_present": {
                "count": len(records_with_group),
                "mean_total_variation": _mean(
                    [
                        float(record[f"tv_zero_{group}"])
                        for record in records_with_group
                    ]
                ),
                "argmax_change_rate": _rate(
                    sum(
                        bool(record[f"argmax_changed_zero_{group}"])
                        for record in records_with_group
                    ),
                    len(records_with_group),
                ),
            },
        }

    latencies = [
        int(event["first_same_host_action_latency"])
        for event in events
        if event["first_same_host_action_latency"] is not None
    ]
    observed_buffer_events = sum(
        int(event["process_old_count"])
        + int(event["connection_old_count"])
        for event in events
    )
    unobserved_buffer_events = sum(
        int(event["process_current_count"])
        + int(event["connection_current_count"])
        for event in events
    )

    return {
        "policy_type": config.get("policy_type", "flat"),
        "canonical_schema": config.get("canonical_schema", "v1"),
        "semantic_action_mask": bool(config.get("semantic_action_mask", False)),
        "training_seed": int(config["seed"]),
        "environment_distribution": config["environment_distribution"],
        "episodes": len(episode_returns),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "episode_returns": episode_returns,
        "ticks": len(ticks),
        "decisions": len(decisions),
        "evidence_occurrences": len(events),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "event_buffer_semantics": {
            "old_observed_event_items_at_evidence_ticks": observed_buffer_events,
            "current_unobserved_event_items_at_evidence_ticks": unobserved_buffer_events,
            "unobserved_fraction": _rate(
                unobserved_buffer_events,
                observed_buffer_events + unobserved_buffer_events,
            ),
        },
        "decision_context": {
            "current_evidence_present": len(current_decisions),
            "current_evidence_present_fraction": _rate(
                len(current_decisions), len(decisions)
            ),
            "memory_evidence_present": len(memory_decisions),
            "memory_evidence_present_fraction": _rate(
                len(memory_decisions), len(decisions)
            ),
            "host_targeted": len(host_decisions),
            "target_current_evidence": len(current_aligned),
            "target_current_evidence_given_host_action_and_current_context": _rate(
                len(current_aligned), len(host_current_context)
            ),
            "uniform_host_current_chance": _mean(
                [
                    float(record["uniform_host_current_chance"])
                    for record in host_current_context
                ]
            ),
            "target_memory_evidence": len(memory_aligned),
            "target_memory_evidence_given_host_action_and_memory_context": _rate(
                len(memory_aligned), len(host_memory_context)
            ),
            "uniform_host_memory_chance": _mean(
                [
                    float(record["uniform_host_memory_chance"])
                    for record in host_memory_context
                ]
            ),
        },
        "action_proportions": {
            "all": _action_proportions(decisions),
            "current_evidence_present": _action_proportions(current_decisions),
            "current_evidence_absent": _action_proportions(no_current_decisions),
        },
        "policy_input_sensitivity": sensitivity,
        "feature_group_input_sensitivity": feature_group_sensitivity,
        "evidence_response": {
            "first_same_host_action_observed": len(latencies),
            "first_same_host_action_observed_fraction": _rate(
                len(latencies), len(events)
            ),
            "median_first_same_host_action_latency": (
                float(median(latencies)) if latencies else None
            ),
            **{
                f"same_host_response_within_{window}_fraction": _rate(
                    sum(
                        bool(event[f"same_host_response_within_{window}"])
                        for event in events
                    ),
                    len(events),
                )
                for window in RESPONSE_WINDOWS
            },
        },
        "reward_association_not_causal": {
            "target_current_evidence": _reward_group(current_aligned),
            "current_context_but_other_host": _reward_group(
                unaligned_current_context
            ),
            "no_current_or_memory_evidence": _reward_group(no_evidence_context),
            "target_memory_evidence": _reward_group(memory_aligned),
        },
        "reward_breakdown": {
            "total": float(sum(float(tick["reward"]) for tick in ticks)),
            "blue_reward_machine": float(
                sum(float(tick["blue_reward_machine"]) for tick in ticks)
            ),
            "action_cost": float(
                sum(float(tick["action_cost"]) for tick in ticks)
            ),
        },
        "action_completion": _completion_summary(decisions),
    }


def _completion_summary(decisions: list[dict]) -> dict[str, object]:
    by_type = defaultdict(list)
    for record in decisions:
        by_type[record["action_type"]].append(record)
    output = {}
    for action_type, records in sorted(by_type.items()):
        completed = [record for record in records if record["completion_tick"] is not None]
        success_counts = Counter(record["completion_success"] for record in completed)
        output[action_type] = {
            "selected": len(records),
            "completed": len(completed),
            "success_counts": dict(sorted(success_counts.items())),
            "mean_observed_duration": _mean(
                [float(record["observed_duration"]) for record in completed]
            ),
        }
    traffic = [
        record
        for record in decisions
        if record["action_type"] in {"AllowTrafficZone", "BlockTrafficZone"}
        and record["completion_tick"] is not None
    ]
    traffic_failures = [
        record for record in traffic if record["completion_success"] == "FALSE"
    ]
    output["semantic_traffic_summary"] = {
        "completed": len(traffic),
        "false_completions": len(traffic_failures),
        "false_completion_rate": _rate(len(traffic_failures), len(traffic)),
    }
    return output


def run_audit(args: argparse.Namespace) -> dict:
    run_path = args.run_dir / f"seed_{args.seed}"
    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)
    checkpoint_path = args.checkpoint_path or run_path / "checkpoint.pt"
    policy = _build_policy(config)
    load_policy_checkpoint(policy, checkpoint_path, device=args.device)
    policy.eval()
    device = next(policy.parameters()).device

    canonical_env = CC4CanonicalFlatEnv(
        episode_steps=int(config["episode_steps"]),
        seed=args.eval_seed,
        canonical_variant=config["canonical_variant"],
        evidence_age_horizon=int(config["evidence_age_horizon"]),
        environment_distribution=config["environment_distribution"],
        canonical_schema=config.get("canonical_schema", "v1"),
        passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
        analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
        semantic_action_mask=bool(config.get("semantic_action_mask", False)),
    )
    policy_type = config.get("policy_type", "flat")
    env = (
        CC4StructuredActionEnv(canonical_env)
        if policy_type in {"structured_nodemlp", "structured_gcn"}
        else canonical_env
    )
    node_feature_names = canonical_env.node_feature_names
    feature_index = {
        name: index for index, name in enumerate(node_feature_names)
    }
    feature_groups = _feature_groups(node_feature_names)

    torch.manual_seed(args.action_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.action_seed)

    all_decisions: list[dict] = []
    all_events: list[dict] = []
    all_ticks: list[dict] = []
    episode_returns: list[float] = []

    for episode_index in range(args.episodes):
        episode_seed = args.eval_seed + episode_index
        observations, infos = env.reset(seed=episode_seed)
        episode_decisions: list[dict] = []
        episode_events: list[dict] = []
        episode_ticks: list[dict] = []
        pending: dict[str, dict] = {}
        episode_return = 0.0
        tick = 0

        while True:
            actions: dict[str, int] = {}
            evidence_by_agent = {
                agent: _evidence_snapshot(canonical_env, agent)
                for agent in env.possible_agents
            }

            for agent, host_evidence in evidence_by_agent.items():
                for hostname, evidence in host_evidence.items():
                    if not evidence["current_any"]:
                        continue
                    current = evidence["current"]
                    episode_events.append(
                        {
                            "episode": episode_index,
                            "episode_seed": episode_seed,
                            "tick": tick,
                            "agent": agent,
                            "hostname": hostname,
                            "decision_ready": bool(infos[agent]["decision_ready"]),
                            **current,
                            "process_old_count": evidence["process_old_count"],
                            "process_current_count": evidence["process_current_count"],
                            "connection_old_count": evidence[
                                "connection_old_count"
                            ],
                            "connection_current_count": evidence[
                                "connection_current_count"
                            ],
                        }
                    )

            for agent in env.possible_agents:
                if not infos[agent]["decision_ready"]:
                    continue
                observation_np = observations[agent]
                observation = torch.as_tensor(
                    observation_np, dtype=torch.float32, device=device
                )
                static_observation = torch.as_tensor(
                    _zero_dynamic_features(observation_np, node_feature_names),
                    dtype=torch.float32,
                    device=device,
                )
                action_mask = torch.as_tensor(
                    infos[agent]["action_mask"], dtype=torch.bool, device=device
                )
                with torch.no_grad():
                    full_logits, full_value = policy(observation)
                    static_logits, static_value = policy(static_observation)
                    full_logits = _masked_logits(full_logits, action_mask)
                    static_logits = _masked_logits(static_logits, action_mask)
                    full_distribution = Categorical(logits=full_logits)
                    action = full_distribution.sample()
                    full_probabilities = full_distribution.probs
                    static_probabilities = Categorical(
                        logits=static_logits
                    ).probs
                    group_observations = torch.stack(
                        [
                            torch.as_tensor(
                                _zero_node_feature_group(
                                    observation_np,
                                    features,
                                    node_feature_names,
                                ),
                                dtype=torch.float32,
                                device=device,
                            )
                            for features in feature_groups.values()
                        ]
                    )
                    group_logits, _ = policy(group_observations)
                    expanded_mask = action_mask.unsqueeze(0).expand_as(group_logits)
                    group_logits = group_logits.masked_fill(
                        ~expanded_mask, torch.finfo(group_logits.dtype).min
                    )
                    group_probabilities = torch.softmax(group_logits, dim=-1)

                action_id = int(action.item())
                metadata = _action_metadata(env, agent, action_id)
                host_evidence = evidence_by_agent[agent]
                current_hosts = [
                    hostname
                    for hostname, evidence in host_evidence.items()
                    if evidence["current_any"]
                ]
                memory_hosts = [
                    hostname
                    for hostname, evidence in host_evidence.items()
                    if evidence["memory_any"]
                ]
                target_host = metadata["target_host"]
                existing_host_count = len(host_evidence)
                canonical_nodes = canonical_env.last_canonical[agent].node_features
                group_presence = {
                    group: bool(
                        canonical_nodes[
                            :,
                            [feature_index[feature] for feature in features],
                        ].max()
                        > 0
                    )
                    for group, features in feature_groups.items()
                }
                record = {
                    "episode": episode_index,
                    "episode_seed": episode_seed,
                    "tick": tick,
                    "agent": agent,
                    **metadata,
                    "current_any": bool(current_hosts),
                    "current_host_count": len(current_hosts),
                    "memory_any": bool(memory_hosts),
                    "memory_host_count": len(memory_hosts),
                    "target_current_evidence": target_host in current_hosts,
                    "target_memory_evidence": target_host in memory_hosts,
                    "existing_host_count": existing_host_count,
                    "uniform_host_current_chance": (
                        len(current_hosts) / existing_host_count
                        if existing_host_count
                        else 0.0
                    ),
                    "uniform_host_memory_chance": (
                        len(memory_hosts) / existing_host_count
                        if existing_host_count
                        else 0.0
                    ),
                    "full_static_total_variation": float(
                        0.5
                        * torch.abs(
                            full_probabilities - static_probabilities
                        ).sum().item()
                    ),
                    "kl_full_static": float(
                        torch.sum(
                            full_probabilities
                            * (
                                torch.log(full_probabilities.clamp_min(1e-12))
                                - torch.log(
                                    static_probabilities.clamp_min(1e-12)
                                )
                            )
                        ).item()
                    ),
                    "argmax_changed": bool(
                        full_logits.argmax().item()
                        != static_logits.argmax().item()
                    ),
                    "full_entropy": float(full_distribution.entropy().item()),
                    "selected_probability_full": float(
                        full_probabilities[action_id].item()
                    ),
                    "selected_probability_static": float(
                        static_probabilities[action_id].item()
                    ),
                    "absolute_value_change": float(
                        torch.abs(full_value - static_value).item()
                    ),
                    "completion_tick": None,
                    "completion_success": None,
                    "observed_duration": None,
                }
                for group_index, group in enumerate(feature_groups):
                    group_probability = group_probabilities[group_index]
                    record[f"{group}_present"] = group_presence[group]
                    record[f"tv_zero_{group}"] = float(
                        0.5
                        * torch.abs(
                            full_probabilities - group_probability
                        ).sum().item()
                    )
                    record[f"argmax_changed_zero_{group}"] = bool(
                        full_logits.argmax().item()
                        != group_logits[group_index].argmax().item()
                    )
                actions[agent] = action_id
                episode_decisions.append(record)
                if agent in pending:
                    raise RuntimeError(f"{agent} selected a new action while one is pending")
                pending[agent] = record

            next_observations, rewards, terminated, truncated, next_infos = env.step(
                actions
            )
            reward_values = [float(rewards[agent]) for agent in env.possible_agents]
            if not np.allclose(reward_values, reward_values[0]):
                raise RuntimeError(f"Blue agents received different team rewards: {rewards}")
            reward = reward_values[0]
            reward_breakdown = (
                canonical_env.env.env.environment_controller.reward.get(
                    "Blue", {}
                )
            )
            blue_reward_machine = float(
                reward_breakdown.get("BlueRewardMachine", 0.0)
            )
            action_cost = float(reward_breakdown.get("action_cost", 0.0))
            if not np.isclose(reward, blue_reward_machine + action_cost):
                raise RuntimeError(
                    "Wrapper reward does not equal BlueRewardMachine + action_cost: "
                    f"reward={reward}, breakdown={reward_breakdown}"
                )

            episode_ticks.append(
                {
                    "episode": episode_index,
                    "episode_seed": episode_seed,
                    "tick": tick,
                    "reward": reward,
                    "blue_reward_machine": blue_reward_machine,
                    "action_cost": action_cost,
                    "decision_count": len(actions),
                    "current_evidence_host_count": sum(
                        int(evidence["current_any"])
                        for host_evidence in evidence_by_agent.values()
                        for evidence in host_evidence.values()
                    ),
                }
            )
            episode_return += reward

            for agent in list(pending):
                if not next_infos[agent]["decision_ready"]:
                    continue
                raw_observation = canonical_env.env.env.get_observation(agent)
                record = pending.pop(agent)
                record["completion_tick"] = tick
                record["completion_success"] = _ternary_name(
                    raw_observation.get("success", TernaryEnum.UNKNOWN)
                )
                record["observed_duration"] = tick - int(record["tick"]) + 1

            episode_done = all(
                terminated.get(agent, False) or truncated.get(agent, False)
                for agent in env.possible_agents
            )
            if episode_done:
                break
            observations = next_observations
            infos = next_infos
            tick += 1

        _enrich_episode(episode_decisions, episode_events, episode_ticks)
        all_decisions.extend(episode_decisions)
        all_events.extend(episode_events)
        all_ticks.extend(episode_ticks)
        episode_returns.append(episode_return)

    summary = _summarize(
        config,
        episode_returns,
        all_decisions,
        all_events,
        all_ticks,
        feature_groups,
    )
    summary["eval_seed"] = args.eval_seed
    summary["action_seed"] = args.action_seed
    summary["checkpoint_path"] = str(checkpoint_path)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv_gz(
        output_prefix.with_name(output_prefix.name + "_decisions.csv.gz"),
        all_decisions,
    )
    _write_csv_gz(
        output_prefix.with_name(output_prefix.name + "_events.csv.gz"),
        all_events,
    )
    _write_csv_gz(
        output_prefix.with_name(output_prefix.name + "_ticks.csv.gz"),
        all_ticks,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_audit(args)
    print(
        f"policy={summary['policy_type']} episodes={summary['episodes']} "
        f"mean_return={summary['mean_return']:.2f} "
        f"evidence={summary['evidence_occurrences']} "
        f"decisions={summary['decisions']}"
    )
    print(
        "mean_full_static_tv="
        f"{summary['policy_input_sensitivity']['all']['mean_total_variation']:.6f}"
    )
    print(
        "same_host_response_within_10="
        f"{summary['evidence_response']['same_host_response_within_10_fraction']}"
    )


if __name__ == "__main__":
    main()
