from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.cyber.canonical_observation import FEATURE_INDEX
from acd_benchmark.cyber.canonical_observation_v2 import V2_FEATURE_INDEX
from acd_benchmark.rl.policies.structured_action import (
    build_structured_gcn_policy_from_config,
    build_structured_nodemlp_policy_from_config,
)
from acd_benchmark.rl.ppo.evaluation import load_policy_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict raw-Blue event evidence with the state event queues "
            "used by BlueFlatWrapper/EnterpriseMAE."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--action-seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _build_policy(config: dict):
    policy_type = config.get("policy_type")
    if policy_type == "structured_nodemlp":
        return build_structured_nodemlp_policy_from_config(config)
    if policy_type == "structured_gcn":
        return build_structured_gcn_policy_from_config(config)
    raise ValueError(
        "Observation-boundary audit currently expects a structured policy, "
        f"received {policy_type!r}"
    )


def _queue_counts(events) -> tuple[int, int, int, int]:
    return (
        len(events.old_process_creation),
        len(events.process_creation),
        len(events.old_network_connections),
        len(events.network_connections),
    )


def _record_boundary(
    canonical_env: CC4CanonicalFlatEnv,
    counters: Counter,
    by_action: Counter,
) -> None:
    state = canonical_env.env.env.environment_controller.state
    for agent in canonical_env.possible_agents:
        raw_observation = canonical_env.last_raw_observations[agent]
        raw_details = canonical_env._observed_host_event_details(raw_observation)
        canonical = canonical_env.last_canonical[agent]
        builder = canonical_env._builders[agent]
        action = raw_observation.get("action")
        action_type = type(action).__name__ if action is not None else "None"
        action_target = getattr(action, "hostname", None)

        for hostname in canonical_env.env.hosts(agent):
            if hostname not in state.hosts or "router" in hostname:
                continue
            details = raw_details.get(hostname)
            parsed_raw_positive = bool(details and details.has_passive_event)
            node_index = builder._node_index(hostname)
            if node_index is None:
                canonical_positive = False
            elif canonical_env.canonical_schema == "v2":
                canonical_positive = bool(
                    canonical.node_features[
                        node_index,
                        V2_FEATURE_INDEX["process_event_count_now"],
                    ]
                    or canonical.node_features[
                        node_index,
                        V2_FEATURE_INDEX["connection_event_count_now"],
                    ]
                )
            else:
                canonical_positive = bool(
                    canonical.node_features[
                        node_index,
                        FEATURE_INDEX["process_event_now"],
                    ]
                    or canonical.node_features[
                        node_index,
                        FEATURE_INDEX["connection_event_now"],
                    ]
                )
            old_proc, current_proc, old_conn, current_conn = _queue_counts(
                state.hosts[hostname].events
            )
            old_positive = old_proc + old_conn > 0
            current_positive = current_proc + current_conn > 0
            union_positive = old_positive or current_positive

            counters["host_ticks"] += 1
            counters["parsed_raw_positive"] += int(parsed_raw_positive)
            counters["canonical_positive"] += int(canonical_positive)
            counters["old_positive"] += int(old_positive)
            counters["current_positive"] += int(current_positive)
            counters["state_union_positive"] += int(union_positive)
            counters["canonical_and_union"] += int(
                canonical_positive and union_positive
            )
            counters["canonical_only"] += int(
                canonical_positive and not union_positive
            )
            counters["union_only"] += int(
                union_positive and not canonical_positive
            )
            counters["current_only"] += int(
                current_positive and not old_positive and not canonical_positive
            )

            if canonical_positive:
                relation = (
                    "target"
                    if action_target == hostname
                    else "non_target"
                )
                boundary = (
                    "canonical_and_state"
                    if union_positive
                    else "canonical_only"
                )
                by_action[f"{boundary}|{action_type}:{relation}"] += 1


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def run_audit(args: argparse.Namespace) -> dict:
    seed_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)
    policy = _build_policy(config)
    checkpoint_path = args.checkpoint_path or seed_dir / "checkpoint.pt"
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
    env = CC4StructuredActionEnv(canonical_env)

    torch.manual_seed(args.action_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.action_seed)

    counters: Counter = Counter()
    by_action: Counter = Counter()
    returns = []
    for episode in range(args.episodes):
        observations, infos = env.reset(seed=args.eval_seed + episode)
        episode_return = 0.0
        while True:
            _record_boundary(canonical_env, counters, by_action)
            actions = {}
            for agent in env.possible_agents:
                if not infos[agent]["decision_ready"]:
                    continue
                observation = torch.as_tensor(
                    observations[agent],
                    dtype=torch.float32,
                    device=device,
                )
                action_mask = torch.as_tensor(
                    infos[agent]["action_mask"],
                    dtype=torch.bool,
                    device=device,
                )
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(
                        observation,
                        action_mask=action_mask,
                    )
                actions[agent] = int(action.item())

            observations, rewards, terminated, truncated, infos = env.step(actions)
            episode_return += float(next(iter(rewards.values())))
            if all(
                terminated.get(agent, False) or truncated.get(agent, False)
                for agent in env.possible_agents
            ):
                break
        returns.append(episode_return)

    host_ticks = counters["host_ticks"]
    union_positive = counters["state_union_positive"]
    payload = {
        "protocol": {
            "name": "cc4_observation_boundary_audit",
            "training_seed": args.seed,
            "episodes": args.episodes,
            "eval_seed": args.eval_seed,
            "action_seed": args.action_seed,
            "checkpoint_path": str(checkpoint_path),
            "policy_type": config["policy_type"],
            "canonical_schema": config.get("canonical_schema", "v1"),
            "environment_distribution": config["environment_distribution"],
            "policy_input": "strict raw combined Blue observation",
            "state_union_definition": (
                "old observed plus current not-yet-observed process/connection queues"
            ),
        },
        "evaluation": {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "episode_returns": returns,
        },
        "counts": dict(sorted(counters.items())),
        "rates": {
            "parsed_raw_positive_per_host_tick": _ratio(
                counters["parsed_raw_positive"], host_ticks
            ),
            "canonical_positive_per_host_tick": _ratio(
                counters["canonical_positive"], host_ticks
            ),
            "state_union_positive_per_host_tick": _ratio(
                union_positive, host_ticks
            ),
            "canonical_only_given_canonical_positive": _ratio(
                counters["canonical_only"], counters["canonical_positive"]
            ),
            "union_only_given_state_union_positive": _ratio(
                counters["union_only"], union_positive
            ),
            "current_only_given_state_union_positive": _ratio(
                counters["current_only"], union_positive
            ),
        },
        "canonical_positive_by_last_action_and_target_relation": dict(
            sorted(by_action.items())
        ),
        "interpretation_limits": [
            "State queue comparisons are diagnostics and are never policy inputs.",
            "Parsed raw events are reported separately from the canonical tensor actually given to the policy.",
            "This audit measures observation-boundary differences, not compromise truth.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_audit(args)
    print(json.dumps(payload["rates"], indent=2))
    print(
        json.dumps(
            payload["canonical_positive_by_last_action_and_target_relation"],
            indent=2,
        )
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
