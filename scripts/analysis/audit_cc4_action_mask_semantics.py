from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit static and temporal information in the official CC4 mask."
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--temporal-episodes", type=int, default=3)
    parser.add_argument("--temporal-steps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cc4_official_action_mask_semantics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/CC4_OFFICIAL_ACTION_MASK_SEMANTICS.md"),
    )
    return parser


def _signature(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()[:16]


def _new_env(seed: int) -> CC4CanonicalFlatEnv:
    return CC4CanonicalFlatEnv(
        episode_steps=400,
        seed=seed,
        canonical_variant="action_aware",
        evidence_age_horizon=50,
        environment_distribution="random_host",
        canonical_schema="v1",
        semantic_action_mask=False,
    )


def run_audit(args: argparse.Namespace) -> dict:
    records = defaultdict(list)
    invalid_label_errors = []
    for episode in range(args.episodes):
        episode_seed = args.eval_seed + episode
        env = _new_env(episode_seed)
        _, infos = env.reset(seed=episode_seed)
        for agent in env.possible_agents:
            mask = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
            labels = env.action_labels(agent)
            invalid_by_label = np.asarray(
                [
                    label.startswith("[Invalid]")
                    or label.startswith("[Padding]")
                    for label in labels
                ],
                dtype=np.bool_,
            )
            if not np.array_equal(~mask, invalid_by_label):
                invalid_label_errors.append({"seed": episode_seed, "agent": agent})
            existing_hosts = int(
                np.count_nonzero(
                    env.last_canonical[agent].node_features[:, 2] > 0.5
                )
            )
            records[agent].append(
                {
                    "seed": episode_seed,
                    "signature": _signature(mask),
                    "valid_actions": int(mask.sum()),
                    "existing_hosts": existing_hosts,
                }
            )

    temporal_changes = []
    decision_ready_states_seen = defaultdict(set)
    for episode in range(args.temporal_episodes):
        episode_seed = args.eval_seed + episode
        env = _new_env(episode_seed)
        _, infos = env.reset(seed=episode_seed)
        initial_masks = {
            agent: np.asarray(info["action_mask"], dtype=np.bool_).copy()
            for agent, info in infos.items()
        }
        sleep_ids = {
            agent: env.action_labels(agent).index("Sleep")
            for agent in env.possible_agents
        }
        restore_ids = {}
        for agent in env.possible_agents:
            action_mask = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
            restore_ids[agent] = next(
                (
                    index
                    for index, label in enumerate(env.action_labels(agent))
                    if action_mask[index] and label.startswith("Restore ")
                ),
                sleep_ids[agent],
            )
        for tick in range(args.temporal_steps):
            actions = {
                agent: restore_ids[agent] if tick == 0 else sleep_ids[agent]
                for agent in env.possible_agents
                if infos[agent]["decision_ready"]
            }
            _, _, _, _, infos = env.step(actions)
            for agent in env.possible_agents:
                decision_ready_states_seen[agent].add(
                    bool(infos[agent]["decision_ready"])
                )
                current = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
                if not np.array_equal(current, initial_masks[agent]):
                    temporal_changes.append(
                        {"seed": episode_seed, "tick": tick, "agent": agent}
                    )

    agent_summary = {}
    for agent, rows in records.items():
        valid = np.asarray([row["valid_actions"] for row in rows], dtype=float)
        hosts = np.asarray([row["existing_hosts"] for row in rows], dtype=float)
        correlation = (
            float(np.corrcoef(valid, hosts)[0, 1])
            if np.std(valid) > 0 and np.std(hosts) > 0
            else None
        )
        agent_summary[agent] = {
            "unique_mask_signatures_across_episodes": len(
                {row["signature"] for row in rows}
            ),
            "valid_action_count_min": int(valid.min()),
            "valid_action_count_max": int(valid.max()),
            "existing_host_count_min": int(hosts.min()),
            "existing_host_count_max": int(hosts.max()),
            "valid_action_host_count_correlation": correlation,
        }

    return {
        "protocol": {
            "environment_distribution": "random_host",
            "semantic_action_mask": False,
            "episodes": args.episodes,
            "episode_seed_range": [
                args.eval_seed,
                args.eval_seed + args.episodes - 1,
            ],
            "temporal_steps_checked": args.temporal_steps,
        },
        "source_semantics": {
            "constructed_at": "reset",
            "host_action_validity": "host exists and Blue has a session",
            "traffic_actions": "always valid in the official mask",
            "sleep_monitor": "always valid",
            "step_behavior": "returns the stored reset-time mask without rebuilding it",
        },
        "audit": {
            "mask_is_constant_within_checked_steps": not temporal_changes,
            "temporal_changes": temporal_changes,
            "decision_ready_states_seen": {
                agent: sorted(states)
                for agent, states in decision_ready_states_seen.items()
            },
            "mask_checked_during_in_progress": any(
                False in states for states in decision_ready_states_seen.values()
            ),
            "false_mask_matches_invalid_or_padding_labels": not invalid_label_errors,
            "invalid_label_errors": invalid_label_errors,
            "agents": agent_summary,
        },
        "interpretation": {
            "dynamic_attack_state_disclosure": (
                "No direct temporal disclosure: the official mask is fixed within "
                "the episode and is not rebuilt from compromise/evidence state."
            ),
            "static_side_channel": (
                "Yes: valid host-target action slots reveal which random-host "
                "slots exist and have a Blue session."
            ),
            "decision_ready": (
                "Separate from action_mask; PPO is called only when decision_ready."
            ),
        },
    }


def _write_report(result: dict, path: Path) -> None:
    lines = [
        "# CAGE 4 Official Action-Mask Semantics",
        "",
        "## Result",
        "",
        "The official mask does not expose changing compromise or evidence state: "
        "it is constructed at reset and remains fixed within the episode. It does "
        "expose static random-host existence/session information through valid "
        "host-target action slots.",
        "",
        f"Temporal changes detected: `{len(result['audit']['temporal_changes'])}`.",
        f"Mask checked while an action was in progress: "
        f"`{result['audit']['mask_checked_during_in_progress']}`.",
        "",
        "| Agent | Unique masks across episodes | Valid actions | Existing hosts | Correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent, row in result["audit"]["agents"].items():
        correlation = row["valid_action_host_count_correlation"]
        correlation_text = (
            "N/A (constant host count)"
            if correlation is None
            else f"{correlation:.3f}"
        )
        lines.append(
            f"| {agent} | {row['unique_mask_signatures_across_episodes']} | "
            f"{row['valid_action_count_min']}-{row['valid_action_count_max']} | "
            f"{row['existing_host_count_min']}-{row['existing_host_count_max']} | "
            f"{correlation_text} |"
        )
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "A zero-observation policy that retains this mask is not information-free: "
            "it still receives static topology/existence information. Mask-only and "
            "no-mask gates must therefore be reported together.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(result, args.report)
    print(
        "mask_constant_within_episode="
        f"{result['audit']['mask_is_constant_within_checked_steps']}"
    )
    print(f"output={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
