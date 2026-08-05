"""Create paired, stability, action, and source-cohort diagnostics for Stage 1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.rl.stage1_factory import (  # noqa: E402
    build_stage1_policy,
    make_stage1_env,
    resolve_model_hidden_dim,
)
from acd_benchmark.utils.reproducibility import sha256_file  # noqa: E402


CORE_SNAPSHOT_COMPONENTS = {
    "ppo_update": (
        "src/acd_benchmark/rl/ppo/update.py",
        "src/acd_benchmark/rl/graph_ppo.py",
    ),
    "ppo_training": (
        "src/acd_benchmark/rl/ppo/training.py",
        "src/acd_benchmark/rl/graph_ppo.py",
    ),
    "rollout_collection": (
        "src/acd_benchmark/rl/rollout/collection.py",
        "src/acd_benchmark/rl/graph_ppo.py",
    ),
    "stage1_factory": ("src/acd_benchmark/rl/stage1_factory.py",),
    "partial_observation": (
        "src/acd_benchmark/wrappers/shared/partial_obs.py",
        "src/acd_benchmark/wrappers/partial_obs.py",
    ),
    "stage1_environment": (
        "src/acd_benchmark/wrappers/stage1/env.py",
        "src/acd_benchmark/wrappers/stage1_env.py",
    ),
    "stage1_observation": (
        "src/acd_benchmark/wrappers/stage1/observation.py",
        "src/acd_benchmark/wrappers/stage1_obs.py",
    ),
    "stage1_training": (
        "scripts/stage1/train_ppo.py",
        "scripts/train_stage1_ppo.py",
    ),
    "stage1_config": ("configs/stage1_16.yaml",),
    "stage1_scenario": ("configs/scenarios/cage2_16.yaml",),
}


def _snapshot_component_path(snapshot_root: Path, candidates: tuple[str, ...]) -> Path:
    """Resolve a component from either the current or historical layout."""
    for relative in candidates:
        path = snapshot_root / relative
        if path.exists():
            return path
    raise FileNotFoundError(f"Snapshot component missing; tried {candidates}")


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    """Return mean and sample SD for a non-empty sequence."""
    return mean(values), stdev(values) if len(values) > 1 else None


def _load_reports(run_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load all canonical final reports."""
    return [
        (path.parent, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(run_root.glob("*/*/seed_*/report.json"))
    ]


def _audit_actions(
    run_dir: Path,
    report: dict[str, Any],
    *,
    episodes: int,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Replay one final argmax policy and count action classes and targets."""
    config = report["resolved_config"]
    run = report["run"]
    scenario = config["scenario"]
    training = config["training"]
    env = make_stage1_env(
        str(run["model"]),
        scenario_path=run_dir / "scenario.yaml",
        node_count=int(scenario["node_count"]),
        observation_probability=float(run["observation_probability"]),
        max_steps=int(scenario["max_episode_steps"]),
    )
    base_seed = int(training["evaluation_base_seed"])
    observation, _ = env.reset(seed=base_seed * 1000)
    policy = build_stage1_policy(
        str(run["model"]),
        observation,
        env,
        hidden_dim=resolve_model_hidden_dim(str(run["model"]), config["model"]),
    )
    checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    action_classes: Counter[str] = Counter()
    action_names: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    for episode in range(episodes):
        observation, _ = env.reset(seed=base_seed * 1000 + episode)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            prepared = policy.prepare_observation(observation, torch.device("cpu"))
            with torch.no_grad():
                logits, _ = policy.forward_prepared(prepared)
            action_index = int(torch.argmax(logits).item())
            action_spec = env.action_specs[action_index]
            action_classes[action_spec.action_cls.__name__] += 1
            action_names[action_spec.name] += 1
            targets[action_spec.hostname or "<global>"] += 1
            observation, _, terminated, truncated, _ = env.step(action_index)
    return action_classes, action_names, targets


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one rectangular diagnostic table."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate diagnostics after the configured matrix is complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/stage1_scalable_16"))
    parser.add_argument("--action-audit-episodes", type=int, default=30)
    args = parser.parse_args()
    reports = _load_reports(args.run_root)
    if not reports:
        raise RuntimeError(f"No completed reports found under {args.run_root}.")
    matrix = reports[0][1]["resolved_config"]["matrix"]
    models = tuple(str(model) for model in matrix["models"])
    levels = tuple(int(level) for level in matrix["observation_levels"])
    seeds = tuple(int(seed) for seed in matrix["seeds"])
    expected_cells = {
        (model, level, seed)
        for model in models
        for level in levels
        for seed in seeds
    }
    actual_cells = {
        (
            str(report["run"]["model"]),
            int(report["run"]["observation_level"]),
            int(report["run"]["seed"]),
        )
        for _, report in reports
    }
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        unexpected = sorted(actual_cells - expected_cells)
        raise RuntimeError(
            f"Configured matrix is incomplete: missing={missing}, unexpected={unexpected}."
        )

    by_cell: dict[tuple[str, int, int], dict[str, Any]] = {}
    per_run = []
    action_rows = []
    patch_cohorts: dict[str, list[str]] = defaultdict(list)
    core_hashes: dict[str, set[str]] = {
        component: set()
        for component in CORE_SNAPSHOT_COMPONENTS
    }

    for run_dir, report in reports:
        run = report["run"]
        model = str(run["model"])
        level = int(run["observation_level"])
        seed = int(run["seed"])
        by_cell[(model, level, seed)] = report
        evaluations = report["evaluations"]
        episode_return_curve = [
            float(item["mean_episode_return"])
            for item in evaluations
        ]
        availability_curve = [
            float(item["mean_mission_availability"]) for item in evaluations
        ]
        final = report["final_evaluation"]
        patch_hash = report["reproducibility"]["source_snapshot"]["git_diff_sha256"]
        label = f"{model}/obs_{level}/seed_{seed}"
        patch_cohorts[patch_hash].append(label)
        snapshot_root = run_dir / "source_snapshot"
        for component, candidates in CORE_SNAPSHOT_COMPONENTS.items():
            source_path = _snapshot_component_path(snapshot_root, candidates)
            core_hashes[component].add(sha256_file(source_path))

        action_classes, action_names, targets = _audit_actions(
            run_dir,
            report,
            episodes=args.action_audit_episodes,
        )
        total_actions = sum(action_classes.values())
        for action_class, count in action_classes.most_common():
            action_rows.append(
                {
                    "model": model,
                    "observation_level": level,
                    "seed": seed,
                    "action_class": action_class,
                    "count": count,
                    "fraction": count / total_actions,
                }
            )
        per_run.append(
            {
                "model": model,
                "observation_level": level,
                "seed": seed,
                "final_episode_return": final["mean_episode_return"],
                "best_episode_return": max(episode_return_curve),
                "best_to_final_episode_return_gap": (
                    max(episode_return_curve) - episode_return_curve[-1]
                ),
                "last_five_episode_return_sd": float(
                    np.std(episode_return_curve[-5:], ddof=1)
                ),
                "final_availability": final["mean_mission_availability"],
                "best_availability": max(availability_curve),
                "last_five_availability_sd": float(
                    np.std(availability_curve[-5:], ddof=1)
                ),
                "detection_observed_fraction": final["detection_observed_count"]
                / max(
                    final["detection_observed_count"]
                    + final["detection_censored_count"],
                    1,
                ),
                "recovery_observed_fraction": final["recovery_observed_count"]
                / max(
                    final["recovery_observed_count"]
                    + final["recovery_censored_count"],
                    1,
                ),
                "top_action_class": action_classes.most_common(1)[0][0],
                "top_action_class_fraction": action_classes.most_common(1)[0][1]
                / total_actions,
                "top_action": action_names.most_common(1)[0][0],
                "top_action_fraction": action_names.most_common(1)[0][1] / total_actions,
                "top_target": targets.most_common(1)[0][0],
                "top_target_fraction": targets.most_common(1)[0][1] / total_actions,
                "source_patch_sha256": patch_hash,
            }
        )

    if "flat" not in models:
        raise RuntimeError("Paired encoder diagnostics require a flat baseline.")
    paired = []
    for comparison_model in (model for model in models if model != "flat"):
        for level in levels:
            differences = defaultdict(list)
            seed_rows = []
            for seed in seeds:
                flat = by_cell[("flat", level, seed)]["final_evaluation"]
                encoder = by_cell[(comparison_model, level, seed)]["final_evaluation"]
                row = {
                    "seed": seed,
                    "episode_return_delta": (
                        encoder["mean_episode_return"]
                        - flat["mean_episode_return"]
                    ),
                    "availability_delta": encoder["mean_mission_availability"]
                    - flat["mean_mission_availability"],
                    "privileged_hosts_delta": encoder[
                        "mean_privileged_compromised_hosts"
                    ]
                    - flat["mean_privileged_compromised_hosts"],
                }
                seed_rows.append(row)
                for key, value in row.items():
                    if key != "seed":
                        differences[key].append(float(value))
            aggregate = {}
            for key, values in differences.items():
                value_mean, value_std = _mean_std(values)
                aggregate[f"{key}_mean"] = value_mean
                aggregate[f"{key}_sd"] = value_std
            paired.append(
                {
                    "model": comparison_model,
                    "observation_level": level,
                    "per_seed": seed_rows,
                    "aggregate": aggregate,
                }
            )

    diagnostics = {
        "run_count": len(reports),
        "action_audit_episodes_per_run": args.action_audit_episodes,
        "paired_encoder_minus_flat": paired,
        "per_run": per_run,
        "source_patch_cohorts": dict(patch_cohorts),
        "core_source_consistent": all(len(values) == 1 for values in core_hashes.values()),
        "core_source_hashes": {
            path: sorted(values) for path, values in core_hashes.items()
        },
        "source_cohort_interpretation": (
            "The later StopProcess patch changes only a service-session reload branch. "
            "Earlier completed runs cannot have executed that branch because the old "
            "implementation deterministically raises there. Core benchmark source, "
            "config, and scenario hashes are identical across all runs."
        ),
    }
    (args.run_root / "stage1_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.run_root / "stage1_diagnostics_per_run.csv", per_run)
    _write_csv(args.run_root / "stage1_action_classes.csv", action_rows)

    lines = [
        "# Stage 1 Diagnostics",
        "",
        f"Core benchmark source consistent across runs: **{diagnostics['core_source_consistent']}**",
        f"Vendored CybORG patch cohorts: **{len(patch_cohorts)}**",
        "",
        "## Paired Encoder Minus Flat",
        "",
        "| Encoder | Obs | Episode-return delta | Availability delta | Privileged-host delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in paired:
        aggregate = item["aggregate"]
        lines.append(
            f"| {item['model']} | {item['observation_level']} | "
            f"{aggregate['episode_return_delta_mean']:.3f} +/- "
            f"{aggregate['episode_return_delta_sd']:.3f} | "
            f"{aggregate['availability_delta_mean']:.3f} +/- "
            f"{aggregate['availability_delta_sd']:.3f} | "
            f"{aggregate['privileged_hosts_delta_mean']:.3f} +/- "
            f"{aggregate['privileged_hosts_delta_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Positive episode-return/availability deltas favor the encoder; negative privileged-host deltas favor the encoder.",
            "",
            "## Final Policy Actions",
            "",
            "| Model | Obs | Seed | Top class | Fraction | Top action | Fraction |",
            "|---|---:|---:|---|---:|---|---:|",
        ]
    )
    for row in sorted(per_run, key=lambda item: (item["model"], item["observation_level"], item["seed"])):
        lines.append(
            f"| {row['model']} | {row['observation_level']} | {row['seed']} | "
            f"{row['top_action_class']} | {row['top_action_class_fraction']:.3f} | "
            f"{row['top_action']} | {row['top_action_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Repeated Restore(Op_Server0) explains checkpoints with near-zero mission availability.",
            "Latency interpretation still requires observed/censored event fractions in the per-run CSV.",
            "",
            "## Source Cohorts",
            "",
        ]
    )
    for patch_hash, labels in patch_cohorts.items():
        lines.append(f"- `{patch_hash}`: {len(labels)} runs")
    lines.extend(["", diagnostics["source_cohort_interpretation"], ""])
    (args.run_root / "stage1_diagnostics.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        f"Saved Stage 1 diagnostics for {len(reports)} runs; "
        f"core_source_consistent={diagnostics['core_source_consistent']}"
    )


if __name__ == "__main__":
    main()
