import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate canonical qualification without pooling episodes.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_record(
    seed: int,
    statistics: dict[str, Any],
    comparison_name: str,
) -> dict[str, Any]:
    comparison = statistics["paired_comparisons"][comparison_name]
    delta = float(comparison["mean_delta"])
    ci95 = [float(value) for value in comparison["bootstrap_ci95"]]
    return {
        "seed": seed,
        "mean_delta": delta,
        "bootstrap_ci95": ci95,
        "point_estimate_positive": delta > 0,
        "ci_gate_pass": ci95[0] > 0,
    }


def _aggregate_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray([record["mean_delta"] for record in records])
    return {
        "per_seed": records,
        "positive_point_estimates": sum(
            record["point_estimate_positive"] for record in records
        ),
        "ci_gate_passes": sum(record["ci_gate_pass"] for record in records),
        "mean_seed_delta": round(float(deltas.mean()), 2),
        "median_seed_delta": round(float(np.median(deltas)), 2),
        "seed_delta_range": [
            round(float(deltas.min()), 2),
            round(float(deltas.max()), 2),
        ],
    }


def main() -> None:
    args = parse_args()
    statistics_by_seed = {
        seed: _load(args.run_dir / f"seed_{seed}" / "qualification_statistics.json")
        for seed in args.seeds
    }

    learning = _aggregate_gate(
        [
            _gate_record(seed, statistics, "best_minus_untrained")
            for seed, statistics in statistics_by_seed.items()
        ],
    )
    observation_use = _aggregate_gate(
        [
            _gate_record(seed, statistics, "best_minus_static_only")
            for seed, statistics in statistics_by_seed.items()
        ],
    )

    seed_count = len(args.seeds)
    minimum_passes = seed_count // 2 + 1
    all_positive = (
        learning["positive_point_estimates"] == seed_count
        and observation_use["positive_point_estimates"] == seed_count
    )
    enough_ci_passes = (
        learning["ci_gate_passes"] >= minimum_passes
        and observation_use["ci_gate_passes"] >= minimum_passes
    )
    too_many_nonpositive = (
        seed_count - learning["positive_point_estimates"] >= minimum_passes
        or seed_count - observation_use["positive_point_estimates"]
        >= minimum_passes
    )
    if all_positive and enough_ci_passes:
        decision = "replicated"
    elif too_many_nonpositive:
        decision = "not_replicated"
    else:
        decision = "mixed"

    deterministic_results = []
    for seed, statistics in statistics_by_seed.items():
        held_out = statistics["deterministic_held_out"]
        deterministic_results.append(
            {
                "seed": seed,
                "untrained_mean": held_out["untrained"]["mean"],
                "static_only_mean": held_out["static_only"]["mean"],
                "best_mean": held_out["best"]["mean"],
                "final_mean": held_out["final"]["mean"],
            },
        )

    payload = {
        "protocol": {
            "training_seeds": args.seeds,
            "replication_unit": "training_seed",
            "episodes_are_not_pooled_across_training_seeds": True,
            "replicated_rule": (
                "all point estimates positive and at least a majority of seeds "
                "pass each paired-bootstrap CI gate"
            ),
            "not_replicated_rule": (
                "a majority of seeds have non-positive point estimates for "
                "either gate"
            ),
            "otherwise": "mixed",
        },
        "deterministic_held_out_by_seed": deterministic_results,
        "learning_gate": learning,
        "dynamic_observation_use_gate": observation_use,
        "decision": decision,
        "claim_boundary": (
            "The action-aware canonical representation is learnable in two of "
            "three PPO training seeds, but qualification is not stable enough "
            "to claim replicated learning across seeds."
        ),
    }
    (args.run_dir / "REPLICATION_SUMMARY.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for result, learning_record, use_record in zip(
        deterministic_results,
        learning["per_seed"],
        observation_use["per_seed"],
        strict=True,
    ):
        rows.append(
            "| {seed} | {untrained_mean:.2f} | {static_only_mean:.2f} | "
            "{best_mean:.2f} | {learning_delta:+.2f} | {learning_gate} | "
            "{use_delta:+.2f} | {use_gate} |".format(
                **result,
                learning_delta=learning_record["mean_delta"],
                learning_gate="PASS" if learning_record["ci_gate_pass"] else "FAIL",
                use_delta=use_record["mean_delta"],
                use_gate="PASS" if use_record["ci_gate_pass"] else "FAIL",
            ),
        )

    summary = f"""# Canonical observation replication summary

## Decision

**{decision.upper()}** under the protocol frozen before seeds 43 and 44.

| Training seed | Untrained | Static-only | Best canonical | Best - untrained | Learning gate | Best - static | Observation-use gate |
|---:|---:|---:|---:|---:|:---:|---:|:---:|
{chr(10).join(rows)}

## Seed-level aggregation

- Learning: {learning['positive_point_estimates']}/{seed_count} positive point estimates and {learning['ci_gate_passes']}/{seed_count} CI-gate passes; mean seed delta {learning['mean_seed_delta']:+.2f}, range {learning['seed_delta_range']}.
- Dynamic observation use: {observation_use['positive_point_estimates']}/{seed_count} positive point estimates and {observation_use['ci_gate_passes']}/{seed_count} CI-gate passes; mean seed delta {observation_use['mean_seed_delta']:+.2f}, range {observation_use['seed_delta_range']}.
- Episodes were paired within each training seed. The 300 held-out episodes were not pooled and treated as independent replication units.

## Claim boundary

The action-aware canonical representation is learnable and its dynamic evidence is useful in seeds 42 and 44. Seed 43 fails both gates. Therefore the result supports feasibility, but not stable multi-seed qualification. It is not yet sufficient to start a definitive Flat-versus-GNN benchmark without addressing or explicitly accepting this optimization variance.
"""
    (args.run_dir / "REPLICATION_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
