from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze matched observation counterfactual evaluations."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "stochastic"),
        default="stochastic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical_observation_counterfactuals.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/CANONICAL_OBSERVATION_COUNTERFACTUALS.md"),
    )
    return parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _t_interval(values: np.ndarray) -> list[float] | None:
    if values.size < 2:
        return None
    standard_error = stats.sem(values)
    if standard_error == 0:
        mean = float(values.mean())
        return [mean, mean]
    margin = float(stats.t.ppf(0.975, values.size - 1) * standard_error)
    mean = float(values.mean())
    return [mean - margin, mean + margin]


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "student_t_ci95": _t_interval(array),
        "values": array.tolist(),
    }


def _action_family_profile(action_counts: dict[str, int]) -> dict[str, dict]:
    family_counts: dict[str, int] = defaultdict(int)
    for label, count in action_counts.items():
        action = label.split(": ", maxsplit=1)[-1].split(maxsplit=1)[0]
        family_counts[action] += int(count)
    total = sum(family_counts.values())
    return {
        action: {
            "count": count,
            "fraction": count / total if total else 0.0,
        }
        for action, count in sorted(family_counts.items())
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    conditions = tuple(VALID_OBSERVATION_CONDITIONS)
    per_seed = {}
    effects = {condition: [] for condition in conditions if condition != "full"}

    for seed in args.seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        evaluations = {
            condition: _load_json(
                seed_dir / f"observation_gate_{condition}_{args.mode}.json"
            )
            for condition in conditions
        }
        full_returns = np.asarray(
            evaluations["full"]["episode_returns"], dtype=np.float64
        )
        rows = {}
        for condition, evaluation in evaluations.items():
            returns = np.asarray(evaluation["episode_returns"], dtype=np.float64)
            if returns.shape != full_returns.shape:
                raise ValueError(
                    f"Evaluation shape differs for seed={seed}, "
                    f"condition={condition}"
                )
            full_minus_condition = float((full_returns - returns).mean())
            rows[condition] = {
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std(ddof=1)),
                "full_minus_condition": full_minus_condition,
                "action_family_profile": _action_family_profile(
                    evaluation.get("action_counts", {})
                ),
            }
            if condition != "full":
                effects[condition].append(full_minus_condition)
        per_seed[str(seed)] = rows

    multi_seed = {
        f"full_minus_{condition}": _summary(values)
        for condition, values in effects.items()
    }
    payload = {
        "protocol": {
            "mode": args.mode,
            "independent_inference_unit": "training seed",
            "paired_environment_and_action_seeds": True,
            "conditions": list(conditions),
            "effect_definition": (
                "Full minus transformed return; positive means Full is better"
            ),
        },
        "per_seed": per_seed,
        "multi_seed": multi_seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table_rows = []
    for seed, rows in per_seed.items():
        full_mean = rows["full"]["mean_return"]
        for condition in conditions:
            row = rows[condition]
            table_rows.append(
                f"| {seed} | `{condition}` | {row['mean_return']:.2f} | "
                f"{full_mean - row['mean_return']:+.2f} |"
            )

    effect_rows = []
    for name, summary in multi_seed.items():
        effect_rows.append(
            f"| `{name}` | {summary['mean']:+.2f} | "
            f"{summary['student_t_ci95']} | {summary['n']} |"
        )

    report = f"""# Canonical Observation Counterfactuals

## Protocol

- Evaluation mode: `{args.mode}`.
- The trained `Full` checkpoint is held fixed.
- Environment seeds and stochastic action seeds are paired across conditions.
- Observation dimension, PPO policy, native reward and official action mask are unchanged.
- Training seed is the independent inference unit; episode-level differences are descriptive.

## Held-Out Results

| Training seed | Condition | Mean return | Full - condition |
|---:|---|---:|---:|
{chr(10).join(table_rows)}

## Across Training Seeds

| Effect | Mean | 95% Student-t CI | Seeds |
|---|---:|---|---:|
{chr(10).join(effect_rows)}

Positive `Full - condition` means the information removed by that condition was
useful to the fixed trained policy. A single-seed result is diagnostic only and
cannot establish a paper-level claim.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(multi_seed, indent=2))
    print(f"json={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
