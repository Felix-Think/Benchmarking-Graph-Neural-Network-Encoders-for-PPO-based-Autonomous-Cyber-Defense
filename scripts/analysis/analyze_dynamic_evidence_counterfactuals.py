from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from acd_benchmark.cyber.dynamic_evidence_counterfactual import (
    COUNTERFACTUAL_TRANSFORMS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze dynamic-evidence counterfactuals across training seeds."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--transforms",
        choices=COUNTERFACTUAL_TRANSFORMS,
        nargs="+",
        default=list(COUNTERFACTUAL_TRANSFORMS),
    )
    parser.add_argument("--time-shift-lag", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _counterfactual_name(transform: str, lag: int) -> str:
    if transform == "time_shift":
        return f"counterfactual_time_shift_lag{lag}_stochastic.json"
    return f"counterfactual_{transform}_stochastic.json"


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
        "positive_zero_negative": [
            int(np.count_nonzero(array > 0)),
            int(np.count_nonzero(array == 0)),
            int(np.count_nonzero(array < 0)),
        ],
        "values": array.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    transforms = tuple(dict.fromkeys(args.transforms))
    effects = {transform: [] for transform in transforms}
    per_seed = {}

    for seed in args.seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        baseline = _load(seed_dir / "evaluation_stochastic.json")
        baseline_returns = np.asarray(
            baseline["episode_returns"],
            dtype=np.float64,
        )
        rows = {
            "full": {
                "mean_return": float(baseline_returns.mean()),
                "std_return": float(baseline_returns.std(ddof=1)),
            }
        }
        for transform in transforms:
            record = _load(
                seed_dir / _counterfactual_name(transform, args.time_shift_lag)
            )
            evaluation = record["evaluation"]
            returns = np.asarray(evaluation["episode_returns"], dtype=np.float64)
            if returns.shape != baseline_returns.shape:
                raise ValueError(
                    f"Episode count mismatch for seed={seed}, transform={transform}"
                )
            effect = float((baseline_returns - returns).mean())
            effects[transform].append(effect)
            rows[transform] = {
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std(ddof=1)),
                "full_minus_counterfactual": effect,
            }
        per_seed[str(seed)] = rows

    multi_seed = {
        f"full_minus_{transform}": _summary(values)
        for transform, values in effects.items()
    }
    gates = {}
    for transform in transforms:
        summary = multi_seed[f"full_minus_{transform}"]
        interval = summary["student_t_ci95"]
        gates[f"uses_{transform}"] = bool(
            len(args.seeds) >= 5
            and interval is not None
            and interval[0] > 0
            and summary["positive_zero_negative"][0] >= 4
        )

    payload = {
        "protocol": {
            "primary_mode": "stochastic",
            "independent_inference_unit": "training seed",
            "training_seeds": args.seeds,
            "paired_environment_seed": 10_000,
            "paired_stochastic_action_seed": 10_000,
            "time_shift_lag": args.time_shift_lag,
            "transforms": list(transforms),
            "effect_definition": (
                "Full minus counterfactual; positive means correctly aligned "
                "dynamic evidence improved return"
            ),
        },
        "per_seed": per_seed,
        "multi_seed": multi_seed,
        "reliance_gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    seed_rows = []
    for seed, rows in per_seed.items():
        full = rows["full"]["mean_return"]
        for transform in transforms:
            row = rows[transform]
            seed_rows.append(
                f"| {seed} | `{transform}` | {full:.2f} | "
                f"{row['mean_return']:.2f} | "
                f"{row['full_minus_counterfactual']:+.2f} |"
            )
    effect_rows = []
    for name, summary in multi_seed.items():
        effect_rows.append(
            f"| `{name}` | {summary['mean']:+.2f} | "
            f"{summary['student_t_ci95']} | "
            f"{summary['positive_zero_negative']} |"
        )
    report = f"""# Dynamic Evidence Counterfactuals

## Protocol

- The trained `Full` checkpoint is held fixed within each training seed.
- Evaluation uses stochastic policy sampling.
- Environment and stochastic-action seeds are paired across transforms.
- Reward, official action mask, policy parameters and scenario distribution are unchanged.
- The training seed is the independent inference unit.

## Per-Seed Results

| Seed | Transform | Full | Counterfactual | Full - counterfactual |
|---:|---|---:|---:|---:|
{chr(10).join(seed_rows)}

## Across Seeds

| Effect | Mean | 95% Student-t CI | Positive / zero / negative |
|---|---:|---|---|
{chr(10).join(effect_rows)}

Positive effects mean that preserving the correct dynamic evidence improved
return. Episode-level differences are descriptive; the confidence interval is
computed across independently trained policies.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(multi_seed, indent=2))
    print(json.dumps(gates, indent=2))
    print(f"json={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
