from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats


CONTROLLED_FIELDS = (
    "policy_type",
    "canonical_schema",
    "canonical_variant",
    "canonical_parser_version",
    "passive_memory_ttl",
    "analysis_memory_ttl",
    "environment_distribution",
    "semantic_action_mask",
    "episode_steps",
    "episodes_per_update",
    "total_simulator_steps",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "minibatch_size",
    "reward_scale",
    "clip_coef",
    "value_coef",
    "entropy_coef",
    "max_grad_norm",
    "target_kl",
    "encoder_hidden_dim",
    "latent_dim",
    "head_hidden_dim",
    "observation_dim",
    "num_actions",
    "parameter_count",
    "action_catalog",
    "node_hidden_dim",
    "node_embedding_dim",
    "context_dim",
    "eval_episodes",
    "eval_seed",
    "runtime_seed_controls",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze matched Full versus No-evidence training runs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm-prefix", type=str, default="flat")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical_observation_qualification.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/CANONICAL_OBSERVATION_QUALIFICATION.md"),
    )
    parser.add_argument(
        "--plateau-slope-threshold",
        type=float,
        default=50.0,
        help="Maximum absolute late return slope per 100k simulator ticks.",
    )
    return parser


def _load_json(path: Path) -> dict | list:
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
        "positive_zero_negative": [
            int(np.count_nonzero(array > 0)),
            int(np.count_nonzero(array == 0)),
            int(np.count_nonzero(array < 0)),
        ],
        "values": array.tolist(),
    }


def _late_periodic_slope(periodic: list[dict]) -> float | None:
    if len(periodic) < 3:
        return None
    late = periodic[-3:]
    x = np.asarray([row["global_step"] for row in late], dtype=np.float64)
    y = np.asarray([row["mean_return"] for row in late], dtype=np.float64)
    if np.unique(x).size < 2:
        return None
    return float(np.polyfit(x / 100_000.0, y, deg=1)[0])


def _training_gain(checkpoint: dict) -> float:
    returns = np.asarray(checkpoint["episode_returns"], dtype=np.float64)
    window = max(1, math.ceil(returns.size * 0.2))
    return float(returns[-window:].mean() - returns[:window].mean())


def _finite_training(checkpoint: dict) -> bool:
    for row in checkpoint.get("history", []):
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def _load_arm(
    run_root: Path,
    arm_prefix: str,
    condition: str,
    seed: int,
) -> dict:
    seed_dir = run_root / f"{arm_prefix}_{condition}" / f"seed_{seed}"
    config = _load_json(seed_dir / "config.json")
    evaluation = _load_json(seed_dir / "evaluation_stochastic.json")
    checkpoint = torch.load(seed_dir / "checkpoint.pt", map_location="cpu")
    periodic_path = seed_dir / "periodic_evaluation.json"
    periodic = _load_json(periodic_path) if periodic_path.exists() else []
    return {
        "seed_dir": seed_dir,
        "config": config,
        "evaluation": evaluation,
        "checkpoint": checkpoint,
        "periodic": periodic,
    }


def _audit_controls(full: dict, no_evidence: dict) -> dict:
    differences = {}
    for field in CONTROLLED_FIELDS:
        left = full["config"].get(field)
        right = no_evidence["config"].get(field)
        if left != right:
            differences[field] = {"full": left, "no_evidence": right}
    return {
        "passed": not differences,
        "expected_difference": {
            "observation_condition": {
                "full": full["config"].get("observation_condition"),
                "no_evidence": no_evidence["config"].get(
                    "observation_condition"
                ),
            }
        },
        "unexpected_differences": differences,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    per_seed = {}
    evidence_effects = []
    full_learning_effects = []
    full_training_gains = []
    late_slopes = []
    control_audits = []
    finite_runs = []

    for seed in args.seeds:
        full = _load_arm(args.run_root, args.arm_prefix, "full", seed)
        no_evidence = _load_arm(
            args.run_root,
            args.arm_prefix,
            "no_evidence",
            seed,
        )
        full_returns = np.asarray(
            full["evaluation"]["episode_returns"], dtype=np.float64
        )
        no_evidence_returns = np.asarray(
            no_evidence["evaluation"]["episode_returns"], dtype=np.float64
        )
        if full_returns.shape != no_evidence_returns.shape:
            raise ValueError(f"Evaluation shapes differ for seed {seed}")

        untrained_path = full["seed_dir"] / "untrained_full_stochastic.json"
        untrained = _load_json(untrained_path)
        untrained_returns = np.asarray(
            untrained["episode_returns"], dtype=np.float64
        )
        if full_returns.shape != untrained_returns.shape:
            raise ValueError(f"Untrained evaluation shape differs for seed {seed}")

        evidence_effect = float((full_returns - no_evidence_returns).mean())
        learning_effect = float((full_returns - untrained_returns).mean())
        training_gain = _training_gain(full["checkpoint"])
        late_slope = _late_periodic_slope(full["periodic"])
        control = _audit_controls(full, no_evidence)
        finite = _finite_training(full["checkpoint"]) and _finite_training(
            no_evidence["checkpoint"]
        )

        evidence_effects.append(evidence_effect)
        full_learning_effects.append(learning_effect)
        full_training_gains.append(training_gain)
        if late_slope is not None:
            late_slopes.append(late_slope)
        control_audits.append(control["passed"])
        finite_runs.append(finite)

        per_seed[str(seed)] = {
            "full_mean_return": float(full_returns.mean()),
            "no_evidence_mean_return": float(no_evidence_returns.mean()),
            "untrained_full_mean_return": float(untrained_returns.mean()),
            "full_minus_no_evidence": evidence_effect,
            "full_minus_untrained": learning_effect,
            "full_training_last20_minus_first20": training_gain,
            "full_late_periodic_slope_per_100k": late_slope,
            "finite_training_metrics": finite,
            "controlled_config_audit": control,
        }

    evidence = _summary(evidence_effects)
    learning = _summary(full_learning_effects)
    training_gain = _summary(full_training_gains)
    slopes = _summary(late_slopes) if late_slopes else None

    evidence_ci = evidence["student_t_ci95"]
    learning_ci = learning["student_t_ci95"]
    gain_ci = training_gain["student_t_ci95"]
    slope_ci = slopes["student_t_ci95"] if slopes is not None else None
    enough_seeds = len(args.seeds) >= 5
    evidence_gate = bool(
        enough_seeds
        and evidence_ci is not None
        and evidence_ci[0] > 0
        and evidence["positive_zero_negative"][0] >= 4
    )
    learning_gate = bool(
        enough_seeds
        and learning_ci is not None
        and learning_ci[0] > 0
        and learning["positive_zero_negative"][0] >= 4
    )
    training_gain_gate = bool(
        enough_seeds
        and gain_ci is not None
        and gain_ci[0] > 0
        and training_gain["positive_zero_negative"][0] >= 4
    )
    plateau_gate = bool(
        enough_seeds
        and slopes is not None
        and slope_ci is not None
        and slope_ci[0] <= 0 <= slope_ci[1]
        and abs(slopes["mean"]) <= args.plateau_slope_threshold
    )
    integrity_gate = all(control_audits) and all(finite_runs)

    payload = {
        "protocol": {
            "primary_mode": "stochastic",
            "independent_unit": "training seed",
            "training_seeds": args.seeds,
            "evaluation_seed_start": 10000,
            "conditions": ["full", "no_evidence"],
            "arm_prefix": args.arm_prefix,
            "full_minus_no_evidence_definition": (
                "positive means sensor evidence improved held-out return"
            ),
            "plateau_slope_threshold_per_100k": args.plateau_slope_threshold,
        },
        "per_seed": per_seed,
        "multi_seed": {
            "full_minus_no_evidence": evidence,
            "full_minus_untrained": learning,
            "full_training_last20_minus_first20": training_gain,
            "full_late_periodic_slope_per_100k": slopes,
        },
        "qualification_gates": {
            "at_least_five_training_seeds": enough_seeds,
            "controlled_and_finite": integrity_gate,
            "trained_full_beats_untrained": learning_gate,
            "full_training_return_improves": training_gain_gate,
            "full_beats_trained_no_evidence": evidence_gate,
            "late_validation_curve_is_flat": plateau_gate,
            "ready_for_encoder_benchmark": bool(
                integrity_gate
                and learning_gate
                and training_gain_gate
                and evidence_gate
                and plateau_gate
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    gates = payload["qualification_gates"]
    seed_rows = "\n".join(
        "| {seed} | {full:.2f} | {no_evidence:.2f} | {delta:+.2f} | "
        "{untrained:.2f} | {learning:+.2f} |".format(
            seed=seed,
            full=row["full_mean_return"],
            no_evidence=row["no_evidence_mean_return"],
            delta=row["full_minus_no_evidence"],
            untrained=row["untrained_full_mean_return"],
            learning=row["full_minus_untrained"],
        )
        for seed, row in per_seed.items()
    )
    report = f"""# Canonical V2 Observation Qualification

## Protocol

- Primary evaluation: stochastic policy sampling.
- Independent inference unit: training seed.
- Conditions are trained independently from initialization: `Full` and `No evidence`.
- PPO, native reward, official mask, architecture, parameter count and evaluation seeds are matched.
- A result is paper-level only with at least five training seeds.

## Held-Out Results

| Seed | Full | No evidence | Full - No evidence | Untrained Full | Full - untrained |
|---:|---:|---:|---:|---:|---:|
{seed_rows}

Multi-seed Full - No evidence: {evidence['mean']:+.2f}, 95% t-CI {evidence_ci}.

Multi-seed Full - untrained: {learning['mean']:+.2f}, 95% t-CI {learning_ci}.

## Qualification Gates

- Five training seeds: {gates['at_least_five_training_seeds']}.
- Controlled configuration and finite metrics: {gates['controlled_and_finite']}.
- Trained Full beats untrained: {gates['trained_full_beats_untrained']}.
- Full training return improves from first to last 20%: {gates['full_training_return_improves']}.
- Trained Full beats trained No evidence: {gates['full_beats_trained_no_evidence']}.
- Late validation curve satisfies the predeclared plateau diagnostic: {gates['late_validation_curve_is_flat']}.
- Ready for encoder benchmark: {gates['ready_for_encoder_benchmark']}.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(payload["multi_seed"], indent=2))
    print(json.dumps(gates, indent=2))
    print(f"json={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
