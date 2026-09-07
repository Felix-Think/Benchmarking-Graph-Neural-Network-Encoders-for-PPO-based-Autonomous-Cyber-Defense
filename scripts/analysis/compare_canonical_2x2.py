from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats


PRIMARY_CONDITIONS = {
    "A": "A_v1_mask_off",
    "B": "B_v2_mask_off_matched",
    "C": "C_v1_mask_on",
    "D": "D_v2_mask_on_matched",
}

CONTROLLED_CONFIG_FIELDS = (
    "canonical_variant",
    "environment_distribution",
    "episode_steps",
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
    "latent_dim",
    "head_hidden_dim",
    "seed",
    "eval_episodes",
    "eval_seed",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the parameter-matched canonical 2x2 qualification."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/canonical_2x2_random_host_20k"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/canonical_2x2_random_host_20k_comparison.json"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_822)
    return parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _action_type_counts(action_counts: dict[str, int]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for key, count in action_counts.items():
        label = key.split(": ", maxsplit=1)[-1]
        result[label.split(maxsplit=1)[0]] += int(count)
    return dict(sorted(result.items()))


def _mean_last(history: list[dict], field: str, count: int = 10) -> float:
    values = [float(row[field]) for row in history[-count:] if field in row]
    return float(np.mean(values)) if values else float("nan")


def _load_condition(path: Path) -> dict:
    config = _load_json(path / "config.json")
    training_summary = _load_json(path / "training_summary.json")
    deterministic = _load_json(path / "evaluation_deterministic.json")
    stochastic = _load_json(path / "evaluation_stochastic.json")
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu")
    history = checkpoint.get("history", [])

    return {
        "path": str(path),
        "config": config,
        "training_summary": training_summary,
        "deterministic": deterministic,
        "stochastic": stochastic,
        "action_type_counts_stochastic": _action_type_counts(
            stochastic["action_counts"]
        ),
        "last_10_update_diagnostics": {
            field: _mean_last(history, field)
            for field in (
                "episode_return_mean",
                "value_loss",
                "policy_loss",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "entropy",
                "normalized_entropy",
                "valid_action_count_mean",
                "decision_rate",
            )
        },
        "early_stop_fraction_last_10": float(
            np.mean(
                [bool(row.get("early_stopped", False)) for row in history[-10:]]
            )
        ),
    }


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _contrast_summary(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    positive = int(np.count_nonzero(values > 0))
    negative = int(np.count_nonzero(values < 0))
    ties = int(np.count_nonzero(values == 0))
    ttest = stats.ttest_1samp(values, popmean=0.0)
    try:
        wilcoxon = stats.wilcoxon(values, zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "paired_bootstrap_95_ci": _bootstrap_ci(
            values,
            samples=samples,
            rng=rng,
        ),
        "wins_ties_losses": {
            "positive": positive,
            "ties": ties,
            "negative": negative,
        },
        "paired_t_test_p": float(ttest.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "episode_differences": values.tolist(),
    }


def _returns(conditions: dict[str, dict], mode: str) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(condition[mode]["episode_returns"], dtype=np.float64)
        for name, condition in conditions.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Evaluation episode counts differ: {sorted(lengths)}")
    return arrays


def _contrasts(
    conditions: dict[str, dict],
    *,
    mode: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    r = _returns(conditions, mode)
    values = {
        "B_minus_A_v2_effect_mask_off": r["B"] - r["A"],
        "C_minus_A_mask_effect_v1": r["C"] - r["A"],
        "D_minus_B_mask_effect_v2": r["D"] - r["B"],
        "D_minus_C_v2_effect_mask_on": r["D"] - r["C"],
        "v2_main_effect": ((r["B"] - r["A"]) + (r["D"] - r["C"])) / 2,
        "mask_main_effect": ((r["C"] - r["A"]) + (r["D"] - r["B"])) / 2,
        "interaction": (r["D"] - r["C"]) - (r["B"] - r["A"]),
    }
    return {
        name: _contrast_summary(
            difference,
            samples=samples,
            rng=rng,
        )
        for name, difference in values.items()
    }


def _controlled_config_audit(conditions: dict[str, dict]) -> dict:
    reference = conditions["A"]["config"]
    differences = {}
    for field in CONTROLLED_CONFIG_FIELDS:
        values = {
            name: condition["config"].get(field)
            for name, condition in conditions.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            differences[field] = values

    parameter_counts = {
        name: int(condition["config"]["parameter_count"])
        for name, condition in conditions.items()
    }
    parameter_spread = max(parameter_counts.values()) - min(parameter_counts.values())
    return {
        "all_controlled_fields_equal": not differences,
        "unexpected_differences": differences,
        "reference_values": {
            field: reference.get(field) for field in CONTROLLED_CONFIG_FIELDS
        },
        "parameter_counts": parameter_counts,
        "parameter_spread": parameter_spread,
        "parameter_spread_fraction_of_A": (
            parameter_spread / parameter_counts["A"]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    conditions = {
        name: _load_condition(
            args.run_dir / directory / f"seed_{args.seed}"
        )
        for name, directory in PRIMARY_CONDITIONS.items()
    }

    rng = np.random.default_rng(args.bootstrap_seed)
    output = {
        "scope": {
            "purpose": "20k technical qualification, not final encoder evidence",
            "training_seeds": [args.seed],
            "evaluation_seed_start": conditions["A"]["config"]["eval_seed"],
            "evaluation_episodes": conditions["A"]["config"]["eval_episodes"],
            "primary_evaluation_mode": "stochastic",
            "higher_return_is_better": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "condition_mapping": PRIMARY_CONDITIONS,
        "controlled_config_audit": _controlled_config_audit(conditions),
        "conditions": conditions,
        "stochastic_contrasts": _contrasts(
            conditions,
            mode="stochastic",
            samples=args.bootstrap_samples,
            rng=rng,
        ),
        "deterministic_contrasts": _contrasts(
            conditions,
            mode="deterministic",
            samples=args.bootstrap_samples,
            rng=rng,
        ),
        "limitations": [
            "Only one PPO training seed was used.",
            "Paired intervals quantify held-out episode-seed variation, not training-seed variation.",
            "The 20k budget is a qualification run and is not sufficient for a final scientific claim.",
            "V2 uses encoder_hidden_dim=92 while v1 uses 128 to match total parameter count within 0.4 percent.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"output={args.output}")
    print(
        "controlled_fields_equal="
        f"{output['controlled_config_audit']['all_controlled_fields_equal']}"
    )
    for name, condition in conditions.items():
        result = condition["stochastic"]
        print(
            f"{name} stochastic_mean={result['mean_return']:.2f} "
            f"std={result['std_return']:.2f} "
            f"parameters={condition['config']['parameter_count']}"
        )
    for name, contrast in output["stochastic_contrasts"].items():
        low, high = contrast["paired_bootstrap_95_ci"]
        print(
            f"{name} mean={contrast['mean']:.2f} "
            f"ci95=[{low:.2f}, {high:.2f}]"
        )


if __name__ == "__main__":
    main()
