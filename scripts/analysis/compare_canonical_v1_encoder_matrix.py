from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats


CONDITIONS = {
    "F-O": {
        "encoder": "flat",
        "semantic_mask": False,
        "run_dir": "canonical_flat_random_pivot_100k",
    },
    "G-O": {
        "encoder": "gcn",
        "semantic_mask": False,
        "run_dir": "canonical_gcn_random_pivot_100k",
    },
    "S-O": {
        "encoder": "graphsage",
        "semantic_mask": False,
        "run_dir": "canonical_v1_graphsage_random_host_official_mask_100k",
    },
    "F-S": {
        "encoder": "flat",
        "semantic_mask": True,
        "run_dir": "canonical_v1_flat_random_host_semantic_mask_100k",
    },
    "G-S": {
        "encoder": "gcn",
        "semantic_mask": True,
        "run_dir": "canonical_v1_gcn_random_host_semantic_mask_100k",
    },
    "S-S": {
        "encoder": "graphsage",
        "semantic_mask": True,
        "run_dir": "canonical_v1_graphsage_random_host_semantic_mask_100k",
    },
}
SEEDS = (42, 43, 44, 45, 46)
CONTROLLED_FIELDS = (
    "canonical_variant",
    "environment_distribution",
    "total_simulator_steps",
    "episode_steps",
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
    "eval_episodes",
    "eval_seed",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the canonical-v1 Flat/GCN multi-seed matrix."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical_v1_encoder_multi_seed.json"),
    )
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_config(config: dict, condition: dict) -> dict:
    output = dict(config)
    output["canonical_schema"] = output.get("canonical_schema", "v1")
    output["semantic_action_mask"] = bool(
        output.get("semantic_action_mask", False)
    )
    output["policy_type"] = output.get("policy_type", condition["encoder"])
    return output


def _action_type_counts(action_counts: dict[str, int]) -> Counter[str]:
    output: Counter[str] = Counter()
    for label, count in action_counts.items():
        action_label = label.split(": ", maxsplit=1)[-1]
        output[action_label.split(maxsplit=1)[0]] += int(count)
    return output


def _last_window(history: list[dict], field: str, window: int = 25) -> float:
    values = [float(row[field]) for row in history[-window:] if field in row]
    return float(np.mean(values)) if values else float("nan")


def _load_run(
    run_path: Path,
    condition: dict,
    results_root: Path,
    condition_name: str,
    seed: int,
) -> dict:
    config = _normalise_config(_read_json(run_path / "config.json"), condition)
    stochastic = _read_json(run_path / "evaluation_stochastic.json")
    deterministic = _read_json(run_path / "evaluation_deterministic.json")
    training_summary = _read_json(run_path / "training_summary.json")
    checkpoint = torch.load(run_path / "checkpoint.pt", map_location="cpu")
    history = checkpoint["history"]
    static_path = run_path / "observation_gate_static_only_stochastic.json"
    audit_path = results_root / (
        f"audit_v1_matrix_{condition_name}_seed_{seed}_summary.json"
    )

    return {
        "path": str(run_path),
        "config": config,
        "training_summary": training_summary,
        "stochastic": stochastic,
        "deterministic": deterministic,
        "action_type_counts": dict(
            sorted(_action_type_counts(stochastic["action_counts"]).items())
        ),
        "last_25_update_diagnostics": {
            field: _last_window(history, field)
            for field in (
                "episode_return_mean",
                "policy_loss",
                "value_loss",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "entropy",
                "normalized_entropy",
                "valid_action_count_mean",
                "decision_rate",
            )
        },
        "early_stop_fraction_last_25": float(
            np.mean(
                [bool(row.get("early_stopped", False)) for row in history[-25:]]
            )
        ),
        "static_only_stochastic": (
            _read_json(static_path) if static_path.exists() else None
        ),
        "dynamic_audit": _read_json(audit_path) if audit_path.exists() else None,
    }


def _sample_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values_by_seed": {
            str(seed): float(value) for seed, value in zip(SEEDS, values, strict=True)
        },
    }


def _paired_summary(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    sem = sample_std / np.sqrt(len(values))
    critical = float(stats.t.ppf(0.975, df=len(values) - 1))
    ttest = stats.ttest_1samp(values, popmean=0.0)
    return {
        **_sample_summary(values),
        "t_95_ci": [mean - critical * sem, mean + critical * sem],
        "wins_ties_losses": {
            "positive": int(np.count_nonzero(values > 0)),
            "ties": int(np.count_nonzero(values == 0)),
            "negative": int(np.count_nonzero(values < 0)),
        },
        "paired_t_test_p": float(ttest.pvalue),
        "warning": (
            f"n={len(values)} training seeds; inference remains sensitive "
            "to individual seeds."
        ),
    }


def _mode_means(runs: dict, mode: str) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [runs[name][seed][mode]["mean_return"] for seed in SEEDS],
            dtype=np.float64,
        )
        for name in CONDITIONS
    }


def _contrasts(values: dict[str, np.ndarray]) -> dict:
    raw = {
        "gcn_minus_flat_official": values["G-O"] - values["F-O"],
        "gcn_minus_flat_semantic": values["G-S"] - values["F-S"],
        "semantic_minus_official_flat": values["F-S"] - values["F-O"],
        "semantic_minus_official_gcn": values["G-S"] - values["G-O"],
        "graphsage_minus_flat_official": values["S-O"] - values["F-O"],
        "graphsage_minus_gcn_official": values["S-O"] - values["G-O"],
        "graphsage_minus_flat_semantic": values["S-S"] - values["F-S"],
        "graphsage_minus_gcn_semantic": values["S-S"] - values["G-S"],
        "semantic_minus_official_graphsage": values["S-S"] - values["S-O"],
        "encoder_main_effect": (
            (values["G-O"] - values["F-O"])
            + (values["G-S"] - values["F-S"])
        )
        / 2,
        "mask_main_effect": (
            (values["F-S"] - values["F-O"])
            + (values["G-S"] - values["G-O"])
        )
        / 2,
        "encoder_by_mask_interaction": (
            (values["G-S"] - values["F-S"])
            - (values["G-O"] - values["F-O"])
        ),
        "graphsage_vs_flat_main_effect": (
            (values["S-O"] - values["F-O"])
            + (values["S-S"] - values["F-S"])
        )
        / 2,
    }
    return {name: _paired_summary(delta) for name, delta in raw.items()}


def _config_audit(runs: dict) -> dict:
    unexpected = {}
    for seed in SEEDS:
        for field in CONTROLLED_FIELDS:
            values = {
                name: runs[name][seed]["config"].get(field)
                for name in CONDITIONS
            }
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
                unexpected[f"seed_{seed}:{field}"] = values

    factor_errors = {}
    for name, condition in CONDITIONS.items():
        for seed in SEEDS:
            config = runs[name][seed]["config"]
            expected = {
                "canonical_schema": "v1",
                "semantic_action_mask": condition["semantic_mask"],
                "policy_type": condition["encoder"],
            }
            for field, value in expected.items():
                if config.get(field) != value:
                    factor_errors[f"{name}:seed_{seed}:{field}"] = {
                        "expected": value,
                        "actual": config.get(field),
                    }
    return {
        "passed": not unexpected and not factor_errors,
        "unexpected_control_differences": unexpected,
        "factor_errors": factor_errors,
        "parameter_counts": {
            name: sorted(
                {
                    int(runs[name][seed]["config"]["parameter_count"])
                    for seed in SEEDS
                }
            )
            for name in CONDITIONS
        },
    }


def _dynamic_audit_summary(runs: dict, name: str) -> dict | None:
    audits = [runs[name][seed]["dynamic_audit"] for seed in SEEDS]
    if any(audit is None for audit in audits):
        return None

    traffic_completed = sum(
        int(audit["action_completion"]["semantic_traffic_summary"]["completed"])
        for audit in audits
    )
    traffic_false = sum(
        int(
            audit["action_completion"]["semantic_traffic_summary"][
                "false_completions"
            ]
        )
        for audit in audits
    )
    traffic_rates = np.asarray(
        [
            audit["action_completion"]["semantic_traffic_summary"][
                "false_completion_rate"
            ]
            for audit in audits
        ],
        dtype=np.float64,
    )

    action_completion: dict[str, dict] = {}
    for audit in audits:
        for action, completion in audit["action_completion"].items():
            if action == "semantic_traffic_summary":
                continue
            output = action_completion.setdefault(
                action,
                {"selected": 0, "completed": 0, "success_counts": Counter()},
            )
            output["selected"] += int(completion["selected"])
            output["completed"] += int(completion["completed"])
            output["success_counts"].update(completion["success_counts"])
    for output in action_completion.values():
        output["success_counts"] = dict(sorted(output["success_counts"].items()))

    sensitivity = {}
    sensitivity_fields = (
        "mean_total_variation",
        "mean_kl_full_static",
        "argmax_change_rate",
        "mean_absolute_value_change",
    )
    sensitivity_groups = set.intersection(
        *(set(audit["policy_input_sensitivity"]) for audit in audits)
    )
    for group in sorted(sensitivity_groups):
        sensitivity[group] = {
            field: _sample_summary(
                np.asarray(
                    [
                        audit["policy_input_sensitivity"][group][field]
                        for audit in audits
                    ],
                    dtype=np.float64,
                )
            )
            for field in sensitivity_fields
        }

    response_fields = (
        "first_same_host_action_observed_fraction",
        "median_first_same_host_action_latency",
        "same_host_response_within_0_fraction",
        "same_host_response_within_1_fraction",
        "same_host_response_within_5_fraction",
        "same_host_response_within_10_fraction",
    )
    evidence_response = {
        field: _sample_summary(
            np.asarray(
                [audit["evidence_response"][field] for audit in audits],
                dtype=np.float64,
            )
        )
        for field in response_fields
    }

    event_type_counts: Counter[str] = Counter()
    for audit in audits:
        event_type_counts.update(audit["event_type_counts"])

    return {
        "audit_episodes_per_training_seed": int(audits[0]["episodes"]),
        "audit_return": _sample_summary(
            np.asarray([audit["mean_return"] for audit in audits])
        ),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "evidence_occurrences": int(
            sum(int(audit["evidence_occurrences"]) for audit in audits)
        ),
        "action_completion": dict(sorted(action_completion.items())),
        "traffic_completion": {
            "completed": traffic_completed,
            "false_completions": traffic_false,
            "pooled_false_completion_rate": (
                traffic_false / traffic_completed if traffic_completed else 0.0
            ),
            "false_completion_rate_by_training_seed": _sample_summary(
                traffic_rates
            ),
        },
        "policy_input_sensitivity": sensitivity,
        "evidence_response": evidence_response,
    }


def _condition_summary(runs: dict, name: str) -> dict:
    stochastic = np.asarray(
        [runs[name][seed]["stochastic"]["mean_return"] for seed in SEEDS]
    )
    deterministic = np.asarray(
        [runs[name][seed]["deterministic"]["mean_return"] for seed in SEEDS]
    )
    action_counts: Counter[str] = Counter()
    for seed in SEEDS:
        action_counts.update(runs[name][seed]["action_type_counts"])
    action_total = sum(action_counts.values())

    diagnostics = {}
    fields = runs[name][SEEDS[0]]["last_25_update_diagnostics"]
    for field in fields:
        per_seed = np.asarray(
            [runs[name][seed]["last_25_update_diagnostics"][field] for seed in SEEDS]
        )
        diagnostics[field] = _sample_summary(per_seed)

    static_values = []
    full_minus_static = []
    for seed in SEEDS:
        static = runs[name][seed]["static_only_stochastic"]
        if static is None:
            continue
        static_values.append(float(static["mean_return"]))
        full_minus_static.append(
            float(runs[name][seed]["stochastic"]["mean_return"])
            - float(static["mean_return"])
        )

    return {
        "factors": CONDITIONS[name],
        "stochastic_return": _sample_summary(stochastic),
        "deterministic_return": _sample_summary(deterministic),
        "action_type_counts": dict(sorted(action_counts.items())),
        "action_type_fractions": {
            action: count / action_total
            for action, count in sorted(action_counts.items())
        },
        "last_25_update_diagnostics": diagnostics,
        "observation_use_gate": (
            {
                "static_only_return": _sample_summary(np.asarray(static_values)),
                "full_minus_static": _paired_summary(
                    np.asarray(full_minus_static)
                ),
            }
            if len(static_values) == len(SEEDS)
            else None
        ),
        "dynamic_audit": _dynamic_audit_summary(runs, name),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = {
        name: {
            seed: _load_run(
                args.runs_root / condition["run_dir"] / f"seed_{seed}",
                condition,
                args.results_root,
                name,
                seed,
            )
            for seed in SEEDS
        }
        for name, condition in CONDITIONS.items()
    }

    stochastic = _mode_means(runs, "stochastic")
    deterministic = _mode_means(runs, "deterministic")
    output = {
        "scope": {
            "training_seeds": list(SEEDS),
            "independent_replication_unit": "training_seed",
            "primary_mode": "stochastic",
            "higher_return_is_better": True,
            "warning": (
                "Five training seeds improve the stability audit but remain "
                "a modest sample for final inference."
            ),
        },
        "config_audit": _config_audit(runs),
        "conditions": {
            name: _condition_summary(runs, name) for name in CONDITIONS
        },
        "stochastic_contrasts": _contrasts(stochastic),
        "deterministic_contrasts": _contrasts(deterministic),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"output={args.output}")
    print(f"config_audit_passed={output['config_audit']['passed']}")
    for name, summary in output["conditions"].items():
        result = summary["stochastic_return"]
        print(
            f"{name} mean={result['mean']:.2f} "
            f"seed_sd={result['sample_std']:.2f}"
        )
    for name, summary in output["stochastic_contrasts"].items():
        low, high = summary["t_95_ci"]
        print(
            f"{name} mean={summary['mean']:.2f} "
            f"t95=[{low:.2f}, {high:.2f}]"
        )


if __name__ == "__main__":
    main()
