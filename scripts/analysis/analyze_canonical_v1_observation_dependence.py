from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import stats

from acd_benchmark.cyber.dynamic_evidence_counterfactual import (
    EVIDENCE_NODE_FEATURES,
)


SEEDS = (42, 43, 44, 45, 46)
CONDITIONS = {
    "F-O": "canonical_flat_random_pivot_100k",
    "G-O": "canonical_gcn_random_pivot_100k",
    "S-O": "canonical_v1_graphsage_random_host_official_mask_100k",
}
EVALUATIONS = {
    "full": ("evaluation_stochastic.json", False),
    "zero_all_dynamic": (
        "observation_gate_static_only_stochastic.json",
        False,
    ),
    "zero_evidence": ("counterfactual_zero_evidence_stochastic.json", True),
    "mask_only": ("observation_gate_zero_all_stochastic.json", False),
    "zero_no_mask": (
        "observation_gate_zero_all_no_action_mask_stochastic.json",
        False,
    ),
    "full_no_mask": (
        "observation_gate_full_no_action_mask_stochastic.json",
        False,
    ),
    "shuffle_hosts": ("counterfactual_shuffle_hosts_stochastic.json", True),
    "time_shift_lag5": (
        "counterfactual_time_shift_lag5_stochastic.json",
        True,
    ),
}
FEATURE_GROUPS = (
    "passive_event_now",
    "analysis_now",
    "passive_event_memory",
    "analysis_memory",
    "observed_now",
    "evidence_age",
    "decoy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze canonical-v1 evidence and observation counterfactuals."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical_v1_observation_dependence.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/CANONICAL_V1_OBSERVATION_DEPENDENCE.md"),
    )
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation(path: Path, nested: bool) -> dict:
    value = _read_json(path)
    return value["evaluation"] if nested else value


def _summary(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "sample_std": None, "t_95_ci": None}
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) > 1:
        half_width = float(
            stats.t.ppf(0.975, len(array) - 1)
            * sample_std
            / np.sqrt(len(array))
        )
        interval = [mean - half_width, mean + half_width]
    else:
        interval = None
    return {
        "n": int(len(array)),
        "mean": mean,
        "sample_std": sample_std,
        "t_95_ci": interval,
        "values": array.tolist(),
    }


def _paired_seed_effect(
    left_by_seed: dict[int, dict],
    right_by_seed: dict[int, dict],
) -> dict:
    seed_means = []
    episode_counts = []
    for seed in SEEDS:
        left = np.asarray(left_by_seed[seed]["episode_returns"], dtype=np.float64)
        right = np.asarray(right_by_seed[seed]["episode_returns"], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"Paired episode mismatch for seed {seed}")
        seed_means.append(float(np.mean(left - right)))
        episode_counts.append(int(len(left)))
    return {
        "definition": "left_minus_right; higher return is better",
        "independent_unit": "training_seed",
        "paired_episode_counts": episode_counts,
        "training_seed_effect": _summary(seed_means),
    }


def _load_evaluations(args: argparse.Namespace) -> dict:
    output = {}
    for condition, run_name in CONDITIONS.items():
        output[condition] = {}
        for seed in SEEDS:
            seed_dir = args.runs_root / run_name / f"seed_{seed}"
            output[condition][seed] = {
                name: _evaluation(seed_dir / filename, nested)
                for name, (filename, nested) in EVALUATIONS.items()
            }
    return output


def _gate_summary(evaluations: dict, condition: str) -> dict:
    by_gate = {
        gate: {
            seed: evaluations[condition][seed][gate]
            for seed in SEEDS
        }
        for gate in EVALUATIONS
    }
    returns = {
        gate: _summary(
            [by_gate[gate][seed]["mean_return"] for seed in SEEDS]
        )
        for gate in EVALUATIONS
    }
    effects = {
        "all_dynamic_value_full_minus_static_only": _paired_seed_effect(
            by_gate["full"], by_gate["zero_all_dynamic"]
        ),
        "evidence_value_full_minus_zero_evidence": _paired_seed_effect(
            by_gate["full"], by_gate["zero_evidence"]
        ),
        "host_alignment_value_full_minus_shuffle": _paired_seed_effect(
            by_gate["full"], by_gate["shuffle_hosts"]
        ),
        "timing_value_full_minus_lag5": _paired_seed_effect(
            by_gate["full"], by_gate["time_shift_lag5"]
        ),
        "official_mask_value_with_zero_observation": _paired_seed_effect(
            by_gate["mask_only"], by_gate["zero_no_mask"]
        ),
        "official_mask_value_with_full_observation": _paired_seed_effect(
            by_gate["full"], by_gate["full_no_mask"]
        ),
    }
    return {"returns": returns, "paired_effects": effects}


def _event_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float:
    return float(value)


def _audit_seed_metrics(summary: dict, event_rows: list[dict[str, str]]) -> dict:
    ticks = int(summary["ticks"])
    decision = summary["decision_context"]
    response = summary["evidence_response"]
    sensitivity = summary["policy_input_sensitivity"]["all"]
    rewards = summary["reward_association_not_causal"]

    event_ticks = np.asarray([int(row["tick"]) for row in event_rows], dtype=int)
    episodes: dict[int, list[int]] = {}
    for row in event_rows:
        episodes.setdefault(int(row["episode"]), []).append(int(row["tick"]))
    first_ticks = [min(values) for values in episodes.values() if values]
    quarter_fractions = {
        f"tick_{start}_{start + 99}": (
            float(np.mean((event_ticks >= start) & (event_ticks < start + 100)))
            if len(event_ticks)
            else 0.0
        )
        for start in (0, 100, 200, 300)
    }
    decision_latencies = [
        int(row["first_decision_latency"])
        for row in event_rows
        if row["first_decision_latency"] != ""
    ]

    feature_sensitivity = {}
    for group in FEATURE_GROUPS:
        present = summary["feature_group_input_sensitivity"][group]["group_present"]
        feature_sensitivity[f"tv_zero_{group}_when_present"] = _float(
            present["mean_total_variation"]
        )

    event_rates = {
        event: count / ticks * 1000.0
        for event, count in summary["event_type_counts"].items()
    }
    action_delta = {}
    present_actions = summary["action_proportions"]["current_evidence_present"]
    absent_actions = summary["action_proportions"]["current_evidence_absent"]
    for action in sorted(set(present_actions) | set(absent_actions)):
        action_delta[action] = present_actions.get(action, 0.0) - absent_actions.get(
            action, 0.0
        )

    target = rewards["target_current_evidence"]
    other = rewards["current_context_but_other_host"]
    no_evidence = rewards["no_current_or_memory_evidence"]
    return {
        "evidence_occurrences_per_1000_ticks": (
            summary["evidence_occurrences"] / ticks * 1000.0
        ),
        "event_type_rates_per_1000_ticks": event_rates,
        "first_evidence_tick_mean": float(np.mean(first_ticks)),
        "event_tick_median": float(np.median(event_ticks)),
        "event_timing_quarter_fractions": quarter_fractions,
        "event_decision_ready_fraction": float(
            np.mean([row["decision_ready"] == "True" for row in event_rows])
        ),
        "first_decision_latency_median": float(np.median(decision_latencies)),
        "current_evidence_decision_fraction": _float(
            decision["current_evidence_present_fraction"]
        ),
        "memory_evidence_decision_fraction": _float(
            decision["memory_evidence_present_fraction"]
        ),
        "current_host_alignment_excess_over_uniform": (
            _float(
                decision[
                    "target_current_evidence_given_host_action_and_current_context"
                ]
            )
            - _float(decision["uniform_host_current_chance"])
        ),
        "memory_host_alignment_excess_over_uniform": (
            _float(
                decision[
                    "target_memory_evidence_given_host_action_and_memory_context"
                ]
            )
            - _float(decision["uniform_host_memory_chance"])
        ),
        "same_host_response_within_10_fraction": _float(
            response["same_host_response_within_10_fraction"]
        ),
        "median_first_same_host_action_latency": _float(
            response["median_first_same_host_action_latency"]
        ),
        "action_fraction_delta_current_evidence_minus_absent": action_delta,
        "full_static_total_variation": _float(
            sensitivity["mean_total_variation"]
        ),
        "full_static_argmax_change_rate": _float(
            sensitivity["argmax_change_rate"]
        ),
        "feature_group_sensitivity": feature_sensitivity,
        "reward_next5_target_minus_other_host": (
            _float(target["mean_reward_next_5"])
            - _float(other["mean_reward_next_5"])
        ),
        "reward_next5_target_minus_no_evidence": (
            _float(target["mean_reward_next_5"])
            - _float(no_evidence["mean_reward_next_5"])
        ),
    }


def _aggregate_nested(seed_metrics: list[dict], field: str) -> dict:
    keys = sorted(
        set().union(*(set(metrics[field]) for metrics in seed_metrics))
    )
    return {
        key: _summary(
            [metrics[field].get(key, 0.0) for metrics in seed_metrics]
        )
        for key in keys
    }


def _audit_summary(args: argparse.Namespace, condition: str) -> dict:
    seed_metrics = []
    for seed in SEEDS:
        prefix = args.results_root / f"audit_v1_matrix_{condition}_seed_{seed}"
        summary = _read_json(prefix.with_name(prefix.name + "_summary.json"))
        events = _event_rows(prefix.with_name(prefix.name + "_events.csv.gz"))
        seed_metrics.append(_audit_seed_metrics(summary, events))

    scalar_fields = [
        key
        for key, value in seed_metrics[0].items()
        if not isinstance(value, dict)
    ]
    return {
        "scope": {
            "training_seeds": list(SEEDS),
            "audit_episodes_per_training_seed": 3,
            "warning": "Reward associations are observational, not causal.",
        },
        "scalars": {
            field: _summary([metrics[field] for metrics in seed_metrics])
            for field in scalar_fields
        },
        "event_type_rates_per_1000_ticks": _aggregate_nested(
            seed_metrics, "event_type_rates_per_1000_ticks"
        ),
        "event_timing_quarter_fractions": _aggregate_nested(
            seed_metrics, "event_timing_quarter_fractions"
        ),
        "action_fraction_delta_current_evidence_minus_absent": _aggregate_nested(
            seed_metrics, "action_fraction_delta_current_evidence_minus_absent"
        ),
        "feature_group_sensitivity": _aggregate_nested(
            seed_metrics, "feature_group_sensitivity"
        ),
        "by_training_seed": {
            str(seed): metrics
            for seed, metrics in zip(SEEDS, seed_metrics, strict=True)
        },
    }


def _format_effect(effect: dict) -> str:
    value = effect["training_seed_effect"]
    low, high = value["t_95_ci"]
    return f"{value['mean']:.2f} [{low:.2f}, {high:.2f}]"


def _write_report(output: dict, path: Path) -> None:
    lines = [
        "# Canonical V1 Observation-Dependence Audit",
        "",
        "## Protocol",
        "",
        "- Primary evaluation uses stochastic policy sampling.",
        "- Interventions share 30 held-out environment seeds per checkpoint.",
        "- The independent inferential unit is the training seed (n=5).",
        "- Positive effects below mean the left/full condition has higher return.",
        "",
        "## Counterfactual Results",
        "",
        "| Encoder | Full-zero evidence | Full-static only | Full-shuffle hosts | Full-lag 5 | Mask-only-no mask | Full-full no mask |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, result in output["conditions"].items():
        effects = result["gates"]["paired_effects"]
        lines.append(
            f"| {condition} | "
            f"{_format_effect(effects['evidence_value_full_minus_zero_evidence'])} | "
            f"{_format_effect(effects['all_dynamic_value_full_minus_static_only'])} | "
            f"{_format_effect(effects['host_alignment_value_full_minus_shuffle'])} | "
            f"{_format_effect(effects['timing_value_full_minus_lag5'])} | "
            f"{_format_effect(effects['official_mask_value_with_zero_observation'])} | "
            f"{_format_effect(effects['official_mask_value_with_full_observation'])} |"
        )

    lines.extend(
        [
            "",
            "## Main Reading",
            "",
            "- No encoder has a full-minus-zero-evidence interval strictly above zero across five training seeds.",
            "- No encoder shows a reliable return loss when evidence is shuffled across hosts.",
            "- GCN performs better under a five-tick evidence lag; this contradicts a claim that the current GCN requires correct evidence timing.",
            "- Flat benefits from the complete dynamic block relative to static-only, but that broader gate also removes `decoy_active` and cannot attribute the gain to sensor evidence alone.",
            "- Evidence memory is active at roughly 95-96% of decisions, indicating saturation rather than sparse discriminative belief.",
            "",
            "## Evidence Availability and Response",
            "",
            "| Encoder | Evidence/1k ticks | Current-evidence decisions | Memory-evidence decisions | Current alignment excess | Same-host response <=10 ticks | Full-vs-static TV |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, result in output["conditions"].items():
        scalars = result["evidence_audit"]["scalars"]
        lines.append(
            f"| {condition} | "
            f"{scalars['evidence_occurrences_per_1000_ticks']['mean']:.2f} | "
            f"{scalars['current_evidence_decision_fraction']['mean']:.3f} | "
            f"{scalars['memory_evidence_decision_fraction']['mean']:.3f} | "
            f"{scalars['current_host_alignment_excess_over_uniform']['mean']:.3f} | "
            f"{scalars['same_host_response_within_10_fraction']['mean']:.3f} | "
            f"{scalars['full_static_total_variation']['mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Evidence Timing",
            "",
            "| Encoder | First evidence tick | Median event tick | Event while decision-ready | First decision latency | Same-host action latency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, result in output["conditions"].items():
        scalars = result["evidence_audit"]["scalars"]
        lines.append(
            f"| {condition} | "
            f"{scalars['first_evidence_tick_mean']['mean']:.2f} | "
            f"{scalars['event_tick_median']['mean']:.2f} | "
            f"{scalars['event_decision_ready_fraction']['mean']:.3f} | "
            f"{scalars['first_decision_latency_median']['mean']:.2f} | "
            f"{scalars['median_first_same_host_action_latency']['mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Action Response",
            "",
            "Values are action-fraction changes when current evidence is present minus absent.",
            "",
            "| Encoder | Analyse | Remove | Restore | BlockTrafficZone |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, result in output["conditions"].items():
        action_delta = result["evidence_audit"][
            "action_fraction_delta_current_evidence_minus_absent"
        ]
        lines.append(
            f"| {condition} | "
            f"{action_delta['Analyse']['mean']:+.3f} | "
            f"{action_delta['Remove']['mean']:+.3f} | "
            f"{action_delta['Restore']['mean']:+.3f} | "
            f"{action_delta['BlockTrafficZone']['mean']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Reward Association",
            "",
            "The following five-tick reward contrasts are observational, not causal.",
            "",
            "| Encoder | Target evidence - other host | Target evidence - no evidence |",
            "|---|---:|---:|",
        ]
    )
    for condition, result in output["conditions"].items():
        scalars = result["evidence_audit"]["scalars"]
        target_other = scalars["reward_next5_target_minus_other_host"]
        target_none = scalars["reward_next5_target_minus_no_evidence"]
        lines.append(
            f"| {condition} | "
            f"{target_other['mean']:.2f} | "
            f"{target_none['mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The official action mask is fixed within an episode, so it does not directly expose changing compromise/evidence state.",
            "- It does expose reset-time valid action slots. No-mask gates also enable padded/invalid actions that the wrapper can convert to no-ops, so mask effects are diagnostic rather than a pure information estimate.",
            "- Host shuffling preserves the per-tick multiset of evidence values but breaks host alignment. Lag-5 uses simulator ticks, not PPO decision steps.",
            "- `decoy_active` is preserved by the evidence interventions. It is removed only by the broader static-only gate.",
            "- Reward-after-action measurements are associations under the attack trajectory, not causal action effects.",
            "- A null intervention effect does not prove that information is useless; it shows that these trained checkpoints do not measurably require it under this protocol.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluations = _load_evaluations(args)
    mask_semantics = _read_json(
        args.results_root / "cc4_official_action_mask_semantics.json"
    )
    output = {
        "protocol": {
            "training_seeds": list(SEEDS),
            "held_out_episodes_per_checkpoint": 30,
            "evaluation_mode": "stochastic",
            "environment_seed_range": [10000, 10029],
            "independent_inferential_unit": "training_seed",
            "evidence_node_features": list(EVIDENCE_NODE_FEATURES),
            "preserved_dynamic_node_features": ["decoy_active"],
        },
        "action_mask_semantics": mask_semantics,
        "conditions": {
            condition: {
                "gates": _gate_summary(evaluations, condition),
                "evidence_audit": _audit_summary(args, condition),
            }
            for condition in CONDITIONS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_report(output, args.report)
    print(f"output={args.output}")
    print(f"report={args.report}")
    for condition, result in output["conditions"].items():
        effects = result["gates"]["paired_effects"]
        print(
            f"{condition} full-shuffle="
            f"{_format_effect(effects['host_alignment_value_full_minus_shuffle'])} "
            f"full-lag5={_format_effect(effects['timing_value_full_minus_lag5'])}"
        )


if __name__ == "__main__":
    main()
