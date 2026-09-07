from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


CONDITIONS = {
    "F-O": {
        "encoder": "Flat",
        "run_dir": "canonical_flat_random_pivot_100k",
    },
    "G-O": {
        "encoder": "GCN",
        "run_dir": "canonical_gcn_random_pivot_100k",
    },
    "S-O": {
        "encoder": "GraphSAGE",
        "run_dir": "canonical_v1_graphsage_random_host_official_mask_100k",
    },
}
SEEDS = (42, 43, 44, 45, 46)
RETENTION_LEVELS = (1.0, 0.75, 0.50, 0.25)
MASK_PROCESSES = {
    "iid": {
        "protocol_name": "iid_fixed_budget_dynamic_host",
        "file_prefix": "dynamic_host_retention",
        "label": "IID",
    },
    "persistent": {
        "protocol_name": "persistent_fixed_budget_dynamic_host",
        "file_prefix": "persistent_dynamic_host_retention",
        "label": "Persistent",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the canonical-v1 dynamic-host degradation curve."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--mask-process",
        choices=tuple(MASK_PROCESSES),
        default="iid",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _q_label(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def _evaluation_path(seed_dir: Path, q: float, mask_process: str) -> Path:
    if q == 1.0:
        return seed_dir / "evaluation_stochastic.json"
    prefix = MASK_PROCESSES[mask_process]["file_prefix"]
    return seed_dir / f"{prefix}_{_q_label(q)}_stochastic.json"


def _load_evaluation(
    path: Path, q: float, episodes: int
) -> tuple[dict, dict | None]:
    record = _read_json(path)
    if q == 1.0:
        episode_returns = record["episode_returns"][:episodes]
        if len(episode_returns) != episodes:
            raise ValueError(f"{path} has fewer than {episodes} episodes")
        record = dict(record)
        record["episode_returns"] = episode_returns
        record["episode_lengths"] = record["episode_lengths"][:episodes]
        record["episode_policy_decisions"] = record[
            "episode_policy_decisions"
        ][:episodes]
        record["mean_return"] = float(np.mean(episode_returns))
        record["std_return"] = float(np.std(episode_returns))
        return record, None
    if int(record["protocol"]["episodes"]) != episodes:
        raise ValueError(
            f"{path} uses {record['protocol']['episodes']} episodes, "
            f"expected {episodes}"
        )
    return record["evaluation"], record["protocol"]


def _sample_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values_by_seed": {
            str(seed): float(value)
            for seed, value in zip(SEEDS, values, strict=True)
        },
    }


def _paired_summary(values: np.ndarray) -> dict:
    summary = _sample_summary(values)
    mean = summary["mean"]
    sample_std = summary["sample_std"]
    sem = sample_std / np.sqrt(len(values))
    critical = float(stats.t.ppf(0.975, df=len(values) - 1))
    test = stats.ttest_1samp(values, popmean=0.0)
    summary.update(
        {
            "t_95_ci": [mean - critical * sem, mean + critical * sem],
            "paired_t_test_p": float(test.pvalue),
            "wins_ties_losses": {
                "positive": int(np.count_nonzero(values > 0)),
                "ties": int(np.count_nonzero(values == 0)),
                "negative": int(np.count_nonzero(values < 0)),
            },
            "warning": "n=5 training seeds; treat inference as preliminary.",
        }
    )
    return summary


def _load_condition(
    runs_root: Path,
    condition: dict,
    episodes: int,
    mask_process: str,
) -> dict:
    evaluations = {}
    protocols = {}
    for seed in SEEDS:
        seed_dir = runs_root / condition["run_dir"] / f"seed_{seed}"
        evaluations[seed] = {}
        protocols[seed] = {}
        for q in RETENTION_LEVELS:
            path = _evaluation_path(seed_dir, q, mask_process)
            if not path.exists():
                raise FileNotFoundError(path)
            evaluation, protocol = _load_evaluation(path, q, episodes)
            evaluations[seed][q] = evaluation
            protocols[seed][q] = protocol
    return {"evaluations": evaluations, "protocols": protocols}


def _return_matrix(condition_data: dict) -> np.ndarray:
    return np.asarray(
        [
            [
                condition_data["evaluations"][seed][q]["mean_return"]
                for q in RETENTION_LEVELS
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )


def _action_type_distribution(action_counts: dict[str, int]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for label, count in action_counts.items():
        action_label = label.split(": ", maxsplit=1)[-1]
        counts[action_label.split(maxsplit=1)[0]] += int(count)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0))
        for key in set(left) | set(right)
    )


def _action_type_tv(condition_data: dict) -> dict[str, dict]:
    values = {q: [] for q in RETENTION_LEVELS}
    for seed in SEEDS:
        baseline = _action_type_distribution(
            condition_data["evaluations"][seed][1.0]["action_counts"]
        )
        for q in RETENTION_LEVELS:
            current = _action_type_distribution(
                condition_data["evaluations"][seed][q]["action_counts"]
            )
            values[q].append(_total_variation(baseline, current))
    return {
        str(q): _sample_summary(np.asarray(rows, dtype=np.float64))
        for q, rows in values.items()
    }


def _condition_summary(returns: np.ndarray, condition_data: dict) -> dict:
    q_values = np.asarray(RETENTION_LEVELS, dtype=np.float64)
    baseline = returns[:, 0]
    degradation = baseline[:, None] - returns
    relative_degradation = degradation / np.abs(baseline[:, None]) * 100.0

    ascending = np.argsort(q_values)
    interval = float(q_values.max() - q_values.min())
    return_auc = np.trapz(
        returns[:, ascending], x=q_values[ascending], axis=1
    ) / interval
    degradation_auc = np.trapz(
        degradation[:, ascending], x=q_values[ascending], axis=1
    ) / interval

    monotonic_violations = []
    for seed_index, seed in enumerate(SEEDS):
        # q decreases from left to right. Improvement at lower q is a violation
        # of the expected monotonic degradation pattern.
        count = int(np.count_nonzero(np.diff(returns[seed_index]) > 0.0))
        monotonic_violations.append({"seed": seed, "count": count})

    return {
        "return_by_retention": {
            str(q): _sample_summary(returns[:, index])
            for index, q in enumerate(RETENTION_LEVELS)
        },
        "degradation_from_q1": {
            str(q): _paired_summary(degradation[:, index])
            for index, q in enumerate(RETENTION_LEVELS)
        },
        "relative_degradation_percent": {
            str(q): _sample_summary(relative_degradation[:, index])
            for index, q in enumerate(RETENTION_LEVELS)
        },
        "normalized_return_auc": _sample_summary(return_auc),
        "normalized_degradation_auc_lower_is_better": _sample_summary(
            degradation_auc
        ),
        "worst_case_return_across_q": _sample_summary(np.min(returns, axis=1)),
        "endpoint_q025_return": _sample_summary(returns[:, -1]),
        "endpoint_q025_degradation": _paired_summary(degradation[:, -1]),
        "monotonicity_diagnostic": {
            "expected": "return should not improve when q decreases",
            "violations_by_seed": monotonic_violations,
            "total_violations": int(
                sum(row["count"] for row in monotonic_violations)
            ),
        },
        "action_type_total_variation_from_q1": _action_type_tv(condition_data),
    }


def _encoder_contrasts(matrices: dict[str, np.ndarray]) -> dict:
    contrasts = {}
    for left, right in (("G-O", "F-O"), ("S-O", "F-O"), ("S-O", "G-O")):
        delta = matrices[left] - matrices[right]
        contrasts[f"{left}_minus_{right}"] = {
            str(q): _paired_summary(delta[:, index])
            for index, q in enumerate(RETENTION_LEVELS)
        }
    return contrasts


def _summary_values(summary: dict) -> np.ndarray:
    return np.asarray(
        [summary["values_by_seed"][str(seed)] for seed in SEEDS],
        dtype=np.float64,
    )


def _persistent_minus_iid_contrasts(
    persistent_output: dict,
    iid_path: Path,
) -> dict:
    iid_output = _read_json(iid_path)
    contrasts = {}
    for name in CONDITIONS:
        persistent = persistent_output["conditions"][name]
        iid = iid_output["conditions"][name]
        contrasts[name] = {
            "return_by_retention": {
                str(q): _paired_summary(
                    _summary_values(persistent["return_by_retention"][str(q)])
                    - _summary_values(iid["return_by_retention"][str(q)])
                )
                for q in RETENTION_LEVELS[1:]
            },
            "endpoint_degradation_difference": _paired_summary(
                _summary_values(persistent["endpoint_q025_degradation"])
                - _summary_values(iid["endpoint_q025_degradation"])
            ),
            "degradation_auc_difference": _paired_summary(
                _summary_values(
                    persistent["normalized_degradation_auc_lower_is_better"]
                )
                - _summary_values(
                    iid["normalized_degradation_auc_lower_is_better"]
                )
            ),
        }
    return {
        "difference_definition": "persistent - iid",
        "positive_return_difference_means": "persistent has higher return",
        "source": str(iid_path),
        "conditions": contrasts,
    }


def _audit_protocol(raw: dict, episodes: int, mask_process: str) -> dict:
    errors = []
    for name, condition_data in raw.items():
        for seed in SEEDS:
            q1 = condition_data["evaluations"][seed][1.0]
            if len(q1["episode_returns"]) != episodes:
                errors.append(f"{name} seed={seed} q=1 episode count")
            for q in RETENTION_LEVELS[1:]:
                evaluation = condition_data["evaluations"][seed][q]
                protocol = condition_data["protocols"][seed][q]
                checks = {
                    "episode_count": len(evaluation["episode_returns"]) == episodes,
                    "retention": float(protocol["retention"]) == q,
                    "mask_seed": int(protocol["mask_seed"]) == 30_000,
                    "environment_seed": int(protocol["environment_seed"]) == 10_000,
                    "action_seed": int(protocol["stochastic_action_seed"]) == 10_000,
                    "mask_process": protocol["mask_process"]
                    == MASK_PROCESSES[mask_process]["protocol_name"],
                }
                errors.extend(
                    f"{name} seed={seed} q={q} {field}"
                    for field, passed in checks.items()
                    if not passed
                )
    return {"passed": not errors, "errors": errors}


def _write_report(output: dict, path: Path) -> None:
    episodes = output["protocol"]["episodes_per_checkpoint"]
    mask_process = output["protocol"]["mask_process"]
    process_label = MASK_PROCESSES[mask_process]["label"]
    process_scope = (
        "The visible host subset is sampled once at episode reset and remains "
        "fixed for all simulator ticks."
        if mask_process == "persistent"
        else "A new visible host subset is sampled at every simulator tick."
    )
    endpoint_cis_include_zero = all(
        summary["endpoint_q025_degradation"]["t_95_ci"][0] <= 0.0
        <= summary["endpoint_q025_degradation"]["t_95_ci"][1]
        for summary in output["conditions"].values()
    )
    contrast_cis_include_zero = all(
        row["t_95_ci"][0] <= 0.0 <= row["t_95_ci"][1]
        for contrast in output["paired_encoder_return_contrasts"].values()
        for row in contrast.values()
    )
    total_monotonicity_violations = sum(
        summary["monotonicity_diagnostic"]["total_violations"]
        for summary in output["conditions"].values()
    )
    process_contrasts = output.get("paired_process_contrasts")
    lines = [
        f"# Canonical v1 {process_label} Dynamic-Host Degradation Curve",
        "",
        "## Scope",
        "",
        "Zero-shot stochastic evaluation of final 100k checkpoints on held-out "
        f"episode seeds 10000-{9999 + episodes}. The intervention retains dynamic host-node "
        "features with an exact subnet-stratified budget. "
        "Static node identity, topology, mission/global features and the official "
        "action mask remain unchanged.",
        "",
        process_scope,
        "",
        "This is an output-level observation-retention stress test. It is not a "
        "claim that CAGE 4 itself implements this sensor-loss process.",
        "",
        "## Return Curve",
        "",
        "| Encoder | q=1.00 | q=0.75 | q=0.50 | q=0.25 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, condition in CONDITIONS.items():
        summary = output["conditions"][name]
        cells = []
        for q in RETENTION_LEVELS:
            row = summary["return_by_retention"][str(q)]
            cells.append(f'{row["mean"]:.2f} +/- {row["sample_std"]:.2f}')
        lines.append(f'| {condition["encoder"]} | ' + " | ".join(cells) + " |")

    if process_contrasts is not None:
        lines.extend(
            [
                "",
                "## Persistent Versus IID",
                "",
                "Paired return difference is `persistent - IID`; positive means "
                "persistent masking produced a higher return.",
                "",
                "| Encoder | q=.75 | q=.50 | q=.25 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, condition in CONDITIONS.items():
            rows = process_contrasts["conditions"][name][
                "return_by_retention"
            ]
            cells = []
            for q in RETENTION_LEVELS[1:]:
                row = rows[str(q)]
                cells.append(
                    f'{row["mean"]:.2f} '
                    f'[{row["t_95_ci"][0]:.2f}, {row["t_95_ci"][1]:.2f}]'
                )
            lines.append(
                f'| {condition["encoder"]} | ' + " | ".join(cells) + " |"
            )

    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "`Degradation = return(q=1) - return(q)`; lower is better. The "
            "degradation AUC averages that loss across q in [0.25, 1.00].",
            "",
            "| Encoder | q=.25 degradation | Relative degradation | Degradation AUC | Worst-case return |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, condition in CONDITIONS.items():
        summary = output["conditions"][name]
        endpoint = summary["endpoint_q025_degradation"]
        relative = summary["relative_degradation_percent"]["0.25"]
        auc = summary["normalized_degradation_auc_lower_is_better"]
        worst = summary["worst_case_return_across_q"]
        lines.append(
            f'| {condition["encoder"]} | {endpoint["mean"]:.2f} +/- '
            f'{endpoint["sample_std"]:.2f} | {relative["mean"]:.2f}% | '
            f'{auc["mean"]:.2f} +/- {auc["sample_std"]:.2f} | '
            f'{worst["mean"]:.2f} +/- {worst["sample_std"]:.2f} |'
        )

    lines.extend(
        [
            "",
            "## Behavioural Response",
            "",
            "Total-variation (TV) distance compares the aggregate action-type "
            "distribution at each q with q=1. Larger values mean a larger policy "
            "response, but are not by themselves evidence of better decisions.",
            "",
            "| Encoder | TV q=.75 | TV q=.50 | TV q=.25 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, condition in CONDITIONS.items():
        summary = output["conditions"][name]
        cells = []
        for q in RETENTION_LEVELS[1:]:
            row = summary["action_type_total_variation_from_q1"][str(q)]
            cells.append(f'{row["mean"]:.3f} +/- {row["sample_std"]:.3f}')
        lines.append(f'| {condition["encoder"]} | ' + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Scientific Interpretation",
            "",
            (
                "- All three q=.25 degradation 95% confidence intervals include zero."
                if endpoint_cis_include_zero
                else "- At least one q=.25 degradation 95% confidence interval excludes zero; inspect encoder-specific intervals before claiming an effect."
            ),
            (
                "- No paired encoder return contrast has a 95% confidence interval excluding zero at any q."
                if contrast_cis_include_zero
                else "- At least one paired encoder return contrast excludes zero; inspect multiplicity and consistency across q before claiming superiority."
            ),
            f"- Total monotonicity violations across all encoders: {total_monotonicity_violations}.",
            "- Robustness does not imply useful observation use or GNN superiority; interpret the curve together with raw return and action response.",
            (
                "- Persistent masking prevents hidden hosts from reappearing within the episode, so it is a stronger observation-loss test than IID masking."
                if mask_process == "persistent"
                else "- IID resampling allows hidden hosts and temporal memory to reappear; persistent masking is required for a stronger test."
            ),
            *(
                [
                    "- No persistent-versus-IID paired return contrast has a 95% confidence interval excluding zero."
                ]
                if process_contrasts is not None
                and all(
                    row["t_95_ci"][0] <= 0.0 <= row["t_95_ci"][1]
                    for condition in process_contrasts["conditions"].values()
                    for row in condition["return_by_retention"].values()
                )
                else []
            ),
            "",
            "## Interpretation Guardrails",
            "",
            f"- The training seed is the inferential unit (`n=5`), not the {episodes} evaluation episodes.",
            "- Raw return and degradation must be interpreted together; robustness alone does not establish a better defender.",
            "- A non-monotonic point can arise from stochastic trajectories and should not be hidden or smoothed away.",
            "- Action-distribution TV is descriptive because no independent q=1 repeat was run to estimate ordinary stochastic rollout divergence.",
            "- Semantic action masking and whole-subnet loss are separate follow-up ablations.",
            "",
            f'Protocol audit passed: `{output["protocol_audit"]["passed"]}`.',
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = args.output or Path(
        "results/canonical_v1_degradation_curve.json"
        if args.mask_process == "iid"
        else "results/canonical_v1_persistent_degradation_curve.json"
    )
    report_path = args.report or Path(
        "results/CANONICAL_V1_DEGRADATION_CURVE_RESULT.md"
        if args.mask_process == "iid"
        else "results/CANONICAL_V1_PERSISTENT_DEGRADATION_CURVE_RESULT.md"
    )
    raw = {
        name: _load_condition(
            args.runs_root,
            condition,
            args.episodes,
            args.mask_process,
        )
        for name, condition in CONDITIONS.items()
    }
    matrices = {name: _return_matrix(data) for name, data in raw.items()}
    output = {
        "protocol": {
            "canonical_schema": "v1",
            "environment_distribution": "random_host",
            "action_mask": "official",
            "evaluation_mode": "stochastic",
            "training_seeds": list(SEEDS),
            "evaluation_seeds": [10_000, 9_999 + args.episodes],
            "episodes_per_checkpoint": args.episodes,
            "mask_process": args.mask_process,
            "mask_seed": 30_000,
            "retention_levels": list(RETENTION_LEVELS),
            "inferential_unit": "training seed",
        },
        "protocol_audit": _audit_protocol(
            raw, args.episodes, args.mask_process
        ),
        "conditions": {
            name: _condition_summary(matrix, raw[name])
            for name, matrix in matrices.items()
        },
        "paired_encoder_return_contrasts": _encoder_contrasts(matrices),
    }
    iid_path = Path("results/canonical_v1_degradation_curve.json")
    if args.mask_process == "persistent" and iid_path.exists():
        output["paired_process_contrasts"] = _persistent_minus_iid_contrasts(
            output,
            iid_path,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_report(output, report_path)
    print(f"protocol_audit_passed={output['protocol_audit']['passed']}")
    for name, condition in CONDITIONS.items():
        summary = output["conditions"][name]
        endpoint = summary["endpoint_q025_degradation"]
        auc = summary["normalized_degradation_auc_lower_is_better"]
        print(
            f"{name} encoder={condition['encoder']} "
            f"q025_degradation={endpoint['mean']:.2f} "
            f"degradation_auc={auc['mean']:.2f}"
        )
    print(f"output={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
