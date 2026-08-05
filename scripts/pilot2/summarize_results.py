"""Summarize Pilot v2 fixed-q curves and paired robustness contrasts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


CORE_METRICS = (
    "mean_mission_availability",
    "mean_episode_return",
    "mean_privileged_compromised_hosts",
    "mean_red_presence_hosts",
    "mean_detection_latency",
    "mean_recovery_latency",
    "action_success_rate_among_known",
    "mean_inference_ms",
)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return across-seed mean and sample standard deviation."""
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else None


def _q_from_label(label: str) -> float:
    """Parse q100, q075, q050, and q025 labels."""
    if not label.startswith("q"):
        raise ValueError(f"Invalid q label: {label}")
    return int(label[1:]) / 100.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular CSV, including a stable empty artifact."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalized_auc(points: dict[float, float]) -> float | None:
    """Compute descriptive trapezoidal AUC normalized by the q interval."""
    ordered = sorted(points.items())
    if len(ordered) < 2:
        return None
    area = 0.0
    for (q_left, y_left), (q_right, y_right) in zip(ordered, ordered[1:]):
        area += (q_right - q_left) * (y_left + y_right) / 2.0
    span = ordered[-1][0] - ordered[0][0]
    return area / span if span > 0 else None


def main() -> None:
    """Write per-seed curves, seed aggregates, and Flat-GNN contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/pilot2_observation_16"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot2_observation_16.yaml"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = [str(model) for model in config["matrix"]["models"]]
    seeds = [int(seed) for seed in config["matrix"]["seeds"]]
    q_values = [float(q) for q in config["evaluation"]["retention_levels"]]
    expected_timesteps = int(config["training"]["total_timesteps"])
    expected = {(model, seed) for model in models for seed in seeds}

    reports: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(args.run_root.glob("*/obs_mixed/seed_*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (str(report["run"]["model"]), int(report["run"]["seed"]))
        report["_report_path"] = str(path)
        reports[key] = report

    completed = {
        key
        for key, report in reports.items()
        if int(report.get("completed_timesteps", -1)) == expected_timesteps
    }
    per_seed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    curves: dict[tuple[str, int], dict[float, dict[str, Any]]] = {}

    for (model, seed), report in sorted(reports.items()):
        final = report["final_evaluation"]
        curve: dict[float, dict[str, Any]] = {}
        for q_label, metrics in final["by_q"].items():
            q = _q_from_label(q_label)
            row = {
                "model": model,
                "seed": seed,
                "retention_q": q,
                "completed_timesteps": report["completed_timesteps"],
                "parameter_count": report["parameter_count"],
                **{metric: metrics.get(metric) for metric in CORE_METRICS},
                "mean_visible_host_count": metrics.get("mean_visible_host_count"),
                "action_success_count": metrics.get("action_success_count"),
                "action_failure_count": metrics.get("action_failure_count"),
                "action_unknown_count": metrics.get("action_unknown_count"),
                "training_steps_per_second": report["compute"].get(
                    "training_steps_per_second"
                ),
                "peak_process_rss_mb": report["compute"].get("peak_process_rss_mb"),
                "report_path": report["_report_path"],
            }
            per_seed_rows.append(row)
            grouped[(model, q)].append(row)
            curve[q] = metrics
        curves[(model, seed)] = curve

    aggregate_rows: list[dict[str, Any]] = []
    for (model, q), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "model": model,
            "retention_q": q,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for metric in (*CORE_METRICS, "training_steps_per_second", "peak_process_rss_mb"):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            metric_mean, metric_std = _mean_std(values)
            aggregate[f"{metric}_seed_mean"] = metric_mean
            aggregate[f"{metric}_seed_std"] = metric_std
        aggregate_rows.append(aggregate)

    curve_rows: list[dict[str, Any]] = []
    for (model, seed), curve in sorted(curves.items()):
        if 1.0 not in curve:
            continue
        availability_at_full = float(curve[1.0]["mean_mission_availability"])
        availability_points = {
            q: float(metrics["mean_mission_availability"])
            for q, metrics in curve.items()
        }
        for q, metrics in sorted(curve.items(), reverse=True):
            curve_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "retention_q": q,
                    "availability": metrics["mean_mission_availability"],
                    "availability_degradation_from_q100": (
                        float(metrics["mean_mission_availability"])
                        - availability_at_full
                    ),
                    "normalized_retention_auc_descriptive": _normalized_auc(
                        availability_points
                    ),
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    if "flat" in models and "graphsage" in models:
        for seed in seeds:
            flat_curve = curves.get(("flat", seed), {})
            graph_curve = curves.get(("graphsage", seed), {})
            if 1.0 not in flat_curve or 1.0 not in graph_curve:
                continue
            flat_full = float(flat_curve[1.0]["mean_mission_availability"])
            graph_full = float(graph_curve[1.0]["mean_mission_availability"])
            for q in q_values:
                if q not in flat_curve or q not in graph_curve:
                    continue
                flat_availability = float(
                    flat_curve[q]["mean_mission_availability"]
                )
                graph_availability = float(
                    graph_curve[q]["mean_mission_availability"]
                )
                flat_degradation = flat_availability - flat_full
                graph_degradation = graph_availability - graph_full
                contrast_rows.append(
                    {
                        "seed": seed,
                        "retention_q": q,
                        "graphsage_minus_flat_absolute_availability": (
                            graph_availability - flat_availability
                        ),
                        "flat_degradation_from_q100": flat_degradation,
                        "graphsage_degradation_from_q100": graph_degradation,
                        "robustness_contrast_positive_favors_graphsage": (
                            graph_degradation - flat_degradation
                        ),
                    }
                )

    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.run_root / "pilot2_per_seed_fixed_q.csv", per_seed_rows)
    _write_csv(args.run_root / "pilot2_aggregate_fixed_q.csv", aggregate_rows)
    _write_csv(args.run_root / "pilot2_retention_curves.csv", curve_rows)
    _write_csv(args.run_root / "pilot2_paired_robustness_contrasts.csv", contrast_rows)

    missing = sorted(expected - completed)
    summary = {
        "protocol": config["protocol"],
        "expected_timesteps": expected_timesteps,
        "expected_run_count": len(expected),
        "completed_run_count": len(completed & expected),
        "complete": not missing,
        "missing_runs": [
            {"model": model, "observation_level": "mixed", "seed": seed}
            for model, seed in missing
        ],
        "fixed_q_groups": aggregate_rows,
        "paired_robustness_contrasts": contrast_rows,
    }
    (args.run_root / "pilot2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Pilot v2 Observation Qualification Summary",
        "",
        f"Protocol complete: **{summary['complete']}** "
        f"({summary['completed_run_count']}/{summary['expected_run_count']} runs)",
        "",
        "| Model | q | Seeds | Availability mean | Return mean | Privileged hosts mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['model']} | {row['retention_q']:.2f} | {row['seed_count']} | "
            f"{row['mean_mission_availability_seed_mean']:.4f} | "
            f"{row['mean_episode_return_seed_mean']:.4f} | "
            f"{row['mean_privileged_compromised_hosts_seed_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The primary robustness contrast is degradation(q) = availability(q) - availability(1.0).",
            "A positive GraphSAGE-minus-Flat degradation contrast means GraphSAGE loses less availability.",
            "The q=0.25 point is a stress/floor diagnostic, not a hard trainability gate.",
            "",
        ]
    )
    if missing:
        lines.extend(["## Missing Runs", ""])
        lines.extend(f"- `{model}/obs_mixed/seed_{seed}`" for model, seed in missing)
        lines.append("")
    (args.run_root / "pilot2_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        f"Summarized {len(reports)} reports: "
        f"{summary['completed_run_count']}/{summary['expected_run_count']} complete"
    )
    if missing and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
