"""Aggregate completed Stage 1 runs across models, observability, and seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


FINAL_METRICS = (
    "mean_episode_return",
    "mean_mission_availability",
    "mean_privileged_compromised_hosts",
    "mean_red_presence_hosts",
    "mean_detection_latency",
    "mean_recovery_latency",
    "operational_server_reach_rate",
    "mean_steps_to_operational_server",
    "mean_inference_ms",
)
COMPUTE_METRICS = (
    "total_wall_seconds",
    "training_steps_per_second",
    "peak_process_rss_mb",
)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return across-seed mean and sample standard deviation."""
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else None


def _load_reports(run_root: Path) -> list[dict[str, Any]]:
    """Load every final report under the canonical matrix directory layout."""
    reports = []
    for path in sorted(run_root.glob("*/*/seed_*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_report_path"] = str(path)
        reports.append(report)
    return reports


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular CSV, preserving an empty file for no rows."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Write machine-readable and human-readable matrix summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/stage1_scalable_16"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_16.yaml"))
    parser.add_argument("--expected-timesteps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_timesteps = args.expected_timesteps or int(config["training"]["total_timesteps"])
    expected_seeds = args.seeds or [int(seed) for seed in config["matrix"]["seeds"]]
    expected = {
        (str(model), int(level), int(seed))
        for model in config["matrix"]["models"]
        for level in config["matrix"]["observation_levels"]
        for seed in expected_seeds
    }
    reports = _load_reports(args.run_root)
    completed = {
        (
            str(report["run"]["model"]),
            int(report["run"]["observation_level"]),
            int(report["run"]["seed"]),
        )
        for report in reports
        if int(report["completed_timesteps"]) == expected_timesteps
    }

    per_seed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        run = report["run"]
        final = report["final_evaluation"]
        compute = report["compute"]
        row = {
            "model": run["model"],
            "observation_level": run["observation_level"],
            "seed": run["seed"],
            "completed_timesteps": report["completed_timesteps"],
            "parameter_count": report["parameter_count"],
            **{metric: final.get(metric) for metric in FINAL_METRICS},
            "detection_observed_count": final.get("detection_observed_count"),
            "detection_censored_count": final.get("detection_censored_count"),
            "recovery_observed_count": final.get("recovery_observed_count"),
            "recovery_censored_count": final.get("recovery_censored_count"),
            **{metric: compute.get(metric) for metric in COMPUTE_METRICS},
            "report_path": report["_report_path"],
        }
        per_seed_rows.append(row)
        grouped[(str(run["model"]), int(run["observation_level"]))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (model, level), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "model": model,
            "observation_level": level,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
        }
        for metric in (*FINAL_METRICS, *COMPUTE_METRICS):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            metric_mean, metric_std = _mean_std(values)
            aggregate[f"{metric}_seed_mean"] = metric_mean
            aggregate[f"{metric}_seed_std"] = metric_std
        aggregate_rows.append(aggregate)

    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.run_root / "stage1_per_seed.csv", per_seed_rows)
    _write_csv(args.run_root / "stage1_aggregate.csv", aggregate_rows)
    missing = sorted(expected - completed)
    summary = {
        "protocol": config["protocol"],
        "expected_timesteps": expected_timesteps,
        "expected_run_count": len(expected),
        "completed_run_count": len(completed & expected),
        "complete": not missing,
        "missing_runs": [
            {"model": model, "observation_level": level, "seed": seed}
            for model, level, seed in missing
        ],
        "groups": aggregate_rows,
    }
    (args.run_root / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Stage 1 Summary",
        "",
        f"Protocol complete: **{summary['complete']}** "
        f"({summary['completed_run_count']}/{summary['expected_run_count']} runs)",
        "",
        "| Model | Obs | Seeds | Episode return (seed mean +/- SD) | Availability | Privileged hosts | SPS | Inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        episode_return_sd = row["mean_episode_return_seed_std"]
        lines.append(
            f"| {row['model']} | {row['observation_level']} | {row['seed_count']} | "
            f"{row['mean_episode_return_seed_mean']:.3f} +/- "
            f"{episode_return_sd:.3f} | " if episode_return_sd is not None else
            f"| {row['model']} | {row['observation_level']} | {row['seed_count']} | "
            f"{row['mean_episode_return_seed_mean']:.3f} +/- n/a | "
        )
        lines[-1] += (
            f"{row['mean_mission_availability_seed_mean']:.3f} | "
            f"{row['mean_privileged_compromised_hosts_seed_mean']:.3f} | "
            f"{row['training_steps_per_second_seed_mean']:.1f} | "
            f"{row['mean_inference_ms_seed_mean']:.3f} |"
        )
    if missing:
        lines.extend(["", "## Missing Runs", ""])
        lines.extend(f"- `{model}/obs_{level}/seed_{seed}`" for model, level, seed in missing)
    lines.extend(
        [
            "",
            "Latency means include horizon-censored compromises; inspect observed/censored counts before interpretation.",
            "",
        ]
    )
    (args.run_root / "stage1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Summarized {len(reports)} reports: "
        f"{summary['completed_run_count']}/{summary['expected_run_count']} complete"
    )
    if missing and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
