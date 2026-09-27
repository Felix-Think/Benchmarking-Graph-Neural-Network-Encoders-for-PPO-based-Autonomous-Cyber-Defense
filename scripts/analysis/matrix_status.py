"""Compact progress plus per-q aggregation across completed training seeds."""

import argparse
import json
from pathlib import Path

from acd_benchmark.metrics.agregate import aggregate_seeds


def read_history(path):
    if not path.exists():
        return []
    # Ignore only an unfinished final append while the training process writes.
    text = path.read_text()
    complete = text[:text.rfind("\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line]


def build_report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    state_path = root / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "starting"}
    cells = []
    grouped = {}
    for job in manifest["jobs"]:
        directory = Path(job["directory"])
        history = read_history(directory / "training_history.jsonl")
        row = {k: job[k] for k in ("encoder", "train_regime", "seed")}
        row["updates"] = len(history)
        row["simulator_steps"] = history[-1]["global_step"] if history else 0
        if history:
            window = history[-50:]
            row["recent_updates"] = len(window)
            for name in ("episode_return_mean", "value_loss", "approx_kl", "entropy", "early_stopped"):
                values = [point[name] for point in window if name in point and point[name] is not None]
                if values:
                    row[name] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
            row["training_seconds"] = history[-1].get("compute/training_seconds")
            row["ticks_per_second"] = history[-1].get("compute/simulator_ticks_per_second")
        milestones = {}
        for step in manifest["config"]["training"]["checkpoint_steps"]:
            path = directory / "milestones" / f"step_{step}" / "validation_retention.json"
            if path.exists():
                curve = json.loads(path.read_text())
                milestones[str(step)] = {
                    "actual_steps": curve["global_step"],
                    "stochastic_returns": {
                        q: result["stochastic"]["mean_return"] for q, result in curve["results"].items()
                    },
                }
        row["milestones"] = milestones
        final_path = directory / "evaluation_retention.json"
        row["final_evaluation_complete"] = final_path.exists()
        if final_path.exists():
            final = json.loads(final_path.read_text())
            row["final_stochastic_returns"] = {
                q: result["stochastic"]["mean_return"] for q, result in final["results"].items()
            }
            for q, result in final["results"].items():
                for mode, evaluation in result.items():
                    key = f"{job['encoder']}/{job['train_regime']}/q_{q}/{mode}"
                    grouped.setdefault(key, {})[job["seed"]] = evaluation["metrics_summary"]
        cells.append(row)
    return {
        "queue_status": state,
        "cells": cells,
        "completed_seed_summaries": {
            key: {"seeds": sorted(seeds), "metrics": aggregate_seeds(seeds)}
            for key, seeds in grouped.items()
        },
        "expected_training_seeds": manifest["config"]["seeds"],
        "note": "Incomplete cells are not results. A per-seed queue does not provide cross-seed uncertainty; combine the five planned training seeds separately. Validation is separate from test.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.run_root)
    path = args.run_root / "analysis_summary.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    print(f"queue={report['queue_status']['status']}")
    for row in report["cells"]:
        if row["updates"] or row["final_evaluation_complete"]:
            print(
                f"{row['encoder']} {row['train_regime']} seed={row['seed']} "
                f"updates={row['updates']} ticks={row['simulator_steps']} "
                f"final_eval={row['final_evaluation_complete']} "
                f"milestones={list(row['milestones'])}"
            )
            if row["milestones"]:
                print(json.dumps(row["milestones"]))
            if "final_stochastic_returns" in row:
                print(json.dumps(row["final_stochastic_returns"]))
    if "error" in report["queue_status"]:
        print(report["queue_status"]["error"])
    print(f"details={path}")


if __name__ == "__main__":
    main()
