"""Gate main benchmarking on reproducible evidence that each policy learns."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_reports(run_root: Path) -> list[dict[str, Any]]:
    """Load canonical reports from a completed qualification matrix."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob("*/*/seed_*/report.json"))
    ]


def main() -> None:
    """Require improvement from initialization and superiority to random."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/learning_qualification_16_v1"),
    )
    parser.add_argument("--minimum-seed-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.minimum_seed_fraction <= 1.0:
        raise ValueError("minimum-seed-fraction must be in (0, 1].")

    reports = _load_reports(args.run_root)
    if not reports:
        raise FileNotFoundError(f"No qualification reports found under {args.run_root}.")
    baseline_path = args.run_root / "baselines" / "baseline_summary.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing qualification baselines: {baseline_path}")
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    random_episode_return_by_level = {
        int(result["observation_level"]): float(result["mean_episode_return"])
        for result in baseline_summary["results"]
        if result["baseline"] == "random"
    }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        run = report["run"]
        grouped[(str(run["model"]), int(run["observation_level"]))].append(report)

    groups = []
    all_passed = True
    for (model, level), model_reports in sorted(grouped.items()):
        if level not in random_episode_return_by_level:
            raise KeyError(f"Random baseline missing for observation level {level}.")
        random_episode_return = random_episode_return_by_level[level]
        per_seed = []
        for report in sorted(model_reports, key=lambda item: int(item["run"]["seed"])):
            evaluations = report["evaluations"]
            initial_monitor_episode_return = float(
                evaluations[0]["mean_episode_return"]
            )
            final_monitor_episode_return = float(
                report.get("monitor_final_evaluation", evaluations[-1])[
                    "mean_episode_return"
                ]
            )
            final_heldout_episode_return = float(
                report["final_evaluation"]["mean_episode_return"]
            )
            per_seed.append(
                {
                    "seed": int(report["run"]["seed"]),
                    "initial_monitor_episode_return": initial_monitor_episode_return,
                    "final_monitor_episode_return": final_monitor_episode_return,
                    "monitor_episode_return_improvement": (
                        final_monitor_episode_return
                        - initial_monitor_episode_return
                    ),
                    "final_heldout_episode_return": final_heldout_episode_return,
                    "heldout_minus_random_episode_return": (
                        final_heldout_episode_return - random_episode_return
                    ),
                }
            )

        required_seed_count = math.ceil(
            args.minimum_seed_fraction * len(per_seed) - 1e-12
        )
        improved_seed_count = sum(
            row["monitor_episode_return_improvement"] > 0.0
            for row in per_seed
        )
        above_random_seed_count = sum(
            row["heldout_minus_random_episode_return"] > 0.0
            for row in per_seed
        )
        mean_monitor_episode_return_improvement = mean(
            row["monitor_episode_return_improvement"] for row in per_seed
        )
        mean_heldout_minus_random_episode_return = mean(
            row["heldout_minus_random_episode_return"] for row in per_seed
        )
        passed = (
            mean_monitor_episode_return_improvement > 0.0
            and mean_heldout_minus_random_episode_return > 0.0
            and improved_seed_count >= required_seed_count
            and above_random_seed_count >= required_seed_count
        )
        all_passed &= passed
        groups.append(
            {
                "model": model,
                "observation_level": level,
                "passed": passed,
                "seed_count": len(per_seed),
                "required_seed_count": required_seed_count,
                "improved_seed_count": improved_seed_count,
                "above_random_seed_count": above_random_seed_count,
                "mean_monitor_episode_return_improvement": (
                    mean_monitor_episode_return_improvement
                ),
                "mean_heldout_minus_random_episode_return": (
                    mean_heldout_minus_random_episode_return
                ),
                "random_baseline_episode_return": random_episode_return,
                "per_seed": per_seed,
            }
        )

    result = {
        "passed": all_passed,
        "minimum_seed_fraction": args.minimum_seed_fraction,
        "groups": groups,
    }
    (args.run_root / "learning_qualification.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Learning Qualification",
        "",
        f"Passed: **{all_passed}**",
        "",
        "| Model | Obs | Improved seeds | Above-random seeds | Mean train delta | Mean random delta | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for group in groups:
        lines.append(
            f"| {group['model']} | {group['observation_level']} | "
            f"{group['improved_seed_count']}/{group['seed_count']} | "
            f"{group['above_random_seed_count']}/{group['seed_count']} | "
            f"{group['mean_monitor_episode_return_improvement']:.3f} | "
            f"{group['mean_heldout_minus_random_episode_return']:.3f} | "
            f"{group['passed']} |"
        )
    (args.run_root / "learning_qualification.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Learning qualification passed={all_passed}")
    if not all_passed and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
