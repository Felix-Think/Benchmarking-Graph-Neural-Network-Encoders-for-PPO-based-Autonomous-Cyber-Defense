"""Run the six Pilot v2 observation-qualification cells sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _is_complete(report_path: Path, expected_timesteps: int) -> bool:
    """Return whether one cell has a complete final report."""
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(report.get("completed_timesteps", -1)) == expected_timesteps


def _write_progress(path: Path, records: list[dict[str, Any]]) -> None:
    """Persist status after each child process for resumable execution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "runs": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run missing cells, fixed-q baselines, summary, and integrity audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot2_observation_16.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=("flat", "graphsage"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["matrix"]["observation_levels"] != ["mixed"]:
        raise ValueError("Pilot v2 matrix must contain only observation level 'mixed'.")
    models = args.models or [str(model) for model in config["matrix"]["models"]]
    seeds = args.seeds or [int(seed) for seed in config["matrix"]["seeds"]]
    output_root = args.output_root or Path(config["output_root"])
    expected_timesteps = int(config["training"]["total_timesteps"])
    progress_path = output_root / "matrix_progress.json"
    records: list[dict[str, Any]] = []

    for seed in seeds:
        for model in models:
            run_dir = output_root / model / "obs_mixed" / f"seed_{seed}"
            report_path = run_dir / "report.json"
            record: dict[str, Any] = {
                "model": model,
                "observation_level": "mixed",
                "seed": seed,
                "run_dir": str(run_dir),
            }
            if not args.overwrite and _is_complete(report_path, expected_timesteps):
                record["status"] = "skipped_complete"
                records.append(record)
                _write_progress(progress_path, records)
                print(f"SKIP complete: {model} obs=mixed seed={seed}", flush=True)
                continue

            command = [
                sys.executable,
                "-m",
                "scripts.stage1.train_ppo",
                "--config",
                str(args.config),
                "--model",
                model,
                "--observation-level",
                "mixed",
                "--seed",
                str(seed),
                "--output-root",
                str(output_root),
            ]
            if args.overwrite:
                command.append("--overwrite")
            if args.dry_run:
                record["status"] = "dry_run"
                record["command"] = command
                records.append(record)
                print(" ".join(command), flush=True)
                continue

            record["status"] = "running"
            records.append(record)
            _write_progress(progress_path, records)
            print(f"START: {model} obs=mixed seed={seed}", flush=True)
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                record["status"] = "failed"
                record["returncode"] = error.returncode
                _write_progress(progress_path, records)
                raise
            record["status"] = "completed"
            _write_progress(progress_path, records)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.pilot2.summarize_results",
                    "--run-root",
                    str(output_root),
                    "--config",
                    str(args.config),
                    "--allow-incomplete",
                ],
                check=True,
            )

    if args.dry_run:
        return
    if not args.skip_baselines:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.pilot2.evaluate_baselines",
                "--config",
                str(args.config),
                "--output-dir",
                str(output_root / "baselines"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pilot2.summarize_results",
            "--run-root",
            str(output_root),
            "--config",
            str(args.config),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pilot2.audit_protocol",
            "--run-root",
            str(output_root),
            "--config",
            str(args.config),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
