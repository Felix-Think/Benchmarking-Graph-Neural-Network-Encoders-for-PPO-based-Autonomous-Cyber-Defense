"""Run the full Stage 1 matrix sequentially with resumable progress."""

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
    """Return whether a run has a valid final report at the expected budget."""
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(report.get("completed_timesteps", -1)) == expected_timesteps


def _write_progress(path: Path, records: list[dict[str, Any]]) -> None:
    """Persist matrix status after every completed or failed child process."""
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
    """Run missing matrix cells, then summarize and audit the full protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_16.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=("flat", "gcn", "graphsage"))
    parser.add_argument("--observation-levels", nargs="+", type=int, choices=(100, 50, 25))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = args.models or [str(model) for model in config["matrix"]["models"]]
    levels = args.observation_levels or [
        int(level) for level in config["matrix"]["observation_levels"]
    ]
    seeds = args.seeds or [int(seed) for seed in config["matrix"]["seeds"]]
    output_root = args.output_root or Path(config["output_root"])
    expected_timesteps = int(config["training"]["total_timesteps"])
    progress_path = output_root / "matrix_progress.json"
    records: list[dict[str, Any]] = []

    for seed in seeds:
        for level in levels:
            for model in models:
                run_dir = output_root / model / f"obs_{level}" / f"seed_{seed}"
                report_path = run_dir / "report.json"
                record = {
                    "model": model,
                    "observation_level": level,
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
                if not args.overwrite and _is_complete(report_path, expected_timesteps):
                    record["status"] = "skipped_complete"
                    records.append(record)
                    _write_progress(progress_path, records)
                    print(f"SKIP complete: {model} obs={level} seed={seed}", flush=True)
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
                    str(level),
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
                print(f"START: {model} obs={level} seed={seed}", flush=True)
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
                        "scripts.stage1.summarize_results",
                        "--run-root",
                        str(output_root),
                        "--config",
                        str(args.config),
                        "--allow-incomplete",
                    ],
                    check=True,
                )

    if args.dry_run:
        _write_progress(progress_path, records)
        return
    if not args.skip_baselines:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.stage1.evaluate_baselines",
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
            "scripts.stage1.summarize_results",
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
            "scripts.stage1.audit_protocol",
            "--run-root",
            str(output_root),
            "--config",
            str(args.config),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
