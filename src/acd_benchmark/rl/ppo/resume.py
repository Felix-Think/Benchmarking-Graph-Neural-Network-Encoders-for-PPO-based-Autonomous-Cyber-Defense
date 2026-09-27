"""Resume committed update boundaries; keep discarded outputs for audit."""

import json
from pathlib import Path
from time import time_ns

import torch


def read_resume_checkpoint(run_dir):
    path = Path(run_dir) / "latest_checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No latest_checkpoint.pt in {run_dir}. Legacy runs cannot be silently "
            "restarted or resumed from weights. Keep the old run and use a new root."
        )
    # Only load trusted checkpoints produced by this project.
    saved = torch.load(path, map_location="cpu")
    if saved.get("resume_version") != 1 or not saved.get("resume_committed", False):
        raise ValueError("Not a committed resume checkpoint")
    return saved


def validate_run_config(previous, current):
    operational = {
        "resume", "progress_every_ticks", "print_every_updates", "no_tensorboard",
        "log_parameter_histograms", "run_dir", "checkpoint_path", "best_checkpoint_path",
        "tensorboard_log_dir",
    }
    for key, value in current.items():
        if key not in operational and previous.get(key) != value:
            raise ValueError(f"Resume run config mismatch: {key}")


def restore_outputs(run_dir, saved, best_path):
    """Roll logs/selection back to the checkpoint's committed transaction."""
    root = Path(run_dir)
    step = saved["total_simulator_steps"]
    archive = root / "discarded_attempts" / str(time_ns())
    to_archive = [
        root / name for name in (
            "training_history.jsonl", "periodic_evaluation.json", "training_summary.json",
            "evaluation_stochastic.json", "evaluation_deterministic.json",
            "evaluation_comparison.json", "evaluation_retention.json", "compute_summary.json",
        )
    ]
    to_archive.append(Path(best_path))
    for path in (root / "milestones").glob("step_*"):
        if int(path.name.split("_")[1]) > step:
            to_archive.append(path)
    for path in to_archive:
        if path.exists():
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive / path.name)
    (root / "training_history.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in saved["history"]),
        encoding="utf-8",
    )
    extra = saved.get("extra_state", {})
    periodic = extra.get("periodic_evaluations", [])
    (root / "periodic_evaluation.json").write_text(
        json.dumps(periodic, indent=2, allow_nan=False), encoding="utf-8",
    )
    if extra.get("best_checkpoint") is not None:
        torch.save(extra["best_checkpoint"], best_path)
    return extra
