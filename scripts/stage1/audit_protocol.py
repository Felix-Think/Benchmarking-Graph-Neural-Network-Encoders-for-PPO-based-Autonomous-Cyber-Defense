"""Audit Stage 1 reports, checkpoints, metric fields, and artifact checksums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.utils.reproducibility import sha256_file  # noqa: E402


REQUIRED_FINAL_METRICS = {
    "mean_episode_return",
    "std_episode_return",
    "mean_mission_availability",
    "mean_privileged_compromised_hosts",
    "mean_detection_latency",
    "detection_observed_count",
    "detection_censored_count",
    "mean_recovery_latency",
    "recovery_observed_count",
    "recovery_censored_count",
    "mean_inference_ms",
}
REQUIRED_COMPUTE_METRICS = {
    "total_wall_seconds",
    "training_seconds_excluding_evaluation",
    "training_steps_per_second",
    "peak_process_rss_mb",
    "peak_gpu_memory_mb",
    "mean_inference_ms",
}
SUPPORTED_PPO_CORES = {
    "acd_benchmark.rl.ppo shared custom PPO",
    "acd_benchmark.rl.graph_ppo shared custom PPO",
}


def _audit_run(
    run_dir: Path,
    *,
    expected_timesteps: int,
    evaluation_frequency: int,
) -> list[str]:
    """Return protocol violations for one expected run directory."""
    errors: list[str] = []

    def require(path: Path) -> bool:
        if path.exists():
            return True
        errors.append(f"missing artifact: {path.name}")
        return False

    report_path = run_dir / "report.json"
    if not require(report_path):
        return errors
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("completed_timesteps", -1)) != expected_timesteps:
        errors.append("completed_timesteps does not match protocol")
    if report.get("ppo_core") not in SUPPORTED_PPO_CORES:
        errors.append("The run is not marked as using the shared PPO core")
    missing_metrics = REQUIRED_FINAL_METRICS - set(report.get("final_evaluation", {}))
    if missing_metrics:
        errors.append(f"missing final metrics: {sorted(missing_metrics)}")
    missing_compute = REQUIRED_COMPUTE_METRICS - set(report.get("compute", {}))
    if missing_compute:
        errors.append(f"missing compute metrics: {sorted(missing_compute)}")

    expected_eval_steps = list(range(0, expected_timesteps + 1, evaluation_frequency))
    if expected_eval_steps[-1] != expected_timesteps:
        expected_eval_steps.append(expected_timesteps)
    actual_eval_steps = [int(item["timesteps"]) for item in report.get("evaluations", [])]
    if actual_eval_steps != expected_eval_steps:
        errors.append(f"evaluation milestones {actual_eval_steps} != {expected_eval_steps}")

    for name in (
        "resolved_config.yaml",
        "source_config.yaml",
        "scenario.yaml",
        "reproducibility.json",
        "package_versions.txt",
        "evaluations.json",
        "evaluation_metrics.csv",
        "episode_metrics.csv",
        "training_metrics.jsonl",
        "progress.json",
        "manifest.json",
        "source_snapshot/snapshot_index.json",
        "checkpoints/latest.pt",
        "checkpoints/best_episode_return.pt",
        "checkpoints/best_availability.pt",
    ):
        require(run_dir / name)
    training = report.get("resolved_config", {}).get("training", {})
    if "final_evaluation_base_seed" in training:
        require(run_dir / "final_evaluation.json")
        require(run_dir / "final_episode_metrics.csv")
        final_episodes = report.get("final_evaluation", {}).get("episode_count")
        if final_episodes != int(training["final_evaluation_episodes"]):
            errors.append("held-out final episode count does not match protocol")
    for step in expected_eval_steps[1:]:
        require(run_dir / "checkpoints" / f"step_{step}.pt")

    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu")
        for key in (
            "policy_state_dict",
            "optimizer_state_dict",
            "resolved_config",
            "ppo_config",
            "normalizer_statistics",
            "reproducibility",
            "evaluation",
        ):
            if key not in checkpoint:
                errors.append(f"checkpoint missing key: {key}")
        if int(checkpoint.get("timesteps", -1)) != expected_timesteps:
            errors.append("latest checkpoint timestep mismatch")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("artifacts", []):
            artifact = run_dir / entry["path"]
            if not artifact.exists():
                errors.append(f"manifest artifact missing: {entry['path']}")
            elif sha256_file(artifact) != entry["sha256"]:
                errors.append(f"manifest checksum mismatch: {entry['path']}")
    return errors


def main() -> None:
    """Audit an expected Stage 1 matrix and write a permanent audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/stage1_scalable_16"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_16.yaml"))
    parser.add_argument("--expected-timesteps", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_timesteps = args.expected_timesteps or int(config["training"]["total_timesteps"])
    evaluation_frequency = args.eval_frequency or int(config["training"]["evaluation_frequency"])
    seeds = args.seeds or [int(seed) for seed in config["matrix"]["seeds"]]
    scenario = yaml.safe_load(Path(config["scenario"]["path"]).read_text(encoding="utf-8"))
    protocol_errors = []
    if len(scenario.get("Hosts", {})) != int(config["scenario"]["node_count"]):
        protocol_errors.append("scenario host count does not match configured node_count")
    if config["observation"].get("topology_masked") is not False:
        protocol_errors.append("topology must remain visible under partial observability")
    if config["observation"].get("visibility_feature") != "visibility.host_visible":
        protocol_errors.append("visibility feature is missing from masking protocol")
    if "evaluation_base_seed" not in config["training"]:
        protocol_errors.append("a common evaluation seed is required across all runs")
    if config.get("baselines", {}).get("required", False):
        baseline_path = args.run_root / "baselines" / "baseline_summary.json"
        if not baseline_path.exists():
            protocol_errors.append("required baseline_summary.json is missing")
        else:
            baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
            expected_baselines = {
                (baseline, int(level))
                for baseline in ("sleep", "monitor", "random")
                for level in config["matrix"]["observation_levels"]
            }
            actual_baselines = {
                (str(result["baseline"]), int(result["observation_level"]))
                for result in baseline_summary.get("results", [])
            }
            if actual_baselines != expected_baselines:
                protocol_errors.append("baseline matrix does not match the protocol")

    runs: list[dict[str, Any]] = []
    for model in config["matrix"]["models"]:
        for level in config["matrix"]["observation_levels"]:
            for seed in seeds:
                run_dir = args.run_root / str(model) / f"obs_{level}" / f"seed_{seed}"
                errors = _audit_run(
                    run_dir,
                    expected_timesteps=expected_timesteps,
                    evaluation_frequency=evaluation_frequency,
                )
                runs.append(
                    {
                        "model": model,
                        "observation_level": level,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "passed": not errors,
                        "errors": errors,
                    }
                )
    passed = not protocol_errors and all(run["passed"] for run in runs)
    audit = {
        "protocol": config["protocol"],
        "passed": passed,
        "expected_timesteps": expected_timesteps,
        "evaluation_frequency": evaluation_frequency,
        "protocol_errors": protocol_errors,
        "run_count": len(runs),
        "passed_run_count": sum(run["passed"] for run in runs),
        "runs": runs,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Stage 1 Protocol Audit",
        "",
        f"Passed: **{passed}**",
        f"Runs: **{audit['passed_run_count']}/{audit['run_count']}**",
        "",
    ]
    for error in protocol_errors:
        lines.append(f"- Protocol: {error}")
    for run in runs:
        if run["passed"]:
            continue
        label = f"{run['model']}/obs_{run['observation_level']}/seed_{run['seed']}"
        lines.extend(f"- `{label}`: {error}" for error in run["errors"])
    (args.run_root / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Stage 1 audit: {audit['passed_run_count']}/{audit['run_count']} runs passed")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
