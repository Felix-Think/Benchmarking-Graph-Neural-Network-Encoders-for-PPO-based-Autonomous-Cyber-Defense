"""Run one reproducible Stage 1 PPO encoder experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._bootstrap import ensure_src_on_path, repository_root

ensure_src_on_path()

import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from acd_benchmark.evaluation.operational_metrics import (  # noqa: E402
    evaluate_policy_with_operational_metrics,
)
from acd_benchmark.rl.ppo.config import PPOConfig  # noqa: E402
from acd_benchmark.rl.ppo.optimizer import make_optimizer  # noqa: E402
from acd_benchmark.rl.ppo.training import train_vector_update  # noqa: E402
from acd_benchmark.rl.pilot2_factory import make_pilot2_env  # noqa: E402
from acd_benchmark.rl.rollout import EpisodeReturnTracker  # noqa: E402
from acd_benchmark.rl.runtime import resolve_device, set_torch_seed  # noqa: E402
from acd_benchmark.rl.stage1_factory import (  # noqa: E402
    SUPPORTED_STAGE1_MODELS,
    build_stage1_policy,
    make_stage1_env,
    resolve_model_hidden_dim,
)
from acd_benchmark.utils.reproducibility import (  # noqa: E402
    collect_reproducibility_metadata,
    sha256_file,
    write_artifact_manifest,
    write_package_versions,
    write_source_snapshot,
)


def parse_args() -> argparse.Namespace:
    """Parse one matrix cell plus smoke-test overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_16.yaml"))
    parser.add_argument("--model", choices=SUPPORTED_STAGE1_MODELS, required=True)
    parser.add_argument(
        "--observation-level",
        choices=("100", "75", "50", "25", "mixed"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--final-eval-episodes", type=int)
    parser.add_argument("--final-eval-base-seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps-per-env", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    """Load the frozen protocol configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_config(args: argparse.Namespace, source: dict[str, Any]) -> dict[str, Any]:
    """Merge one matrix cell and explicit smoke overrides into the config."""
    config = deepcopy(source)
    observation_level = str(args.observation_level)
    run_config: dict[str, Any] = {
        "model": args.model,
        "observation_level": observation_level,
        "seed": args.seed,
    }
    if observation_level == "mixed":
        run_config["training_q_values"] = [
            float(value) for value in config["observation"]["q_values"]
        ]
    else:
        run_config["observation_probability"] = int(observation_level) / 100.0
    config["run"] = run_config
    overrides = {
        "total_timesteps": args.total_timesteps,
        "evaluation_frequency": args.eval_frequency,
        "evaluation_episodes": args.eval_episodes,
        "final_evaluation_episodes": args.final_eval_episodes,
        "final_evaluation_base_seed": args.final_eval_base_seed,
        "num_envs": args.num_envs,
        "rollout_steps_per_env": args.rollout_steps_per_env,
        "ppo_epochs": args.ppo_epochs,
    }
    for key, value in overrides.items():
        if value is not None:
            config["training"][key] = value
    if args.device is not None:
        config["compute"]["device"] = args.device
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Reject settings that cannot hit exact vectorized evaluation milestones."""
    training = config["training"]
    num_envs = int(training["num_envs"])
    for key in ("total_timesteps", "evaluation_frequency"):
        if int(training[key]) % num_envs != 0:
            raise ValueError(f"{key} must be divisible by num_envs.")
    if int(training["evaluation_frequency"]) > int(training["total_timesteps"]):
        raise ValueError("evaluation_frequency cannot exceed total_timesteps.")
    final_keys = {
        "final_evaluation_episodes",
        "final_evaluation_base_seed",
    }
    configured_final_keys = final_keys & set(training)
    if configured_final_keys and configured_final_keys != final_keys:
        raise ValueError(
            "final_evaluation_episodes and final_evaluation_base_seed must be set together."
        )
    if configured_final_keys:
        if int(training["final_evaluation_episodes"]) <= 0:
            raise ValueError("final_evaluation_episodes must be positive.")
        if int(training["final_evaluation_base_seed"]) == int(
            training["evaluation_base_seed"]
        ):
            raise ValueError("Final evaluation seeds must be held out from monitoring.")


def _write_evaluation_artifacts(run_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    """Persist raw JSON plus aggregate and per-episode CSV tables."""
    (run_dir / "evaluations.json").write_text(
        json.dumps(evaluations, indent=2),
        encoding="utf-8",
    )
    aggregate_rows = []
    episode_rows = []
    for evaluation in evaluations:
        fixed_q_results = evaluation.get("by_q")
        if fixed_q_results is None:
            fixed_q_results = {"fixed": evaluation}
        for q_label, metrics in fixed_q_results.items():
            aggregate_rows.append(
                {
                    "checkpoint_timesteps": evaluation["timesteps"],
                    "evaluation_q": q_label,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"episodes", "timesteps", "by_q"}
                    },
                }
            )
            for episode in metrics["episodes"]:
                episode_rows.append(
                    {
                        "checkpoint_timesteps": evaluation["timesteps"],
                        "evaluation_q": q_label,
                        **episode,
                    }
                )
    with (run_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (run_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)


def _write_final_evaluation_artifacts(
    run_dir: Path,
    evaluation: dict[str, Any],
) -> None:
    """Persist a held-out final evaluation separately from monitor checkpoints."""
    (run_dir / "final_evaluation.json").write_text(
        json.dumps(evaluation, indent=2),
        encoding="utf-8",
    )
    episode_rows = []
    fixed_q_results = evaluation.get("by_q")
    if fixed_q_results is None:
        fixed_q_results = {"fixed": evaluation}
    for q_label, metrics in fixed_q_results.items():
        episode_rows.extend(
            {"evaluation_q": q_label, **episode}
            for episode in metrics.get("episodes", [])
        )
    if not episode_rows:
        return
    with (run_dir / "final_episode_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)


def _gpu_peak_memory_mb(device: torch.device) -> float | None:
    """Return allocated GPU peak when the selected backend exposes it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0**2)
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return float(torch.mps.driver_allocated_memory()) / (1024.0**2)
    return None


def _q_label(q: float) -> str:
    """Return the stable q label used by Pilot v2 artifacts."""
    return f"q{int(round(q * 100)):03d}"


def _training_q_exposure(
    envs: list[Any],
    q_values: tuple[float, ...],
) -> dict[str, Any]:
    """Aggregate realized mixed-q exposure across vector environments."""
    labels = [_q_label(q) for q in q_values]
    episode_counts = {label: 0 for label in labels}
    timestep_counts = {label: 0 for label in labels}
    for env in envs:
        for label, count in env.q_episode_counts.items():
            episode_counts[label] += int(count)
        for label, count in env.q_step_counts.items():
            timestep_counts[label] += int(count)
    episode_total = sum(episode_counts.values())
    timestep_total = sum(timestep_counts.values())
    return {
        "episode_counts": episode_counts,
        "timestep_counts": timestep_counts,
        "episode_proportions": {
            label: count / episode_total if episode_total else 0.0
            for label, count in episode_counts.items()
        },
        "timestep_proportions": {
            label: count / timestep_total if timestep_total else 0.0
            for label, count in timestep_counts.items()
        },
    }


def _aggregate_retention_curve(by_q: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Expose simple checkpoint-selection summaries without hiding fixed-q data."""
    if not by_q:
        raise ValueError("A retention curve requires at least one fixed-q result.")
    metrics = list(by_q.values())
    return {
        "evaluation_mode": "fixed_q_retention_curve",
        "mean_episode_return": sum(x["mean_episode_return"] for x in metrics) / len(metrics),
        "mean_mission_availability": (
            sum(x["mean_mission_availability"] for x in metrics) / len(metrics)
        ),
        "mean_privileged_compromised_hosts": (
            sum(x["mean_privileged_compromised_hosts"] for x in metrics) / len(metrics)
        ),
        "by_q": by_q,
    }


def _retention_curve_report_lines(evaluation: dict[str, Any]) -> list[str]:
    """Format every fixed-q point instead of hiding it behind a curve mean."""
    lines = [
        f"- curve_mean_episode_return: `{evaluation['mean_episode_return']:.4f}`",
        f"- curve_mean_mission_availability: "
        f"`{evaluation['mean_mission_availability']:.4f}`",
        f"- curve_mean_privileged_compromised_hosts: "
        f"`{evaluation['mean_privileged_compromised_hosts']:.4f}`",
        "",
        "## Fixed-q evaluation",
        "",
        "| q | Episode return | Mission availability | Privileged compromised hosts |",
        "|---:|---:|---:|---:|",
    ]
    for q_label, metrics in evaluation["by_q"].items():
        lines.append(
            f"| {q_label} | {metrics['mean_episode_return']:.4f} | "
            f"{metrics['mean_mission_availability']:.4f} | "
            f"{metrics['mean_privileged_compromised_hosts']:.4f} |"
        )
    return lines


def _save_checkpoint(
    path: Path,
    *,
    policy,
    optimizer,
    timesteps: int,
    resolved_config: dict[str, Any],
    ppo_config: PPOConfig,
    reproducibility: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """Save policy/optimizer state and metadata for independent evaluation.

    The checkpoint does not claim bit-exact mid-rollout resume because CybORG
    simulator states for every vector environment are not serialized.
    """
    torch.save(
        {
            "timesteps": timesteps,
            "model": resolved_config["run"]["model"],
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "resolved_config": resolved_config,
            "ppo_config": asdict(ppo_config),
            "normalizer_statistics": None,
            "reproducibility": reproducibility,
            "evaluation": evaluation,
        },
        path,
    )


def main() -> None:
    """Train, evaluate, checkpoint, and fully describe one Stage 1 run."""
    args = parse_args()
    source_config = _load_config(args.config)
    config = _resolve_config(args, source_config)
    _validate_config(config)
    run = config["run"]
    training = config["training"]
    scenario = config["scenario"]
    model_config = config["model"]
    device = resolve_device(str(config["compute"]["device"]))
    torch_threads = int(config["compute"]["torch_threads"])
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)
    set_torch_seed(int(run["seed"]))

    output_root = Path(config["output_root"])
    run_dir = (
        output_root
        / str(run["model"])
        / f"obs_{run['observation_level']}"
        / f"seed_{run['seed']}"
    )
    if (run_dir / "report.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed run already exists: {run_dir}")
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    scenario_path = Path(str(scenario["path"]))
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    shutil.copy2(scenario_path, run_dir / "scenario.yaml")
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    repo_root = repository_root()
    reproducibility = collect_reproducibility_metadata(repo_root, device=device)
    reproducibility["resolved_config_sha256"] = sha256_file(resolved_config_path)
    reproducibility["scenario_sha256"] = sha256_file(run_dir / "scenario.yaml")
    protocol_note = (
        "notes/pilot2_protocol.md"
        if str(config["protocol"]).startswith("pilot2_")
        else "notes/stage1_protocol.md"
    )
    reproducibility["source_snapshot"] = write_source_snapshot(
        run_dir,
        repo_root,
        (
            "src/acd_benchmark",
            "scripts/__init__.py",
            "scripts/_bootstrap.py",
            "scripts/stage1",
            "scripts/pilot2",
            str(args.config),
            str(scenario_path),
            protocol_note,
            "pyproject.toml",
            "uv.lock",
        ),
    )
    (run_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2),
        encoding="utf-8",
    )
    write_package_versions(run_dir / "package_versions.txt")

    is_pilot2 = str(config["protocol"]).startswith("pilot2_")
    q_values = tuple(float(value) for value in config["observation"].get("q_values", ()))
    if is_pilot2 and run["observation_level"] != "mixed":
        raise ValueError("Pilot v2 training must use --observation-level mixed.")

    def make_env(*, fixed_q: float | None = None):
        if is_pilot2:
            return make_pilot2_env(
                str(run["model"]),
                scenario_path=scenario_path,
                node_count=int(scenario["node_count"]),
                q_values=q_values,
                fixed_q=fixed_q,
                max_steps=int(scenario["max_episode_steps"]),
                flat_include_adjacency=bool(
                    config["observation"]["flat_include_adjacency"]
                ),
            )
        return make_stage1_env(
            str(run["model"]),
            scenario_path=scenario_path,
            node_count=int(scenario["node_count"]),
            observation_probability=float(run["observation_probability"]),
            max_steps=int(scenario["max_episode_steps"]),
        )

    num_envs = int(training["num_envs"])
    train_envs = [make_env() for _ in range(num_envs)]
    episode_return_tracker = EpisodeReturnTracker.for_envs(num_envs)
    observations = [
        env.reset(seed=int(run["seed"]) * 10_000 + env_index)[0]
        for env_index, env in enumerate(train_envs)
    ]
    policy = build_stage1_policy(
        str(run["model"]),
        observations[0],
        train_envs[0],
        hidden_dim=resolve_model_hidden_dim(str(run["model"]), model_config),
    ).to(device)
    ppo_config = PPOConfig(
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        ppo_clip=float(training["ppo_clip"]),
        ppo_epochs=int(training["ppo_epochs"]),
        n_steps=int(training["rollout_steps_per_env"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        entropy_coef=float(training["entropy_coef"]),
        value_coef=float(training["value_coef"]),
        max_grad_norm=float(training["max_grad_norm"]),
    )
    optimizer = make_optimizer(policy, ppo_config)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process(os.getpid())
    peak_rss_mb = process.memory_info().rss / (1024.0**2)
    evaluations = []

    def evaluate(
        timesteps: int,
        *,
        episodes: int | None = None,
        base_seed: int | None = None,
    ) -> dict[str, Any]:
        evaluation_episodes = (
            int(training["evaluation_episodes"])
            if episodes is None
            else int(episodes)
        )
        evaluation_seed = (
            int(training["evaluation_base_seed"])
            if base_seed is None
            else int(base_seed)
        )
        if is_pilot2:
            by_q = {}
            for q in (float(value) for value in config["evaluation"]["retention_levels"]):
                metrics = evaluate_policy_with_operational_metrics(
                    make_env(fixed_q=q),
                    policy,
                    episodes=evaluation_episodes,
                    seed=evaluation_seed,
                    device=device,
                )
                by_q[_q_label(q)] = metrics
            return {"timesteps": timesteps, **_aggregate_retention_curve(by_q)}

        metrics = evaluate_policy_with_operational_metrics(
            make_env(),
            policy,
            episodes=evaluation_episodes,
            seed=evaluation_seed,
            device=device,
        )
        return {"timesteps": timesteps, **metrics}

    total_wall_started = time.perf_counter()
    evaluations.append(evaluate(0))
    _write_evaluation_artifacts(run_dir, evaluations)
    completed_timesteps = 0
    train_elapsed_seconds = 0.0
    next_evaluation = int(training["evaluation_frequency"])
    total_timesteps = int(training["total_timesteps"])
    best_episode_return = float("-inf")
    best_availability = float("-inf")
    update_index = 0

    with (run_dir / "training_metrics.jsonl").open("w", encoding="utf-8") as train_log:
        while completed_timesteps < total_timesteps:
            remaining = min(next_evaluation, total_timesteps) - completed_timesteps
            steps_per_env = min(int(training["rollout_steps_per_env"]), remaining // num_envs)
            if steps_per_env <= 0:
                raise RuntimeError("Vector rollout cannot reach the requested milestone exactly.")
            update_started = time.perf_counter()
            observations, losses = train_vector_update(
                train_envs,
                policy,
                optimizer,
                observations,
                ppo_config,
                device,
                steps_per_env=steps_per_env,
                episode_return_tracker=episode_return_tracker,
            )
            update_seconds = time.perf_counter() - update_started
            train_elapsed_seconds += update_seconds
            collected_steps = steps_per_env * num_envs
            completed_timesteps += collected_steps
            update_index += 1
            peak_rss_mb = max(
                peak_rss_mb,
                process.memory_info().rss / (1024.0**2),
            )
            train_record = {
                "update": update_index,
                "timesteps": completed_timesteps,
                "collected_steps": collected_steps,
                "update_seconds": update_seconds,
                "train_steps_per_second": collected_steps / update_seconds,
                **losses,
            }
            if is_pilot2:
                q_exposure = _training_q_exposure(train_envs, q_values)
                train_record.update(
                    {
                        f"train_episodes_{label}": count
                        for label, count in q_exposure["episode_counts"].items()
                    }
                )
                train_record.update(
                    {
                        f"train_timesteps_{label}": count
                        for label, count in q_exposure["timestep_counts"].items()
                    }
                )
            train_log.write(json.dumps(train_record) + "\n")
            train_log.flush()

            if completed_timesteps == next_evaluation or completed_timesteps == total_timesteps:
                evaluation = evaluate(completed_timesteps)
                evaluations.append(evaluation)
                _write_evaluation_artifacts(run_dir, evaluations)
                checkpoint_path = checkpoint_dir / f"step_{completed_timesteps}.pt"
                _save_checkpoint(
                    checkpoint_path,
                    policy=policy,
                    optimizer=optimizer,
                    timesteps=completed_timesteps,
                    resolved_config=config,
                    ppo_config=ppo_config,
                    reproducibility=reproducibility,
                    evaluation=evaluation,
                )
                shutil.copy2(checkpoint_path, checkpoint_dir / "latest.pt")
                if evaluation["mean_episode_return"] > best_episode_return:
                    best_episode_return = float(evaluation["mean_episode_return"])
                    shutil.copy2(
                        checkpoint_path,
                        checkpoint_dir / "best_episode_return.pt",
                    )
                if evaluation["mean_mission_availability"] > best_availability:
                    best_availability = float(evaluation["mean_mission_availability"])
                    shutil.copy2(checkpoint_path, checkpoint_dir / "best_availability.pt")
                progress = {
                    "completed_timesteps": completed_timesteps,
                    "latest_evaluation": evaluation,
                    "checkpoint": str(checkpoint_path.relative_to(run_dir)),
                }
                (run_dir / "progress.json").write_text(
                    json.dumps(progress, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[{run['model']} obs={run['observation_level']} seed={run['seed']}] "
                    f"steps={completed_timesteps} "
                    f"episode_return={evaluation['mean_episode_return']:.4f} "
                    f"availability={evaluation['mean_mission_availability']:.4f}",
                    flush=True,
                )
                next_evaluation = min(
                    next_evaluation + int(training["evaluation_frequency"]),
                    total_timesteps,
                )

    monitor_final_evaluation = evaluations[-1]
    if "final_evaluation_base_seed" in training:
        final_evaluation = evaluate(
            completed_timesteps,
            episodes=int(training["final_evaluation_episodes"]),
            base_seed=int(training["final_evaluation_base_seed"]),
        )
        _write_final_evaluation_artifacts(run_dir, final_evaluation)
    else:
        final_evaluation = monitor_final_evaluation
    total_wall_seconds = time.perf_counter() - total_wall_started
    compute = {
        "device": str(device),
        "num_envs": num_envs,
        "total_wall_seconds": total_wall_seconds,
        "training_seconds_excluding_evaluation": train_elapsed_seconds,
        "training_steps_per_second": completed_timesteps / train_elapsed_seconds,
        "peak_process_rss_mb": peak_rss_mb,
        "peak_gpu_memory_mb": _gpu_peak_memory_mb(device),
        "mean_inference_ms": (
            float(
                sum(
                    metrics["mean_inference_ms"]
                    for metrics in final_evaluation["by_q"].values()
                )
                / len(final_evaluation["by_q"])
            )
            if is_pilot2
            else final_evaluation["mean_inference_ms"]
        ),
    }
    report = {
        "protocol": config["protocol"],
        "run": run,
        "completed_timesteps": completed_timesteps,
        "parameter_count": parameter_count,
        "ppo_core": "acd_benchmark.rl.ppo shared custom PPO",
        "resolved_config": config,
        "ppo_config": asdict(ppo_config),
        "compute": compute,
        "reproducibility": reproducibility,
        "evaluations": evaluations,
        "monitor_final_evaluation": monitor_final_evaluation,
        "final_evaluation": final_evaluation,
    }
    if is_pilot2:
        report["training_q_exposure"] = _training_q_exposure(train_envs, q_values)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        f"# {config['protocol']} - {str(run['model']).upper()} - "
        f"Obs {run['observation_level']} - Seed {run['seed']}",
        "",
        f"- completed_timesteps: `{completed_timesteps}`",
    ]
    if is_pilot2:
        report_lines.extend(_retention_curve_report_lines(final_evaluation))
    else:
        report_lines.extend(
            [
                f"- final_episode_return: "
                f"`{final_evaluation['mean_episode_return']:.4f} +/- "
                f"{final_evaluation['std_episode_return']:.4f}`",
                f"- mission_availability: "
                f"`{final_evaluation['mean_mission_availability']:.4f}`",
                f"- privileged_compromised_hosts: "
                f"`{final_evaluation['mean_privileged_compromised_hosts']:.4f}`",
                f"- detection_latency: `{final_evaluation['mean_detection_latency']}`",
                f"- recovery_latency: `{final_evaluation['mean_recovery_latency']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            f"- training_steps_per_second: `{compute['training_steps_per_second']:.2f}`",
            f"- peak_process_rss_mb: `{peak_rss_mb:.2f}`",
            f"- git_commit: `{reproducibility['git_commit']}`",
            f"- git_dirty: `{reproducibility['git_dirty']}`",
            "",
        ]
    )
    (run_dir / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    manifest = write_artifact_manifest(run_dir)
    print(f"Saved {config['protocol']} run to {run_dir} ({manifest['artifact_count']} artifacts)")


if __name__ == "__main__":
    main()
