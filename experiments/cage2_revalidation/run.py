"""Run a new, explicitly scoped CAGE-2 three-encoder diagnostic."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

from acd_benchmark.cyber.action_space import ACTION_SPACE, build_action, describe_action
from acd_benchmark.cyber.blue_table_env import BlueTableEnv, HOST_NAMES
from acd_benchmark.rl.ppo.training import PPOTrainingConfig, seed_runtime, train_policy
from experiments.cage2_revalidation.policies import make_policy, public_graph


BASE_COMMIT = "291977eec20c0f223b9e507ad41482bd0db268c7"
REPO = Path(__file__).resolve().parents[2]
BASELINES = (
    "sleep",
    "monitor",
    "random",
    "restore_op_server",
    "restore_round_robin",
    "analyse_user0",
)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if commit != BASE_COMMIT:
        raise RuntimeError(
            "This diagnostic must be run on the explicitly pinned main commit"
        )
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=REPO, check=True)
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(filter(None, files)):
        digest.update(name.encode() + b"\0" + (REPO / name).read_bytes() + b"\0")
    extensions = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "experiments").rglob("*.py"))
    }
    return {
        "base_commit": commit,
        "tracked_source_sha256": digest.hexdigest(),
        "extension_sha256": extensions,
    }


def make_env(horizon: int) -> BlueTableEnv:
    env = BlueTableEnv(max_step=horizon)
    env.reset(seed=0)
    rows = env.env._create_blue_table(success=None)._rows
    if tuple(row[2] for row in rows) != HOST_NAMES:
        raise RuntimeError(
            "Native observation row order does not match the declared host slots"
        )
    return env


def action_id(kind: str, host: str | None = None) -> int:
    return ACTION_SPACE.index((kind, host))


def intervention_observations(observation: np.ndarray) -> dict[str, np.ndarray]:
    activity = observation.copy()
    activity[:, :2] = 0
    compromise = observation.copy()
    compromise[:, 2:] = 0
    return {
        "zero_all": np.zeros_like(observation),
        "zero_activity": activity,
        "zero_compromise": compromise,
        "rotate_host_rows": np.roll(observation, 1, axis=0),
    }


def concentration(counts: Counter) -> dict:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot summarize an empty evaluation")
    targets = Counter()
    for index, count in counts.items():
        _, target = ACTION_SPACE[index]
        if target is not None:
            targets[target] += count
    top, share = max(counts.items(), key=lambda pair: pair[1])
    return {
        "decisions": total,
        "action_counts": {
            describe_action(index): count for index, count in sorted(counts.items())
        },
        "restore_share": sum(
            count
            for index, count in counts.items()
            if ACTION_SPACE[index][0] == "Restore"
        )
        / total,
        "top_action": describe_action(top),
        "top_action_share": share / total,
        "distinct_actions": len(counts),
        "target_host_counts": dict(targets),
        "dominant_target_host": targets.most_common(1)[0][0] if targets else None,
        "dominant_target_share_of_targeted_actions": max(targets.values())
        / sum(targets.values())
        if targets
        else None,
    }


@torch.no_grad()
def evaluate(
    *,
    policy,
    baseline,
    episodes,
    first_seed,
    action_seed,
    horizon,
    deterministic,
    output: Path,
) -> dict:
    env = make_env(horizon)
    counter = Counter()
    records = []
    probes = {name: [] for name in intervention_observations(np.zeros((13, 4)))}
    probe_inputs, probe_logits, probe_locations = [], [], []
    entropy_values = []
    started = time.perf_counter()
    if policy is not None:
        policy.eval()
    with torch.random.fork_rng(devices=[]):
        for episode in range(episodes):
            seed = first_seed + episode
            observation, _ = env.reset(seed=seed)
            torch.manual_seed(action_seed + episode)
            rng = np.random.default_rng(action_seed + episode)
            reward_sum, tick, done = 0.0, 0, False
            episode_counts = Counter()
            while not done:
                if policy is not None:
                    logits, _ = policy(
                        torch.as_tensor(observation, dtype=torch.float32)
                    )
                    distribution = Categorical(logits=logits[0])
                    selected = (
                        int(logits[0].argmax())
                        if deterministic
                        else int(distribution.sample())
                    )
                    entropy_values.append(float(distribution.entropy()))
                    if tick % 20 == 0:
                        original = distribution.probs
                        variants = intervention_observations(observation)
                        changed_logits, _ = policy(
                            torch.as_tensor(
                                np.stack(list(variants.values())), dtype=torch.float32
                            )
                        )
                        for position, name in enumerate(variants):
                            changed = changed_logits[position].softmax(-1)
                            probes[name].append(
                                (
                                    float(0.5 * (original - changed).abs().sum()),
                                    int(original.argmax() != changed.argmax()),
                                )
                            )
                        probe_inputs.append(observation.copy())
                        probe_logits.append(logits[0].numpy().copy())
                        probe_locations.append((seed, tick))
                elif baseline == "sleep":
                    selected = action_id("Sleep")
                elif baseline == "monitor":
                    selected = action_id("Monitor")
                elif baseline == "random":
                    selected = int(rng.integers(len(ACTION_SPACE)))
                elif baseline == "restore_op_server":
                    selected = action_id("Restore", "Op_Server0")
                elif baseline == "restore_round_robin":
                    selected = action_id("Restore", HOST_NAMES[tick % len(HOST_NAMES)])
                elif baseline == "analyse_user0":
                    selected = action_id("Analyse", "User0")
                else:
                    raise ValueError(baseline)
                episode_counts[selected] += 1
                observation, reward, terminated, truncated, _ = env.step(
                    build_action(selected)
                )
                reward_sum += float(reward)
                tick += 1
                done = terminated or truncated
            counter.update(episode_counts)
            records.append(
                {
                    "seed": seed,
                    "return": reward_sum,
                    "ticks": tick,
                    "action_counts": {
                        str(key): value for key, value in episode_counts.items()
                    },
                }
            )
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                print(
                    f"evaluation={output.stem} episodes={episode + 1}/{episodes} mean_return={statistics.mean(row['return'] for row in records):.3f}",
                    flush=True,
                )
    returns = [row["return"] for row in records]
    result = {
        "baseline": baseline,
        "deterministic": deterministic if policy is not None else None,
        "evaluation_seed_start": first_seed,
        "action_seed_start": action_seed,
        "episodes": episodes,
        "episode_horizon": horizon,
        "mean_return": statistics.mean(returns),
        "sample_std_return_across_evaluation_episodes": statistics.stdev(returns)
        if episodes > 1
        else None,
        "episode_records": records,
        **concentration(counter),
        "mean_action_entropy": statistics.mean(entropy_values)
        if entropy_values
        else None,
        "same_state_observation_probes": {
            name: {
                "n": len(values),
                "mean_total_variation": statistics.mean(row[0] for row in values),
                "max_total_variation": max(row[0] for row in values),
                "argmax_change_fraction": statistics.mean(row[1] for row in values),
            }
            for name, values in probes.items()
            if values
        },
        "evaluation_wall_seconds": time.perf_counter() - started,
    }
    save_json(output, result)
    if probe_inputs:
        np.savez_compressed(
            output.with_suffix(".probes.npz"),
            observations=np.stack(probe_inputs),
            logits=np.stack(probe_logits),
            seed_tick=np.asarray(probe_locations),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("flat", "gcn", "graphsage"),
        default=("flat", "gcn", "graphsage"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--episode-horizon", type=int, default=200)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.run_root.resolve()
    if root.exists():
        raise FileExistsError(
            f"Use a new output directory; existing artifacts are never overwritten: {root}"
        )
    identity = source_identity()
    graph = public_graph()
    config = PPOTrainingConfig(
        total_timesteps=args.train_steps, seed=args.seed, device="cpu"
    )
    models = {}
    for model in args.models:
        seed_runtime(args.seed)
        candidate = make_policy(model, graph)
        models[model] = sum(parameter.numel() for parameter in candidate.parameters())
    protocol = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "new engineering diagnostic, not recovery of historical pilot artifacts",
        "argv": sys.argv,
        "models_parameter_counts": models,
        "training_config": asdict(config),
        "episode_horizon": args.episode_horizon,
        "training_seeds": [args.seed],
        "evaluation_episodes_per_mode": args.eval_episodes,
        "evaluation_seeds": [10_000, 10_000 + args.eval_episodes - 1],
        "evaluation_action_seed_start": 60_000,
        "action_catalogue": [
            describe_action(index) for index in range(len(ACTION_SPACE))
        ],
        "observation": "native BlueTable 13 x 4, including BlueTable's own state bookkeeping",
        "additional_temporal_belief": False,
        "synthetic_masking": False,
        "action_mask": False,
        "red_agent": "RedMeanderAgent",
        "green_agent": "SleepAgent, unchanged Scenario2.yaml default",
        "graph": graph,
        "baseline_controls": [] if args.skip_baselines else list(BASELINES),
        "probe_interpretation": "same-state input sensitivity, not proof of useful observation use or physical sensor failure",
        "selection_rule": "final fixed-budget policy; no selection or extension based on Restore frequency",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torch-geometric", "numpy", "gym", "cyborg")
            },
        },
    }
    save_json(root / "manifest.json", protocol)
    report = {"protocol": "manifest.json", "baselines": {}, "models": {}}
    if not args.skip_baselines:
        for baseline in BASELINES:
            print(f"BASELINE {baseline}", flush=True)
            report["baselines"][baseline] = evaluate(
                policy=None,
                baseline=baseline,
                episodes=args.eval_episodes,
                first_seed=10_000,
                action_seed=60_000,
                horizon=args.episode_horizon,
                deterministic=False,
                output=root / "baselines" / f"{baseline}.json",
            )
            save_json(root / "summary.json", report)
    for model in args.models:
        print(
            f"TRAIN {model} seed={args.seed} steps={args.train_steps} parameters={models[model]}",
            flush=True,
        )
        directory = root / model / f"seed_{args.seed}"
        directory.mkdir(parents=True)
        checkpoint_path = directory / "checkpoint.pt"
        cell_config = PPOTrainingConfig(
            **{**asdict(config), "checkpoint_path": str(checkpoint_path)}
        )
        seed_runtime(args.seed)
        policy = make_policy(model, graph)
        training_started = time.perf_counter()

        def on_update(metrics):
            row = {**metrics, "wall_seconds": time.perf_counter() - training_started}
            with (directory / "training_history.jsonl").open("a") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"model={model} update={metrics['update']} step={metrics['global_step']} return={metrics['episode_return_mean']} entropy={metrics['entropy']:.4f} approx_kl={metrics['approx_kl']:.6f} elapsed={row['wall_seconds']:.1f}s",
                flush=True,
            )

        result = train_policy(
            make_env(args.episode_horizon), policy, cell_config, on_update=on_update
        )
        train_seconds = time.perf_counter() - training_started
        save_json(
            directory / "training_summary.json",
            {
                "config": asdict(cell_config),
                "total_timesteps": result.total_timesteps,
                "updates": result.total_updates,
                "completed_episodes": len(result.episode_returns),
                "wall_seconds": train_seconds,
                "timesteps_per_second": result.total_timesteps / train_seconds,
                "parameter_count": models[model],
            },
        )
        model_report = {}
        for name, deterministic in (("greedy", True), ("stochastic", False)):
            model_report[name] = evaluate(
                policy=policy,
                baseline=None,
                episodes=args.eval_episodes,
                first_seed=10_000,
                action_seed=60_000,
                horizon=args.episode_horizon,
                deterministic=deterministic,
                output=directory / f"evaluation_{name}.json",
            )
            save_json(
                root / "summary.json",
                {**report, "models": {**report["models"], model: model_report}},
            )
        report["models"][model] = model_report
        save_json(root / "summary.json", report)
        print(
            f"COMPLETE {model}: greedy_return={model_report['greedy']['mean_return']:.3f} greedy_restore={model_report['greedy']['restore_share']:.4f} top_action={model_report['greedy']['top_action']} top_share={model_report['greedy']['top_action_share']:.4f}",
            flush=True,
        )
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_json(root / "summary.json", report)
    if source_identity() != identity:
        raise RuntimeError(
            "Source changed during execution; results need a provenance review"
        )
    save_json(
        root / "completed.json",
        {
            "status": "completed",
            "models": list(args.models),
            "completed_at": report["completed_at"],
        },
    )
    print(f"EXPERIMENT COMPLETED {root}", flush=True)


if __name__ == "__main__":
    main()
