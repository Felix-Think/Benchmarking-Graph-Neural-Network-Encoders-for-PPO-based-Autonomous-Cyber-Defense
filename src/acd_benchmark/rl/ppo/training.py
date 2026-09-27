from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Mapping
import random

import numpy as np
import torch

from acd_benchmark.rl.ppo.update import ppo_update
from acd_benchmark.rl.rollout.rollout import collect_rollout, concatenate_batches
from acd_benchmark.rl.rollout.parallel import episode_action_rng, rollout_action_seed
from acd_benchmark.metrics.agregate import aggregate_episodes
from acd_benchmark.metrics.compute import ComputeMetrics

MetricValue = float | int | bool | None
UpdateCallback = Callable[[dict[str, MetricValue]], None]


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_simulator_steps: int = 100_000
    episodes_per_update: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 64
    reward_scale: float = 0.01
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str | None = None
    checkpoint_steps: tuple[int, ...] = ()
    rollout_workers: int = 1
    action_seed_protocol: str = "global"


@dataclass
class TrainingResult:
    history: list[dict[str, MetricValue]]
    episode_returns: list[float]
    episode_lengths: list[int]
    total_simulator_steps: int
    total_policy_decisions: int
    total_episodes: int
    total_updates: int
    episode_cyber_metrics: list[dict]
    compute_metrics: dict

    @property
    def total_timesteps(self) -> int:
        """Backward-compatible alias for simulator-step based logging."""
        return self.total_simulator_steps


def seed_runtime(seed: int) -> None:
    """Seed action sampling and the numerical libraries used by training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: PPOTrainingConfig,
    result: TrainingResult,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "history": result.history,
            "episode_returns": result.episode_returns,
            "episode_lengths": result.episode_lengths,
            "episode_cyber_metrics": result.episode_cyber_metrics,
            "compute_metrics": result.compute_metrics,
            "total_simulator_steps": result.total_simulator_steps,
            "total_policy_decisions": result.total_policy_decisions,
            "total_episodes": result.total_episodes,
            "total_updates": result.total_updates,
            # Compatibility with existing checkpoint readers.
            "total_timesteps": result.total_simulator_steps,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def save_best_evaluation_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: PPOTrainingConfig,
    *,
    update: int,
    global_step: int,
    training_metrics: Mapping[str, MetricValue],
    evaluation: Mapping[str, object],
    metric_name: str,
    metric_value: float,
) -> None:
    """Save the policy with the best fixed-seed periodic evaluation score."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "best_periodic_evaluation",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "selection_metric": {
                "name": metric_name,
                "mode": "max",
                "value": float(metric_value),
            },
            "training_metrics": dict(training_metrics),
            "evaluation": dict(evaluation),
            "total_updates": int(update),
            "total_simulator_steps": int(global_step),
            # Compatibility with existing checkpoint readers.
            "total_timesteps": int(global_step),
        },
        checkpoint_path,
    )


def train_policy(
    env,
    policy: torch.nn.Module,
    config: PPOTrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    on_update: UpdateCallback | None = None,
    on_milestone: Callable[[TrainingResult, int], None] | None = None,
    episode_collector=None,
) -> TrainingResult:
    """Train one shared policy using complete CAGE 4 episodes.

    Each PPO update combines ``episodes_per_update`` on-policy trajectories.
    An optional collector runs complete episodes concurrently, not PPO updates.
    The budget is measured in simulator ticks and rollouts are never cut in the
    middle of an episode. The final update can therefore exceed the requested
    budget by at most one episode.
    """
    assert config.total_simulator_steps > 0
    assert config.episodes_per_update > 0
    assert config.minibatch_size > 0
    assert config.update_epochs > 0
    assert config.reward_scale > 0
    if config.action_seed_protocol not in ("global", "per_episode_v1"):
        raise ValueError("Unknown action RNG protocol")
    from acd_benchmark.rl.runtime import learner_device

    device = learner_device(config.device)
    if device.type == "cuda" and episode_collector is None:
        raise ValueError("CUDA learner requires CPU episode workers, even with one worker")
    if config.rollout_workers not in (1, 2, 3):
        raise ValueError("Use 1-3 rollout workers")
    if config.rollout_workers > 1 and episode_collector is None:
        raise ValueError("Parallel training requires an episode collector")
    if episode_collector is not None:
        if config.action_seed_protocol != "per_episode_v1":
            raise ValueError("Parallel collection requires per_episode_v1 RNG")
        if episode_collector.workers != config.rollout_workers:
            raise ValueError("Configured and actual rollout worker counts differ")
    if config.checkpoint_steps and not config.checkpoint_path:
        raise ValueError("Milestones require checkpoint_path")
    if any(step <= 0 or step > config.total_simulator_steps for step in config.checkpoint_steps):
        raise ValueError("Checkpoint steps must lie within the training budget")
    seed_runtime(config.seed)
    policy.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
        )

    history: list[dict[str, MetricValue]] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    global_simulator_steps = 0
    total_policy_decisions = 0
    episode_index = 0
    update_index = 0

    episode_cyber_metrics = []
    compute = ComputeMetrics(device)
    pending_milestones = sorted(set(config.checkpoint_steps))

    def current_result():
        return TrainingResult(
            history=list(history),
            episode_returns=list(episode_returns),
            episode_lengths=list(episode_lengths),
            total_simulator_steps=global_simulator_steps,
            total_policy_decisions=total_policy_decisions,
            total_episodes=episode_index,
            total_updates=update_index,
            episode_cyber_metrics=list(episode_cyber_metrics),
            compute_metrics=compute.summary(),
        )

    while global_simulator_steps < config.total_simulator_steps:
        rollout_batches = []
        update_returns = []
        update_lengths = []
        update_simulator_steps = 0
        update_cyber_metrics = []

        compute.start_training()
        def serial_episodes():
            for local_index in range(config.episodes_per_update):
                if rollout_batches and global_simulator_steps + update_simulator_steps >= config.total_simulator_steps:
                    break
                action_seed = rollout_action_seed(config.seed, episode_index + local_index)
                independent_rng = config.action_seed_protocol == "per_episode_v1"
                with episode_action_rng(action_seed) if independent_rng else nullcontext():
                    batch, score, ticks = collect_rollout(
                        env=env, policy=policy,
                        seed=config.seed + episode_index + local_index,
                        gamma=config.gamma, gae_lambda=config.gae_lambda,
                        reward_scale=config.reward_scale,
                    )
                cyber = env.cyber_summary()
                if independent_rng:
                    cyber["rollout_action_seed"] = action_seed
                yield batch, score, ticks, cyber

        with compute.measure("rollout"):
            if episode_collector is None:
                episodes = serial_episodes()
            else:
                count = episode_collector.episode_count(
                    config.episodes_per_update, config.total_simulator_steps - global_simulator_steps,
                )
                indices = range(episode_index, episode_index + count)
                episodes = episode_collector.collect(
                    policy, episode_seeds=[config.seed + i for i in indices],
                    action_seeds=[rollout_action_seed(config.seed, i) for i in indices],
                    gamma=config.gamma, gae_lambda=config.gae_lambda,
                    reward_scale=config.reward_scale,
                )

            for batch, team_return, simulator_steps, cyber in episodes:
                if simulator_steps <= 0:
                    raise RuntimeError(
                        "A rollout must contain at least one simulator tick."
                    )
                assert cyber["ticks"] == simulator_steps
                update_cyber_metrics.append(cyber)

                rollout_batches.append(batch)
                update_returns.append(float(team_return))
                update_lengths.append(int(simulator_steps))
                update_simulator_steps += int(simulator_steps)

            batch = concatenate_batches(rollout_batches)
        completed_episodes = len(rollout_batches)

        policy_decisions = int(batch.actions.shape[0])
        if policy_decisions <= 0:
            raise RuntimeError("A rollout must contain at least one policy decision.")

        with compute.measure("ppo_update"):
            update_metrics = ppo_update(
                policy=policy,
                batch=batch,
                optimizer=optimizer,
                update_epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
                clip_coef=config.clip_coef,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
            )
        compute.end_training(update_simulator_steps)

        episode_returns.extend(update_returns)
        episode_lengths.extend(update_lengths)
        
        episode_cyber_metrics.extend(update_cyber_metrics)

        global_simulator_steps += update_simulator_steps
        total_policy_decisions += policy_decisions
        episode_index += completed_episodes
        update_index += 1

        mean_return = float(np.mean(update_returns))

        metrics: dict[str, MetricValue] = {
            "update": update_index,
            "global_step": global_simulator_steps,
            "episode": episode_index,
            "completed_episodes": completed_episodes,
            "episode_return_mean": mean_return,
            "episode_return_std": float(np.std(update_returns)),
            "episode_return_min": float(np.min(update_returns)),
            "episode_return_max": float(np.max(update_returns)),
            "episode_length_mean": float(np.mean(update_lengths)),
            "rollout_reward_mean": float(np.sum(update_returns))
            / update_simulator_steps,
            "policy_decisions": policy_decisions,
            "total_policy_decisions": total_policy_decisions,
            "decision_rate": policy_decisions / update_simulator_steps,
            "duration_mean": float(batch.durations.mean()),
            "duration_min": float(batch.durations.min()),
            "duration_max": float(batch.durations.max()),
            "reward_scale": config.reward_scale,
            "rollout_workers": config.rollout_workers,
            "scaled_transition_reward_mean": float(batch.rewards.mean()),
        }
        metrics.update(update_metrics)
        metrics.update({f"compute/{key}": value for key, value in compute.summary().items()})
        retentions = [
            row["observation_mask"]["retention"]
            for row in update_cyber_metrics if "observation_mask" in row
        ]
        if retentions:
            metrics["observation/retention_mean"] = float(np.mean(retentions))
            for q in (1.0, 0.75, 0.5, 0.25):
                metrics[f"observation/q_{q:g}_episodes"] = retentions.count(q)

        summary = aggregate_episodes(update_returns, update_cyber_metrics)
        for name, stats in summary.items():
            if name == "episode_return":
                continue
            for stat, value in stats.items():
                metrics[f"cyber/{name}/{stat}"] = value

        history.append(metrics)

        if on_update is not None:
            on_update(metrics)

        # Checkpoint only at completed PPO updates; do not change rollout grouping.
        while pending_milestones and global_simulator_steps >= pending_milestones[0]:
            requested_step = pending_milestones.pop(0)
            snapshot = current_result()
            milestone_path = (
                Path(config.checkpoint_path).parent / "milestones"
                / f"step_{requested_step}" / "checkpoint.pt"
            )
            save_training_checkpoint(milestone_path, policy, optimizer, config, snapshot)
            if on_milestone is not None:
                on_milestone(snapshot, requested_step)

    result = current_result()

    if config.checkpoint_path is not None:
        save_training_checkpoint(
            path=config.checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            config=config,
            result=result,
        )

    return result
