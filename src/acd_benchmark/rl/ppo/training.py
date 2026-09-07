from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping
import random

import numpy as np
import torch

from acd_benchmark.rl.ppo.update import ppo_update
from acd_benchmark.rl.rollout.rollout import collect_rollout, concatenate_batches


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


@dataclass
class TrainingResult:
    history: list[dict[str, MetricValue]]
    episode_returns: list[float]
    episode_lengths: list[int]
    total_simulator_steps: int
    total_policy_decisions: int
    total_episodes: int
    total_updates: int

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
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "history": result.history,
            "episode_returns": result.episode_returns,
            "episode_lengths": result.episode_lengths,
            "total_simulator_steps": result.total_simulator_steps,
            "total_policy_decisions": result.total_policy_decisions,
            "total_episodes": result.total_episodes,
            "total_updates": result.total_updates,
            # Compatibility with existing checkpoint readers.
            "total_timesteps": result.total_simulator_steps,
        },
        checkpoint_path,
    )


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
) -> TrainingResult:
    """Train one shared policy using complete CAGE 4 episodes.

    Each PPO update combines ``episodes_per_update`` sequential trajectories.
    The budget is measured in simulator ticks and rollouts are never cut in the
    middle of an episode. The final update can therefore exceed the requested
    budget by at most one episode.
    """
    assert config.total_simulator_steps > 0
    assert config.episodes_per_update > 0
    assert config.minibatch_size > 0
    assert config.update_epochs > 0
    assert config.reward_scale > 0
    seed_runtime(config.seed)
    device = torch.device(config.device)
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

    while global_simulator_steps < config.total_simulator_steps:
        rollout_batches = []
        update_returns = []
        update_lengths = []
        update_simulator_steps = 0

        for local_episode_index in range(config.episodes_per_update):
            if (
                rollout_batches
                and global_simulator_steps + update_simulator_steps
                >= config.total_simulator_steps
            ):
                break

            batch, team_return, simulator_steps = collect_rollout(
                env=env,
                policy=policy,
                seed=config.seed + episode_index + local_episode_index,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                reward_scale=config.reward_scale,
            )
            if simulator_steps <= 0:
                raise RuntimeError(
                    "A rollout must contain at least one simulator tick."
                )
            rollout_batches.append(batch)
            update_returns.append(float(team_return))
            update_lengths.append(int(simulator_steps))
            update_simulator_steps += int(simulator_steps)

        batch = concatenate_batches(rollout_batches)
        completed_episodes = len(rollout_batches)

        policy_decisions = int(batch.actions.shape[0])
        if policy_decisions <= 0:
            raise RuntimeError("A rollout must contain at least one policy decision.")

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

        episode_returns.extend(update_returns)
        episode_lengths.extend(update_lengths)

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
            "scaled_transition_reward_mean": float(batch.rewards.mean()),
        }
        metrics.update(update_metrics)
        history.append(metrics)

        if on_update is not None:
            on_update(metrics)

    result = TrainingResult(
        history=history,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        total_simulator_steps=global_simulator_steps,
        total_policy_decisions=total_policy_decisions,
        total_episodes=episode_index,
        total_updates=update_index,
    )

    if config.checkpoint_path is not None:
        save_training_checkpoint(
            path=config.checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            config=config,
            result=result,
        )

    return result
