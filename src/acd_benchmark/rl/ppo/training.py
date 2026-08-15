from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import random

import numpy as np
import torch

from acd_benchmark.rl.ppo.update import ppo_update
from acd_benchmark.rl.rollout.rollout import RolloutBatch, collect_rollout


MetricValue = float | int | bool | None
UpdateCallback = Callable[[dict[str, MetricValue]], None]


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 100_000
    rollout_steps: int = 2_048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 64
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
    total_timesteps: int
    total_updates: int


def seed_runtime(seed: int) -> None:
    """Seed action sampling and the numerical libraries used by training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _track_completed_episodes(
    batch: RolloutBatch,
    running_return: float,
    running_length: int,
) -> tuple[list[float], list[int], float, int]:
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    rewards = batch.rewards.detach().cpu().tolist()
    terminated = batch.terminated.detach().cpu().tolist()
    truncated = batch.truncated.detach().cpu().tolist()

    for reward, was_terminated, was_truncated in zip(
        rewards,
        terminated,
        truncated,
    ):
        running_return += float(reward)
        running_length += 1

        if was_terminated or was_truncated:
            completed_returns.append(running_return)
            completed_lengths.append(running_length)
            running_return = 0.0
            running_length = 0

    return (
        completed_returns,
        completed_lengths,
        running_return,
        running_length,
    )


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
            "total_timesteps": result.total_timesteps,
            "total_updates": result.total_updates,
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
    """Train one policy with sequential single-environment PPO rollouts."""
    assert config.total_timesteps > 0
    assert config.rollout_steps > 0
    assert config.minibatch_size > 0
    assert config.update_epochs > 0

    seed_runtime(config.seed)
    device = torch.device(config.device)
    policy.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
        )

    observation, _ = env.reset(seed=config.seed)
    history: list[dict[str, MetricValue]] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    running_episode_return = 0.0
    running_episode_length = 0
    global_step = 0
    update_index = 0

    while global_step < config.total_timesteps:
        steps_this_rollout = min(
            config.rollout_steps,
            config.total_timesteps - global_step,
        )
        batch, observation = collect_rollout(
            env=env,
            policy=policy,
            initial_observation=observation,
            rollout_steps=steps_this_rollout,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

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

        (
            completed_returns,
            completed_lengths,
            running_episode_return,
            running_episode_length,
        ) = _track_completed_episodes(
            batch,
            running_episode_return,
            running_episode_length,
        )
        episode_returns.extend(completed_returns)
        episode_lengths.extend(completed_lengths)

        global_step += steps_this_rollout
        update_index += 1
        metrics: dict[str, MetricValue] = {
            "update": update_index,
            "global_step": global_step,
            "rollout_reward_mean": float(batch.rewards.mean()),
            "completed_episodes": len(completed_returns),
            "episode_return_mean": (
                float(np.mean(completed_returns))
                if completed_returns
                else None
            ),
            "episode_length_mean": (
                float(np.mean(completed_lengths))
                if completed_lengths
                else None
            ),
        }
        metrics.update(update_metrics)
        history.append(metrics)

        if on_update is not None:
            on_update(metrics)

    result = TrainingResult(
        history=history,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        total_timesteps=global_step,
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
