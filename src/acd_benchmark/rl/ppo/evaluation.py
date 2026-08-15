from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from acd_benchmark.cyber.action_space import ACTION_SPACE, build_action, describe_action


@dataclass
class EvaluationResult:
    episode_returns: list[float]
    episode_lengths: list[int]
    action_counts: dict[str, int]
    deterministic: bool

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns))

    @property
    def min_return(self) -> float:
        return float(np.min(self.episode_returns))

    @property
    def max_return(self) -> float:
        return float(np.max(self.episode_returns))

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean(self.episode_lengths))

    def to_dict(self) -> dict:
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_episode_length": self.mean_episode_length,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "action_counts": self.action_counts,
            "deterministic": self.deterministic,
        }


def load_policy_checkpoint(
    policy: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict:
    if device is None:
        device = next(policy.parameters()).device
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)
    return checkpoint


def evaluate_policy(
    env,
    policy: torch.nn.Module,
    *,
    episodes: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
) -> EvaluationResult:
    """Evaluate a policy without gradients or simulator true-state access."""
    assert episodes > 0

    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    action_counts = {
        describe_action(action_id): 0
        for action_id in range(len(ACTION_SPACE))
    }

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0

            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    if deterministic:
                        logits, _ = policy(observation_tensor)
                        action_id = int(
                            torch.argmax(logits, dim=-1).item()
                        )
                    else:
                        action, _, _, _ = policy.get_action_and_value(
                            observation_tensor
                        )
                        action_id = int(action.item())

                action_counts[describe_action(action_id)] += 1
                observation, reward, terminated, truncated, _ = env.step(
                    build_action(action_id)
                )
                episode_return += float(reward)
                episode_length += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
    finally:
        policy.train(was_training)

    return EvaluationResult(
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        action_counts=action_counts,
        deterministic=deterministic,
    )
