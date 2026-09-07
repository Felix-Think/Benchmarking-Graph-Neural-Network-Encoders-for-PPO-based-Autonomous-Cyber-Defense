from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EvaluationResult:
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_policy_decisions: list[int]
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

    @property
    def mean_policy_decisions(self) -> float:
        return float(np.mean(self.episode_policy_decisions))

    def to_dict(self) -> dict:
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_episode_length": self.mean_episode_length,
            "mean_policy_decisions": self.mean_policy_decisions,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "episode_policy_decisions": self.episode_policy_decisions,
            "action_counts": self.action_counts,
            "deterministic": self.deterministic,
        }


@dataclass
class EvaluationModeComparison:
    deterministic: EvaluationResult
    stochastic: EvaluationResult
    environment_seed: int
    stochastic_action_seed: int

    @property
    def paired_return_differences(self) -> list[float]:
        """Return stochastic minus deterministic for each shared episode seed."""
        return [
            stochastic_return - deterministic_return
            for deterministic_return, stochastic_return in zip(
                self.deterministic.episode_returns,
                self.stochastic.episode_returns,
                strict=True,
            )
        ]

    @property
    def mean_paired_return_difference(self) -> float:
        return float(np.mean(self.paired_return_differences))

    @property
    def std_paired_return_difference(self) -> float:
        return float(np.std(self.paired_return_differences))

    def to_dict(self) -> dict:
        return {
            "environment_seed": self.environment_seed,
            "stochastic_action_seed": self.stochastic_action_seed,
            "difference_definition": "stochastic_return - deterministic_return",
            "mean_paired_return_difference": self.mean_paired_return_difference,
            "std_paired_return_difference": self.std_paired_return_difference,
            "paired_return_differences": self.paired_return_differences,
            "deterministic": self.deterministic.to_dict(),
            "stochastic": self.stochastic.to_dict(),
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
    """Evaluate one shared policy across all five CAGE 4 Blue agents."""
    assert episodes > 0

    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_policy_decisions: list[int] = []
    action_counts: dict[str, int] = {}
    reward_agent = env.possible_agents[0]

    try:
        for episode_index in range(episodes):
            observations, infos = env.reset(seed=seed + episode_index)
            action_labels = {
                agent: env.action_labels(agent)
                for agent in env.possible_agents
            }

            episode_return = 0.0
            simulator_steps = 0
            policy_decisions = 0

            while True:
                actions: dict[str, int] = {}

                for agent in env.possible_agents:
                    if not infos[agent]["decision_ready"]:
                        continue

                    observation = torch.as_tensor(
                        observations[agent],
                        dtype=torch.float32,
                        device=device,
                    )
                    action_mask = torch.as_tensor(
                        infos[agent]["action_mask"],
                        dtype=torch.bool,
                        device=device,
                    )

                    with torch.no_grad():
                        action, _, _, _ = policy.get_action_and_value(
                            observation,
                            action_mask=action_mask,
                            deterministic=deterministic,
                        )

                    action_id = int(action.item())
                    if not bool(action_mask[action_id]):
                        raise RuntimeError(
                            f"Policy selected masked action {action_id} for {agent}."
                        )

                    actions[agent] = action_id
                    policy_decisions += 1

                    action_key = f"{agent}: {action_labels[agent][action_id]}"
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1

                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    next_infos,
                ) = env.step(actions)

                episode_return += float(rewards[reward_agent])
                simulator_steps += 1

                episode_done = all(
                    terminated.get(agent, False)
                    or truncated.get(agent, False)
                    for agent in env.possible_agents
                )
                if episode_done:
                    break

                observations = next_observations
                infos = next_infos

            episode_returns.append(episode_return)
            episode_lengths.append(simulator_steps)
            episode_policy_decisions.append(policy_decisions)
    finally:
        policy.train(was_training)

    return EvaluationResult(
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_policy_decisions=episode_policy_decisions,
        action_counts=action_counts,
        deterministic=deterministic,
    )


def evaluate_policy_modes(
    deterministic_env,
    stochastic_env,
    policy: torch.nn.Module,
    *,
    episodes: int = 10,
    seed: int = 10_000,
    stochastic_action_seed: int | None = None,
) -> EvaluationModeComparison:
    """Evaluate masked argmax and policy sampling on identical episode seeds."""
    if stochastic_action_seed is None:
        stochastic_action_seed = seed

    deterministic_result = evaluate_policy(
        deterministic_env,
        policy,
        episodes=episodes,
        seed=seed,
        deterministic=True,
    )

    stochastic_result = evaluate_policy_with_preserved_rng(
        stochastic_env,
        policy,
        episodes=episodes,
        seed=seed,
        deterministic=False,
        action_seed=stochastic_action_seed,
    )

    return EvaluationModeComparison(
        deterministic=deterministic_result,
        stochastic=stochastic_result,
        environment_seed=seed,
        stochastic_action_seed=stochastic_action_seed,
    )


def evaluate_policy_with_preserved_rng(
    env,
    policy: torch.nn.Module,
    *,
    episodes: int,
    seed: int,
    deterministic: bool,
    action_seed: int,
) -> EvaluationResult:
    """Evaluate without changing the action-sampling RNG used by training."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        torch.manual_seed(action_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(action_seed)

        return evaluate_policy(
            env,
            policy,
            episodes=episodes,
            seed=seed,
            deterministic=deterministic,
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
