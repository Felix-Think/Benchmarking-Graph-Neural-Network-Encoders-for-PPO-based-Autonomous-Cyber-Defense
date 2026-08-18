
import torch
import torch.nn as nn

from acd_benchmark.rl.policies.actor_critic import ActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight


class FlatActorCritic(ActorCritic):
    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        encoder_hidden_dim: int = 128,
        latent_dim: int = 128,
        head_hidden_dim: int = 128,
    ):
        super().__init__(
            latent_dim=latent_dim,
            num_actions=num_actions,
            head_hidden_dim=head_hidden_dim,
        )

        self.observation_dim = observation_dim

        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.encoder.apply(init_weight)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        if observation.ndim != 2:
            raise ValueError(
                "Flat observation must have shape "
                f"(observation_dim,) or (batch, observation_dim), "
                f"received {tuple(observation.shape)}"
            )

        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observation.shape[-1]}"
            )

        return self.encoder(observation.float())

if __name__ == '__main__':

    from acd_benchmark.cyber.cc4_flat_env import CC4FlatEnv

    env = CC4FlatEnv(episode_steps=500, seed=42)

    agent = env.possible_agents[0]

    observation_dim = env.observation_space(agent).shape[0]
    num_actions = env.action_space(agent).n

    policy = FlatActorCritic(
        observation_dim=observation_dim,
        num_actions=num_actions,
        encoder_hidden_dim=128,
        latent_dim=128,
        head_hidden_dim=128,
    )

    observations, infos = env.reset(seed=42)
    observation = torch.as_tensor(
        observations[agent],
        dtype=torch.float32,
    )

    action_mask = torch.as_tensor(
        infos[agent]["action_mask"],
        dtype=torch.bool,
    )
    action, log_prob, entropy, value = policy.get_action_and_value(
        observation,
        action_mask=action_mask,
    )

    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert action_mask[action.item()]

    observation_batch = torch.stack([
    torch.as_tensor(observations[agent], dtype=torch.float32)
    for agent in env.possible_agents
    ])

    mask_batch = torch.stack([
        torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)
        for agent in env.possible_agents
    ])

    actions, log_probs, entropies, values = (
        policy.get_action_and_value(
            observation_batch,
            action_mask=mask_batch,
        )
    )

    assert actions.shape == (5,)
    assert log_probs.shape == (5,)
    assert entropies.shape == (5,)
    assert values.shape == (5,)