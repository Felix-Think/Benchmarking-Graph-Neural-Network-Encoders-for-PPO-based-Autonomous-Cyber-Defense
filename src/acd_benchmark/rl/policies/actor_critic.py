import torch
import torch.nn as nn
from torch.distributions import Categorical
from acd_benchmark.rl.ppo.init_weight import init_weight

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
        
class ActorCritic(nn.Module):
    def __init__(self, latent_dim : int, num_actions : int):
        super().__init__()
        self.actor = MLP(input_dim = latent_dim, output_dim = num_actions) 
        self.critic = MLP(input_dim = latent_dim, output_dim = 1)

        self.actor.apply(init_weight)
        self.critic.apply(init_weight)

    def encode(self, observation : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, observation):
        latent = self.encode(observation)
        return self.actor(latent), self.critic(latent)
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits, value = self.forward(observation)
        distribution = Categorical(logits=logits)

        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


    def get_value(self, observation: torch.Tensor):
        _, value = self.forward(observation)
        return value.squeeze(-1)
    
