
import torch.nn as nn 
import torch

from acd_benchmark.rl.policies.actor_critic import ActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight
class FlatActorCritic(ActorCritic):
    def __init__(self, num_nodes:int = 13, node_features:int = 4, latent_dim:int = 64 , num_actions: int = 41):
        super().__init__(latent_dim = latent_dim, num_actions = num_actions)
        input_dim = num_nodes * node_features 

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
            nn.Tanh(),

        )
        self.encoder.apply(init_weight)
    def encode(self, obs : torch.Tensor):
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        return self.encoder(obs)
