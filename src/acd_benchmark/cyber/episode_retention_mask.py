import numpy as np

from acd_benchmark.cyber.dynamic_host_observation_mask import FixedBudgetDynamicHostMask


class EpisodeRetentionMask(FixedBudgetDynamicHostMask):
    """Sample q per episode, independently of PPO and simulator RNG streams."""

    def __init__(self, env, *, retentions=(1.0, 0.75, 0.5, 0.25), q_seed=40_000, mask_seed=30_000):
        self.retentions = tuple(float(q) for q in retentions)
        if not self.retentions or any(not 0 <= q <= 1 for q in self.retentions):
            raise ValueError("Provide retention levels in [0, 1]")
        self.q_seed = int(q_seed)
        super().__init__(env, retention=1.0, mask_seed=mask_seed, mask_process="persistent")

    def reset(self, seed=None):
        episode_seed = 0 if seed is None else int(seed)
        rng = np.random.default_rng(np.random.SeedSequence((self.q_seed, episode_seed)))
        self.retention = float(rng.choice(self.retentions))
        return super().reset(seed=seed)

    def cyber_summary(self):
        return {
            **super().cyber_summary(),
            "observation_mask": {
                "retention": self.retention,
                "process": self.process_name,
                "episode_seed": self._episode_seed,
                "q_seed": self.q_seed,
                "mask_seed": self.mask_seed,
                "visible_host_nodes": {
                    agent: np.flatnonzero(visible).tolist()
                    for agent, visible in self.last_visibility.items()
                },
            },
        }
