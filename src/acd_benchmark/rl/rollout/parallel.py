"""Synchronous whole-episode CPU workers with independent action RNG streams."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import fields
import multiprocessing as mp
import os
from math import ceil
import random
import tempfile

import numpy as np
import torch
from torch_geometric import set_home_dir

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.episode_retention_mask import EpisodeRetentionMask
from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.rollout.rollout import collect_rollout, RolloutBatch
from acd_benchmark.rl.runtime import cpu_state_dict


_ENV = None
_POLICY = None
_PYG_CACHE = None


def make_training_env(config):
    env = make_benchmark_env(config, seed=config["seed"])
    if config.get("train_regime", "full") == "mixed-q":
        env = EpisodeRetentionMask(
            env, retentions=config["train_retentions"],
            q_seed=config["q_seed"] + config["seed"],
            mask_seed=config["train_mask_seed"] + config["seed"],
        )
    return env


def _initialize_worker(config):
    global _ENV, _POLICY, _PYG_CACHE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # PyG 2.5 writes generated Python modules non-atomically during construction.
    _PYG_CACHE = tempfile.TemporaryDirectory(prefix="acd-pyg-worker-")
    set_home_dir(_PYG_CACHE.name)
    # Rollout is dominated by Python simulator work, not large BLAS operations.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _ENV = make_training_env(config)
    _POLICY = build_policy(config).cpu()


def _episode(payload):
    state, seed, action_seed, gamma, gae_lambda, reward_scale = payload
    _POLICY.load_state_dict(state)
    random.seed(action_seed)
    np.random.seed(action_seed)
    torch.manual_seed(action_seed)
    batch, score, ticks = collect_rollout(
        _ENV, _POLICY, seed=seed, gamma=gamma,
        gae_lambda=gae_lambda, reward_scale=reward_scale,
    )
    # NumPy payloads avoid shared-memory file-descriptor lifetime issues on macOS.
    arrays = {
        field.name: getattr(batch, field.name).detach().cpu().numpy()
        for field in fields(RolloutBatch)
    }
    cyber = _ENV.cyber_summary()
    cyber["rollout_action_seed"] = action_seed
    return arrays, score, ticks, cyber


@contextmanager
def episode_action_rng(action_seed):
    """Serial reference uses exactly the same per-episode action streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(action_seed)
        np.random.seed(action_seed)
        torch.manual_seed(action_seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def rollout_action_seed(training_seed, episode_index):
    return int(np.random.SeedSequence((70_000, training_seed, episode_index)).generate_state(1)[0])


class ParallelEpisodeCollector:
    def __init__(self, config, workers):
        self.workers = workers
        self.episode_ticks = int(config["episode_steps"]) - 1
        if self.episode_ticks <= 0:
            raise ValueError("episode_steps must be at least 2")
        if not 1 <= workers <= 3:
            raise ValueError("Use 1-3 episode workers on this CPU protocol")
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker, initargs=({**config, "device": "cpu"},),
        )

    def collect(self, policy, *, episode_seeds, action_seeds, gamma, gae_lambda, reward_scale):
        state = cpu_state_dict(policy)
        payloads = [
            (state, seed, action_seed, gamma, gae_lambda, reward_scale)
            for seed, action_seed in zip(episode_seeds, action_seeds, strict=True)
        ]
        # map preserves episode order even when workers finish out of order.
        results = []
        for arrays, score, ticks, cyber in self.pool.map(_episode, payloads):
            if ticks != self.episode_ticks:
                raise RuntimeError("Parallel budget assumes fixed-length CAGE 4 episodes")
            batch = RolloutBatch(**{name: torch.from_numpy(array) for name, array in arrays.items()})
            results.append((batch, score, ticks, cyber))
        return results

    def episode_count(self, requested, remaining_ticks):
        """Keep the serial budget rule for fixed-length CAGE 4 episodes."""
        return min(requested, ceil(remaining_ticks / self.episode_ticks))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
