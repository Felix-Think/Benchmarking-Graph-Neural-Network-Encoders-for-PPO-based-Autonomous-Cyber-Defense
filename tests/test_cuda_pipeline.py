import json
from pathlib import Path

import pytest
import torch

from acd_benchmark.rl.runtime import cpu_inference_policy, cpu_state_dict, learner_device
from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.rollout.parallel import ParallelEpisodeCollector, rollout_action_seed
from acd_benchmark.rl.ppo.update import ppo_update
from scripts.training.train import build_parser, run_training


ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real NVIDIA CUDA hardware")


def test_cuda_unavailable_fails_before_creating_run(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = build_parser().parse_args(["--encoder", "nodemlp", "--device", "cuda", "--run-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        run_training(args)
    assert not (tmp_path / "seed_42").exists()


def test_cpu_snapshots_are_independent_and_do_not_consume_rng():
    policy = torch.nn.Linear(3, 2)
    rng = torch.get_rng_state().clone()
    snapshot = cpu_state_dict(policy)
    replica = cpu_inference_policy(policy)
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.no_grad():
        policy.weight.add_(1)
    assert not torch.equal(snapshot["weight"], policy.weight)
    assert torch.equal(replica.weight, snapshot["weight"])
    replica.load_state_dict(cpu_state_dict(policy))
    assert torch.equal(replica.weight, policy.weight)
    assert not replica.training


def test_cuda_config_initializes_cpu_workers(monkeypatch):
    from acd_benchmark.rl.rollout import parallel

    seen = {}
    class Pool:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    collector = ParallelEpisodeCollector({"device": "cuda", "episode_steps": 10}, 2)
    collector.close()
    assert seen["initargs"][0]["device"] == "cpu"
    assert seen["mp_context"].get_start_method() == "spawn"


@GPU
@pytest.mark.parametrize("encoder", ["nodemlp", "gcn", "graphsage"])
def test_real_cuda_update_accepts_cpu_worker_batch(encoder):
    torch.set_num_threads(1)
    device = learner_device("cuda")
    config = json.loads((ROOT / "configs/experiments/encoder_retention_500k_cuda.json").read_text())["training"]
    config.update(
        episode_steps=10, seed=42, train_regime="mixed-q",
        policy_type=f"structured_{encoder}", canonical_schema="v2",
        observation_dim=5185, num_actions=242,
    )
    learner = build_policy(config).to(device)
    collector = ParallelEpisodeCollector(config, 2)
    try:
        batch, _, _, _ = collector.collect(
            learner, episode_seeds=[42], action_seeds=[rollout_action_seed(42, 0)],
            gamma=0.99, gae_lambda=0.95, reward_scale=0.01,
        )[0]
    finally:
        collector.close()
    assert batch.observations.device.type == "cpu"
    with torch.no_grad():
        _, log_probs, _, values = learner.get_action_and_value(
            batch.observations.to(device),
            action=batch.actions.to(device), action_mask=batch.action_masks.to(device),
        )
    # Same frozen policy across devices must start PPO near ratio=1.
    torch.testing.assert_close(log_probs.cpu(), batch.old_log_probs, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(values.cpu(), batch.values, atol=1e-4, rtol=1e-4)
    before = cpu_state_dict(learner)
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    metrics = ppo_update(learner, batch, optimizer, update_epochs=1)
    assert all(torch.isfinite(torch.tensor(float(value))) for value in metrics.values())
    assert any(not torch.equal(before[k], v.cpu()) for k, v in learner.state_dict().items())
    assert all(s["exp_avg"].is_cuda for s in optimizer.state.values())


@GPU
@pytest.mark.parametrize("encoder", ["nodemlp", "gcn", "graphsage"])
def test_real_cuda_cli_checkpoints_cpu_eval_and_metrics(tmp_path, encoder):
    args = build_parser().parse_args([
        "--encoder", encoder, "--device", "cuda", "--train-regime", "mixed-q",
        "--rollout-workers", "2", "--action-seed-protocol", "per_episode_v1",
        "--run-dir", str(tmp_path), "--total-simulator-steps", "36",
        "--episode-steps", "10", "--episodes-per-update", "4", "--update-epochs", "1",
        "--checkpoint-steps", "36", "--eval-retentions", "1", "0.25",
        "--periodic-eval-episodes", "1", "--eval-episodes", "1", "--eval-every-updates", "1",
    ])
    run_training(args)
    directory = tmp_path / "seed_42"
    saved = json.loads((directory / "config.json").read_text())
    assert saved["execution_pipeline"] == "cpu_rollout_cuda_learner_v1"
    assert saved["evaluation_device"] == "cpu"
    assert saved["compute_metadata"]["gpu_name"]
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu")
    assert checkpoint["total_simulator_steps"] == 36
    assert checkpoint["config"]["device"] == "cuda"
    assert (directory / "best_checkpoint.pt").is_file()
    assert (directory / "milestones/step_36/validation_retention.json").is_file()
    curve = json.loads((directory / "evaluation_retention.json").read_text())
    assert set(curve["results"]) == {"1", "0.25"}
    training = json.loads((directory / "training_summary.json").read_text())
    assert training["compute_metrics"]["cuda_peak_allocated_bytes"] > 0
    evaluation = json.loads((directory / "evaluation_stochastic.json").read_text())
    assert evaluation["compute_metrics"]["cuda_peak_allocated_bytes"] is None
