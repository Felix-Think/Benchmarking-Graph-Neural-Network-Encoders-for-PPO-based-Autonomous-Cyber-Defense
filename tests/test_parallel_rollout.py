import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.ppo.training import PPOTrainingConfig, seed_runtime, train_policy
from acd_benchmark.rl.rollout.parallel import ParallelEpisodeCollector, make_training_env
from scripts.training.train import build_parser, run_training


def settings(encoder, regime):
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/experiments/encoder_retention_500k_cuda.json").read_text())["training"]
    return {
        **base, "device": "cpu", "episode_steps": 10, "seed": 42, "train_regime": regime,
        "policy_type": f"structured_{encoder}", "canonical_schema": "v2",
        "observation_dim": 5185, "num_actions": 242,
    }


@pytest.mark.parametrize("encoder", ["nodemlp", "gcn", "graphsage"])
@pytest.mark.parametrize("regime", ["full", "mixed-q"])
def test_parallel_training_matches_serial_with_partial_final_update(tmp_path, encoder, regime):
    torch.set_num_threads(1)
    run_config = settings(encoder, regime)
    seed_runtime(42)
    reference = build_policy(run_config)
    initial = copy.deepcopy(reference.state_dict())
    config = PPOTrainingConfig(
        total_simulator_steps=40, episodes_per_update=4, update_epochs=2,
        action_seed_protocol="per_episode_v1", checkpoint_steps=(20, 40),
        checkpoint_path=str(tmp_path / "serial" / "checkpoint.pt"),
    )
    serial = train_policy(make_training_env(run_config), reference, config)
    candidate = build_policy(run_config)
    candidate.load_state_dict(initial)
    collector = ParallelEpisodeCollector(run_config, workers=2)
    try:
        parallel = train_policy(
            None, candidate,
            replace(config, rollout_workers=2, checkpoint_path=str(tmp_path / "parallel" / "checkpoint.pt")),
            episode_collector=collector,
        )
    finally:
        collector.close()
    assert parallel.total_simulator_steps == serial.total_simulator_steps == 45
    assert parallel.total_episodes == serial.total_episodes == 5
    assert parallel.total_policy_decisions == serial.total_policy_decisions
    assert [row["completed_episodes"] for row in parallel.history] == [4, 1]
    assert parallel.episode_returns == serial.episode_returns
    assert parallel.episode_cyber_metrics == serial.episode_cyber_metrics
    for a, b in zip(serial.history, parallel.history, strict=True):
        assert {k: v for k, v in a.items() if not k.startswith("compute/") and k != "rollout_workers"} == {
            k: v for k, v in b.items() if not k.startswith("compute/") and k != "rollout_workers"
        }
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    for requested, actual in ((20, 36), (40, 45)):
        checkpoint = torch.load(tmp_path / "parallel" / "milestones" / f"step_{requested}" / "checkpoint.pt")
        assert checkpoint["total_simulator_steps"] == actual
        assert checkpoint["config"]["rollout_workers"] == 2
        assert checkpoint["config"]["action_seed_protocol"] == "per_episode_v1"


def test_parallel_cli_saves_milestone_and_final_evaluations(tmp_path):
    args = build_parser().parse_args([
        "--encoder", "graphsage", "--train-regime", "mixed-q",
        "--rollout-workers", "2", "--action-seed-protocol", "per_episode_v1",
        "--run-dir", str(tmp_path), "--total-simulator-steps", "40",
        "--episode-steps", "10", "--episodes-per-update", "4", "--update-epochs", "1",
        "--checkpoint-steps", "20", "40", "--eval-retentions", "1", "0.5",
        "--periodic-eval-episodes", "1", "--eval-episodes", "1", "--eval-every-updates", "1",
    ])
    run_training(args)
    directory = tmp_path / "seed_42"
    saved = json.loads((directory / "config.json").read_text())
    assert saved["rollout_workers"] == 2
    assert saved["action_seed_protocol"] == "per_episode_v1"
    assert (directory / "best_checkpoint.pt").is_file()
    assert list((directory / "tensorboard").glob("events.out.tfevents.*"))
    for milestone in (20, 40):
        assert (directory / "milestones" / f"step_{milestone}" / "validation_retention.json").is_file()
    curve = json.loads((directory / "evaluation_retention.json").read_text())
    assert set(curve["results"]) == {"1", "0.5"}
    assert set(curve["results"]["0.5"]) == {"stochastic", "deterministic"}


def test_parallel_cli_rejects_legacy_rng_before_creating_run(tmp_path):
    args = build_parser().parse_args([
        "--encoder", "nodemlp", "--rollout-workers", "2", "--run-dir", str(tmp_path),
    ])
    with pytest.raises(ValueError, match="per_episode_v1"):
        run_training(args)
    assert not (tmp_path / "seed_42").exists()
