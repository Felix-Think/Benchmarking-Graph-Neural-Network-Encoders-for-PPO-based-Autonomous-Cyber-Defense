import json
from dataclasses import replace

import pytest
import torch

from acd_benchmark.rl.ppo.training import PPOTrainingConfig, train_policy
from acd_benchmark.rl.ppo.resume import read_resume_checkpoint
from acd_benchmark.rl.rollout.rollout import collect_rollout
from test_ppo_training_evaluation import TinyPolicy, TinyEpisodeEnv


def tiny_policy():
    torch.manual_seed(123)
    return TinyPolicy(4, 2, 8, 8, 8)


def assert_nested_equal(left, right, approximate=False):
    if isinstance(left, torch.Tensor):
        if approximate:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key], approximate)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b, approximate)
    else:
        assert left == right


@pytest.mark.parametrize("protocol", ["global", "per_episode_v1"])
def test_interruption_restores_weights_optimizer_rng_and_episode_counters(tmp_path, protocol):
    torch.set_num_threads(1)
    config = PPOTrainingConfig(
        total_simulator_steps=15, episodes_per_update=2,
        update_epochs=2, minibatch_size=4, action_seed_protocol=protocol,
        checkpoint_path=str(tmp_path / "reference/checkpoint.pt"),
    )
    reference = tiny_policy()
    expected = train_policy(TinyEpisodeEnv(), reference, config)
    interrupted_config = replace(config, checkpoint_path=str(tmp_path / "resumed/checkpoint.pt"))
    def interrupt(metrics):
        if metrics["update"] == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        train_policy(TinyEpisodeEnv(), tiny_policy(), interrupted_config, on_update=interrupt)
    saved = read_resume_checkpoint(tmp_path / "resumed")
    assert saved["total_updates"] == 1
    assert saved["total_simulator_steps"] == 6
    actual_policy = tiny_policy()
    result = train_policy(
        TinyEpisodeEnv(), actual_policy, interrupted_config, resume_checkpoint=saved,
    )
    assert result.total_simulator_steps == expected.total_simulator_steps == 15
    assert result.episode_cyber_metrics == expected.episode_cyber_metrics
    for a, b in zip(result.history, expected.history):
        assert {k: v for k, v in a.items() if not k.startswith("compute/")} == {
            k: v for k, v in b.items() if not k.startswith("compute/")
        }
    actual = torch.load(tmp_path / "resumed/checkpoint.pt")
    baseline = torch.load(tmp_path / "reference/checkpoint.pt")
    assert_nested_equal(actual["policy_state_dict"], baseline["policy_state_dict"])
    assert_nested_equal(actual["optimizer_state_dict"], baseline["optimizer_state_dict"])
    assert torch.equal(actual["rng_state"]["torch"], baseline["rng_state"]["torch"])


def test_resume_rejects_changed_hyperparameters_and_legacy_checkpoint(tmp_path):
    config = PPOTrainingConfig(total_simulator_steps=3, checkpoint_path=str(tmp_path / "checkpoint.pt"))
    train_policy(TinyEpisodeEnv(), tiny_policy(), config)
    saved = read_resume_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="gamma"):
        train_policy(TinyEpisodeEnv(), tiny_policy(), replace(config, gamma=0.9), resume_checkpoint=saved)
    saved.pop("resume_version")
    with pytest.raises(ValueError, match="boundary"):
        train_policy(TinyEpisodeEnv(), tiny_policy(), config, resume_checkpoint=saved)


def test_interruption_in_first_rollout_has_initial_checkpoint(tmp_path):
    class InterruptedEnv(TinyEpisodeEnv):
        def step(self, actions):
            raise KeyboardInterrupt

    config = PPOTrainingConfig(
        total_simulator_steps=3, checkpoint_path=str(tmp_path / "checkpoint.pt"),
    )
    with pytest.raises(KeyboardInterrupt):
        train_policy(InterruptedEnv(), tiny_policy(), config)
    saved = read_resume_checkpoint(tmp_path)
    assert saved["total_updates"] == saved["total_simulator_steps"] == 0
    recovered = train_policy(TinyEpisodeEnv(), tiny_policy(), config, resume_checkpoint=saved)
    assert recovered.total_simulator_steps == 3


def test_queue_resume_refuses_changed_source_without_starting_child(tmp_path, monkeypatch):
    from scripts.experiments import run_encoder_matrix as module
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source_sha256": "old", "config": {"seeds": [42]}, "jobs": [],
    }))
    monkeypatch.setattr(module, "source_files", lambda: [])
    monkeypatch.setattr(module, "fingerprint", lambda paths: "new")
    with pytest.raises(RuntimeError, match="Source differs"):
        module.main(["--resume", "--foreground", "--run-root", str(tmp_path)])
    assert not (tmp_path / "status.json").exists()


def test_progress_every_50_ticks_preserves_rollout(capsys):
    p = tiny_policy()
    torch.manual_seed(44)
    before = torch.get_rng_state()
    batch, score, ticks = collect_rollout(TinyEpisodeEnv(101), p, seed=42, progress_every_ticks=50)
    output = capsys.readouterr().out
    assert output.count("rollout episode_seed=") == 3
    assert "episode_tick=50 " in output and "episode_tick=100 " in output
    assert "episode_tick=101 " in output
    assert "return_so_far=50.00" in output
    assert "availability=1.0000" in output
    torch.set_rng_state(before)
    reference, rscore, rticks = collect_rollout(TinyEpisodeEnv(101), p, seed=42)
    assert (score, ticks) == (rscore, rticks)
    for key in vars(batch):
        assert torch.equal(getattr(batch, key), getattr(reference, key))


@pytest.mark.parametrize("encoder", ["nodemlp", "gcn", "graphsage"])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")),
])
def test_native_parallel_mixed_q_resume_rolls_back_failed_validation(tmp_path, monkeypatch, encoder, device):
    from scripts.training import train
    base = [
        "--encoder", encoder, "--device", device, "--train-regime", "mixed-q",
        "--total-simulator-steps", "81", "--episode-steps", "10",
        "--episodes-per-update", "4", "--rollout-workers", "2",
        "--action-seed-protocol", "per_episode_v1", "--update-epochs", "1",
        "--eval-every-updates", "1", "--periodic-eval-episodes", "1",
        "--eval-episodes", "1", "--checkpoint-steps", "36", "72", "81",
        "--eval-retentions", "1", "--progress-every-ticks", "0",
    ]
    args = train.build_parser().parse_args(base + ["--run-dir", str(tmp_path / "reference")])
    train.run_training(args)
    args = train.build_parser().parse_args(base + ["--run-dir", str(tmp_path / "candidate")])
    evaluate = train.evaluate_retention_curve
    def fail_second_milestone(*a, **kw):
        if kw["global_step"] == 72:
            raise RuntimeError("simulated disconnect")
        return evaluate(*a, **kw)
    monkeypatch.setattr(train, "evaluate_retention_curve", fail_second_milestone)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        train.run_training(args)
    candidate = tmp_path / "candidate/seed_42"
    assert read_resume_checkpoint(candidate)["total_simulator_steps"] == 36
    monkeypatch.setattr(train, "evaluate_retention_curve", evaluate)
    args.resume = True
    train.run_training(args)
    actual = torch.load(candidate / "checkpoint.pt")
    baseline = torch.load(tmp_path / "reference/seed_42/checkpoint.pt")
    assert_nested_equal(actual["policy_state_dict"], baseline["policy_state_dict"], device == "cuda")
    assert_nested_equal(actual["optimizer_state_dict"], baseline["optimizer_state_dict"], device == "cuda")
    assert actual["episode_cyber_metrics"] == baseline["episode_cyber_metrics"]
    assert actual["total_episodes"] == 9
    history = [json.loads(line) for line in (candidate / "training_history.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in history] == [36, 72, 81]
    periodic = json.loads((candidate / "periodic_evaluation.json").read_text())
    assert [row["global_step"] for row in periodic] == [36, 72, 81]
    assert (candidate / "discarded_attempts").is_dir()
    # Final-evaluation failure/retry must not perform any further PPO updates.
    original_update = train.train_policy
    def assert_budget_done(*a, **kw):
        assert kw["resume_checkpoint"]["total_simulator_steps"] == 81
        return original_update(*a, **kw)
    monkeypatch.setattr(train, "train_policy", assert_budget_done)
    train.run_training(args)
    repeated = torch.load(candidate / "checkpoint.pt")
    assert_nested_equal(actual["policy_state_dict"], repeated["policy_state_dict"])


def test_queue_resume_skips_completed_jobs_and_passes_resume_flag(tmp_path, monkeypatch):
    from scripts.experiments import run_encoder_matrix as module
    done, pending = tmp_path / "done", tmp_path / "pending"
    done.mkdir()
    pending.mkdir()
    (done / "compute_summary.json").write_text('{"status": "completed"}')
    (done / "evaluation_retention.json").write_text("{}")
    (pending / "config.json").write_text("{}")
    (pending / "console.log").write_text("old log\n")
    jobs = [
        {"directory": str(p), "seed": 42, "arguments": ["--encoder", "nodemlp"]}
        for p in (done, pending)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"source_sha256": "x", "jobs": jobs}))
    monkeypatch.setattr(module, "source_files", lambda: [])
    monkeypatch.setattr(module, "fingerprint", lambda paths: "x")
    calls = []
    class Child:
        pid = 1234
        def __init__(self, command, **kwargs):
            calls.append(command)
            (pending / "evaluation_retention.json").write_text("{}")
        def wait(self):
            return 0
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    assert module.execute_queue(tmp_path, resume=True) == 0
    assert len(calls) == 1 and calls[0][-1] == "--resume"
    assert (pending / "console.log").read_text() == "old log\n"
