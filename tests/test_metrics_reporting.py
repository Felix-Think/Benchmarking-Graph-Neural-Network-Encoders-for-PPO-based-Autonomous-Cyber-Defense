import json
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.dynamic_evidence_counterfactual import DynamicEvidenceCounterfactual
from acd_benchmark.cyber.dynamic_host_observation_mask import FixedBudgetDynamicHostMask
from acd_benchmark.metrics.agregate import _summary, aggregate_episodes, aggregate_seeds
from acd_benchmark.metrics.cyber import CyberMetrics
from acd_benchmark.rl.ppo.evaluation import EvaluationModeComparison, EvaluationResult
from acd_benchmark.rl.ppo.training import (
    PPOTrainingConfig,
    TrainingResult,
    save_training_checkpoint,
)
from acd_benchmark.rl.tensorboard_logger import (
    TensorBoardLogger,
    write_checkpoint_history_to_tensorboard,
)
from scripts.training import train as train_script
from scripts.evaluation.evaluate import main as evaluate_command


def episode_metrics(*, recovered):
    metrics = CyberMetrics()
    metrics.reset(
        {"A": {"subnet": "zone_a", "has_ot": True, "protected": True}}
    )
    active_hosts = [{"A"}, set(), {"A"} if recovered else set()]
    for tick, active in enumerate(active_hosts):
        red_hosts = {"A"} if tick == 1 else set()
        metrics.update(0, active, red_hosts)
        metrics.update_compromised_host(red_hosts)
        metrics.update_ot_recovery(active)
        metrics.update_green(0, "zone_a", "GreenAccessService", tick != 1)
    return metrics.summary()


def evaluation_result(*, deterministic):
    return EvaluationResult(
        episode_returns=[-3.0, -6.0],
        episode_lengths=[3, 3],
        episode_policy_decisions=[4, 5],
        action_counts={"blue_0: Sleep": 9},
        deterministic=deterministic,
        episode_cyber_metrics=[
            episode_metrics(recovered=True),
            episode_metrics(recovered=False),
        ],
    )


def training_result():
    history = [
        {
            "update": update,
            "global_step": update * 3,
            "cyber/availability/mean": 1.0 / update,
            "cyber/GreenLocalWork_success_rate/mean": None,
            "compute/training_seconds": float(update),
            "compute/simulator_ticks": update * 3,
            "compute/simulator_ticks_per_second": 3.0,
        }
        for update in (1, 2)
    ]
    cyber_summaries = [
        episode_metrics(recovered=True),
        episode_metrics(recovered=False),
    ]
    for metrics, cyber in zip(history, cyber_summaries):
        summary = aggregate_episodes([0.0], [cyber])
        for name, stats in summary.items():
            if name.startswith(("ot_ttr_", "ot_recovery_", "ot_unrecovered_")):
                for stat, value in stats.items():
                    metrics[f"cyber/{name}/{stat}"] = value
    return TrainingResult(
        history=history,
        episode_returns=[-3.0, -6.0],
        episode_lengths=[3, 3],
        total_simulator_steps=6,
        total_policy_decisions=9,
        total_episodes=2,
        total_updates=2,
        episode_cyber_metrics=cyber_summaries,
        compute_metrics={
            "training_seconds": 2.0,
            "simulator_ticks": 6,
            "simulator_ticks_per_second": 3.0,
        },
    )


def read_events(path):
    events = EventAccumulator(str(path), size_guidance={"scalars": 0})
    events.Reload()
    return events


def test_training_tensorboard_logs_dynamic_metrics_and_skips_missing(tmp_path):
    result = training_result()
    with TensorBoardLogger(tmp_path) as logger:
        for metrics in result.history:
            logger({**metrics, "value_loss": 0.25})

    events = read_events(tmp_path)
    values = events.Scalars("training/cyber/availability/mean")
    assert [entry.step for entry in values] == [3, 6]
    assert [entry.value for entry in values] == pytest.approx([1.0, 0.5])
    assert events.Scalars("loss/value")[-1].value == pytest.approx(0.25)
    assert events.Scalars("training/compute/training_seconds")[-1].value == 2.0
    assert "training/cyber/GreenLocalWork_success_rate/mean" not in events.Tags()["scalars"]
    ttr = events.Scalars("training/cyber/ot_ttr_completed_ticks/mean")
    assert [(entry.step, entry.value) for entry in ttr] == [(3, 1.0)]
    assert events.Scalars("training/cyber/ot_unrecovered_count/mean")[-1].value == 1.0
    assert events.Scalars("training/cyber/ot_ttr_completed_ticks/missing")[-1].value == 1.0


def test_evaluation_tensorboard_separates_modes_and_keeps_missing_counts(tmp_path):
    with TensorBoardLogger(tmp_path) as logger:
        for mode in ("deterministic", "stochastic"):
            logger.log_evaluation(
                evaluation_result(deterministic=mode == "deterministic"),
                global_step=6,
                namespace=f"final_evaluation/{mode}",
            )

    events = read_events(tmp_path)
    for mode in ("deterministic", "stochastic"):
        prefix = f"final_evaluation/{mode}"
        assert events.Scalars(f"{prefix}/cyber/availability/mean")[0].value == 0.5
        assert events.Scalars(f"{prefix}/cyber/compromised_host_count/mean")[0].value == 0.0
        assert events.Scalars(f"{prefix}/cyber/unique_compromised_host_count/mean")[0].value == 1.0
        assert events.Scalars(f"{prefix}/cyber/ot_ttr_completed_ticks/mean")[0].value == 1.0
        assert events.Scalars(f"{prefix}/cyber/ot_ttr_completed_ticks/missing")[0].value == 1.0
        assert events.Scalars(f"{prefix}/cyber/ot_recovery_count/mean")[0].value == 0.5
        assert events.Scalars(f"{prefix}/cyber/ot_unrecovered_count/mean")[0].value == 0.5
        assert events.Scalars(f"{prefix}/cyber/ot_recovery_rate/mean")[0].value == 0.5
        missing_prefix = f"{prefix}/cyber/GreenLocalWork_success_rate"
        assert events.Scalars(f"{missing_prefix}/n")[0].value == 0.0
        assert events.Scalars(f"{missing_prefix}/missing")[0].value == 2.0
        assert f"{missing_prefix}/mean" not in events.Tags()["scalars"]
        assert f"{prefix}/episode_returns" in events.Tags()["histograms"]


def test_evaluation_json_preserves_recovery_and_unrecovered_events():
    result = evaluation_result(deterministic=False)
    record = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    recovered, unrecovered = record["episode_cyber_metrics"]
    assert recovered["ot_recovery_events"][0]["ttr_ticks"] == 1
    assert unrecovered["ot_unrecovery_events"][0]["ttr_ticks"] is None
    assert record["metrics_summary"]["GreenLocalWork_success_rate"]["mean"] is None
    assert record["deterministic"] is False
    assert recovered["unique_compromised_host_count"] == 1
    assert record["metrics_summary"]["ot_ttr_completed_ticks"] == {
        "n": 1, "missing": 1, "mean": 1.0, "std": 0.0,
    }


@pytest.mark.parametrize(
    "completed_ticks,unrecovered_count,expected_ttr,expected_rate",
    [
        ([], 0, None, None),
        ([], 2, None, 0.0),
        ([2, 8], 0, 5.0, 1.0),
        ([2, 8], 1, 5.0, 2 / 3),
    ],
)
def test_recovery_summary_distinguishes_no_outages_from_unrecovered_outages(
    completed_ticks, unrecovered_count, expected_ttr, expected_rate
):
    row = episode_metrics(recovered=False)
    row["ot_recovery_events"] = [
        {"hostname": "A", "start_tick": 1, "recovery_tick": ticks + 1, "ttr_ticks": ticks}
        for ticks in completed_ticks
    ]
    row["ot_unrecovery_events"] = [
        {"hostname": f"host_{index}", "start_tick": 1, "observed_tick": 99, "ttr_ticks": None}
        for index in range(unrecovered_count)
    ]
    summary = aggregate_episodes([-1.0], [row])
    ttr = summary["ot_ttr_completed_ticks"]
    assert ttr["mean"] == expected_ttr
    assert ttr["n"] == int(expected_ttr is not None)
    assert ttr["missing"] == int(expected_ttr is None)
    assert summary["ot_recovery_rate"]["mean"] == expected_rate
    assert summary["ot_recovery_count"]["mean"] == len(completed_ticks)
    assert summary["ot_unrecovered_count"]["mean"] == unrecovered_count
    json.dumps(summary, allow_nan=False)
    # Do not mutate the raw event records or substitute censored durations.
    assert all(event["ttr_ticks"] is None for event in row["ot_unrecovery_events"])


def test_ttr_uses_equal_episode_weights_then_equal_seed_weights():
    first = episode_metrics(recovered=True)
    first["ot_recovery_events"].append({
        "hostname": "A", "start_tick": 4, "recovery_tick": 11, "ttr_ticks": 7,
    })
    second = episode_metrics(recovered=True)
    second["ot_recovery_events"][0].update(recovery_tick=11, ttr_ticks=9)
    third = episode_metrics(recovered=False)
    summary = aggregate_episodes([-1.0] * 3, [first, second, third])
    # Episode means are 4, 9, None, not the pooled event mean (1 + 7 + 9) / 3.
    assert summary["ot_ttr_completed_ticks"] == {
        "n": 2, "missing": 1, "mean": 6.5, "std": 2.5,
    }
    assert summary["ot_recovery_count"]["mean"] == 1.0
    assert summary["ot_unrecovered_count"]["mean"] == pytest.approx(1 / 3)
    assert summary["ot_recovery_rate"]["mean"] == pytest.approx(2 / 3)

    across_seeds = aggregate_seeds({
        42: aggregate_episodes([-1.0], [first]),
        43: aggregate_episodes([-1.0], [second]),
        44: aggregate_episodes([-1.0], [third]),
    })
    ttr = across_seeds["ot_ttr_completed_ticks"]
    assert ttr["n"] == 2
    assert ttr["missing"] == 1
    assert ttr["mean"] == 6.5
    assert ttr["std"] == pytest.approx(5 / 2 ** 0.5)


def test_episode_and_training_seed_std_use_distinct_conventions():
    result = evaluation_result(deterministic=False)
    summary = result.to_dict()["metrics_summary"]
    assert summary["episode_return"]["std"] == result.std_return
    assert _summary([1.0])["std"] == 0.0
    assert _summary([None]) == {
        "n": 0, "missing": 1, "mean": None, "std": None,
    }
    seeds = {42: {"score": {"mean": 1.0}}, 43: {"score": {"mean": 3.0}}}
    assert aggregate_seeds(seeds)["score"]["std"] == pytest.approx(2 ** 0.5)
    assert aggregate_seeds({42: seeds[42]})["score"]["std"] is None


@pytest.mark.parametrize(
    "condition", ["iid", "persistent", "zero_evidence", "shuffle_hosts", "time_shift"]
)
def test_observation_wrappers_forward_metrics_without_extra_collection(condition):
    base_env = make_benchmark_env({"episode_steps": 10}, seed=123)
    if condition in ("iid", "persistent"):
        env = FixedBudgetDynamicHostMask(
            base_env, retention=0.5, mask_process=condition
        )
    else:
        env = DynamicEvidenceCounterfactual(
            base_env, transform=condition, time_shift_lag=2
        )
    env.reset(seed=123)
    for tick in range(1, 10):
        _, _, terminated, truncated, _ = env.step({})
        assert env.cyber_summary()["ticks"] == tick
        assert env.cyber_summary() == base_env.cyber_summary()
    assert all(
        terminated.get(agent, False) or truncated.get(agent, False)
        for agent in env.possible_agents
    )
    saved = env.cyber_summary()
    env.reset(seed=124)
    assert env.cyber_summary()["ticks"] == 0
    assert saved["ticks"] == 9


def test_checkpoint_preserves_cyber_compute_and_rebuilds_tensorboard(tmp_path):
    policy = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(policy.parameters())
    result = training_result()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path, policy, optimizer, PPOTrainingConfig(), result
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["episode_cyber_metrics"] == result.episode_cyber_metrics
    assert checkpoint["compute_metrics"] == result.compute_metrics
    assert checkpoint["history"] == result.history
    assert checkpoint["total_timesteps"] == 6

    log_dir = tmp_path / "rebuilt_tensorboard"
    assert write_checkpoint_history_to_tensorboard(checkpoint_path, log_dir) == 2
    assert len(read_events(log_dir).Scalars("training/cyber/availability/mean")) == 2


def configure_reporting_run(monkeypatch, tmp_path, *, tensorboard, fail_evaluation=False):
    result = training_result()
    comparison = EvaluationModeComparison(
        deterministic=evaluation_result(deterministic=True),
        stochastic=evaluation_result(deterministic=False),
        environment_seed=10_000,
        stochastic_action_seed=10_000,
    )
    env = SimpleNamespace(
        possible_agents=("blue_0",),
        observation_space=lambda agent: SimpleNamespace(shape=(4,)),
        action_space=lambda agent: SimpleNamespace(n=2),
    )
    monkeypatch.setattr(train_script, "make_benchmark_env", lambda *a, **kw: env)
    monkeypatch.setattr(train_script, "build_policy", lambda config: torch.nn.Linear(4, 2))
    monkeypatch.setattr(
        train_script,
        "evaluate_policy_with_preserved_rng",
        lambda *a, **kw: comparison.stochastic,
    )

    def fake_train(env, policy, config, *, optimizer, on_update, **kwargs):
        # Exercise reporting only; no PPO updates or simulator training.
        for metrics in result.history:
            on_update(metrics)
        return result

    run_dir = tmp_path / "run" / "seed_42"

    def fake_final_evaluation(*args, **kwargs):
        assert (run_dir / "training_summary.json").is_file()
        if fail_evaluation:
            raise RuntimeError("Simulated final evaluation failure")
        return comparison

    monkeypatch.setattr(train_script, "train_policy", fake_train)
    monkeypatch.setattr(train_script, "evaluate_policy_modes", fake_final_evaluation)
    argv = [
        "--encoder", "nodemlp",
        "--run-dir", str(tmp_path / "run"),
        "--eval-every-updates", "1",
        "--print-every-updates", "0",
    ]
    if not tensorboard:
        argv.append("--no-tensorboard")
    args = train_script.build_parser().parse_args(argv)
    return args, run_dir, result


@pytest.mark.parametrize("tensorboard", [False, True])
def test_cli_exports_history_episode_metrics_and_both_eval_modes(
    monkeypatch, tmp_path, tensorboard
):
    args, run_dir, result = configure_reporting_run(
        monkeypatch, tmp_path, tensorboard=tensorboard
    )
    train_script.run_training(args)
    history = [
        json.loads(line)
        for line in (run_dir / "training_history.jsonl").read_text().splitlines()
    ]
    assert history == result.history
    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["episode_returns"] == result.episode_returns
    assert summary["episode_lengths"] == result.episode_lengths
    assert summary["episode_cyber_metrics"] == result.episode_cyber_metrics
    assert summary["compute_metrics"] == result.compute_metrics
    assert summary["metrics_summary"]["ot_ttr_completed_ticks"]["mean"] == 1.0
    assert summary["metrics_summary"]["ot_unrecovered_count"]["mean"] == 0.5
    periodic = json.loads((run_dir / "periodic_evaluation.json").read_text())
    assert len(periodic) == 2
    assert periodic[0]["metrics_summary"]["availability"]["mean"] == 0.5
    assert periodic[0]["metrics_summary"]["ot_recovery_rate"]["mean"] == 0.5
    for mode in ("deterministic", "stochastic"):
        record = json.loads((run_dir / f"evaluation_{mode}.json").read_text())
        assert record["deterministic"] is (mode == "deterministic")
        assert len(record["episode_cyber_metrics"]) == 2
        assert record["metrics_summary"]["ot_ttr_completed_ticks"]["missing"] == 1
    if tensorboard:
        events = read_events(run_dir / "tensorboard")
        assert len(events.Scalars("training/compute/training_seconds")) == 2
        assert len(events.Scalars("evaluation/cyber/availability/mean")) == 2


def test_cli_keeps_training_json_if_final_evaluation_fails(monkeypatch, tmp_path):
    args, run_dir, result = configure_reporting_run(
        monkeypatch, tmp_path, tensorboard=True, fail_evaluation=True
    )
    with pytest.raises(RuntimeError, match="Simulated final evaluation failure"):
        train_script.run_training(args)
    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["compute_metrics"] == result.compute_metrics
    assert (run_dir / "training_history.jsonl").is_file()
    assert not (run_dir / "evaluation_deterministic.json").exists()


def test_native_cc4_short_run_reports_training_and_evaluation_metrics(tmp_path):
    args = train_script.build_parser().parse_args([
        "--encoder", "nodemlp",
        "--run-dir", str(tmp_path / "native"),
        "--total-simulator-steps", "10",
        "--episode-steps", "10",
        "--episodes-per-update", "2",
        "--update-epochs", "1",
        "--node-hidden-dim", "16",
        "--node-embedding-dim", "8",
        "--context-dim", "16",
        "--head-hidden-dim", "16",
        "--eval-every-updates", "1",
        "--periodic-eval-episodes", "1",
        "--eval-episodes", "1",
        "--print-every-updates", "0",
    ])
    train_script.run_training(args)
    run_dir = tmp_path / "native" / "seed_42"
    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["total_simulator_steps"] == 18
    assert summary["compute_metrics"]["simulator_ticks"] == 18
    compute = summary["compute_metrics"]
    assert compute["rollout/wall_seconds"] > 0
    assert compute["ppo_update/wall_seconds"] > 0
    assert compute["peak_process_tree_rss_bytes_sampled"] > 0
    assert compute["cuda_peak_allocated_bytes"] is None
    run_compute = json.loads((run_dir / "compute_summary.json").read_text())
    assert run_compute["compute_metadata"]["device"] == "cpu"
    assert run_compute["training_with_validation_seconds"] >= compute["training_seconds"]
    assert run_compute["run_wall_seconds"] >= run_compute["final_evaluation_wall_seconds"] > 0
    assert [row["ticks"] for row in summary["episode_cyber_metrics"]] == [9, 9]
    assert "ot_ttr_completed_ticks" in summary["metrics_summary"]
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["episode_cyber_metrics"] == summary["episode_cyber_metrics"]
    for mode in ("deterministic", "stochastic"):
        record = json.loads((run_dir / f"evaluation_{mode}.json").read_text())
        assert [row["ticks"] for row in record["episode_cyber_metrics"]] == record["episode_lengths"]
        assert "ot_unrecovered_count" in record["metrics_summary"]
        assert record["compute_metrics"]["simulator_ticks"] == 9
        assert record["compute_metrics"]["wall_seconds"] > 0
        assert record["compute_metrics"]["cuda_peak_allocated_bytes"] is None
    events = read_events(run_dir / "tensorboard")
    assert "training/compute/training_seconds" in events.Tags()["scalars"]
    assert "final_evaluation/stochastic/cyber/availability/mean" in events.Tags()["scalars"]
    assert "training/cyber/compromised_host_count/mean" in events.Tags()["scalars"]
    assert "training/cyber/ot_ttr_completed_ticks/n" in events.Tags()["scalars"]
    assert "training/cyber/ot_unrecovered_count/mean" in events.Tags()["scalars"]
    assert "final_evaluation/stochastic/cyber/ot_ttr_completed_ticks/n" in events.Tags()["scalars"]
    assert "training/compute/peak_process_tree_rss_bytes_sampled" in events.Tags()["scalars"]
    assert "final_evaluation/stochastic/compute/wall_seconds" in events.Tags()["scalars"]
    assert "final_evaluation/stochastic/compute/cuda_peak_allocated_bytes" not in events.Tags()["scalars"]
    for name, options in (
        ("retention", ["--retention", "0.5"]),
        ("counterfactual", ["--counterfactual", "zero_evidence"]),
    ):
        output = tmp_path / f"{name}.json"
        evaluate_command([
            "--run-dir", str(tmp_path / "native"),
            "--episodes", "1",
            "--output", str(output),
            *options,
        ])
        record = json.loads(output.read_text())
        for mode in ("deterministic", "stochastic"):
            assert record[mode]["episode_cyber_metrics"][0]["ticks"] == 9
            assert "unique_compromised_host_count" in record[mode]["metrics_summary"]
