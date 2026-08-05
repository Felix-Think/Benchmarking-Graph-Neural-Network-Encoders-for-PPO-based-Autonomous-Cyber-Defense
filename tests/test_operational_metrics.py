from pathlib import Path

import torch

from acd_benchmark.cyber.scalable_scenario import write_scalable_scenario
from acd_benchmark.evaluation.operational_metrics import (
    EpisodeMetricTracker,
    GroundTruthSnapshot,
    evaluate_policy_with_operational_metrics,
)
from acd_benchmark.rl.policies import FlatActorCritic
from acd_benchmark.wrappers.stage1.env import Stage1FlatEnv
from scripts.stage1.evaluate_baselines import _fixed_action_index


def _snapshot(presence=(), privileged=(), available=("Op_Server0",)):
    return GroundTruthSnapshot(
        red_presence_hosts=frozenset(presence),
        red_privileged_hosts=frozenset(privileged),
        available_critical_hosts=frozenset(available),
    )


def test_latency_tracker_records_detection_recovery_and_censoring():
    tracker = EpisodeMetricTracker(
        episode=0,
        environment_seed=1,
        critical_hosts=("Op_Server0",),
        initial_snapshot=_snapshot(presence=("User0",)),
    )
    tracker.record_step(
        timestep=1,
        action_name="Sleep",
        action_hostname=None,
        snapshot=_snapshot(presence=("User0", "Enterprise0")),
        inference_ms=1.0,
    )
    tracker.record_step(
        timestep=2,
        action_name="Analyse",
        action_hostname="User0",
        snapshot=_snapshot(presence=("User0", "Enterprise0"), privileged=("Enterprise0",)),
        inference_ms=2.0,
    )
    tracker.record_step(
        timestep=3,
        action_name="Remove",
        action_hostname="User0",
        snapshot=_snapshot(presence=("Enterprise0",)),
        inference_ms=3.0,
    )
    metrics = tracker.finalize(episode_return=-3.0)

    assert metrics.episode_return == -3.0
    assert metrics.detection_observed_count == 1
    assert metrics.detection_censored_count == 1
    assert metrics.recovery_observed_count == 1
    assert metrics.recovery_censored_count == 1
    assert metrics.detection_latency == 2.0
    assert metrics.recovery_latency == 2.5
    assert metrics.mission_availability == 1.0


def test_live_operational_evaluation_returns_all_metric_groups(tmp_path: Path):
    scenario_path = write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)
    env = Stage1FlatEnv(
        scenario_path,
        observation_probability=0.25,
        max_steps=3,
    )
    observation, _ = env.reset(seed=1)
    policy = FlatActorCritic(
        input_dim=observation.shape[0],
        hidden_dim=16,
        action_count=env.action_count,
    )
    metrics = evaluate_policy_with_operational_metrics(
        env,
        policy,
        episodes=1,
        seed=1,
        device=torch.device("cpu"),
    )

    assert metrics["episode_count"] == 1
    assert 0.0 <= metrics["mean_mission_availability"] <= 1.0
    assert metrics["mean_privileged_compromised_hosts"] >= 0.0
    assert "mean_detection_latency" in metrics
    assert "mean_recovery_latency" in metrics
    assert metrics["mean_inference_ms"] >= 0.0


def test_stage1_baseline_actions_resolve_after_reset(tmp_path: Path):
    scenario_path = write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)
    env = Stage1FlatEnv(
        scenario_path,
        observation_probability=1.0,
        max_steps=3,
    )
    env.reset(seed=1)

    sleep_index = _fixed_action_index(env, "sleep")
    monitor_index = _fixed_action_index(env, "monitor")

    assert env.action_specs[sleep_index].name == "Sleep"
    assert env.action_specs[monitor_index].name == "Monitor"
