import torch

from acd_benchmark.rl.ppo.evaluation import load_policy_checkpoint
from acd_benchmark.rl.ppo.training import (
    PPOTrainingConfig,
    save_best_evaluation_checkpoint,
)


def test_best_evaluation_checkpoint_records_selection_metadata(tmp_path):
    policy = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    config = PPOTrainingConfig(total_simulator_steps=100)
    checkpoint_path = tmp_path / "best_checkpoint.pt"

    save_best_evaluation_checkpoint(
        path=checkpoint_path,
        policy=policy,
        optimizer=optimizer,
        config=config,
        update=10,
        global_step=3990,
        training_metrics={"update": 10, "global_step": 3990},
        evaluation={"mean_return": -123.0, "deterministic": True},
        metric_name="periodic_deterministic_mean_return",
        metric_value=-123.0,
    )

    restored_policy = torch.nn.Linear(3, 2)
    checkpoint = load_policy_checkpoint(restored_policy, checkpoint_path)

    assert checkpoint["checkpoint_type"] == "best_periodic_evaluation"
    assert checkpoint["selection_metric"] == {
        "name": "periodic_deterministic_mean_return",
        "mode": "max",
        "value": -123.0,
    }
    assert checkpoint["total_updates"] == 10
    assert checkpoint["total_simulator_steps"] == 3990
    assert checkpoint["evaluation"]["deterministic"] is True
    for expected, restored in zip(
        policy.parameters(),
        restored_policy.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, restored)
