"""Independently check saved evaluations and replay each final checkpoint."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np
import torch

from acd_benchmark.cyber.action_space import describe_action
from experiments.cage2_revalidation.policies import make_policy
from experiments.cage2_revalidation.run import (
    concentration, evaluate, intervention_observations, save_json, source_identity,
)


ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
manifest = json.loads((ROOT / "manifest.json").read_text())
summary = json.loads((ROOT / "summary.json").read_text())
assert json.loads((ROOT / "completed.json").read_text())["status"] == "completed"
for key, value in source_identity().items():
    assert manifest[key] == value, key

evaluations = list((ROOT / "baselines").glob("*.json")) + list(ROOT.glob("*/seed_0/evaluation_*.json"))
checks = []
for path in evaluations:
    data = json.loads(path.read_text())
    records = data["episode_records"]
    assert len(records) == data["episodes"] == 100
    assert [row["seed"] for row in records] == list(range(10_000, 10_100))
    counts = Counter()
    for row in records:
        assert row["ticks"] == 200
        assert sum(row["action_counts"].values()) == row["ticks"]
        counts.update({int(key): value for key, value in row["action_counts"].items()})
    assert {describe_action(key): value for key, value in counts.items()} == data["action_counts"]
    recomputed = concentration(counts)
    for key in ("restore_share", "top_action", "top_action_share", "decisions", "distinct_actions"):
        assert data[key] == recomputed[key], (path, key)
    assert abs(statistics.mean(row["return"] for row in records) - data["mean_return"]) < 1e-9
    assert abs(statistics.stdev(row["return"] for row in records) - data["sample_std_return_across_evaluation_episodes"]) < 1e-9
    checks.append(str(path.relative_to(ROOT)))
assert len(checks) == 12

replays = []
rows = []
for name in ("flat", "gcn", "graphsage"):
    directory = ROOT / name / "seed_0"
    policy = make_policy(name, manifest["graph"])
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["total_timesteps"] == 100_000
    assert checkpoint["total_updates"] == 49
    assert len(checkpoint["episode_returns"]) == 500
    assert all(length == 200 for length in checkpoint["episode_lengths"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    assert sum(parameter.numel() for parameter in policy.parameters()) == manifest["models_parameter_counts"][name]
    for mode in ("greedy", "stochastic"):
        data = summary["models"][name][mode]
        saved = json.loads((directory / f"evaluation_{mode}.json").read_text())
        assert data == saved
        probes = np.load(directory / f"evaluation_{mode}.probes.npz")
        assert probes["observations"].shape == (1000, 13, 4)
        assert probes["logits"].shape == (1000, 41)
        inputs = torch.as_tensor(probes["observations"])
        with torch.no_grad():
            logits, _ = policy(inputs)
            assert np.allclose(logits.numpy(), probes["logits"], atol=1e-5, rtol=1e-5)
            original = logits.softmax(-1)
            for intervention in data["same_state_observation_probes"]:
                changed_inputs = torch.as_tensor(np.stack([
                    intervention_observations(observation)[intervention]
                    for observation in probes["observations"]
                ]))
                changed_logits, _ = policy(changed_inputs)
                tv = 0.5 * (original - changed_logits.softmax(-1)).abs().sum(-1)
                rates = (original.argmax(-1) != changed_logits.argmax(-1)).float()
                reported = data["same_state_observation_probes"][intervention]
                assert abs(float(tv.mean()) - reported["mean_total_variation"]) < 1e-6
                assert abs(float(rates.mean()) - reported["argmax_change_fraction"]) < 1e-6
        with tempfile.TemporaryDirectory(prefix="cage2_replay_", dir="/tmp") as temporary:
            replay = evaluate(policy=policy, baseline=None, episodes=1, first_seed=10_000,
                              action_seed=60_000, horizon=200, deterministic=mode == "greedy",
                              output=Path(temporary) / "replay.json")
            assert abs(replay["mean_return"] - data["episode_records"][0]["return"]) < 1e-9
            assert replay["episode_records"][0]["action_counts"] == data["episode_records"][0]["action_counts"]
            replays.append({"model": name, "mode": mode, "seed": 10_000, "matched": True})
        rows.append({"model": name, "mode": mode,
                     "mean_return": data["mean_return"],
                     "evaluation_episode_sample_std": data["sample_std_return_across_evaluation_episodes"],
                     "restore_percent": 100 * data["restore_share"],
                     "top_action": data["top_action"],
                     "top_action_percent": 100 * data["top_action_share"],
                     "distinct_action_targets": data["distinct_actions"],
                     "episodes_with_target_or_action_changes": sum(len(row["action_counts"]) > 1 for row in data["episode_records"]),
                     "zero_compromise_probe": data["same_state_observation_probes"]["zero_compromise"]})

evidence_files = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix in (".pt", ".json", ".jsonl", ".npz") and path.name != "validation_report.json"]
report = {"status": "passed", "checked_evaluation_files": checks,
          "source_identity_unchanged": True,
          "checkpoint_replays": replays, "results": rows,
          "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(evidence_files)},
          "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
save_json(ROOT / "validation_report.json", report)
print(json.dumps({key: report[key] for key in ("status", "checkpoint_replays", "results")}, indent=2))
