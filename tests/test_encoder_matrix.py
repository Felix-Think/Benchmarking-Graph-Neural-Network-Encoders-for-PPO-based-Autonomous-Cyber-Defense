import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.experiments.run_encoder_matrix import build_jobs
from scripts.training.train import build_parser
from scripts.analysis.matrix_status import read_history


def test_matrix_contains_eighteen_matched_serial_jobs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/experiments/encoder_retention_500k_cuda.json").read_text())
    jobs = build_jobs(config, tmp_path)
    assert len(jobs) == 18
    assert len({row["directory"] for row in jobs}) == 18
    assert [row["seed"] for row in jobs] == [42] * 6 + [43] * 6 + [44] * 6
    for job in jobs:
        args = build_parser().parse_args(job["arguments"])
        assert args.total_simulator_steps == 500000
        assert args.checkpoint_steps == [100000, 200000, 300000, 400000, 500000]
        assert args.device == "cuda"
        assert args.rollout_workers == 2
        assert args.episodes_per_update == 4
        assert args.action_seed_protocol == "per_episode_v1"
        assert args.eval_retentions == [1, 0.75, 0.5, 0.25]
        assert args.torch_threads == 1
        assert not args.semantic_action_mask
        assert args.periodic_eval_seed != args.eval_seed
        assert args.train_mask_seed != args.eval_mask_seed


def test_progress_reader_ignores_only_incomplete_last_record(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"global_step": 10}\n{"global_step":')
    assert read_history(path) == [{"global_step": 10}]


def test_portable_queue_uses_same_python_without_zsh(tmp_path, monkeypatch):
    from scripts.experiments import run_encoder_matrix as module
    directory = tmp_path / "nodemlp/full/seed_42"
    directory.mkdir(parents=True)
    (directory / "evaluation_retention.json").write_text("{}")
    manifest = {
        "source_sha256": "test", "jobs": [{
            "directory": str(directory), "seed": 42, "arguments": ["--encoder", "nodemlp"],
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "source_files", lambda: [])
    monkeypatch.setattr(module, "fingerprint", lambda paths: "test")
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    calls = []

    class Child:
        pid = 1234
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    assert module.execute_queue(tmp_path) == 0
    command, kwargs = calls[0]
    assert command[:5] == [sys.executable, "-u", "-m", "scripts.training.train", "--encoder"]
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"


def test_foreground_queue_supports_source_archive_without_git(tmp_path, monkeypatch):
    from scripts.experiments import run_encoder_matrix as module
    config = Path(__file__).resolve().parents[1] / "configs/experiments/encoder_retention_500k_cuda.json"
    monkeypatch.setattr(module, "source_files", lambda: [])
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 128, stdout="", stderr="not a repository"),
    )
    seen = []
    monkeypatch.setattr(module, "execute_queue", lambda root: seen.append(root) or 0)
    monkeypatch.setattr("acd_benchmark.rl.runtime.learner_device", lambda name: name)
    root = tmp_path / "queue"
    assert module.main(["--config", str(config), "--run-root", str(root), "--seeds", "43", "--foreground"]) == 0
    assert seen == [root]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["git_head"] is None
    assert len(manifest["jobs"]) == 6
    assert all(job["seed"] == 43 for job in manifest["jobs"])


def test_queue_rejects_duplicate_or_unknown_seeds(tmp_path):
    from scripts.experiments.run_encoder_matrix import main
    for seeds in (["42", "42"], ["999"]):
        with pytest.raises(SystemExit):
            main(["--run-root", str(tmp_path / "unused"), "--seeds", *seeds])
    assert not (tmp_path / "unused").exists()
