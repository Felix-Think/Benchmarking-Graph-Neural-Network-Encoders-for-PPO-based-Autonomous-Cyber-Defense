import json

import pytest
import torch

from tools.cage2_evidence import BASE_COMMIT, normalize_paths, publish, verify_payload


def test_path_normalization_preserves_tensors_and_other_strings(tmp_path):
    source = tmp_path / "repo" / "runs" / "original"
    repo = tmp_path / "repo"
    tensor = torch.arange(3)
    value = {
        "checkpoint": str(source / "flat/checkpoint.pt"),
        "source": str(repo / "file.py"),
        "tensor": tensor,
        "label": "Restore(Op_Server0)",
    }
    result = normalize_paths(value, source, repo)
    assert result["checkpoint"] == "{evidence}/flat/checkpoint.pt"
    assert result["source"] == "{repo}/file.py"
    assert result["tensor"] is tensor
    assert result["label"] == value["label"]
    assert value["checkpoint"].startswith(str(source))


def test_publication_preserves_raw_evidence_and_checkpoint_state(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    files = {
        "completed.json": {"status": "completed"},
        "validation_report.json": {"status": "passed"},
        "manifest.json": {
            "base_commit": BASE_COMMIT,
            "argv": [str(source / "test.py")],
        },
    }
    for name, value in files.items():
        (source / name).write_text(json.dumps(value))
    original = {
        "config": {"checkpoint_path": str(source / "checkpoint.pt")},
        "policy_state_dict": {"weight": torch.arange(4)},
    }
    torch.save(original, source / "checkpoint.pt")
    raw_bytes = (source / "checkpoint.pt").read_bytes()
    destination = tmp_path / "published"
    publish(source, destination)
    restored = torch.load(destination / "checkpoint.pt", weights_only=True)
    assert restored["config"]["checkpoint_path"] == "{evidence}/checkpoint.pt"
    assert torch.equal(
        restored["policy_state_dict"]["weight"], original["policy_state_dict"]["weight"]
    )
    assert (source / "checkpoint.pt").read_bytes() == raw_bytes
    assert verify_payload(destination)["base_commit"] == BASE_COMMIT
    with pytest.raises(ValueError, match="new evidence"):
        publish(source, destination)
    (destination / "completed.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_payload(destination)
