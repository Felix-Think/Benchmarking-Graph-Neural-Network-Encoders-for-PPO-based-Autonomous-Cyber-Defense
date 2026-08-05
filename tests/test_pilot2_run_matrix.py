import sys
from pathlib import Path

from scripts.pilot2 import run_matrix


def test_dry_run_does_not_create_output_directory(tmp_path, monkeypatch):
    output_root = tmp_path / "pilot2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix",
            "--config",
            str(Path("configs/pilot2_observation_16.yaml")),
            "--output-root",
            str(output_root),
            "--models",
            "flat",
            "--seeds",
            "0",
            "--dry-run",
        ],
    )

    run_matrix.main()

    assert not output_root.exists()
