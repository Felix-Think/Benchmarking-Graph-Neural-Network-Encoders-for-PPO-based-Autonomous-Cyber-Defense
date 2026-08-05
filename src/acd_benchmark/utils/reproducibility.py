"""Write deterministic run metadata, package versions, and checksums."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    """Run one read-only git query, returning an empty string on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def collect_reproducibility_metadata(
    repo_root: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Collect source, software, and hardware identity for one run."""
    git_status = _git(repo_root, "status", "--short")
    metadata = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_geometric_version": importlib.metadata.version("torch-geometric"),
        "numpy_version": importlib.metadata.version("numpy"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines(),
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        metadata["gpu_name"] = "Apple Metal Performance Shaders"
    else:
        metadata["gpu_name"] = None
    return metadata


def write_package_versions(path: Path) -> None:
    """Write an alphabetized package snapshot without invoking pip."""
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        },
        key=str.lower,
    )
    path.write_text("\n".join(packages) + "\n", encoding="utf-8")


def write_source_snapshot(
    run_dir: Path,
    repo_root: Path,
    relative_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Archive the code/config used by a run plus the tracked Git diff.

    A commit hash cannot reconstruct an experiment launched from a dirty
    worktree. The snapshot preserves uncommitted and untracked benchmark code,
    while ``git_diff.patch`` captures modifications to tracked dependencies
    such as the vendored CybORG source.
    """
    snapshot_root = run_dir / "source_snapshot"
    copied_files: list[str] = []
    for relative in relative_paths:
        source = repo_root / relative
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            destination = snapshot_root / candidate.relative_to(repo_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            copied_files.append(str(candidate.relative_to(repo_root)))

    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    patch_path = snapshot_root / "git_diff.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(diff.stdout if diff.returncode == 0 else b"")
    index = {
        "copied_file_count": len(copied_files),
        "copied_files": copied_files,
        "git_diff_patch": str(patch_path.relative_to(run_dir)),
        "git_diff_sha256": sha256_file(patch_path),
    }
    (snapshot_root / "snapshot_index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    return index


def write_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """Hash every run artifact except the manifest itself."""
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"artifact_count": len(entries), "artifacts": entries}
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
