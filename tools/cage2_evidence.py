"""Publish and independently verify the frozen CAGE-2 diagnostic evidence."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "291977eec20c0f223b9e507ad41482bd0db268c7"
RELEASE_METADATA = {"ARCHIVE.json", "README.md", "verification_report.json"}
RENAMED = {
    "validation_report.json": "original_validation_report.json",
    "validate_results.py": "original_validate_results.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize_paths(value, source, repo=REPO):
    if isinstance(value, dict):
        return {key: normalize_paths(item, source, repo) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item, source, repo) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_paths(item, source, repo) for item in value)
    if isinstance(value, str):
        for prefix, marker in ((source, "{evidence}"), (repo, "{repo}")):
            if value.startswith(str(prefix) + "/"):
                return marker + "/" + value[len(str(prefix)) + 1 :]
    return value


def private_paths(value):
    if isinstance(value, dict):
        return any(private_paths(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(private_paths(item) for item in value)
    return isinstance(value, str) and any(
        prefix in value for prefix in ("/Users/", "/home/", "/content/", "C:\\Users\\")
    )


def equal_payload(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal_payload(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal_payload(a, b) for a, b in zip(left, right))
        )
    return left == right


def publish(source, destination):
    source, destination = source.resolve(), destination.resolve()
    require(not destination.exists(), "Use a new evidence directory.")
    require(
        not destination.is_relative_to(source), "Evidence must be outside the raw run."
    )
    require(
        read_json(source / "completed.json")["status"] == "completed",
        "Run is incomplete.",
    )
    require(
        read_json(source / "validation_report.json")["status"] == "passed",
        "Raw validation failed.",
    )
    manifest = read_json(source / "manifest.json")
    require(manifest["base_commit"] == BASE_COMMIT, "Unexpected diagnostic base.")
    destination.mkdir(parents=True)
    records, changed = {}, []
    for original in sorted(source.rglob("*")):
        require(not original.is_symlink(), "Do not publish symlinked evidence.")
        if not original.is_file():
            continue
        relative = original.relative_to(source).as_posix()
        published = RENAMED.get(relative, relative)
        target = destination / published
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.suffix == ".json":
            value = read_json(original)
            normalized = normalize_paths(value, source)
            require(
                not private_paths(normalized), "Unredacted JSON metadata: " + relative
            )
            if normalized == value:
                shutil.copy2(original, target)
            else:
                write_json(target, normalized)
                changed.append(published)
        elif original.suffix == ".pt":
            value = torch.load(original, map_location="cpu", weights_only=True)
            normalized = normalize_paths(value, source)
            require(
                not private_paths(normalized),
                "Unredacted checkpoint metadata: " + relative,
            )
            torch.save(normalized, target)
            restored = torch.load(target, map_location="cpu", weights_only=True)
            require(equal_payload(normalized, restored), "Checkpoint payload changed.")
            changed.append(published)
        else:
            require(
                original.suffix in {".jsonl", ".npz", ".py"},
                "Unexpected evidence file: " + relative,
            )
            shutil.copy2(original, target)
        records[published] = {
            "original_name": relative,
            "original_sha256": sha256(original),
            "published_sha256": sha256(target),
        }
    metadata = {
        "archive_version": 1,
        "base_commit": BASE_COMMIT,
        "original_run": source.name,
        "files": records,
        "path_metadata_normalized": changed,
        "normalization": "Only repository/run path strings are replaced by {repo}/{evidence} tokens. Checkpoint tensors, optimizer state, metrics, and probe arrays are unchanged.",
        "original_validation_report": "original_validation_report.json",
    }
    write_json(destination / "ARCHIVE.json", metadata)
    verify_payload(destination)
    return metadata


def verify_payload(root):
    metadata = read_json(root / "ARCHIVE.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(
        actual - RELEASE_METADATA == set(metadata["files"]),
        "Evidence inventory mismatch.",
    )
    for name, record in metadata["files"].items():
        require(
            sha256(root / name) == record["published_sha256"],
            "Checksum mismatch: " + name,
        )
        if name.endswith(".json"):
            require(
                not private_paths(read_json(root / name)),
                "Private JSON metadata: " + name,
            )
        elif name.endswith(".pt"):
            value = torch.load(root / name, map_location="cpu", weights_only=True)
            require(not private_paths(value), "Private checkpoint metadata: " + name)
    original = read_json(root / "original_validation_report.json")
    originals = {
        record["original_name"]: record for record in metadata["files"].values()
    }
    for name, digest in original.get("sha256", {}).items():
        require(
            originals[name]["original_sha256"] == digest,
            "Original audit mismatch: " + name,
        )
    if "validation_script_sha256" in original:
        require(
            originals["validate_results.py"]["original_sha256"]
            == original["validation_script_sha256"],
            "Original validator identity mismatch.",
        )
    return metadata


def verify_source(manifest):
    require(manifest["base_commit"] == BASE_COMMIT, "Unexpected source base.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=REPO,
        check=True,
    )
    files = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", BASE_COMMIT], cwd=REPO
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(filter(None, files)):
        digest.update(name.encode() + b"\0" + (REPO / name).read_bytes() + b"\0")
    require(
        digest.hexdigest() == manifest["tracked_source_sha256"],
        "Frozen simulator/core source changed.",
    )
    for name, expected in manifest["extension_sha256"].items():
        require(sha256(REPO / name) == expected, "As-run extension changed: " + name)


def verify(root, report_path=None):
    from acd_benchmark.cyber.action_space import describe_action
    from experiments.cage2_revalidation.policies import make_policy
    from experiments.cage2_revalidation.run import (
        concentration,
        evaluate,
        intervention_observations,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    metadata = verify_payload(root)
    manifest, summary = (
        read_json(root / "manifest.json"),
        read_json(root / "summary.json"),
    )
    verify_source(manifest)
    require(
        read_json(root / "completed.json")["status"] == "completed",
        "Incomplete evidence.",
    )
    evaluations = sorted((root / "baselines").glob("*.json")) + sorted(
        root.glob("*/seed_0/evaluation_*.json")
    )
    for path in evaluations:
        data = read_json(path)
        records, counts = data["episode_records"], Counter()
        require(len(records) == data["episodes"] == 100, "Episode count mismatch.")
        require(
            [row["seed"] for row in records] == list(range(10000, 10100)),
            "Evaluation seeds changed.",
        )
        for row in records:
            require(
                row["ticks"] == sum(row["action_counts"].values()) == 200,
                "Decision/tick mismatch.",
            )
            counts.update(
                {int(key): value for key, value in row["action_counts"].items()}
            )
        require(
            {describe_action(key): value for key, value in counts.items()}
            == data["action_counts"],
            "Action count mismatch.",
        )
        recomputed = concentration(counts)
        for key in (
            "restore_share",
            "top_action",
            "top_action_share",
            "decisions",
            "distinct_actions",
        ):
            require(data[key] == recomputed[key], "Action summary mismatch: " + key)
        require(
            abs(statistics.mean(row["return"] for row in records) - data["mean_return"])
            < 1e-9,
            "Mean return mismatch.",
        )
        require(
            abs(
                statistics.stdev(row["return"] for row in records)
                - data["sample_std_return_across_evaluation_episodes"]
            )
            < 1e-9,
            "Return standard deviation mismatch.",
        )
        if path.parent.name == "baselines":
            require(
                summary["baselines"][path.stem] == data, "Baseline summary mismatch."
            )
    require(len(evaluations) == 12, "Missing model or baseline evaluations.")
    replays, results = [], []
    for model in ("flat", "gcn", "graphsage"):
        directory = root / model / "seed_0"
        checkpoint = torch.load(
            directory / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        require(
            checkpoint["total_timesteps"] == 100000
            and checkpoint["total_updates"] == 49,
            "Training budget mismatch.",
        )
        require(
            len(checkpoint["episode_returns"]) == 500
            and set(checkpoint["episode_lengths"]) == {200},
            "Training episodes changed.",
        )
        policy = make_policy(model, manifest["graph"])
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        require(
            sum(p.numel() for p in policy.parameters())
            == manifest["models_parameter_counts"][model],
            "Parameter count mismatch.",
        )
        for mode in ("greedy", "stochastic"):
            data = read_json(directory / ("evaluation_" + mode + ".json"))
            require(summary["models"][model][mode] == data, "Model summary mismatch.")
            with np.load(
                directory / ("evaluation_" + mode + ".probes.npz"), allow_pickle=False
            ) as probes:
                inputs = torch.as_tensor(probes["observations"])
                require(
                    tuple(inputs.shape) == (1000, 13, 4), "Probe input shape mismatch."
                )
                with torch.no_grad():
                    logits, _ = policy(inputs)
                    require(
                        np.allclose(
                            logits.numpy(), probes["logits"], atol=1e-5, rtol=1e-5
                        ),
                        "Saved probe logits mismatch.",
                    )
                    original = logits.softmax(-1)
                    for name, reported in data["same_state_observation_probes"].items():
                        changed = torch.as_tensor(
                            np.stack(
                                [
                                    intervention_observations(obs)[name]
                                    for obs in probes["observations"]
                                ]
                            )
                        )
                        probabilities = policy(changed)[0].softmax(-1)
                        tv = 0.5 * (original - probabilities).abs().sum(-1)
                        rate = (
                            (original.argmax(-1) != probabilities.argmax(-1))
                            .float()
                            .mean()
                        )
                        require(
                            abs(float(tv.mean()) - reported["mean_total_variation"])
                            < 1e-6,
                            "Probe distribution mismatch.",
                        )
                        require(
                            abs(float(rate) - reported["argmax_change_fraction"])
                            < 1e-6,
                            "Probe argmax mismatch.",
                        )
            with tempfile.TemporaryDirectory(
                prefix="cage2-evidence-replay-"
            ) as temporary:
                replay = evaluate(
                    policy=policy,
                    baseline=None,
                    episodes=1,
                    first_seed=10000,
                    action_seed=60000,
                    horizon=200,
                    deterministic=mode == "greedy",
                    output=Path(temporary) / "replay.json",
                )
                require(
                    abs(replay["mean_return"] - data["episode_records"][0]["return"])
                    < 1e-9,
                    "Checkpoint replay return mismatch.",
                )
                require(
                    replay["episode_records"][0]["action_counts"]
                    == data["episode_records"][0]["action_counts"],
                    "Checkpoint replay actions mismatch.",
                )
            replays.append(
                {"model": model, "mode": mode, "seed": 10000, "matched": True}
            )
            results.append(
                {
                    "model": model,
                    "mode": mode,
                    "mean_return": data["mean_return"],
                    "evaluation_episode_sample_std": data[
                        "sample_std_return_across_evaluation_episodes"
                    ],
                    "restore_percent": 100 * data["restore_share"],
                    "top_action": data["top_action"],
                    "top_action_percent": 100 * data["top_action_share"],
                }
            )
    report = {
        "status": "passed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "payload_files_checked": len(metadata["files"]),
        "evaluation_files_checked": len(evaluations),
        "frozen_source_verified": True,
        "as_run_extension_verified": True,
        "private_path_scan_passed": True,
        "checkpoint_replays": replays,
        "results": results,
        "scope": "All stored evaluation summaries and observation probes; one replayed episode per model/mode, not all 100 episodes.",
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("publish")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--evidence", type=Path, required=True)
    check.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "publish":
        result = publish(args.source, args.destination)
        print(
            json.dumps(
                {
                    "payload_files": len(result["files"]),
                    "normalized": result["path_metadata_normalized"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(verify(args.evidence.resolve(), args.report), indent=2))


if __name__ == "__main__":
    main()
