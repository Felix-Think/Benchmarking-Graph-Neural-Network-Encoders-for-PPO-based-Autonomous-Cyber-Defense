"""One policy at a time, immutable configuration, fail-fast experiment queue."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile


REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_files():
    paths = []
    for directory in ("src", "scripts", "CybORG", "configs", "tests"):
        for root, directories, files in os.walk(REPO / directory):
            directories[:] = [
                d for d in directories
                if d not in (".git", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache")
            ]
            paths.extend(
                Path(root) / name for name in files
                if name != ".DS_Store" and not name.endswith((".pyc", ".pyo"))
            )
    paths.extend(REPO / name for name in ("pyproject.toml", "uv.lock", ".python-version"))
    return sorted(paths)


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_jobs(config, run_root):
    jobs = []
    # Complete all six cells for a seed before proceeding to the next seed.
    for seed in config["seeds"]:
        for encoder in config["encoders"]:
            for regime in config["train_regimes"]:
                run_dir = run_root / encoder / regime
                settings = {
                    **config["training"], "encoder": encoder, "train_regime": regime,
                    "seed": seed, "run_dir": str(run_dir),
                }
                arguments = []
                for name, value in settings.items():
                    option = "--" + name.replace("_", "-")
                    if isinstance(value, bool):
                        if value:
                            arguments.append(option)
                    elif isinstance(value, list):
                        arguments.extend([option, *map(str, value)])
                    else:
                        arguments.extend([option, str(value)])
                jobs.append({
                    "encoder": encoder, "train_regime": regime, "seed": seed,
                    "directory": str(run_dir / f"seed_{seed}"), "arguments": arguments,
                })
    return jobs


def execute_queue(root):
    manifest = json.loads((root / "manifest.json").read_text())
    state = {"status": "running", "pid": os.getpid(), "started_at": now(), "completed": []}
    write_json(root / "status.json", state)
    child = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, job in enumerate(manifest["jobs"]):
            if (root / "STOP_AFTER_CURRENT").exists():
                state["status"] = "stopped_after_current"
                break
            if fingerprint(source_files()) != manifest["source_sha256"]:
                raise RuntimeError("Source changed after launch; queue stopped, not mixing code versions")
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            state.update(current_index=index, current=job, current_started_at=now())
            environment = {
                **os.environ, "PYTHONHASHSEED": str(job["seed"]), "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYG_HOME": "/tmp/pyg",
            }
            command = [
                sys.executable, "-u", "-m", "scripts.training.train", *job["arguments"],
            ]
            with (directory / "console.log").open("xb") as log:
                child = subprocess.Popen(
                    command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                state["child_pid"] = child.pid
                write_json(root / "status.json", state)
                print(f"START {index + 1}/{len(manifest['jobs'])} {directory}", flush=True)
                exit_code = child.wait()
            child = None
            if exit_code:
                raise RuntimeError(f"Training exited with code {exit_code}: {directory / 'console.log'}")
            if not (directory / "evaluation_retention.json").is_file():
                raise RuntimeError(f"Missing final retention evaluation: {directory}")
            state["completed"].append({**job, "finished_at": now()})
            state.pop("child_pid", None)
            write_json(root / "status.json", state)
        else:
            state["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = repr(exc)
        print(state["error"], flush=True)
    finally:
        state["updated_at"] = now()
        write_json(root / "status.json", state)
    return 0 if state["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/encoder_retention_500k_cuda.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launch", action="store_true", help="Start a detached serial queue")
    parser.add_argument("--foreground", action="store_true", help="Run the queue in this terminal/notebook")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a subset of the configured training seeds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.launch and args.foreground:
        parser.error("Choose --launch or --foreground, not both")
    root = args.run_root.resolve()
    if args.worker:
        return execute_queue(root)
    config = json.loads(args.config.read_text())
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(config["seeds"]):
            parser.error("--seeds must be a unique subset of configured seeds")
        config["seeds"] = args.seeds
    jobs = build_jobs(config, root)
    for index, job in enumerate(jobs, 1):
        print(f"{index:02d}. {job['encoder']} {job['train_regime']} seed={job['seed']}")
    if not (args.launch or args.foreground):
        return 0
    if config["training"].get("device", "cpu").startswith("cuda"):
        from acd_benchmark.rl.runtime import learner_device
        learner_device(config["training"]["device"])
    # Refuse overwrites, including a previously failed experiment queue.
    root.mkdir(parents=True, exist_ok=False)
    paths = source_files()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
    )
    manifest = {
        "created_at": now(), "config": config, "jobs": jobs, "source_sha256": fingerprint(paths),
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "python": sys.version, "max_concurrent_policies": 1,
    }
    write_json(root / "manifest.json", manifest)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(REPO))
        for name in ("README.md", "docs/ENCODER_RETENTION_PROTOCOL.md"):
            archive.add(REPO / name, arcname=name)
    if git_head.returncode == 0:
        (root / "working_changes.patch").write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        )
    if args.foreground:
        return execute_queue(root)
    with (root / "queue.log").open("xb") as log:
        worker = subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.experiments.run_encoder_matrix", "--worker", "--run-root", str(root)],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"queue_pid={worker.pid} run_root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
