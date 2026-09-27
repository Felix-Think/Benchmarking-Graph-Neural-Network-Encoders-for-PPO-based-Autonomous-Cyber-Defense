"""Fail-fast CUDA qualification; never mistake skipped GPU tests for a pass."""

import os
from pathlib import Path
import subprocess
import sys

import torch

from acd_benchmark.rl.runtime import learner_device


def main():
    learner_device("cuda")
    print(f"CUDA smoke test: {torch.cuda.get_device_name(0)}", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_cuda_pipeline.py", "tests/test_resume.py", "-q"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )


if __name__ == "__main__":
    raise SystemExit(main())
