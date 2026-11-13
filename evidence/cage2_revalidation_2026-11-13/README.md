# CAGE-2 diagnostic evidence, 2026-11-13

This directory preserves fresh Flat, GCN, and GraphSAGE diagnostic runs for the
paper's CAGE-2 validation discussion. It is not the CAGE-4 main benchmark and
does not recover the historical temporal-belief pilot.

## Provenance

- Evidence branch: `codex/cage2-evidence-2026-11-13`.
- Frozen simulator/core: `291977eec20c0f223b9e507ad41482bd0db268c7`.
- As-run extension: the five Python files named and hashed in `manifest.json`.
- Original run: `cage2_three_encoder_100k_seed0_2026-11-13`.
- Artifact payload: 32 files, approximately 2.6 MiB, inventoried in `ARCHIVE.json`.
- Original raw files remain unchanged outside this published evidence directory.

Each payload record includes the original and published SHA-256 checksums.
Absolute repository/run path strings in the manifest, training summaries and
checkpoint metadata were replaced with `{repo}`/`{evidence}` tokens. Checkpoint
tensors, optimizer state, trajectories, metrics and probe arrays were preserved.
The checkpoint files were reserialized to remove their path metadata; they are
therefore not byte-identical to the originals.

`original_validation_report.json` and `original_validate_results.py` preserve
the pre-publication audit. Their original-checksum claims refer to the raw files,
not the path-normalized copies. Use `ARCHIVE.json` for the published payload and
`verification_report.json` for the publication audit. These administrative
metadata files and this README are versioned by Git rather than self-checksummed.

## Contents

- `manifest.json`: source hashes, graph, action catalogue, seed protocol and runtime.
- `{flat,gcn,graphsage}/seed_0/checkpoint.pt`: final policy and optimizer state.
- `training_history.jsonl` and `training_summary.json`: update history and budget.
- `evaluation_{greedy,stochastic}.json`: all 100 episode records in each mode.
- `evaluation_{greedy,stochastic}.probes.npz`: 1,000 inputs/logits per model and mode.
- `baselines/`: six predeclared simple controls, with full episode records.
- `summary.json` and `completed.json`: combined results and completion marker.

Only one training seed (0) is represented. Evaluation-episode standard deviations
are not uncertainty across training seeds. Neither these runs nor the probes
establish a general encoder ranking or the causal usefulness of observations.

## Verify Without Training

From the repository root, recreate the recorded Python 3.10 environment:

```sh
uv sync --locked --all-groups --python 3.10.19
env PYTHONPATH=CybORG:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python tools/cage2_evidence.py verify \
  --evidence evidence/cage2_revalidation_2026-11-13
```

The verifier checks all payload checksums, frozen core source and the original
extension hashes. It recomputes all 12 model/baseline evaluation summaries,
validates all six saved probe sets against the checkpoints, and replays one
200-tick episode for every model/mode pair (six replays). It does not replay all
100 evaluation episodes. The default invocation writes no evidence files.

The recorded runtime was macOS arm64, Python 3.10.19, Torch 2.3.1, PyG 2.5.3,
NumPy 1.23.5, Gym 0.23.1 and vendored CybORG 3.1, with one CPU Torch/BLAS
thread. Cross-platform bitwise equivalence is not promised.

Tests, including archive normalization and tamper detection:

```sh
env PYTHONPATH=CybORG:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m pytest tests \
  experiments/cage2_revalidation/test_protocol.py tools/test_cage2_evidence.py -q
```

## Rerun Training in Isolation

The original entry point insists on the frozen base commit, so create a separate
worktree and overlay only the versioned as-run extension. Run these commands from
the evidence checkout, with an unused sibling destination:

```sh
git worktree add --detach ../cage2-rerun 291977eec20c0f223b9e507ad41482bd0db268c7
git archive HEAD experiments | tar -x -C ../cage2-rerun
cd ../cage2-rerun
uv sync --locked --all-groups --python 3.10.19
env PYTHONPATH=CybORG:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -u -m experiments.cage2_revalidation.run \
  --run-root runs/cage2_new_reproduction
```

The command trains all three models for 100,000 environment steps each and runs
the six controls and both evaluation modes. It refuses an existing output root.
Do not rerun into this evidence directory or silently replace archived results.

## Citation

Cite the GitHub URL of the final evidence commit and this directory, not only the
mutable branch name or the base commit. The base commit alone does not contain
the GNN adapters or the evidence.
