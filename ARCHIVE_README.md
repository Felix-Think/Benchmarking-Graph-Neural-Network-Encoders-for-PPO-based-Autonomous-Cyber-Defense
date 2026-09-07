# Canonical research snapshot before main cleanup

Archive branch: `archive/canonical-v1-before-main-2026-09-07`.

This snapshot preserves the working source, vendored CybORG changes, notes,
results and all existing run artifacts before the main-benchmark cleanup.
It includes canonical v1 experiments as well as the later v2 schema / v4
parser qualification, so no intervening research work is lost.

`runs/` was explicitly staged despite `.gitignore`: configs, histories,
evaluation JSON, checkpoints, logs and TensorBoard event files are preserved.
Virtual environments, package caches and Python bytecode are not part of this
snapshot.

Historical checkpoints can have older parser semantics even when their schema
is named v1 or v2. Use each run's `config.json` and protocol when interpreting
it; the source in this snapshot is the latest pre-cleanup source, not a claim
that every historical run used an identical parser. The current parser rejects
incompatible historical checkpoints intentionally.

Read canonical v1 in `src/acd_benchmark/cyber/canonical_observation.py` and the
corresponding `results/CANONICAL_V1_*` reports. The latest five-seed observation
qualification is documented in `results/OBSERVATION_LEARNING_QUALIFICATION_STATUS.md`.

The later user decision is a fixed-budget encoder benchmark (300k simulator
ticks per seed, seeds 42-46, stochastic primary evaluation), without a
convergence claim. Reports in this archive retain their original wording.
