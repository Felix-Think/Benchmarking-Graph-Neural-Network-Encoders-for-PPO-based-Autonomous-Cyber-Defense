# New CAGE-2 diagnostic, 2026-11-13

This extension is pinned to the GitHub `main` commit
`291977eec20c0f223b9e507ad41482bd0db268c7`. It does not change the frozen
simulator, Flat policy, rollout collector, GAE or PPO optimizer. The separate
`main-benchmark` worktree and its existing changes are untouched.

## Scope

- Three fresh runs: Flat, GCN, GraphSAGE; paired training seed 0.
- Fixed 100,000 native environment steps per run; horizon 200.
- Native `BlueTableEnv` input, 13 hosts by four features. This wrapper already
  maintains its own observed compromise bookkeeping.
- The existing 41-action subset (Sleep, Monitor, Analyse/Restore/Remove on each
  of 13 hosts), not the full 158-action pilot catalogue. No action mask.
- No additional temporal-belief module, synthetic masking, q mixture or
  recovered historical checkpoint. This is not an exact replication of the
  old 16-by-31, 300k-step temporal-belief experiment.
- Unmodified main PPO settings, one CPU Torch/BLAS thread, sequential models.
- Greedy and stochastic evaluation: 100 episodes each, environment seeds
  10000--10099, per-episode action seeds 60000--60099.
- Six predeclared controls: Sleep, Monitor, uniform random action,
  Restore(Op_Server0), round-robin Restore, Analyse(User0). No best-action
  search and no policy/checkpoint selection based on evaluation scores.

## GNN adapters

Both new GNNs use two PyG convolution layers of width 8, ReLU, ordered host
readout and a projection into the same latent width 64 and original actor and
critic heads. Shared-head initial parameters are identical under a paired seed.
Whole-policy counts: Flat 29,610; GCN 28,890; GraphSAGE 28,986. These are close
but not identical and are not a new matched main benchmark.

The static 16-node graph is built only from public Scenario2.yaml: the 13 host
slots plus three router nodes with zero features. Its 32 directed links match
the native host-router/router-router wiring. Runtime Red state and dynamic
firewall state do not alter these edges or enter policy input.

## Evidence

Outputs include a source hash, package versions, full configuration, per-update
training history, final checkpoints, per-episode returns/action counts and
saved observation-probe inputs/logits. Distribution sensitivity probes zero
activity, compromise, or all features, or rotate host rows on the same state.
They measure input sensitivity only, not useful information use or physical
sensor failures. Argmax invariance and probability-distribution invariance are
reported separately.

There is only one training seed. Evaluation-episode standard deviations must
not be presented as training-seed uncertainty or used for encoder rankings.
Failure or success to concentrate on Restore is an outcome, not a tuning
criterion. The old historical result cannot be relabelled as recovered data.

## Commands

The published evidence and portable verifier are documented in
`evidence/cage2_revalidation_2026-11-13/README.md`. That directory contains the
three checkpoints, all evaluation records, probe arrays, and checksum metadata.
The original training entry point below deliberately requires checkout of the
pinned base commit. On the evidence branch, use the documented isolated-worktree
procedure to rerun training; do not bypass the source guard.

The vendored import path must be explicit with modern editable installation:

```sh
PYTHONPATH=CybORG:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests experiments/cage2_revalidation/test_protocol.py -q

PYTHONPATH=CybORG:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m experiments.cage2_revalidation.run \
  --run-root runs/cage2_three_encoder_100k_seed0_2026-11-13
```

Existing output directories are never overwritten. Original tests initially
showed two import-resolution failures with implicit editable paths; all 20
passed with the explicit vendored import path and no simulator source edits.
