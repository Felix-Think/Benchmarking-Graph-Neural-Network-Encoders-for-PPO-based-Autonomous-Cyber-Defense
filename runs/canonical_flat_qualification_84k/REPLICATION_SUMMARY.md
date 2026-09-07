# Canonical observation replication summary

## Decision

**MIXED** under the protocol frozen before seeds 43 and 44.

| Training seed | Untrained | Static-only | Best canonical | Best - untrained | Learning gate | Best - static | Observation-use gate |
|---:|---:|---:|---:|---:|:---:|---:|:---:|
| 42 | -3686.11 | -3542.40 | -3183.12 | +502.99 | PASS | +359.28 | PASS |
| 43 | -3373.13 | -3393.55 | -3393.34 | -20.21 | FAIL | +0.21 | FAIL |
| 44 | -3704.22 | -3402.65 | -3122.30 | +581.92 | PASS | +280.35 | PASS |

## Seed-level aggregation

- Learning: 2/3 positive point estimates and 2/3 CI-gate passes; mean seed delta +354.90, range [-20.21, 581.92].
- Dynamic observation use: 3/3 positive point estimates and 2/3 CI-gate passes; mean seed delta +213.28, range [0.21, 359.28].
- Episodes were paired within each training seed. The 300 held-out episodes were not pooled and treated as independent replication units.

## Claim boundary

The action-aware canonical representation is learnable and its dynamic evidence is useful in seeds 42 and 44. Seed 43 fails both gates. Therefore the result supports feasibility, but not stable multi-seed qualification. It is not yet sufficient to start a definitive Flat-versus-GNN benchmark without addressing or explicitly accepting this optimization variance.
