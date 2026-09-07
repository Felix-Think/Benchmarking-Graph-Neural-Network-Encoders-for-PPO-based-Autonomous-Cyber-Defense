# Canonical V1 Feature-Group Gate Summary

## Protocol

- Policy: `gcn`.
- Checkpoint: `checkpoint.pt` after
  100000 requested simulator steps.
- Evaluation: stochastic policy sampling, 30 paired held-out episodes.
- Environment seeds: 10000-10029; action-sampling seed: 10000.
- Uncertainty: 20,000 paired bootstrap resamples.
- `topology_only` is not pure adjacency. It retains structural adjacency, host
  membership, and static node identity/type/existence/control features.
- No-action-mask gates replace the mask with all-valid entries. Padded action
  indices can therefore be converted to `Sleep` by the fixed-action wrapper.

## Returns

| Gate | Mean return | Std | Action-type TV from full |
|---|---:|---:|---:|
| `full` | -867.60 | 317.07 | 0.0000 |
| `static_only` | -864.90 | 302.96 | 0.0683 |
| `zero_all` | -3674.67 | 744.18 | 0.2970 |
| `zero_all_no_action_mask` | -3915.27 | 782.62 | 0.7226 |
| `mission_only` | -3478.00 | 606.96 | 0.2843 |
| `node_identity_only` | -1110.87 | 414.51 | 0.1329 |
| `structure_only` | -3516.00 | 627.55 | 0.2952 |
| `topology_only` | -1256.40 | 407.77 | 0.1530 |
| `topology_no_adjacency` | -1408.40 | 519.87 | 0.1510 |
| `topology_mission` | -1255.67 | 420.31 | 0.1506 |
| `full_no_action_mask` | -1469.93 | 509.58 | 0.7764 |

## Incremental Paired Comparisons

Positive deltas favor the feature group named before `given`.

| Comparison | Mean delta [paired bootstrap 95% CI] | W/T/L |
|---|---:|---:|
| `action_mask_given_zero_observation` | 240.60 [-73.87, 567.00] | 17/0/13 |
| `mission_given_zero_observation` | 196.67 [-70.53, 474.20] | 17/0/13 |
| `node_identity_given_zero_observation` | 2563.80 [2204.06, 2943.13] | 30/0/0 |
| `structure_given_zero_observation` | 158.67 [-68.27, 391.60] | 17/0/13 |
| `topology_static_given_zero_observation` | 2418.27 [2151.66, 2697.00] | 30/0/0 |
| `adjacency_given_identity_and_membership` | 152.00 [-73.23, 382.33] | 18/0/12 |
| `mission_given_topology_static` | 0.73 [-209.47, 210.60] | 13/0/17 |
| `policy_firewall_globals_given_topology_mission` | 390.77 [190.63, 590.67] | 23/0/7 |
| `dynamic_features_given_static_context` | -2.70 [-130.47, 127.17] | 17/0/13 |
| `action_mask_given_full_observation` | 602.33 [412.23, 795.93] | 26/0/4 |

## Interpretation Boundary

These are evaluation-time interventions on one trained seed. They identify
feature groups that are sufficient or behaviorally influential for this
checkpoint, but they do not establish unique causal attribution because the
groups can interact nonlinearly. Replication across training seeds remains
necessary before making a paper-level claim.
