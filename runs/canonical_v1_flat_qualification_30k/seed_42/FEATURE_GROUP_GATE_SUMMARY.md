# Canonical V1 Feature-Group Gate Summary

## Protocol

- Checkpoint: `checkpoint.pt` after the 30k simulator-step qualification run.
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
| `full` | -834.53 | 271.06 | 0.0000 |
| `static_only` | -854.00 | 355.09 | 0.0163 |
| `zero_all` | -3751.90 | 556.14 | 0.2676 |
| `zero_all_no_action_mask` | -3951.27 | 935.49 | 0.7088 |
| `mission_only` | -3646.87 | 652.76 | 0.2671 |
| `node_identity_only` | -1176.73 | 374.22 | 0.0899 |
| `structure_only` | -2734.50 | 541.24 | 0.2100 |
| `topology_only` | -937.07 | 325.75 | 0.0512 |
| `topology_no_adjacency` | -997.77 | 406.85 | 0.0797 |
| `topology_mission` | -952.93 | 302.98 | 0.0537 |
| `full_no_action_mask` | -1322.43 | 487.36 | 0.6853 |

## Incremental Paired Comparisons

Positive deltas favor the feature group named before `given`.

| Comparison | Mean delta [paired bootstrap 95% CI] | W/T/L |
|---|---:|---:|
| `action_mask_given_zero_observation` | 199.37 [-135.44, 533.30] | 17/0/13 |
| `mission_given_zero_observation` | 105.03 [-106.04, 329.80] | 17/0/13 |
| `node_identity_given_zero_observation` | 2575.17 [2373.73, 2781.57] | 30/0/0 |
| `structure_given_zero_observation` | 1017.40 [728.30, 1307.10] | 27/0/3 |
| `topology_static_given_zero_observation` | 2814.83 [2615.86, 3016.10] | 30/0/0 |
| `adjacency_given_identity_and_membership` | 60.70 [-92.17, 218.17] | 17/0/13 |
| `mission_given_topology_static` | -15.87 [-169.70, 147.00] | 13/0/17 |
| `policy_firewall_globals_given_topology_mission` | 98.93 [-52.81, 234.57] | 19/1/10 |
| `dynamic_features_given_static_context` | 19.47 [-137.20, 170.53] | 17/0/13 |
| `action_mask_given_full_observation` | 487.90 [277.97, 700.77] | 23/0/7 |

## Interpretation Boundary

These are evaluation-time interventions on one trained seed. They identify
feature groups that are sufficient or behaviorally influential for this
checkpoint, but they do not establish unique causal attribution because the
groups can interact nonlinearly. Replication across training seeds remains
necessary before making a paper-level claim.
