# Matched Flat versus GCN on Random-Pivot CAGE 4

## Protocol

- Training seed: 42.
- Training distribution: `random_host`.
- Requested simulator budget: 100,000 ticks per encoder.
- Actual simulator budget: 100,149 ticks and 251 complete episodes per encoder.
- Held-out evaluation: 30 episodes, environment seeds 10000-10029.
- Stochastic action seed: 10000.
- Primary evaluation: stochastic sampling from the learned PPO policy.
- Canonical observation: action-aware V1, 1651 dimensions.
- Action catalog and masks: padded 242-action catalog with official masks.
- Flat parameters: 292,339.
- GCN parameters: 291,735.
- Relative parameter-count difference: 0.21%.

The comparison script verified equality of the PPO, environment, canonical,
training-budget and evaluation control fields before calculating statistics.

## Held-Out Returns

| Encoder/condition | Mean | Standard deviation |
|---|---:|---:|
| Flat full | -762.83 | 323.20 |
| Flat static-only | -822.77 | 266.86 |
| GCN full | -1034.33 | 397.51 |
| GCN static-only | -1004.33 | 423.25 |

## Paired Comparisons

| Comparison | Mean paired delta | Bootstrap 95% CI | Wins/ties/losses |
|---|---:|---:|---:|
| GCN full minus Flat full | -271.50 | [-458.17, -88.56] | 9/0/21 |
| GCN static minus Flat static | -181.57 | [-385.90, 8.23] | 13/0/17 |
| GCN-minus-Flat dynamic-use interaction | -89.93 | [-297.90, 123.77] | 13/0/17 |

The paired t-test for GCN-full minus Flat-full gives p=0.0083 and the Wilcoxon
test gives p=0.0099. These tests quantify environment-seed variation for the
two fixed trained policies. They do not account for PPO training-seed variance.

## Observation-Use Gates

- Flat full minus static-only: +59.93, bootstrap 95% CI [-61.80, 177.77],
  17/0/13 wins/ties/losses.
- GCN full minus static-only: -30.00, bootstrap 95% CI [-194.57, 135.67],
  14/0/16 wins/ties/losses.

Neither encoder passes the strict observation-use gate. Randomizing pivot hosts
removes the fixed `server_host_0` correlation, but it does not make dynamic
process, connection or analysis evidence necessary for good reward.

## Action-Type Diagnostic

| Action type | Flat full | GCN full |
|---|---:|---:|
| Restore | 59.03% | 44.48% |
| Remove | 12.33% | 15.81% |
| Analyse | 9.12% | 9.81% |
| AllowTrafficZone | 10.64% | 24.13% |
| DeployDecoy | 6.77% | 4.05% |

The action-type total-variation distance between full Flat and full GCN is
0.1766. Flat still relies heavily on broad Restore behaviour. GCN restores less
but allocates substantially more decisions to AllowTrafficZone; this difference
does not translate into better return.

## Scientific Decision

1. The random-pivot distribution succeeds as a structural intervention: a
   fixed pivot slot predicts only about 11% of held-out pivots.
2. It fails as a sufficient observation-use intervention: both 100k policies
   remain competitive when dynamic canonical features are removed.
3. For training seed 42, Flat performs better than GCN under the matched
   protocol. This is a pilot result, not an encoder-superiority claim.
4. No claim about average encoder performance is valid until multiple training
   seeds quantify optimization variance.
5. Before adding Red profiles, denser networks or reward shaping, inspect why
   blanket Restore/traffic policies score well and whether canonical events are
   timely and action-relevant. Otherwise additional environment diversity will
   bundle multiple confounds without solving the observation-use problem.

Machine-readable statistics are stored in
`results/matched_random_pivot_100k_comparison.json`.
