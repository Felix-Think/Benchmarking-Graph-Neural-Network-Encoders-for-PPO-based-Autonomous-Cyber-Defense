# Dynamic Evidence Counterfactuals

## Protocol

- The trained `Full` checkpoint is held fixed within each training seed.
- Evaluation uses stochastic policy sampling.
- Environment and stochastic-action seeds are paired across transforms.
- Reward, official action mask, policy parameters and scenario distribution are unchanged.
- The training seed is the independent inference unit.

## Per-Seed Results

| Seed | Transform | Full | Counterfactual | Full - counterfactual |
|---:|---|---:|---:|---:|
| 42 | `shuffle_hosts` | -770.70 | -1084.97 | +314.27 |
| 42 | `time_shift` | -770.70 | -977.43 | +206.73 |
| 42 | `zero_evidence` | -770.70 | -1188.27 | +417.57 |
| 43 | `shuffle_hosts` | -1109.90 | -1096.63 | -13.27 |
| 43 | `time_shift` | -1109.90 | -1101.43 | -8.47 |
| 43 | `zero_evidence` | -1109.90 | -1289.67 | +179.77 |
| 44 | `shuffle_hosts` | -719.30 | -919.00 | +199.70 |
| 44 | `time_shift` | -719.30 | -754.40 | +35.10 |
| 44 | `zero_evidence` | -719.30 | -1438.97 | +719.67 |
| 45 | `shuffle_hosts` | -799.33 | -1004.40 | +205.07 |
| 45 | `time_shift` | -799.33 | -784.53 | -14.80 |
| 45 | `zero_evidence` | -799.33 | -1349.33 | +550.00 |
| 46 | `shuffle_hosts` | -608.27 | -811.77 | +203.50 |
| 46 | `time_shift` | -608.27 | -656.00 | +47.73 |
| 46 | `zero_evidence` | -608.27 | -1016.33 | +408.07 |

## Across Seeds

| Effect | Mean | 95% Student-t CI | Positive / zero / negative |
|---|---:|---|---|
| `full_minus_shuffle_hosts` | +181.85 | [33.72140576930394, 329.9852608973627] | [4, 0, 1] |
| `full_minus_time_shift` | +53.26 | [-58.41512930658486, 164.93512930658483] | [3, 0, 2] |
| `full_minus_zero_evidence` | +455.01 | [207.9012778205677, 702.125388846099] | [5, 0, 0] |

Positive effects mean that preserving the correct dynamic evidence improved
return. Episode-level differences are descriptive; the confidence interval is
computed across independently trained policies.
