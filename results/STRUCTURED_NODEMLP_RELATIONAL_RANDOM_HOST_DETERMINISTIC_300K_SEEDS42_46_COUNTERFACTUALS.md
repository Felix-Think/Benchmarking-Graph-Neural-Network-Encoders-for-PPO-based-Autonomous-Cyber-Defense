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
| 42 | `zero_evidence` | -497.77 | -1799.53 | +1301.77 |
| 42 | `shuffle_hosts` | -497.77 | -929.47 | +431.70 |
| 43 | `zero_evidence` | -224.90 | -5852.73 | +5627.83 |
| 43 | `shuffle_hosts` | -224.90 | -662.00 | +437.10 |
| 44 | `zero_evidence` | -434.23 | -4552.43 | +4118.20 |
| 44 | `shuffle_hosts` | -434.23 | -1574.20 | +1139.97 |
| 45 | `zero_evidence` | -352.07 | -2343.20 | +1991.13 |
| 45 | `shuffle_hosts` | -352.07 | -876.47 | +524.40 |
| 46 | `zero_evidence` | -780.23 | -2745.03 | +1964.80 |
| 46 | `shuffle_hosts` | -780.23 | -1447.27 | +667.03 |

## Across Seeds

| Effect | Mean | 95% Student-t CI | Positive / zero / negative |
|---|---:|---|---|
| `full_minus_zero_evidence` | +3000.75 | [751.1986116634926, 5250.29472166984] | [5, 0, 0] |
| `full_minus_shuffle_hosts` | +640.04 | [273.4766235618826, 1006.6033764381173] | [5, 0, 0] |

Positive effects mean that preserving the correct dynamic evidence improved
return. Episode-level differences are descriptive; the confidence interval is
computed across independently trained policies.
