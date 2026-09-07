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
| 42 | `shuffle_hosts` | -497.77 | -929.47 | +431.70 |
| 42 | `time_shift` | -497.77 | -513.70 | +15.93 |
| 42 | `zero_evidence` | -497.77 | -1799.53 | +1301.77 |

## Across Seeds

| Effect | Mean | 95% Student-t CI | Positive / zero / negative |
|---|---:|---|---|
| `full_minus_shuffle_hosts` | +431.70 | None | [1, 0, 0] |
| `full_minus_time_shift` | +15.93 | None | [1, 0, 0] |
| `full_minus_zero_evidence` | +1301.77 | None | [1, 0, 0] |

Positive effects mean that preserving the correct dynamic evidence improved
return. Episode-level differences are descriptive; the confidence interval is
computed across independently trained policies.
