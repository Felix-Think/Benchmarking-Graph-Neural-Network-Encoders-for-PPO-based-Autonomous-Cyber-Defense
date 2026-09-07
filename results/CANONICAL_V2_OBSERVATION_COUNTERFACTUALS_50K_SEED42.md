# Canonical Observation Counterfactuals

## Protocol

- Evaluation mode: `stochastic`.
- The trained `Full` checkpoint is held fixed.
- Environment seeds and stochastic action seeds are paired across conditions.
- Observation dimension, PPO policy, native reward and official action mask are unchanged.
- Training seed is the independent inference unit; episode-level differences are descriptive.

## Held-Out Results

| Training seed | Condition | Mean return | Full - condition |
|---:|---|---:|---:|
| 42 | `full` | -1281.87 | +0.00 |
| 42 | `no_evidence` | -1235.80 | -46.07 |
| 42 | `action_state_only` | -4030.57 | +2748.70 |
| 42 | `topology_only` | -1399.20 | +117.33 |
| 42 | `zero_with_mask` | -4724.57 | +3442.70 |

## Across Training Seeds

| Effect | Mean | 95% Student-t CI | Seeds |
|---|---:|---|---:|
| `full_minus_no_evidence` | -46.07 | None | 1 |
| `full_minus_action_state_only` | +2748.70 | None | 1 |
| `full_minus_topology_only` | +117.33 | None | 1 |
| `full_minus_zero_with_mask` | +3442.70 | None | 1 |

Positive `Full - condition` means the information removed by that condition was
useful to the fixed trained policy. A single-seed result is diagnostic only and
cannot establish a paper-level claim.
