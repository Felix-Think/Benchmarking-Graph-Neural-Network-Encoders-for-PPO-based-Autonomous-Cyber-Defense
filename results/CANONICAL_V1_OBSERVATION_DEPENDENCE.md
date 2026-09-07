# Canonical V1 Observation-Dependence Audit

## Protocol

- Primary evaluation uses stochastic policy sampling.
- Interventions share 30 held-out environment seeds per checkpoint.
- The independent inferential unit is the training seed (n=5).
- Positive effects below mean the left/full condition has higher return.

## Counterfactual Results

| Encoder | Full-zero evidence | Full-static only | Full-shuffle hosts | Full-lag 5 | Mask-only-no mask | Full-full no mask |
|---|---:|---:|---:|---:|---:|---:|
| F-O | 32.92 [-29.07, 94.91] | 66.93 [13.07, 120.78] | -66.49 [-172.00, 39.01] | -8.18 [-95.98, 79.62] | 387.21 [207.95, 566.47] | 339.58 [150.25, 528.91] |
| G-O | 78.64 [-44.45, 201.73] | 54.05 [-86.17, 194.27] | -29.78 [-138.69, 79.13] | -80.61 [-137.37, -23.84] | 449.45 [207.01, 691.89] | 455.61 [195.29, 715.94] |
| S-O | 68.36 [-2.96, 139.68] | 111.69 [-115.01, 338.38] | 32.35 [-214.58, 279.28] | 48.99 [-0.86, 98.84] | 440.21 [264.53, 615.89] | 714.74 [363.43, 1066.05] |

## Main Reading

- No encoder has a full-minus-zero-evidence interval strictly above zero across five training seeds.
- No encoder shows a reliable return loss when evidence is shuffled across hosts.
- GCN performs better under a five-tick evidence lag; this contradicts a claim that the current GCN requires correct evidence timing.
- Flat benefits from the complete dynamic block relative to static-only, but that broader gate also removes `decoy_active` and cannot attribute the gain to sensor evidence alone.
- Evidence memory is active at roughly 95-96% of decisions, indicating saturation rather than sparse discriminative belief.

## Evidence Availability and Response

| Encoder | Evidence/1k ticks | Current-evidence decisions | Memory-evidence decisions | Current alignment excess | Same-host response <=10 ticks | Full-vs-static TV |
|---|---:|---:|---:|---:|---:|---:|
| F-O | 882.37 | 0.212 | 0.961 | 0.075 | 0.228 | 0.092 |
| G-O | 767.92 | 0.173 | 0.955 | 0.048 | 0.199 | 0.158 |
| S-O | 1072.35 | 0.268 | 0.961 | 0.138 | 0.285 | 0.228 |

## Evidence Timing

| Encoder | First evidence tick | Median event tick | Event while decision-ready | First decision latency | Same-host action latency |
|---|---:|---:|---:|---:|---:|
| F-O | 2.87 | 178.70 | 0.451 | 0.80 | 27.10 |
| G-O | 2.53 | 181.20 | 0.411 | 1.00 | 28.60 |
| S-O | 2.53 | 182.60 | 0.488 | 0.60 | 22.20 |

## Action Response

Values are action-fraction changes when current evidence is present minus absent.

| Encoder | Analyse | Remove | Restore | BlockTrafficZone |
|---|---:|---:|---:|---:|
| F-O | +0.060 | -0.026 | -0.054 | +0.020 |
| G-O | +0.024 | -0.009 | -0.023 | +0.006 |
| S-O | +0.109 | -0.032 | -0.038 | +0.014 |

## Reward Association

The following five-tick reward contrasts are observational, not causal.

| Encoder | Target evidence - other host | Target evidence - no evidence |
|---|---:|---:|
| F-O | -0.61 | -6.06 |
| G-O | -2.53 | -6.88 |
| S-O | 0.42 | -6.20 |

## Interpretation Boundary

- The official action mask is fixed within an episode, so it does not directly expose changing compromise/evidence state.
- It does expose reset-time valid action slots. No-mask gates also enable padded/invalid actions that the wrapper can convert to no-ops, so mask effects are diagnostic rather than a pure information estimate.
- Host shuffling preserves the per-tick multiset of evidence values but breaks host alignment. Lag-5 uses simulator ticks, not PPO decision steps.
- `decoy_active` is preserved by the evidence interventions. It is removed only by the broader static-only gate.
- Reward-after-action measurements are associations under the attack trajectory, not causal action effects.
- A null intervention effect does not prove that information is useless; it shows that these trained checkpoints do not measurably require it under this protocol.
