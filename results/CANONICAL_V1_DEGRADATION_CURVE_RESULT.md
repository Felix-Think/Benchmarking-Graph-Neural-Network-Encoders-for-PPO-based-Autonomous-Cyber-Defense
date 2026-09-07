# Canonical v1 IID Dynamic-Host Degradation Curve

## Scope

Zero-shot stochastic evaluation of final 100k checkpoints on held-out episode seeds 10000-10029. The intervention retains dynamic host-node features with an exact subnet-stratified budget. Static node identity, topology, mission/global features and the official action mask remain unchanged.

A new visible host subset is sampled at every simulator tick.

This is an output-level observation-retention stress test. It is not a claim that CAGE 4 itself implements this sensor-loss process.

## Return Curve

| Encoder | q=1.00 | q=0.75 | q=0.50 | q=0.25 |
|---|---:|---:|---:|---:|
| Flat | -865.75 +/- 143.44 | -890.27 +/- 138.56 | -808.89 +/- 89.05 | -883.69 +/- 139.74 |
| GCN | -866.82 +/- 199.08 | -891.81 +/- 189.62 | -867.25 +/- 147.22 | -880.25 +/- 161.74 |
| GraphSAGE | -948.35 +/- 207.23 | -987.74 +/- 146.92 | -971.34 +/- 183.07 | -987.39 +/- 205.71 |

## Robustness

`Degradation = return(q=1) - return(q)`; lower is better. The degradation AUC averages that loss across q in [0.25, 1.00].

| Encoder | q=.25 degradation | Relative degradation | Degradation AUC | Worst-case return |
|---|---:|---:|---:|---:|
| Flat | 17.94 +/- 37.30 | 2.26% | -7.79 +/- 43.06 | -925.50 +/- 131.83 |
| GCN | 13.43 +/- 66.97 | 2.78% | 10.71 +/- 63.51 | -952.63 +/- 172.96 |
| GraphSAGE | 39.04 +/- 81.10 | 4.82% | 27.30 +/- 49.93 | -1017.93 +/- 200.49 |

## Behavioural Response

Total-variation (TV) distance compares the aggregate action-type distribution at each q with q=1. Larger values mean a larger policy response, but are not by themselves evidence of better decisions.

| Encoder | TV q=.75 | TV q=.50 | TV q=.25 |
|---|---:|---:|---:|
| Flat | 0.008 +/- 0.002 | 0.015 +/- 0.003 | 0.021 +/- 0.007 |
| GCN | 0.015 +/- 0.008 | 0.028 +/- 0.013 | 0.041 +/- 0.019 |
| GraphSAGE | 0.015 +/- 0.003 | 0.023 +/- 0.004 | 0.034 +/- 0.007 |

## Scientific Interpretation

- All three q=.25 degradation 95% confidence intervals include zero.
- No paired encoder return contrast has a 95% confidence interval excluding zero at any q.
- Total monotonicity violations across all encoders: 21.
- Robustness does not imply useful observation use or GNN superiority; interpret the curve together with raw return and action response.
- IID resampling allows hidden hosts and temporal memory to reappear; persistent masking is required for a stronger test.

## Interpretation Guardrails

- The training seed is the inferential unit (`n=5`), not the 30 evaluation episodes.
- Raw return and degradation must be interpreted together; robustness alone does not establish a better defender.
- A non-monotonic point can arise from stochastic trajectories and should not be hidden or smoothed away.
- Action-distribution TV is descriptive because no independent q=1 repeat was run to estimate ordinary stochastic rollout divergence.
- Semantic action masking and whole-subnet loss are separate follow-up ablations.

Protocol audit passed: `True`.
