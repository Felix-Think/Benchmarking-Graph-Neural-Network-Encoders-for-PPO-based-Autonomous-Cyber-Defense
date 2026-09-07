# Canonical v1 Persistent Dynamic-Host Degradation Curve

## Scope

Zero-shot stochastic evaluation of final 100k checkpoints on held-out episode seeds 10000-10029. The intervention retains dynamic host-node features with an exact subnet-stratified budget. Static node identity, topology, mission/global features and the official action mask remain unchanged.

The visible host subset is sampled once at episode reset and remains fixed for all simulator ticks.

This is an output-level observation-retention stress test. It is not a claim that CAGE 4 itself implements this sensor-loss process.

## Return Curve

| Encoder | q=1.00 | q=0.75 | q=0.50 | q=0.25 |
|---|---:|---:|---:|---:|
| Flat | -865.75 +/- 143.44 | -881.65 +/- 176.16 | -871.86 +/- 111.42 | -877.41 +/- 158.00 |
| GCN | -866.82 +/- 199.08 | -830.85 +/- 201.74 | -886.97 +/- 148.90 | -892.03 +/- 201.30 |
| GraphSAGE | -948.35 +/- 207.23 | -972.65 +/- 237.28 | -968.75 +/- 165.18 | -943.18 +/- 134.64 |

## Persistent Versus IID

Paired return difference is `persistent - IID`; positive means persistent masking produced a higher return.

| Encoder | q=.75 | q=.50 | q=.25 |
|---|---:|---:|---:|
| Flat | 8.62 [-121.03, 138.27] | -62.97 [-131.41, 5.47] | 6.28 [-82.24, 94.80] |
| GCN | 60.97 [-5.01, 126.94] | -19.72 [-108.16, 68.72] | -11.78 [-81.17, 57.61] |
| GraphSAGE | 15.09 [-117.17, 147.35] | 2.59 [-44.81, 49.99] | 44.21 [-105.57, 193.98] |

## Robustness

`Degradation = return(q=1) - return(q)`; lower is better. The degradation AUC averages that loss across q in [0.25, 1.00].

| Encoder | q=.25 degradation | Relative degradation | Degradation AUC | Worst-case return |
|---|---:|---:|---:|---:|
| Flat | 11.66 +/- 41.99 | 1.18% | 9.28 +/- 36.72 | -909.85 +/- 146.04 |
| GCN | 25.21 +/- 96.41 | 3.85% | -1.07 +/- 62.16 | -948.28 +/- 165.39 |
| GraphSAGE | -5.17 +/- 80.79 | 0.76% | 14.04 +/- 32.17 | -1007.65 +/- 209.79 |

## Behavioural Response

Total-variation (TV) distance compares the aggregate action-type distribution at each q with q=1. Larger values mean a larger policy response, but are not by themselves evidence of better decisions.

| Encoder | TV q=.75 | TV q=.50 | TV q=.25 |
|---|---:|---:|---:|
| Flat | 0.008 +/- 0.002 | 0.015 +/- 0.007 | 0.025 +/- 0.009 |
| GCN | 0.015 +/- 0.007 | 0.027 +/- 0.012 | 0.039 +/- 0.019 |
| GraphSAGE | 0.017 +/- 0.005 | 0.024 +/- 0.007 | 0.034 +/- 0.007 |

## Scientific Interpretation

- All three q=.25 degradation 95% confidence intervals include zero.
- No paired encoder return contrast has a 95% confidence interval excluding zero at any q.
- Total monotonicity violations across all encoders: 21.
- Robustness does not imply useful observation use or GNN superiority; interpret the curve together with raw return and action response.
- Persistent masking prevents hidden hosts from reappearing within the episode, so it is a stronger observation-loss test than IID masking.
- No persistent-versus-IID paired return contrast has a 95% confidence interval excluding zero.

## Interpretation Guardrails

- The training seed is the inferential unit (`n=5`), not the 30 evaluation episodes.
- Raw return and degradation must be interpreted together; robustness alone does not establish a better defender.
- A non-monotonic point can arise from stochastic trajectories and should not be hidden or smoothed away.
- Action-distribution TV is descriptive because no independent q=1 repeat was run to estimate ordinary stochastic rollout divergence.
- Semantic action masking and whole-subnet loss are separate follow-up ablations.

Protocol audit passed: `True`.
