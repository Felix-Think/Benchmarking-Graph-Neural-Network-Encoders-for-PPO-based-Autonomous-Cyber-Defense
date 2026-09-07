# Canonical V2 Observation Qualification

## Protocol

- Primary evaluation: stochastic policy sampling.
- Independent inference unit: training seed.
- Conditions are trained independently from initialization: `Full` and `No evidence`.
- PPO, native reward, official mask, architecture, parameter count and evaluation seeds are matched.
- A result is paper-level only with at least five training seeds.

## Held-Out Results

| Seed | Full | No evidence | Full - No evidence | Untrained Full | Full - untrained |
|---:|---:|---:|---:|---:|---:|
| 43 | -1109.90 | -1069.50 | -40.40 | -4725.03 | +3615.13 |
| 44 | -719.30 | -866.63 | +147.33 | -5135.63 | +4416.33 |
| 45 | -799.33 | -760.80 | -38.53 | -5057.70 | +4258.37 |
| 46 | -608.27 | -908.60 | +300.33 | -4879.67 | +4271.40 |

Multi-seed Full - No evidence: +92.18, 95% t-CI [-169.3347803378207, 353.70144700448736].

Multi-seed Full - untrained: +4140.31, 95% t-CI [3571.666699415215, 4708.949967251449].

## Qualification Gates

- Five training seeds: False.
- Controlled configuration and finite metrics: True.
- Trained Full beats untrained: False.
- Full training return improves from first to last 20%: False.
- Trained Full beats trained No evidence: False.
- Late validation curve satisfies the predeclared plateau diagnostic: False.
- Ready for encoder benchmark: False.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
