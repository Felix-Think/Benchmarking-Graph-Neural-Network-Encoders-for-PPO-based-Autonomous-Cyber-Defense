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
| 42 | -770.70 | -990.07 | +219.37 | -4914.70 | +4144.00 |
| 43 | -1109.90 | -1069.50 | -40.40 | -4725.03 | +3615.13 |
| 44 | -719.30 | -866.63 | +147.33 | -5135.63 | +4416.33 |
| 45 | -799.33 | -760.80 | -38.53 | -5057.70 | +4258.37 |
| 46 | -608.27 | -908.60 | +300.33 | -4879.67 | +4271.40 |

Multi-seed Full - No evidence: +117.62, 95% t-CI [-72.69676822110418, 307.9367682211041].

Multi-seed Full - untrained: +4141.05, 95% t-CI [3756.766062438286, 4525.327270895048].

## Qualification Gates

- Five training seeds: True.
- Controlled configuration and finite metrics: True.
- Trained Full beats untrained: True.
- Full training return improves from first to last 20%: True.
- Trained Full beats trained No evidence: False.
- Late validation curve satisfies the predeclared plateau diagnostic: False.
- Ready for encoder benchmark: False.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
