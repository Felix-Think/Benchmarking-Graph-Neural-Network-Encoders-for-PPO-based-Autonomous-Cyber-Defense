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
| 42 | -978.73 | -955.60 | -23.13 | -4974.57 | +3995.83 |

Multi-seed Full - No evidence: -23.13, 95% t-CI None.

Multi-seed Full - untrained: +3995.83, 95% t-CI None.

## Qualification Gates

- Five training seeds: False.
- Controlled configuration and finite metrics: True.
- Trained Full beats untrained: False.
- Full training return improves from first to last 20%: False.
- Trained Full beats trained No evidence: False.
- Late validation curve satisfies the predeclared plateau diagnostic: False.
- Ready for encoder benchmark: False.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
