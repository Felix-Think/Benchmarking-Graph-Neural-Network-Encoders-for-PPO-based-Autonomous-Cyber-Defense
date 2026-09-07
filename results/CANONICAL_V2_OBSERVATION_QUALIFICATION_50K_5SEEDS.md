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
| 42 | -1281.87 | -1155.53 | -126.33 | -4904.67 | +3622.80 |
| 43 | -1266.53 | -1292.97 | +26.43 | -4803.67 | +3537.13 |
| 44 | -1166.53 | -1306.87 | +140.33 | -5187.10 | +4020.57 |
| 45 | -1151.27 | -1161.30 | +10.03 | -5236.90 | +4085.63 |
| 46 | -1407.93 | -1145.37 | -262.57 | -4812.73 | +3404.80 |

Multi-seed Full - No evidence: -42.42, 95% t-CI [-235.17500575618342, 150.33500575618345].

Multi-seed Full - untrained: +3734.19, 95% t-CI [3358.975587590703, 4109.397745742631].

## Qualification Gates

- Five training seeds: True.
- Controlled configuration and finite metrics: True.
- Trained Full beats untrained: True.
- Full training return improves from first to last 20%: True.
- Trained Full beats trained No evidence: False.
- Late validation curve satisfies the predeclared plateau diagnostic: False.
- Ready for encoder benchmark: False.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
