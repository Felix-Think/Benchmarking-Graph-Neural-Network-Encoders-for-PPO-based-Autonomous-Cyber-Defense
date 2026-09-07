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
| 42 | -497.77 | -1182.37 | +684.60 | -5170.10 | +4672.33 |
| 43 | -224.90 | -1369.27 | +1144.37 | -4671.33 | +4446.43 |
| 44 | -434.23 | -1495.50 | +1061.27 | -5127.87 | +4693.63 |
| 45 | -352.07 | -958.23 | +606.17 | -5175.47 | +4823.40 |
| 46 | -780.23 | -1597.70 | +817.47 | -5353.53 | +4573.30 |

Multi-seed Full - No evidence: +862.77, 95% t-CI [572.6781164257047, 1152.868550240962].

Multi-seed Full - untrained: +4641.82, 95% t-CI [4466.818128516046, 4816.821871483953].

## Qualification Gates

- Five training seeds: True.
- Controlled configuration and finite metrics: True.
- Trained Full beats untrained: True.
- Full training return improves from first to last 20%: True.
- Trained Full beats trained No evidence: True.
- Late validation curve satisfies the predeclared plateau diagnostic: False.
- Ready for encoder benchmark: False.

The plateau gate is a diagnostic, not a mathematical proof of convergence. It requires the multi-seed late-slope confidence interval to include zero and the mean absolute slope to remain below the declared threshold.
