# Canonical Flat Learning Qualification - Seed 42

## Protocol

- Observation: action-aware HostCompressed canonical representation, 1,489 values.
- Policy: shared Flat PPO for all five Blue agents.
- Training seed: 42.
- Requested budget: 84,000 simulator steps.
- Actual budget: 84,189 simulator steps, 211 complete episodes/updates.
- Episode length: 400 configured steps, 399 executed simulator ticks.
- PPO: learning rate 3e-4, reward scale 0.01, entropy coefficient 0.001,
  target KL 0.015, four update epochs and minibatch size 64.
- Model selection: deterministic evaluation every 10 updates on 10 fixed
  episodes with seeds 20,000-20,009.
- Held-out evaluation: 100 episodes with seeds 10,000-10,099.
- Primary metric: deterministic cumulative team return. Higher is better.
- Secondary metric: stochastic cumulative team return.

The best checkpoint was selected only on the periodic seed set. Held-out results
were not used to choose a checkpoint.

## Held-Out Results

| Policy/condition | Deterministic mean | Deterministic SD | Stochastic mean | Stochastic SD |
|---|---:|---:|---:|---:|
| Canonical untrained | -3686.11 | 996.03 | not measured at 100 episodes | - |
| Canonical best, static-only gate | -3542.40 | 751.80 | not measured | - |
| Canonical final checkpoint | -3637.00 | 780.90 | -1150.47 | 431.38 |
| Canonical best checkpoint | -3183.12 | 787.68 | -1348.71 | 526.22 |
| Prior Flat best checkpoint | -3208.70 | 919.75 | -1357.92 | 569.88 |

## Paired Evidence

All comparisons below use the same 100 held-out episode seeds. Confidence
intervals are paired bootstrap 95% intervals with 50,000 resamples.

| Comparison | Mean delta | 95% CI | Paired dz | Wins/Ties/Losses | Paired t p |
|---|---:|---:|---:|---:|---:|
| Canonical best - untrained | +502.99 | [293.78, 715.92] | 0.466 | 66/0/34 | 1.01e-5 |
| Canonical best - static-only | +359.28 | [249.84, 475.66] | 0.621 | 65/4/31 | 1.23e-8 |
| Static-only - untrained | +143.71 | [-45.31, 333.82] | 0.147 | 55/0/45 | 0.144 |
| Canonical final - untrained | +49.11 | [-126.85, 222.89] | 0.055 | 55/0/45 | 0.585 |
| Canonical best - Flat best | +25.58 | [-177.22, 230.09] | 0.024 | 55/0/45 | 0.807 |

For stochastic evaluation, canonical best minus Flat best was +9.21 with a
95% paired bootstrap interval of [-123.97, 144.39] and paired t p=0.894.

## Learning Curve Finding

- Untrained deterministic return on the 10 periodic seeds: -3842.70.
- First periodic checkpoint at 3,990 steps: -4413.30.
- Best periodic checkpoint at update 80 / 31,920 steps: -3208.50.
- Last periodic checkpoint at update 210 / 83,790 steps: -3771.80.

The policy therefore learned a useful canonical-observation policy, but the
gain was not stable through the full budget. The final checkpoint was not
significantly better than the untrained policy on held-out seeds.

## Decision

**PASS for the seed-42 learning qualification:** PPO can learn a policy from the
canonical representation, and removing dynamic evidence causes a statistically
clear performance reduction. This rules out the narrow explanation that the
best result comes only from static topology, action masks or action priors.

**NOT PASS for encoder superiority:** canonical Flat and prior official Flat are
statistically indistinguishable on both deterministic and stochastic held-out
evaluation. No GNN claim can be made from this run.

**NOT YET paper-level evidence:** only one training seed was qualified. The
canonical Flat model also has 271,603 parameters versus 107,891 for the prior
Flat model. Training stability, multiple training seeds and parameter-budget
handling remain required before the main benchmark.
