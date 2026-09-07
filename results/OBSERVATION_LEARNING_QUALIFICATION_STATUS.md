# Observation Learning Qualification Status

## Decision

Canonical parser v4 passes the predeclared **multi-seed observation-learning
qualification**. Across seeds 42--46, PPO learns in every Full run, trained Full
beats matched No evidence in 5/5 seeds, removing evidence causes a large return
loss in 5/5 seeds, and shuffling evidence between hosts causes a material return
loss in 5/5 seeds. The seed-level confidence intervals for all primary evidence
effects have positive lower bounds.

The strict encoder benchmark gate remains blocked only by the predeclared
plateau diagnostic. Four of five Full policies still have an absolute late
validation slope above 50 return per 100k ticks. This does not invalidate the
observation: it means 300k ticks cannot yet support a claim that training has
converged. Short-lag temporal ordering also remains unqualified because the
seed-42 time-shift interval contains zero and that intervention is no longer a
primary gate.

Primary evaluation uses stochastic sampling from the learned policy. Training
seed is the independent inference unit; evaluation episodes only quantify the
uncertainty of a fixed checkpoint.

## Canonical V4 Boundary

- Input is the raw combined Blue observation, not simulator compromise state.
- Successful `DeployDecoy` process feedback is excluded from passive evidence.
- Reset inventory is excluded from passive evidence.
- Relational connection features retain direction, port class, visible remote
  address and locally known remote subnet.
- IP-to-subnet memory is private to each Blue agent and reset every episode.
- `Full` and `No evidence` keep identical dimensions, parameter count, PPO,
  native reward, official action mask, structured action catalog and seeds.
- Focused canonical/PPO tests: 53 passed; lint checks passed.

Two nominally identical long-run starts diverged before process-level seed
controls were logged. A dedicated 10k-tick replication with
`PYTHONHASHSEED=42`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` produced exact
agreement in 21 episode returns, all PPO history records and final policy weights
(`SHA-256 1026f4d12d42c8857ba7682b05cb702b9532f4de44137a881579d0fa22a7ba0e`).
All future qualification runs must use and record these controls; the two
interrupted 300k attempts are explicitly marked unusable.

## Official-Mask Seed-42 Screen

Both policies used `random_host`, canonical v4, passive TTL 5, 500-tick episodes,
50k simulator ticks and 30 held-out stochastic episodes.

| Condition | Mean return | Standard deviation |
|---|---:|---:|
| Full | -576.70 | 151.01 |
| No evidence, trained from scratch | -762.37 | 330.11 |
| Untrained Full | -5170.10 | 809.44 |

- PPO learning effect: `Full - untrained = +4593.40`.
- Evidence training effect: `Full - No evidence = +185.67`.
- Training last-20%-minus-first-20% return: `+3187.43`.
- Only one late periodic evaluation exists, so convergence is not established.

Counterfactuals used the same Full checkpoint, environment seeds and stochastic
action seed:

| Intervention | Mean return | Full minus intervention | Paired 95% t-CI |
|---|---:|---:|---:|
| Zero evidence | -678.17 | +101.47 | [-4.92, 207.86] |
| Shuffle evidence between hosts | -623.13 | +46.43 | [-34.57, 127.44] |
| Shift evidence by 6 ticks | -684.93 | +108.23 | [11.79, 204.67] |

All effects point in the expected direction, but only the time-shift interval is
strictly above zero in this single-checkpoint diagnostic.

## Deterministic-Runtime 300k Qualification

Both matched conditions used the official mask, `random_host`, canonical v4,
passive TTL 5, 500-tick episodes, 300k simulator ticks, validation every 25 PPO
updates and the recorded single-thread runtime controls. Final evaluation used
30 held-out stochastic episodes with the same environment and action-sampling
seeds.

| Condition | Mean return | Standard deviation |
|---|---:|---:|
| Full | -497.77 | 217.32 |
| No evidence, trained from scratch | -1182.37 | 394.00 |
| Untrained Full | -5170.10 | 809.44 |

- Evidence training effect: `Full - No evidence = +684.60`.
- Descriptive paired episode interval: `[+511.07, +858.13]`; Full wins 29/30
  paired episodes. Evaluation episodes are not independent training replicates,
  so this interval cannot replace a seed-level interval.
- PPO learning effect: `Full - untrained = +4672.33`.
- Training last-20%-minus-first-20% return: `+995.90`.
- Full late periodic slope: `+43.89 return/100k ticks`, within the predeclared
  one-seed plateau threshold of absolute slope at most 50.

The Full periodic means at updates 25/50/75/100/125/150 were respectively
`-598.9`, `-1519.2`, `-440.6`, `-407.7`, `-499.1` and `-363.9`. The update-50
collapse and later recovery show why checkpoint selection and convergence must
not be inferred from a single evaluation point. The late curve is compatible
with a one-seed plateau, not proof of convergence across seeds.

Counterfactuals held the final Full checkpoint, environment seeds, official
action mask and stochastic action seed fixed:

| Intervention | Mean return | Full minus intervention | Paired 95% t-CI | Positive episodes |
|---|---:|---:|---:|---:|
| Zero evidence | -1799.53 | +1301.77 | [1074.97, 1528.56] | 30/30 |
| Shuffle evidence between hosts | -929.47 | +431.70 | [297.72, 565.68] | 27/30 |
| Shift evidence by 6 ticks | -513.70 | +15.93 | [-115.19, 147.06] | 14/30 |

Thus this checkpoint depends strongly on the presence and host placement of
evidence. It does not show reliable dependence on evidence timing at a six-tick
lag. These are checkpoint-level diagnostics; the training seed remains the
independent inference unit.

## Five-Seed Qualification

Seeds 42--46 used identical canonical schema, PPO, parameter count, reward,
official mask, `random_host` distribution, simulator budget and evaluation
seeds. Only the training seed and the intended Full/No-evidence condition
differed. All runs used the recorded single-thread deterministic-runtime
controls.

| Seed | Full | No evidence | Full - No evidence | Late Full slope / 100k |
|---:|---:|---:|---:|---:|
| 42 | -497.77 | -1182.37 | +684.60 | +43.89 |
| 43 | -224.90 | -1369.27 | +1144.37 | +142.38 |
| 44 | -434.23 | -1495.50 | +1061.27 | +752.20 |
| 45 | -352.07 | -958.23 | +606.17 | +189.68 |
| 46 | -780.23 | -1597.70 | +817.47 | +118.44 |

- Mean Full return across seeds: `-457.84 +/- 207.09`.
- Mean No-evidence return across seeds: `-1320.61 +/- 255.03`.
- Mean matched evidence effect: `+862.77 +/- 233.63`.
- Seed-level 95% t-interval for Full minus No evidence:
  `[+572.68, +1152.87]`; 5/5 effects are positive.
- Mean Full minus untrained effect: `+4641.82`; seed-level 95% t-interval
  `[+4466.82, +4816.82]`; 5/5 effects are positive.
- Mean training last-20%-minus-first-20% improvement: `+1236.09`; seed-level
  95% t-interval `[+945.29, +1526.88]`; 5/5 effects are positive.

The observation-utility claim is therefore qualified at the training-seed
level. Absolute Full performance still varies materially, especially for seed
46, and the periodic trajectories contain collapses and recoveries. This is a
PPO stability/convergence issue, not evidence that the observation is ignored.

Primary counterfactuals used each final Full checkpoint with paired environment
and stochastic-action seeds:

| Effect across five seeds | Mean | Seed-level 95% t-CI | Positive seeds |
|---|---:|---:|---:|
| Full minus zero evidence | +3000.75 | [+751.20, +5250.29] | 5/5 |
| Full minus shuffled-host evidence | +640.04 | [+273.48, +1006.60] | 5/5 |

The zero-evidence intervention is an out-of-distribution perturbation of a
trained policy, so its magnitude must not be interpreted as the return that a
policy would learn from scratch without evidence. The host-shuffle result is the
cleaner evidence that the policy uses which host generated the observation,
rather than only the total amount of suspicious activity.

## Operational Audit

The original 50k audit showed weak alignment: 10.22% current-evidence targeting
versus 9.12% chance, 14.13% memory-evidence targeting versus 12.43% chance, and
17.97% same-host response within 10 ticks.

Across five held-out episodes for the 300k final Full checkpoint (5,333
decisions and 3,543 evidence occurrences):

- current-evidence host targeting: 22.02% versus 8.65% uniform chance;
- memory-evidence host targeting: 38.87% versus 28.69% uniform chance;
- median first same-host defensive response: 14 ticks;
- same-host response within 10 ticks: 39.91%;
- Full-versus-static policy total variation: 0.3379, with argmax changing on
  68.80% of decisions;
- zeroing relational-now features changes argmax on 31.82% of decisions where
  those features are present;
- zeroing relational-memory features changes argmax on 30.70% of decisions
  where those features are present;
- `AllowTrafficZone` falls to 6.34% of decisions, although 95.10% of completed
  traffic actions still return `FALSE`.

The long-budget policy therefore uses dynamic and relational evidence in an
operationally measurable, host-conditioned way. This is substantially stronger
than mere neural-network input sensitivity, but causal usefulness must still be
replicated across training seeds.

The observation-boundary audit reports zero canonical-only and zero
state-union-only positive events in the five-episode sample. This supports the
parser implementation for that sample; it is not a universal proof.

## Semantic-Mask Diagnostic

A matched 50k seed-42 diagnostic removed redundant traffic actions for both
conditions:

| Condition | Mean return | Standard deviation |
|---|---:|---:|
| Full | -967.80 | 372.46 |
| No evidence | -684.57 | 249.38 |
| Untrained Full | -5000.20 | 848.65 |

`Full - No evidence = -283.23`. Semantic validity handling removes the invalid
traffic-action attractor but does not make evidence easier to exploit. It remains
an ablation and is not promoted to the native benchmark configuration.

## Cybermonic Inheritance

Already inherited cleanly:

- 500-tick episodes;
- complete multi-tick action reward accumulation;
- temporal observation memory;
- node/edge/global action scoring;
- independent per-agent observation boundaries.

Not inherited into the main qualification:

- five independent actor-critic networks;
- variable port/file/internet entity nodes;
- heuristic inter-agent messages;
- EnterpriseMAE compromised/scanned features;
- local reward shaping and very large parallel training budget.

Cybermonic reports independent local policies and convergence near 50k training
episodes. The present 50k-tick screen contains only 101 episodes. Directly
copying all Cybermonic components would confound observation, policy
factorization, parameter count, reward and training budget.

## Next Gate

1. Treat observation utility as qualified; do not add Cybermonic-style messages,
   reward shaping, semantic action masking or five independent policies to make
   this effect larger.
2. Resolve the convergence decision before the encoder benchmark. Either extend
   Full policies under a predeclared continuation budget and re-test the plateau,
   or explicitly redefine the paper as a fixed-budget/sample-efficiency
   benchmark where convergence is not claimed.
3. If strict convergence remains required, keep No evidence fixed: its causal
   qualification is complete. Continue only the five Full seeds from their final
   checkpoints if the training implementation can preserve optimizer and RNG
   state exactly; otherwise a fresh longer-budget run is required.
4. Once the convergence protocol is fixed, build matched GCN and GraphSAGE
   encoders. Keep PPO, canonical schema, reward, official mask, environment
   distribution, action scorer, training seeds and parameter budget fixed.
5. Time shift, five independent local policies, bounded per-event slots and
   messages remain separate secondary interventions, not prerequisites for the
   encoder comparison.
