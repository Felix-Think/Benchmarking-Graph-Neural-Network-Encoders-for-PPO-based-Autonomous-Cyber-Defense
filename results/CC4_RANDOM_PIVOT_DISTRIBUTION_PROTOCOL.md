# CAGE 4 Random-Pivot Distribution Protocol

## Problem Being Controlled

The official CAGE 4 `EnterpriseScenarioGenerator` attaches every logical
cross-subnet discovery link to `server_host_0`. Environment seeds randomize
host counts, addresses, services and Red choices, but do not randomize this
lateral-movement pivot slot. A policy can therefore learn a fixed-slot defence
without using current process, connection or analysis evidence.

## Intervention

`random_host` first creates a standard CAGE 4 scenario. It then samples one
existing non-router pivot host independently for each participating subnet and
moves the original cross-subnet information links to those pivots. The logical
subnet graph remains unchanged.

The pivot sampler uses a local RNG derived from a stable fingerprint of the
completed scenario. It does not advance the simulator RNG used later by Red or
Green agents.

## Controlled Components

The following remain fixed between Flat and GCN:

- PPO implementation and hyperparameters;
- canonical V1 schema and dimensionality;
- official CAGE 4 reward machine and action costs;
- padded 242-action catalog and per-episode action masks;
- mission phases and FiniteStateRedAgent;
- 100,000 simulator-step training budget;
- approximately matched parameter budget: Flat 292,339 versus GCN 291,735.

Only the scenario distribution changes from fixed to randomized pivot hosts.

## Train/Test Split

- Train distribution: `random_host`, seeds 42 through 292 for a 100k run with
  400-step episodes (251 complete rollouts because the budget may be exceeded
  by one complete episode).
- Held-out test distribution: `random_host`, seeds 10000 through 10029.
- Secondary stress test: `official` fixed-pivot CAGE 4 on the same test seeds.
- Stochastic PPO evaluation is primary; deterministic argmax remains a
  diagnostic.

No test seed or checkpoint-selection score may be used for training or model
selection.

## Distribution Audit

The exact 251/30 split produced:

- 251 unique train pivot maps and 30 unique test pivot maps;
- zero seed overlap and zero exact pivot-map overlap;
- 15-16 observed pivot slots per subnet in training;
- normalized pivot-slot entropy between 0.916 and 0.939 in training;
- held-out hit rate of 10.83% for a strategy that selects each subnet's modal
  train pivot slot;
- held-out hit rate of 12.50% for the official `server_host_0` slot.

The full machine-readable audit is in
`results/cc4_random_pivot_distribution_audit.json`.

## Qualification Gates

After training separately on `random_host`:

1. Learning gate: trained full-observation policy must outperform its untrained
   baseline on paired held-out seeds.
2. Observation-use gate: full observation must outperform `static_only` on
   paired held-out seeds. Report the paired mean, bootstrap 95% confidence
   interval and episode win counts.
3. Encoder comparison: compare Flat and GCN only after both use the same
   distribution, budget, seeds and control settings.
4. Generalisation diagnostic: evaluate the same checkpoints on `official`
   without retraining.

The old fixed-pivot GCN checkpoint scored -891.70 with full observations and
-825.37 under `static_only` on `random_host` (paired full-minus-static -66.33).
This is a pre-training stress test, not evidence against the new distribution:
the checkpoint learned on the old official distribution and therefore cannot
be expected to acquire evidence use at evaluation time.

### 20k pre-qualification

A GCN retrained for 20,349 actual simulator steps on `random_host` produced:

- full stochastic return: -1419.93 +/- 560.08;
- `static_only` stochastic return: -1631.17 +/- 600.48;
- paired full-minus-static mean: +211.23;
- bootstrap 95% confidence interval: [-35.18, 455.73];
- episode wins/ties/losses: 19/0/11.

This is a positive observation-use signal, but it does not pass the strict gate
because the confidence interval includes zero. It justifies the planned 100k
run; it does not support a final claim by itself.

### Matched 100k result

The subsequent matched 100k run did not strengthen the observation-use signal:

- Flat full minus static-only: +59.93, CI [-61.80, 177.77].
- GCN full minus static-only: -30.00, CI [-194.57, 135.67].
- GCN full minus Flat full: -271.50, CI [-458.17, -88.56].

Therefore `random_host` should be retained as a useful structural stress test,
but it is not sufficient to make dynamic evidence necessary. Detailed results
and training-seed limitations are in
`results/MATCHED_RANDOM_PIVOT_100K_RESULT.md`.

## Stop Rule

Do not add Red-agent mixtures, denser networks, reward shaping or observation
features before this isolated intervention is tested. If a retrained policy
still fails the observation-use gate, randomizing pivot slots is insufficient
to defeat observation-independent blanket defence. That result should trigger
an action/reward and evidence-timing audit, not another uncontrolled bundle of
environment modifications.
