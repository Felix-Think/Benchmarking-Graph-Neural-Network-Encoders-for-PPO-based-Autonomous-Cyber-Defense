# Canonical Flat Replication Protocol

Protocol frozen after the exploratory qualification of training seed 42 and
before running training seeds 43 and 44.

## Fixed Training Conditions

- Training seeds: 42, 43 and 44.
- Canonical variant: `action_aware`.
- Evidence-age horizon: 50.
- Requested budget: 84,000 simulator steps per seed.
- Episode steps: 400.
- Learning rate: 3e-4.
- Gamma: 0.99.
- GAE lambda: 0.95.
- PPO epochs: 4.
- Minibatch size: 64.
- Reward scale: 0.01.
- Clip coefficient: 0.2.
- Value coefficient: 0.5.
- Entropy coefficient: 0.001.
- Maximum gradient norm: 0.5.
- Target KL: 0.015.
- Encoder/head dimensions: 128/128/128.
- Device: CPU.

No hyperparameter or representation change is permitted between replication
seeds.

## Model Selection

- Evaluate every 10 PPO updates.
- Use 10 fixed periodic episodes with seeds 20,000-20,009.
- Select the checkpoint with the highest deterministic mean team return.
- Do not inspect held-out results when selecting a checkpoint.

## Held-Out Evaluation

- Use 100 episodes with seeds 10,000-10,099.
- Primary result: deterministic best-checkpoint return.
- Secondary result: stochastic best-checkpoint return.
- Evaluate the untrained canonical policy deterministically using the same
  initialization seed and held-out episode seeds.
- Evaluate the best policy under the `static_only` observation-use gate, which
  preserves structural features, global features and action masks while zeroing
  dynamic current/belief evidence.

## Per-Seed Qualification

For each training seed, report paired episode-level bootstrap confidence
intervals and effect sizes for:

1. best canonical minus untrained canonical;
2. best canonical minus static-only gate;
3. best canonical minus prior Flat best;
4. final canonical minus untrained canonical.

A seed passes the learning gate when the 95% paired bootstrap confidence
interval for best minus untrained excludes zero on the positive side.

A seed passes the observation-use gate when the 95% paired bootstrap confidence
interval for best minus static-only excludes zero on the positive side.

## Replication Decision

- `Replicated`: all three seeds have positive learning and observation-use point
  estimates, and at least two of three seeds pass each corresponding confidence-
  interval gate.
- `Mixed`: point estimates or confidence-interval gates disagree across seeds.
- `Not replicated`: at least two seeds have non-positive point estimates for
  either learning or observation use.

Training seed is the replication unit. Episode-level samples from different
training seeds must not be pooled and treated as 300 independent observations.
Three seeds are sufficient for this pilot replication decision, not for a final
paper-level uncertainty estimate.
