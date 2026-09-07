# Canonical V1 Flat Qualification Protocol

## Purpose

Test whether PPO can learn with the corrected canonical V1 representation and
whether the learned stochastic policy uses its dynamic cyber evidence.

## Training

- Training seed: 42.
- Canonical variant: `action_aware`.
- Requested budget: 30,000 simulator steps.
- Actual budget: 30,324 simulator steps (76 complete 400-step episodes).
- Episode horizon: 400 simulator steps.
- PPO: learning rate 3e-4, gamma 0.99, GAE lambda 0.95, four epochs,
  minibatch size 64, clip coefficient 0.2, value coefficient 0.5.
- Entropy coefficient: 0.001.
- Reward scale: 0.01.
- Maximum gradient norm: 0.5.
- Target KL: 0.015.
- Encoder/latent/head dimensions: 128/128/128.
- Observation dimension: 1,651.
- Shared action space: 242 actions.

## Model Selection

No periodic model selection was used. Evaluation uses the final checkpoint so
that deterministic checkpoint selection cannot confound the stochastic-primary
qualification.

## Held-Out Evaluation

- Environment seeds: 10,000 through 10,029.
- Episodes: 30.
- Primary mode: stochastic sampling from the learned policy.
- Secondary mode: deterministic masked argmax.
- Stochastic action seed: 10,000.
- Baselines use the same episode seeds and action-sampling seed.

## Gates

1. Learning gate: trained stochastic return minus reproducibly initialized
   untrained stochastic return has a paired-bootstrap 95% confidence interval
   above zero.
2. Observation-use gate: full canonical stochastic return minus `static_only`
   stochastic return has a paired-bootstrap 95% confidence interval above zero.
3. `static_only` keeps node identity/type, host membership, topology,
   communication policy, firewall matrix, global mission/action-success
   features and the official action mask. It removes current cyber evidence,
   temporal evidence memory, evidence age and decoy state.

This is one training seed and is a pilot qualification, not paper-level
multi-seed evidence.
