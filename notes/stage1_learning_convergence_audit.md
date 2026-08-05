# Stage 1 Learning and Convergence Audit

## Audit question

The primary Stage 1 validity question is not whether one observation mask is
better than another. It is whether the shared PPO pipeline learns useful
policies, whether training is numerically stable, whether policies converge,
and whether the evaluation protocol measures learned defensive behavior.

The 100% and 25% observation settings are secondary stress conditions. They
must not be used to hide a failure to learn in either encoder.

## Evidence inspected

- All 12 `training_metrics.jsonl` files.
- All 12 checkpoint evaluation histories at 0, 100k, ..., 1M timesteps.
- Final operational metrics and action audits.
- The common 30 deterministic evaluation episodes used by every run.
- A provisional direct baseline probe on the 16-host Stage 1 scenario using
  the same 30 environment seeds.

## 1. Did PPO optimize without numerical failure?

Yes, at the numerical level.

- Every recorded policy loss, value loss, entropy, and rollout reward is
  finite. There are no NaN or infinite values.
- Mean rollout reward improved from the first 25 to the last 25 PPO updates in
  all 12 runs.
- Mean value loss decreased over the same interval in all 12 runs.
- Entropy did not collapse to zero. The main exception is GraphSAGE, 100%
  observation, seed 1: its entropy remained much lower than the successful
  GraphSAGE seeds and its rollout reward improved only from about -634 to -608.
  This is a stable but poor local policy, not a numerical crash.

Conclusion: the PPO implementation trains, but numerical stability alone does
not establish useful learning or convergence.

## 2. Did final policies improve over their untrained initialization?

Yes by reward, in all 12 runs. The aggregate deterministic evaluation changes
are:

| Encoder | Observation | Reward at step 0 | Reward at 1M | Delta | Availability at step 0 -> 1M | Privileged hosts at step 0 -> 1M |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 100% | -725.4 | -439.3 | +286.1 | 0.570 -> 0.625 | 7.114 -> 6.055 |
| Flat | 25% | -700.7 | -527.9 | +172.8 | 0.601 -> 0.436 | 7.160 -> 7.068 |
| GraphSAGE | 100% | -728.8 | -272.6 | +456.2 | 0.568 -> 0.760 | 7.111 -> 5.026 |
| GraphSAGE | 25% | -751.4 | -171.5 | +579.9 | 0.551 -> 0.585 | 7.196 -> 6.136 |

This proves that the optimization pipeline changes every policy in a direction
that improves CybORG return relative to its deterministic untrained policy.
It does not prove that every final policy is better than a random policy or is
operationally useful.

## 3. Did policies beat simple baselines?

A provisional baseline probe was run directly on the 16-host Stage 1 scenario
with the same 30 environment seeds:

- Sleep: approximately -769.
- Monitor: approximately -720.
- Uniform random actions: approximately -507 across repeated 30-episode action
  seed batches.

The probe has not yet been stored as a formal reproducibility artifact, so it
must not be quoted as a final paper result. It is nevertheless enough to expose
the current trainability issue:

- Mean Flat 100% (-439) is better than the provisional random baseline, but
  only two of three individual Flat seeds are better than the random mean.
- Mean Flat 25% (-528) is worse than the provisional random baseline; only one
  of three individual seeds is better.
- GraphSAGE 100% is better on average, but one of three seeds (-666) is worse.
- All three GraphSAGE 25% seeds are better by reward than the random mean.

Therefore the current results do not support the statement that every trained
Stage 1 agent learned a useful policy. In particular, Flat 25% fails the
trainability gate at the aggregate level.

## 4. Did the policies converge?

Not uniformly.

### Flat, 100% observation

- Every seed improves over initialization.
- Seed 1 reaches its best reward at 1M steps.
- Seeds 0 and 2 reach their best rewards at 700k and regress by roughly 256 and
  250 reward points by 1M.
- The aggregate curve peaks near 700k and remains noisy afterward.

Verdict: learned, but weak convergence and checkpoint instability.

### Flat, 25% observation

- Every seed improves over initialization, but the deterministic curves are
  highly volatile.
- No seed has its best checkpoint at 1M.
- The final aggregate reward does not beat the provisional random baseline.
- Seed 0 ends with mission availability 0.018 because the policy repeatedly
  restores the operational server.

Verdict: not a convincingly converged or useful baseline.

### GraphSAGE, 100% observation

- Seeds 0 and 2 learn strong policies and are near their best reward at 1M.
- Seed 1 remains in a poor local policy throughout training.
- The aggregate curve broadly plateaus after roughly 400k-600k steps, but its
  large seed variance is caused by this bimodal outcome.

Verdict: convergence is demonstrated for two of three seeds, not robustly
across seeds.

### GraphSAGE, 25% observation

- All three final checkpoints have their best recorded reward at 1M.
- Seeds 0 and 2 show strong late-training policies.
- Seed 1 remains volatile; its reward improves but availability falls from
  0.519 to 0.307 and privileged compromises increase from 7.12 to 7.55.
- Because the aggregate curve is still improving at the final checkpoint, a
  plateau has not been demonstrated.

Verdict: useful reward learning is present, but full convergence and
operational alignment are not established.

## 5. Is the evaluation protocol appropriate?

### Valid design choices

- All models use the same PPO core, environment, action space, reward, training
  budget, and final-checkpoint rule.
- Evaluation uses a separate environment and a common fixed set of seeds.
- Evaluation is deterministic (`argmax`), which is a common PPO reporting
  choice and measures the deployable greedy policy.
- True simulator state is used only for post-action evaluation metrics, not as
  policy input.
- The summary uses final checkpoints. This avoids choosing and reporting the
  same best checkpoint on the same repeated evaluation set.

### Remaining weaknesses

- Thirty evaluation episodes per training seed are adequate for a pilot but
  modest for a paper-level result. Closely related work commonly uses 100 or
  1000 episodes.
- Random and no-action baselines are not yet first-class Stage 1 artifacts with
  reward and operational metrics.
- Training rollouts use sampled PPO actions, while evaluation uses argmax. Flat
  policies show a large gap between late sampled-rollout reward and final
  deterministic reward. A stochastic-policy evaluation should be reported as
  a sensitivity check, not as a replacement for deterministic evaluation.
- Reusing the same 30 episodes at every checkpoint is acceptable for monitoring
  the final checkpoint, but `best_reward.pt` must be evaluated on a new held-out
  seed set if it is ever reported.
- Reward and mission availability can disagree. Reward cannot be the only
  success criterion.

## 6. Consistency with PPO-GNN literature

The literature supports the plausibility of mixed convergence and high
variance; it does not validate our numerical rewards.

- Nyberg and Johnson train CAGE-2 MLP-PPO and MPNN-PPO agents for 500k steps
  and evaluate each configuration over 1000 episodes. Their specialized MLP
  performs better on its trained topology, while MPNN provides topology-size
  generalization. They explicitly report high score deviations and about 10x
  slower MPNN training. This shows that a GNN is not expected to beat Flat in
  every trained setting. Source: https://arxiv.org/html/2407.05775
- King, Bowman, and Huang use 5M steps for their Yawning Titan comparison and
  hundreds of zero-shot evaluation episodes. Their inductive graph PPO models
  can strongly outperform standard observations on unseen graphs, while a
  transductive graph model can fail to generalize. This supports checking
  learned behavior per seed instead of treating `GNN` as a guarantee. Source:
  https://arxiv.org/html/2509.16151
- Dudman and Bull train vanilla PPO and HGT-based TERLA for about 1M steps in
  CAGE-4 and evaluate deterministically over 100 episodes. Vanilla PPO and a
  single TERLA policy converge around 300k steps, while some separate TERLA
  agents need the full budget. Reported evaluation is PPO -2825 +/- 912,
  separate TERLA -2773 +/- 917, and single TERLA -2048 +/- 799. This supports a
  1M-step pilot budget but also shows that graph policies may require the whole
  budget. Source: https://proceedings.mlr.press/v299/dudman25a.html
- Campbell, Eirinaki, and Park train GCN-PPO for 5M timesteps and use confidence
  intervals when evaluating scalability under partial observability. Their
  result is not a clean encoder-only comparison because the GCN system also
  changes reward design, curriculum, and memory. It should not be used as a
  numerical target for Stage 1. Source:
  https://doi.org/10.1109/ACCESS.2024.3418931

Absolute rewards cannot be compared across these papers because the simulator,
reward function, action space, episode horizon, topology, and evaluation
aggregation differ.

## Current scientific verdict

Stage 1 demonstrates that the shared PPO implementation can learn and that the
GraphSAGE pipeline can produce strong defensive policies. It does not yet
demonstrate uniform convergence across seeds, a reliable Flat baseline under
25% observation, or statistically established GraphSAGE superiority.

The observation matrix should therefore be treated as a stress test after a
trainability gate, not as the main evidence that the pilot works.

## Required gate before paper claims

1. Save reproducible Sleep, Monitor, and random-action Stage 1 baselines with
   the same reward and operational metrics.
2. Improve or retune the Flat baseline until it reliably beats random across
   seeds in the selected primary observation setting.
3. Add held-out deterministic evaluation with at least 100 episodes per seed.
4. Add stochastic-policy evaluation as a sensitivity analysis.
5. Define convergence before rerunning: final performance near the late-window
   plateau, no large best-to-final collapse, and no operational metric
   regression hidden by reward.
6. Only after both encoders pass the trainability gate should the observation
   stress condition be used to compare robustness.

## Safe paper wording for the current pilot

> Across all twelve pilot runs, PPO optimization remained numerically stable
> and final deterministic returns exceeded those of the corresponding
> untrained policies. However, convergence was not uniform across training
> seeds, and the Flat policy under the strongest observation restriction did
> not outperform a provisional random-action baseline. The pilot therefore
> validates end-to-end trainability of the implementation, but does not yet
> establish robust encoder superiority.
