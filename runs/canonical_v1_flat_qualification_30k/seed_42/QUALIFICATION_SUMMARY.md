# Canonical V1 Flat Qualification - Seed 42

## Result

- Trained stochastic final checkpoint: -834.53 +/- 271.06.
- Untrained stochastic baseline: -3776.13 +/- 624.59.
- Static-only stochastic gate: -854.00 +/- 355.09.
- Deterministic final checkpoint (secondary): -3670.57 +/- 873.02.

## Paired Gates

- Learning delta, trained minus untrained: +2941.60, paired-bootstrap 95%
  CI [2707.96, 3179.94], wins/ties/losses 30/0/0. **PASS**.
- Dynamic observation-use delta, trained minus static-only: +19.47,
  paired-bootstrap 95% CI [-139.23, 170.17], wins/ties/losses 17/0/13.
  **FAIL**.
- Full/static-only action-type total variation: 0.0164, indicating nearly
  identical aggregate action-type distributions.

## Interpretation

The corrected 1,651-dimensional canonical V1 is trainable: PPO improves very
strongly over its reproducibly initialized untrained policy. However, this
30k-step run does not show that the policy uses dynamic process, connection,
analysis or temporal-belief evidence. Most of the learned performance survives
when those features are removed, so the policy may primarily use static
identity/topology information, mission features, action masks and learned action
priors.

Canonical V1 therefore passes technical and learning validation but does not
yet pass the dynamic observation-use qualification required to freeze it for a
Flat-versus-GNN encoder benchmark. This is one seed and must not be presented as
paper-level evidence.
