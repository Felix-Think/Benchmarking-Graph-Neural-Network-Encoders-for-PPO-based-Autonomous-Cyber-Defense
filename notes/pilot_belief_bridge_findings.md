# Temporal-Belief Bridge Findings

Date: 2026-08-10

## Evidence Status

This is engineering evidence from a dirty worktree, not paper evidence. The
bridge used one paired seed and 300,000 environment steps per cell. Its purpose
was mechanism diagnosis before spending more compute.

The four cells were:

- Flat with raw current-step control;
- Flat with observable temporal belief;
- GraphSAGE with raw current-step control;
- GraphSAGE with observable temporal belief.

Both arms used a `16 x 31` observation. The raw control's five belief channels
were fixed at zero, so paired models had identical shapes, parameter counts and
seeded initialization. Flat had 117,151 parameters and GraphSAGE had 106,783.
Both used all 158 native actions, no action mask, native reward, mixed q, and
the same official CAGE-2 plus CybORG++ patchset.

## 300k Result

Every cell had the same monitoring result at 100k, 200k and 300k:

- aggregate return: `-398.205`;
- mission availability: `1.000`;
- privileged compromise burden: `6.955`.

In final held-out evaluation, every cell and every q level produced:

- return: `-400.341`;
- mission availability: `1.000`;
- privileged compromise burden: `6.962`;
- Restore fraction: `1.000`;
- maximum action-target share: `1.000`.

All 40,000 final evaluation decisions in each cell were
`Restore(Op_Server0)`. There was no q-response.

Target-KL early-stop update fractions were:

| Arm | Flat | GraphSAGE |
|---|---:|---:|
| Raw control | 0.0204 | 0.0748 |
| Temporal belief | 0.0340 | 0.0884 |

All values stayed below the configured 0.20 blocker threshold. PPO numerical
instability therefore does not explain the behavioral collapse, although the
belief cells stopped slightly more often.

## Observation-Use Diagnostic

The engineering gate used 20 outcome episodes and 5 sensitivity episodes per
cell, not the preregistered full 100/20 episode budget. All four checkpoints
failed outcome, sensitivity, and joint gates.

Maximum mean total variation under same-state interventions was:

| Arm | Flat | GraphSAGE | Argmax change |
|---|---:|---:|---:|
| Raw control | 0.0338 | 0.0115 | 0.000 |
| Temporal belief | 0.3290 | 0.1506 | 0.000 |

Belief therefore changed the policy logits substantially, especially for
Flat, but never changed the selected action. The precise conclusion is not
that the network ignored belief numerically. Restore remained dominant at the
decision boundary, so belief had no behavioral or outcome effect.

## Open-Loop Controls

Full 158-action calibration and 100-episode evaluation produced the same
baselines under both arms:

| Control at q=1 | Return | Availability | Privileged burden |
|---|---:|---:|---:|
| Sleep | -983.136 | 0.338 | 6.904 |
| Monitor | -986.792 | 0.330 | 6.838 |
| Random | -794.661 | 0.472 | 5.858 |
| Restore Op_Server0 | -400.341 | 1.000 | 6.962 |
| Restore round-robin | -291.636 | 0.881 | 2.262 |
| Best fixed action | -319.954 | 1.000 | 6.075 |

The calibrated best fixed action was `Restore(Enterprise2)`. It strictly
dominated every learned cell on the availability/burden pair: equal
availability and lower privileged burden. The learned policies also failed to
Pareto-dominate Sleep or Monitor because their privileged burden was slightly
worse despite much higher availability.

## Conclusion

The tested claim is rejected at bridge scope:

> Adding the clean observable temporal-belief module alone is not sufficient
> to reproduce the useful learning behavior attributed to the old code.

This result does not prove that temporal memory is irrelevant to cyber-defense
POMDPs. It shows that this belief representation cannot overcome the current
unmasked Restore shortcut under the tested PPO/action/reward protocol.

The old result may instead depend on one or more legacy differences:

- the full latest-host cache, including pre-mask telemetry retention;
- the custom scalable scenario and old topology;
- the old action surface and reward dynamics;
- Stage1 target-visibility behavior;
- older PPO learning rate and update schedule;
- interactions among these factors rather than temporal belief alone.

The full legacy cache should not be ported without redesign because its
pre-mask update order can leak synthetically hidden telemetry. Do not spend
1M steps or additional seeds on this bridge. The next Pilot decision must
control the Restore/open-loop action shortcut first; temporal or recurrent
memory can then be tested as a separate ablation.

## Artifacts

- `runs/pilot_raw_bridge_control/qualification/`;
- `runs/pilot_belief_bridge/qualification/`;
- `runs/pilot_raw_bridge_control/baselines/`;
- `runs/pilot_belief_bridge/baselines/`;
- `notes/pilot2_legacy_reference.md`;
- `notes/pilot_belief_bridge_protocol.md`.
