# Pilot Seed-0 Action-Mask Findings

Date: 2026-08-09

Protocol: `pilot_cc2pp_observation_qualification_v1_4`

## Status

Pilot seed 0 is complete for Flat and GraphSAGE at 1,000,000 environment
steps. It is a failed qualification, not main-benchmark evidence. Seeds 1 and
2 must not be run until the blockers below are resolved.

The worktree was dirty when these engineering runs were created. Each run
contains a source snapshot and diff, but the artifacts do not satisfy the
clean-worktree reproducibility gate.

## Controlled Intervention

- Native `HybridAvailabilityConfidentiality` reward; no reward shaping.
- Full frozen 158-entry action catalogue.
- Shared `observable_response_v1` mask for Flat and GraphSAGE.
- Only Remove and Restore are gated.
- A response is available only when the exact policy observation contains
  host-local evidence; the mask does not read true state.
- Observation-use sensitivity is measured on raw pre-mask policy logits, so
  the mask cannot manufacture a PASS.
- Raw unmasked Restore policies are retained only as exploit sentinels.

The simulator also includes the local Restore service-process invariant fix.
This repairs operational metric consistency without changing Restore reward,
cost, duration, or session-removal semantics.

## PPO Tuning

With learning rate `1e-4`, Flat reached a 65.5% target-KL early-stop fraction
by 174k steps. A 100k probe at `5e-5` reduced this to 22.45%, still above the
locked 20% limit. The final shared learning rate was therefore fixed at
`3e-5` before the 1M runs.

Final target-KL early-stop fractions:

| Model | Fraction | Gate |
|---|---:|---|
| Flat | 1.84% | PASS |
| GraphSAGE | 20.20% | FAIL by 0.20 percentage points |

## Final Held-Out Outcomes

Each q cell contains 100 episodes of 100 steps.

| Model | q | Return | Availability | Privileged burden | Deterministic actions |
|---|---:|---:|---:|---:|---|
| Flat | 1.00 | -332.471 | 1.000 | 6.962 | 67.87% Analyse Op_Server0; 32.13% Restore Op_Server0 |
| Flat | 0.75 | -354.131 | 0.982 | 8.362 | 76.68% Analyse Op_Server0; 23.32% Restore Op_Server0 |
| Flat | 0.50 | -426.258 | 0.910 | 8.486 | 82.07% Analyse Op_Server0; 17.93% Restore Op_Server0 |
| Flat | 0.25 | -573.442 | 0.760 | 8.488 | 89.42% Analyse Op_Server0; 10.58% Restore Op_Server0 |
| GraphSAGE | 1.00 | -986.792 | 0.330 | 6.838 | 100% DecoyHarakaSMPT Enterprise2 |
| GraphSAGE | 0.75 | -986.792 | 0.330 | 6.838 | 100% DecoyHarakaSMPT Enterprise2 |
| GraphSAGE | 0.50 | -986.792 | 0.330 | 6.838 | 100% DecoyHarakaSMPT Enterprise2 |
| GraphSAGE | 0.25 | -986.792 | 0.330 | 6.838 | 100% DecoyHarakaSMPT Enterprise2 |

Flat's high availability is not sufficient evidence of observation-dependent
cyber defence. Its target entropy is zero in every q cell, and privileged
burden worsens as q decreases. GraphSAGE collapses to one q-invariant decoy.

## Observation-Use Gate

Both models fail the joint gate.

- Flat: outcome intervention PASS; raw-logit sensitivity FAIL.
- GraphSAGE: outcome intervention FAIL; raw-logit sensitivity FAIL.
- Raw argmax change fraction is 0.0 in every tested cell for both models.
- Mean total-variation distances are below 0.003 in every final cell, well
  below the locked 0.01 threshold.

Flat therefore learns an Analyse/Restore ordering that the semantic action
mask turns into a useful cycle. The evidence-dependent behavior is supplied
mainly by the mask, not demonstrated by the encoder. This is the action-mask
confound identified in the literature review.

## Baseline Evidence

On q=1 held-out episodes:

| Baseline | Return | Availability | Privileged burden |
|---|---:|---:|---:|
| Random, mask-aware | -848.365 | 0.452 | 6.687 |
| Masked Restore critical | -332.471 | 1.000 | 6.962 |
| Masked Restore round-robin | -218.388 | 1.000 | 5.025 |
| Raw Restore critical exploit | -400.341 | 1.000 | 6.962 |
| Raw Restore round-robin exploit | -291.636 | 0.881 | 2.262 |

Flat exactly matches the q=1 masked-Restore-critical baseline and is weakly
Pareto-dominated by controlled mask-aware baselines. GraphSAGE does not beat
the random baseline on availability or return.

## Decision

1. Do not run seeds 1 and 2 under protocol v1.4.
2. Do not claim that action masking solved Restore overfitting.
3. Do not claim a Flat-versus-GNN ranking from these runs.
4. Treat semantic evidence-based masking as a benchmark confound, not a neutral
   invalid-action mask.
5. The next intervention must be isolated from this result. Reward shaping,
   action abstraction, or structural-only validity handling requires a new
   protocol identity and an explicit ablation against v1.4.

## Artifacts

- `runs/pilot/qualification/flat/seed_0/`
- `runs/pilot/qualification/graphsage/seed_0/`
- `runs/pilot/baselines/`
- `runs/pilot/qualification/summary/summary.json`
