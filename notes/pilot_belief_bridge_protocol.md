# Observable Temporal-Belief Bridge

## Purpose

This is a diagnostic A/B bridge, not the main encoder benchmark. It asks:

> Does a compact, policy-observable temporal belief help a feed-forward PPO
> policy learn under the current partial-observation process?

It does not assume that the answer is yes and it does not require GraphSAGE to
beat Flat.

The completed seed-0 engineering result is recorded in
`notes/pilot_belief_bridge_findings.md`.

## Paired Protocols

| Contract | Raw control | Temporal-belief bridge |
|---|---|---|
| Config | `configs/pilot_raw_bridge_control.yaml` | `configs/pilot_belief_bridge.yaml` |
| Simulator | official CAGE-2 plus audited CybORG++ patchset | same |
| Reward | native only | same |
| Actions | all 158 native entries | same |
| Action mask | none | none |
| q schedule | mixed 1/.75/.50/.25 | same |
| Retention | post-native fixed-budget stratified mask | same |
| Policy memory | none; five belief channels fixed to zero | observable five-state host belief |
| Observation | 16 x 31 | 16 x 31 |
| Seed | 0 | paired seed 0 |
| Budget | 300,000 environment steps | same |

Both arms use the same input width, parameter count and seeded initialization.
The zero channels in the control carry no information. The bridge remains a
mechanism diagnosis rather than a publication-quality encoder ranking.

## No-Leakage Update Order

For each reset or step:

1. encode only the current raw Blue observation;
2. sample and apply the synthetic host-retention mask;
3. apply selected Restore/Remove feedback without reading success;
4. update temporal belief only from retained native telemetry;
5. append the five belief channels;
6. apply any observation-use intervention;
7. provide the matched tensor to Flat or GraphSAGE.

Hidden raw evidence cannot update belief. A hidden host can retain belief that
was acquired at an earlier visible timestep; that persistence is the intended
memory treatment rather than leakage.

## Belief Semantics

- A retained host with native telemetry and no anomaly initializes as clean.
- Exploit or anomalous-session evidence changes it to user compromise.
- Malware-file or privileged-session evidence changes it to privileged
  compromise.
- Absence of later evidence does not erase an existing suspicion.
- Selected Restore predicts clean state.
- Selected Remove changes user/privileged suspicion to uncertain.
- Action success and evaluator true state are unavailable.

## Execution Order

1. Run unit tests and development audits for both configs.
2. Run 4,096-step smoke cells for Flat and GraphSAGE under both configs.
3. Run the paired 300k seed-0 controls only if smoke artifacts are valid.
4. Compare learning curves, final q response, action concentration, Restore
   fraction, mission availability, compromise burden, and observation-use
   sensitivity.
5. Do not run 1M or additional seeds until the bridge shows useful learning
   without open-loop control dominance.

Commands:

```bash
uv run --python 3.10 python -m scripts.pilot.run_matrix \
  --config configs/pilot_raw_bridge_control.yaml --smoke \
  --models flat graphsage --seeds 0 --skip-baselines \
  --skip-observation-gates --overwrite

uv run --python 3.10 python -m scripts.pilot.run_matrix \
  --config configs/pilot_belief_bridge.yaml --smoke \
  --models flat graphsage --seeds 0 --skip-baselines \
  --skip-observation-gates --overwrite
```

## Decision Rules

- If belief improves both encoders relative to paired raw controls and the
  policies pass the observation-use gate, temporal state is a plausible
  contributor to the old learning difference.
- If only Flat or only GraphSAGE improves, inspect model-specific optimization
  and action distributions before making a POMDP claim.
- If belief policies collapse to Restore or another fixed action and are
  dominated by open-loop controls, the bridge fails even if native return is
  high.
- If no model improves, temporal belief alone does not explain the old result;
  remaining candidates include the old scenario, full host cache, action
  surface, reward dynamics, and PPO settings.
- Bridge success permits a controlled Pilot redesign. It is not evidence that
  GNN is superior and does not make the project ready for main benchmarking.
