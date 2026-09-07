# Canonical v1 Degradation Curve Protocol

## Question

How quickly do parameter-matched Flat, GCN and GraphSAGE PPO policies degrade
when progressively less dynamic host evidence is available at evaluation time?

## Controlled Design

- Existing 100k checkpoints only; no retraining or checkpoint reselection.
- Canonical v1 `action_aware`, `random_host`, episode horizon 400.
- Official CAGE 4 action mask only in the primary experiment.
- Training seeds 42-46.
- Stage A uses 3 paired held-out episodes (10000-10002) for every checkpoint.
- Stage B extends every checkpoint uniformly to 30 episodes (10000-10029) if
  Stage A uncertainty is too large; no outcome-dependent seed selection.
- Stochastic policy evaluation with action RNG seed 10000.
- Retention levels `q = 1.00, 0.75, 0.50, 0.25`.
- One policy evaluation process at a time.

## Observation Intervention

At each simulator tick and for each Blue agent, the evaluator selects an exact
fixed budget of existing host nodes. Selection is proportional and stratified
by local subnet, with deterministic tie-breaking from mask seed 30000. All 11
dynamic canonical-v1 node features are retained for selected hosts and set to
zero for hidden hosts.

The following are held fixed:

- host/subnet existence and role features;
- host-to-subnet membership and subnet adjacency;
- communication policy and firewall state;
- mission phase and action-success global features;
- official action-validity mask, reward and action space.

Therefore `q` is specifically the **dynamic host-feature retention ratio**. It
is not the percentage of the whole vector retained, not topology removal, and
not a native CAGE 4 sensor model.

## Metrics

- Return response curve at each q.
- Endpoint degradation: `R(q=1) - R(q=.25)`; lower is better.
- Relative endpoint degradation divided by `abs(R(q=1))`.
- Normalized degradation AUC across `[.25, 1]`; lower is better.
- Raw return AUC and worst-case return; higher is better.
- Paired encoder contrasts at each q using training seed as the unit (`n=5`).
- Monotonicity violations retained as a stochastic-noise diagnostic.

Robustness and absolute performance are reported separately. A policy is not
declared better merely because it starts from a poor baseline and therefore has
less room to degrade.

## Follow-Up, Not Part of the Primary Curve

- Repeat under semantic action masking as an explicit ablation.
- Persistent subnet masking at one matched severity to test observation-process
  generalisation.
- Mixed-q retraining only after this zero-shot diagnostic is interpreted.
