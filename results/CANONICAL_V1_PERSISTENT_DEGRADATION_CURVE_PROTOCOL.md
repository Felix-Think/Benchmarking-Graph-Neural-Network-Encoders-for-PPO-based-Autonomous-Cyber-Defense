# Canonical v1 Persistent Degradation Curve Protocol

## Question

Do parameter-matched Flat, GCN and GraphSAGE PPO checkpoints degrade when a
fixed subset of hosts remains dynamically unobservable for the whole episode?

## Design

- Reuse final 100k checkpoints; no retraining or checkpoint reselection.
- Canonical v1 `action_aware`, `random_host`, official action mask.
- Training seeds 42-46 and held-out evaluation seeds 10000-10029.
- Stochastic evaluation with paired action RNG seed 10000.
- `q = 1.00, 0.75, 0.50, 0.25`.
- One evaluation policy process at a time.

At reset, an exact host budget is sampled proportionally within each local
subnet. The selected host slots remain visible for all 400 simulator ticks;
all dynamic features of the remaining host slots remain zero for the entire
episode. Static identity, topology, communication/firewall state, mission/global
features, action mask, reward and action space remain unchanged.

This prevents hidden evidence and canonical memory from reappearing later in
the same episode. It remains an output-level sensor-loss model rather than a
native CAGE 4 observation process.

## Metrics

- Return response curve and worst-case return.
- Endpoint and relative degradation at `q=.25`.
- Normalized degradation AUC.
- Action-type total-variation distance from `q=1`.
- Paired encoder contrasts using training seed as the inferential unit (`n=5`).
- Monotonicity violations retained rather than smoothed away.
