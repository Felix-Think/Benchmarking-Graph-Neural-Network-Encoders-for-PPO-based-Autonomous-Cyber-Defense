# Benchmarking GNN Encoders for PPO-based Cyber Defense

Code workspace for **Benchmarking Graph Neural Network Encoders for PPO-based
Autonomous Cyber Defense under Partial Observability**.

## Current Scope

The active simulator is CAGE 4. The current code provides:

- a shared-parameter PPO policy for all five Blue agents;
- the official `BlueFlatWrapper` baseline;
- a matched canonical observation consumed by either a Flat MLP or a GCN;
- action masking and asynchronous Blue decision handling;
- fixed reward scaling, GAE, PPO updates, checkpoints and TensorBoard logging;
- paired deterministic and stochastic evaluation.

The canonical Flat and GCN pipelines use the same PPO implementation. Feature
gates support evaluation-time ablations of dynamic evidence, node identity,
topology, mission phase and action masking.

## Environment

Python 3.10 is required. The local `CybORG/` directory is the simulator source
used by the project.

```bash
uv sync --all-groups
```

The examples below use the local `uvr` shell alias. Replace `uvr` with
`uv run --python 3.10` if the alias is unavailable.

## Code Structure

```text
.
|-- CybORG/                               local CAGE 4 simulator
|-- scripts/
|   |-- inspection/inspect_cc4_env.py     inspect agents and native spaces
|   |-- training/train_flat.py            official BlueFlat PPO baseline
|   |-- training/train_canonical_flat.py  matched canonical Flat PPO
|   |-- training/train_canonical_gcn.py   matched canonical GCN PPO
|   |-- evaluation/evaluate_flat.py       evaluate official Flat checkpoint
|   |-- evaluation/evaluate_canonical_flat.py
|                                           evaluate canonical checkpoint
|   |-- evaluation/evaluate_canonical_gcn.py
|   |                                       evaluate canonical GCN checkpoint
|   `-- evaluation/evaluate_canonical_observation_gate.py
|                                           canonical feature-group ablations
|-- src/acd_benchmark/
|   |-- cyber/cc4_env.py                  construct native CAGE 4
|   |-- cyber/cc4_flat_env.py             official Flat adapter
|   |-- cyber/canonical_observation.py    canonical v1 representation builder
|   |-- cyber/canonical_observation_v2.py canonical v2 typed alerts and TTL memory
|   |-- cyber/cc4_canonical_env.py        canonical Flat PPO adapter
|   |-- rl/policies/                      shared actor-critic, Flat and GCN
|   |-- rl/rollout/                       multi-agent rollout collection
|   |-- rl/ppo/                           GAE, PPO update, train and evaluation
|   `-- rl/tensorboard_logger.py          scalar and histogram logging
|-- tests/                                focused regression tests
|-- runs/                                 generated training artifacts
|-- pyproject.toml                        project and dependency definitions
`-- uv.lock                               locked dependency versions
```

## Canonical Observation

Each Blue agent receives a fixed representation built from its raw Blue
observation, the process/connection event channels used by CAGE 4's official
`BlueFlatWrapper`, public mission rules, and the outcomes of its own actions.
It does not use compromise state, Red sessions, or other hidden simulator state.
The representation contains:

- 57 nodes with 17 features per node;
- a 48-by-9 host-to-subnet membership matrix;
- a 9-by-9 physical subnet-topology matrix;
- a separate 9-by-9 desired communication-policy matrix;
- a separate 9-by-9 agent-known firewall-block matrix;
- 7 global mission-phase and action-success features.

The node features distinguish current process/connection events from temporal
memory. A completed `Analyse` action is represented separately from its maximum
observed file density. `Remove` does not declare a host clean; only a successful
`Restore` clears the corresponding action-aware belief.

The resulting Flat input has 1,651 values:

```text
57 * 17 + 48 * 9 + 3 * (9 * 9) + 7 = 1651
```

Canonical checkpoints produced before this V1 schema are dimensionally
incompatible and are retained only as historical qualification evidence.

Available variants:

- `action_aware`: primary candidate; retains evidence and updates memory after
  successful defensive actions;
- `belief`: retains observed evidence without action-conditioned clearing;
- `instant`: keeps current-step evidence only and is a diagnostic condition.

Flat and GNN policies consume the same canonical information. The Flat policy
receives its serialization. The GCN uses host-membership and physical-topology
relations as message-passing edges, while retaining communication policy,
firewall state and global features as graph-level context.
Inter-agent messages are intentionally excluded from both encoders in V1.

The current GCN concatenates the 57 ordered node embeddings before its latent
projection. This preserves host-slot identity required by the fixed 242-action
head and makes it suitable for same-scale qualification. It is not the final
cross-scale design; cross-scale experiments require a node-conditioned action
head rather than a fixed global action-logit vector.

Canonical v1 remains the backward-compatible baseline. Canonical v2 can be
selected with `--canonical-schema v2`; it separates passive alert memory from
Analyse history, decays both memories under independently configurable TTLs,
and retains observable event-attribute indicators. The optional
`--semantic-action-mask` removes redundant `AllowTrafficZone` and
`BlockTrafficZone` choices using only successful Blue action history. These are
independent switches and must be evaluated with the 2x2 protocol in
`results/CANONICAL_V2_SEMANTIC_MASK_PROTOCOL.md`.

V2 has 29 node features and a 2,335-dimensional Flat serialization. With the
current 128-dimensional Flat network, use `--gcn-hidden-dim 40` for the matched
V2 GCN comparison (379,891 versus 380,811 trainable parameters).

Validate that schema and mask remain independent before training:

```bash
uvr python -m scripts.analysis.validate_canonical_v2_interventions
```

Add these options to either canonical Flat or canonical GCN training command:

```text
--canonical-schema v2
--passive-memory-ttl 20
--analysis-memory-ttl 20
--semantic-action-mask
```

## Inspect CAGE 4

```bash
uvr python -m scripts.inspection.inspect_cc4_env
```

## Train Official Flat

```bash
uvr python -m scripts.training.train_flat \
  --total-simulator-steps 100000 \
  --episode-steps 500 \
  --reward-scale 0.01 \
  --seed 42 \
  --device cpu \
  --run-dir runs/flat_cc4
```

## Train Canonical Flat

This command uses the same policy, rollout, GAE, PPO update and evaluation code
as the official Flat run. Only the environment adapter and observation width
change.

```bash
uvr python -m scripts.training.train_canonical_flat \
  --canonical-variant action_aware \
  --evidence-age-horizon 50 \
  --total-simulator-steps 100000 \
  --episode-steps 500 \
  --reward-scale 0.01 \
  --seed 42 \
  --device cpu \
  --run-dir runs/canonical_flat_cc4
```

This is a learning-qualification run, not a final encoder benchmark. The
canonical representation is qualified only if trained Flat consistently beats
untrained/random baselines on held-out seeds.

## Evaluate Checkpoints

Official Flat:

```bash
uvr python -m scripts.evaluation.evaluate_flat \
  --run-dir runs/flat_cc4 \
  --seed 42 \
  --eval-episodes 100 \
  --eval-seed 10000
```

Canonical Flat:

```bash
uvr python -m scripts.evaluation.evaluate_canonical_flat \
  --run-dir runs/canonical_flat_cc4 \
  --seed 42 \
  --eval-episodes 100 \
  --eval-seed 10000
```

## Train Canonical GCN

The 28-dimensional GCN configuration has approximately the same parameter
count as the current canonical Flat policy.

```bash
PYG_HOME=/tmp/pyg uvr python -m scripts.training.train_canonical_gcn \
  --canonical-variant action_aware \
  --evidence-age-horizon 50 \
  --total-simulator-steps 100000 \
  --episode-steps 400 \
  --gcn-hidden-dim 28 \
  --gcn-layers 2 \
  --reward-scale 0.01 \
  --entropy-coef 0.001 \
  --target-kl 0.015 \
  --seed 42 \
  --device cpu \
  --eval-every-updates 0 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --run-dir runs/canonical_gcn_v1_qualification_100k
```

## Random-Pivot Train/Test Distribution

Official CAGE 4 fixes every cross-subnet lateral-movement information link at
`server_host_0`. The `random_host` distribution keeps the logical subnet graph
and all benchmark controls fixed, but samples the actual pivot host from the
existing non-router hosts in each subnet for every episode. This removes the
fixed pivot-slot shortcut without changing canonical or action dimensions.

Audit the exact 100k train and held-out test seed ranges:

```bash
uvr python -m scripts.analysis.audit_cc4_pivot_distribution \
  --train-seed 42 \
  --train-episodes 251 \
  --test-seed 10000 \
  --test-episodes 30 \
  --episode-steps 400 \
  --output results/cc4_random_pivot_distribution_audit.json
```

Add this option to both matched Flat and GCN training commands:

```bash
--environment-distribution random_host
```

Evaluation reconstructs the distribution stored in `config.json`. To run the
secondary official-CAGE-4 stress test without retraining, add:

```bash
--environment-distribution official --output-prefix evaluation_official
```

The complete split, controls, qualification gates and stop rule are recorded
in `results/CC4_RANDOM_PIVOT_DISTRIBUTION_PROTOCOL.md`.

Summarize the paired full-versus-static gate after evaluation:

```bash
uvr python -m scripts.analysis.summarize_observation_use_gate \
  --run-dir runs/canonical_gcn_random_pivot_qualification_20k \
  --seed 42
```

Evaluate a saved GCN checkpoint:

```bash
PYG_HOME=/tmp/pyg uvr python -m scripts.evaluation.evaluate_canonical_gcn \
  --run-dir runs/canonical_gcn_v1_qualification_100k \
  --seed 42 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --stochastic-action-seed 10000
```

Run all feature-group gates using one checkpoint load:

```bash
PYG_HOME=/tmp/pyg uvr python \
  -m scripts.evaluation.evaluate_canonical_observation_gate \
  --run-dir runs/canonical_gcn_v1_qualification_100k \
  --seed 42 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --stochastic-action-seed 10000 \
  --transform all \
  --mode stochastic
```

`PYG_HOME` gives PyTorch Geometric a writable location for its generated
message-passing cache. It does not change the model or experiment semantics.

The evaluation scripts report deterministic argmax actions and stochastic samples on
paired environment seeds. The run configuration determines the canonical
variant used to reconstruct the evaluation environment.

## Canonical Learning Qualification

Evaluate the reproducibly initialized canonical policy before training:

```bash
uvr python -m scripts.evaluation.evaluate_untrained_canonical_flat \
  --run-dir runs/canonical_flat_qualification_84k \
  --seed 42 \
  --eval-episodes 100 \
  --eval-seed 10000 \
  --episode-steps 400 \
  --mode deterministic \
  --output-prefix untrained_heldout_evaluation
```

Evaluate a best checkpoint without replacing the final-checkpoint JSON files:

```bash
uvr python -m scripts.evaluation.evaluate_canonical_flat \
  --run-dir runs/canonical_flat_qualification_84k \
  --seed 42 \
  --eval-episodes 100 \
  --eval-seed 10000 \
  --checkpoint-path \
    runs/canonical_flat_qualification_84k/seed_42/best_checkpoint.pt \
  --output-prefix best_evaluation
```

Test whether the best policy uses dynamic canonical evidence:

```bash
uvr python -m scripts.evaluation.evaluate_canonical_observation_gate \
  --run-dir runs/canonical_flat_qualification_84k \
  --seed 42 \
  --eval-episodes 100 \
  --eval-seed 10000 \
  --transform static_only
```

`static_only` preserves structural identity, topology, global features and the
official action mask while zeroing all current and remembered cyber evidence.
It is an observation-use diagnostic, not a training condition.

## TensorBoard

```bash
uvr tensorboard --logdir runs
```

Training stores each seed under `<run-dir>/seed_<seed>/`, including config,
checkpoint, final evaluations, periodic evaluations and TensorBoard events.

## Validation

Run the current canonical/PPO integration tests:

```bash
uvr pytest -q \
  tests/test_canonical_observation.py \
  tests/test_cc4_canonical_env.py \
  tests/test_canonical_ppo_integration.py \
  tests/test_best_evaluation_checkpoint.py
```

Some retained CAGE 2 patch tests still import APIs that no longer exist in the
vendored CAGE 4 tree. Therefore, `pytest tests -q` currently fails during test
collection for those legacy tests; this is separate from the canonical/PPO
integration.
