# Canonical V2 Observation Qualification Protocol

## Objective

Before comparing Flat, GCN and GraphSAGE, establish that the canonical
observation supports PPO learning and that sensor evidence adds held-out utility.
This protocol does not optimize leaderboard reward and does not claim GNN
superiority.

## Reference Calibration

The unmodified pretrained KEEP/Cybermonic submission was evaluated sequentially
on the local CybORG v4.0 installation. A two-episode, 500-tick smoke evaluation
at environment seed 30000 produced mean return -117.5 and standard deviation
30.41. This is not a statistical estimate, but it confirms that the reference
code, weights and current environment are operationally compatible.

## Controlled Training Arms

Two Flat PPO arms are initialized and trained independently for every seed:

1. `Full`: complete canonical v2 observation.
2. `No evidence`: sensor outputs and beliefs derived from sensor outputs are
   zeroed during both training and evaluation.

`No evidence` retains decoy state, firewall state, mission phase, action success,
Analyse completion/history, topology, host identity and the official CAGE 4
action mask. File density is removed because it is the sensor result returned by
Analyse, not merely action feedback.

The following remain identical between arms: PPO implementation and
hyperparameters, shared policy, network architecture and parameter count,
native reward, official action mask, action catalogue, environment distribution,
500-tick episode setting, training budget and evaluation seeds.

Every training process must also set `PYTHONHASHSEED` to the training seed and
set both `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to one. These process-level
controls are recorded in `config.json`. A two-replica 10k-tick test produced
identical episode returns, PPO histories and policy-weight SHA-256 hashes.

## Inherited Reference Practices

- Official 500-tick CAGE 4 episodes.
- Complete action-duration reward accumulation before storing a decision.
- Several complete trajectories per PPO update. Episodes are collected
  sequentially to keep only one policy process active.
- Temporal v2 state separates passive alerts from Analyse history and applies
  bounded memory TTLs.
- Stochastic policy evaluation is the primary result, matching the KEEP code's
  default sampling behavior.

The common node/edge/global action scorer is now inherited for both the
no-message-passing control and the GNN encoders. It scores host actions from host
embeddings, traffic actions from subnet-pair embeddings, and global actions from
the shared context. Cybermonic's five independent policies, variable port/file
entity nodes, inter-agent messages and local reward shaping remain excluded.
Importing them together would change multi-agent factorization, observation
information, total parameter count and reward semantics at the same time.

Canonical parser v4 adds only relational attributes available in raw Blue
observations: connection direction, local/remote port classes, whether a remote
address has been observed, and its subnet when that mapping was locally visible.
Each Blue agent maintains its own IP-to-subnet map; observations are not merged
across agents. The fixed host slots preserve the same information for the
NodeMLP and future GNN encoder.

## Budget and Evaluation

- Training seeds: 42, 43, 44, 45 and 46.
- Requested budget: 300,000 simulator ticks per arm and seed.
- Episodes per PPO update: 4.
- Validation: 10 fixed episodes every 25 updates, stochastic actions, seeds
  beginning at 20,000.
- Held-out test: 30 episodes, stochastic actions, seeds 10,000-10,029.
- Final checkpoint is the primary checkpoint. Periodic validation is used only
  to inspect the learning curve and plateau; it must not select the reported
  test result.

## Predeclared Gates

The independent inference unit is the training seed, not the evaluation episode.

1. **Integrity:** all controlled configuration fields match and all optimization
   metrics are finite.
2. **PPO learning:** trained `Full` beats its identically initialized untrained
   policy in at least four of five seeds and the 95% Student-t interval across
   seed-level paired effects is above zero.
3. **Training improvement:** the last 20% of `Full` training episodes exceeds the
   first 20% in at least four of five seeds and its seed-level 95% interval is
   above zero.
4. **Evidence utility:** trained `Full` beats independently trained
   `No evidence` in at least four of five seeds and the seed-level 95% interval
   is above zero.
5. **Plateau diagnostic:** using the last three fixed-seed validation points per
   seed, the multi-seed late-slope interval includes zero and the absolute mean
   slope is at most 50 return points per 100,000 simulator ticks.

The plateau rule is an operational diagnostic, not a mathematical convergence
proof. Failure of the evidence gate blocks encoder claims but does not imply that
the environment contains no useful evidence; it may identify the state estimator,
multi-agent policy factorization or training budget as the next bottleneck.

## Intervention Order After A Failed Gate

Do not import all Cybermonic components at once. Apply the following order:

1. Train the matched shared-policy `Full`/`No evidence` pair for the declared
   budget and collect at least three late validation points.
2. If evidence utility remains weak, test five independent local NodeMLP
   policies against five independent no-evidence controls. Match per-agent and
   total parameter counts across later encoders.
3. If local specialization still fails, replace host-level relational OR
   aggregation with bounded, fixed event slots that preserve each observed
   connection's host, direction, port class and remote-subnet relation. Flatten
   the same slots for NodeMLP and expose the same edges to GNNs.
4. Treat messages as a separate communication ablation. Do not add
   EnterpriseMAE compromise/scanned bits to the strict partial-observation main
   benchmark because that changes the observation boundary.

## Five Diagnostic Conditions

After matched training, checkpoints may be evaluated under five dimension-
preserving conditions. Every condition retains the official mask:

1. `Full`.
2. `No evidence`.
3. `Action-state only`: decoy, firewall, action success and mission phase.
4. `Topology only`: static topology and host identity.
5. `Zero + mask`: policy biases and official action mask only.

Only `Full` and `No evidence` are trained from scratch in this qualification.
The other three are diagnostic interventions, not separately optimized agents.
