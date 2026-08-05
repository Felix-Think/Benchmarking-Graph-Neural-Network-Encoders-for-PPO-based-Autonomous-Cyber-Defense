# Stage 1 Protocol - Scalable 16-Host Partial Observability

## Scientific question

Stage 1 compares a flat MLP encoder and a GraphSAGE encoder while holding the
simulator, observations, action mapping, reward, PPO implementation, training
budget, evaluation seeds, and static topology constant. A difference in result
is therefore attributed to encoder inductive bias only within the limits listed
under Threats and Deviations.

This is a 12-run matrix:

- encoder: `flat`, `graphsage`
- dynamic host visibility: `100%`, `25%`
- training seeds: `0`, `1`, `2`
- budget: `1,000,000` environment timesteps per run
- evaluation: 30 deterministic episodes at step 0 and every 100,000 steps

Training seeds alter initialization, action sampling, simulator training streams,
and masking streams. Evaluation uses one common fixed set of 30 environment and
mask seeds for every matrix cell, preventing evaluation-scenario variance from
being folded into the across-training-seed comparison.

The 50% masking path is implemented and tested for later stages, but it is not
part of the Proposal's Stage 1 matrix.

## Scalable scenario

`configs/scenarios/cage2_16.yaml` expands the official 13-host Scenario 2 to 16
hosts while retaining the CAGE-2 host images, agents, rewards, Blue
Velociraptor clients, operational firewall setting, and Meander Red agent. The
frozen host order is:

1. 5 Enterprise hosts: `Defender`, `Enterprise0` to `Enterprise3`
2. 5 Operational hosts: `Op_Server0`, `Op_Host0` to `Op_Host3`
3. 6 User hosts: `User0` to `User5`

The graph contains directed edges in both directions for every conceptual
undirected relation. Hosts within each subnet form a clique. Every User host is
connected to `Enterprise0`; `Enterprise0` is connected to `Op_Host0`. This gives
84 directed edges and one connected graph.

This gateway graph is an explicit benchmark construction, not topology learned
from raw Blue observations and not a claim about a physical router recorded by
CybORG. Flat receives the identical 16x16 adjacency matrix, so topology data are
matched even though only GraphSAGE applies message passing to it.

## Matched observations

Each host has 15 features in a fixed order:

- 5 static role one-hots
- 3 static subnet one-hots
- 1 static criticality bit
- 5 dynamic semantic Blue-observation features: scan, exploit, unknown
  compromise, user compromise, privileged compromise
- 1 host-visibility bit

The graph observation is `[16, 15]` plus an edge index and the frozen adjacency.
The flat observation is exactly the same node matrix flattened in canonical host
order followed by the same adjacency flattened, for 496 values. Neither policy
receives simulator true state.

## Partial-observation intervention

At every timestep, host visibility is sampled independently from a Bernoulli
distribution with probability 1.0, 0.5, or 0.25. If a host is hidden, only its
five dynamic features are replaced with zero. Role, subnet, criticality,
adjacency, and the visibility bit remain observable.

The host targeted by the current or previous Blue action is forced visible,
implementing the Proposal's action-feedback exception. Mask RNGs are seeded and
separate from the simulator seed stream. Flat and GraphSAGE use the same masking
implementation, but each independent run receives its own deterministic stream.

## Shared PPO core

Both policies use the modular shared core in `acd_benchmark.rl.ppo` and
`acd_benchmark.rl.rollout` with identical PPO hyperparameters and synchronous
8-environment rollouts. Scripts and tests import the owning modules directly;
there is no aggregate compatibility facade. Environments are stepped
sequentially inside one process; policy minibatches are vectorized.
Therefore, `num_envs=8` is a sampling/vectorization design and must not be
described as eight multiprocessing workers.

GraphSAGE uses two `SAGEConv` layers. Node embeddings are flattened in canonical
host order before the actor and critic heads. This deliberately differs from a
pure global pooled readout: global pooling would erase hostname identity even
though 177 of 178 Blue actions target a specific host. The ordered readout keeps
the action problem identifiable, but it also means this Stage 1 policy is fixed
to 16 hosts and does not itself test zero-shot topology-size generalization.

## Evaluation-only metrics

True simulator state is read only after policy actions for evaluation logging:

- reward: CybORG episode return
- mission availability: fraction of timesteps where the critical `OTService`
  exists and is active
- privileged compromised hosts: mean number of hosts with active Red `root`
  or Windows `SYSTEM` sessions per timestep
- Red-presence hosts: mean number of hosts with any active Red session
- detection latency: first Red presence to first Blue Analyse, Remove, or
  Restore targeting that host
- recovery latency: first Red presence to disappearance of all Red sessions on
  that host
- operational-server reach time/rate
- training wall time, training throughput, process RSS, accelerator memory when
  available, parameter count, and deterministic-policy inference latency

Undetected and unrecovered compromises are right-censored at the episode horizon
and included using the remaining-horizon value. Reports retain observed and
censored counts. Latency must not be interpreted without those counts; a lower
censored mean is not automatically better detection.

## Reproducibility artifacts

Every run stores resolved and source YAML, generated scenario, package versions,
hardware/software metadata, Git commit and dirty status, a source snapshot plus
tracked Git binary diff, raw evaluation JSON/CSV, per-update JSONL, periodic
checkpoints, latest/best-reward/best-availability checkpoints, report JSON/MD,
progress, and a SHA-256 manifest.

Checkpoints contain model and optimizer state, seed/config metadata, metrics,
and an explicit `normalizer_statistics: null` because no learned normalizer is
used. They support independent reconstruction and evaluation. They do not
support bit-exact mid-rollout resume because live CybORG states for all eight
environments are not serialized.

## Threats and deviations

- The 16-host topology is a controlled CAGE-2-style extension, not an official
  DARPA scenario. Results must be labelled accordingly.
- Added hosts clone official role-compatible host definitions; this preserves
  semantics but reduces host-image diversity.
- Static gateway edges are benchmark assumptions. Supplying the same adjacency
  to Flat controls information access, but not how efficiently each architecture
  uses relational structure.
- Flat and GraphSAGE do not have equal parameter counts. Results compare the
  specified architectures, not parameter-matched capacity.
- Ordered GraphSAGE readout is necessary for host-specific actions but gives up
  size-invariant pooling. Topology generalization requires a later host-wise
  action head or another permutation-aware readout.
- The repository is currently dirty. Run snapshots and patches preserve the
  actual source, but a clean tagged commit remains preferable for final paper
  release.
- Gym and this CybORG version emit deprecation warnings. Tests pass, but the
  dependency stack should remain pinned for these experiments.

## Post-run protocol notes

The 12-run matrix completed, but one deterministic GraphSAGE seed-2 trajectory
exposed an upstream `StopProcess` bug after 700k steps. `get_session_from_pid()`
returned an integer ID that was treated as a session object; after resolving the
ID, the reload path also had to use the real `Session.username` field. A
regression test using a real CybORG `Session` now covers this path.

Nine completed runs retain the pre-fix CybORG patch snapshot and three retain
the post-fix snapshot. The core benchmark, scenario, and configuration hashes
are identical. The changed branch cannot have executed in a pre-fix completed
run because it deterministically raises when reached. This is therefore a
documented execution-compatible hotfix, but the two source cohorts must still be
reported rather than described as one byte-identical simulator build.

Final and intermediate action audits also found reward-mission misalignment.
Policies can repeatedly choose `Restore(Op_Server0)`, improving or preserving
CybORG return while keeping the critical service unavailable. Reward alone is
therefore not a valid policy ranking for this pilot; mission availability and
compromise metrics are necessary co-primary evidence.
