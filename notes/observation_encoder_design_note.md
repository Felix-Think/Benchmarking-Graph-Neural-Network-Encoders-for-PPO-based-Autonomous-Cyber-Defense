# Observation Encoder Design Note

Last updated: 2026-07-14

## Purpose

This note records the implemented observation encoder used as the
dynamic-signal source for the scalable protocol. The goal is to make raw Blue observations
usable by flat and GNN policies without giving either side extra host-state
information.

The encoder is implemented in:

- `src/acd_benchmark/wrappers/shared/host_obs.py`
- `src/acd_benchmark/wrappers/shared/graph_obs.py`
- `src/acd_benchmark/wrappers/shared/flat_obs.py`
- `src/acd_benchmark/wrappers/shared/partial_obs.py`
- `src/acd_benchmark/wrappers/stage1/observation.py`

The audit script is:

- `scripts/inspection/audit_observation_encoder.py`

## Core Decision

Use raw Blue observations directly instead of `BlueTableWrapper` or
`ChallengeWrapper` vector observations.

Reason:

- `BlueTableWrapper` table mode keeps `Subnet`, `IP Address`, `Hostname`,
  `Activity`, and `Compromised`, but vector mode keeps only compressed
  `Activity` and `Compromised` bits.
- The project needs to compare flat and graph encoders, so wrapper-induced
  compression should not decide the result before the encoder comparison.
- Raw Blue observations let us define a matched graph and flat observation from
  the same visible data.

Safe claim:

> We implement a raw-observation host-level encoder that constructs matched
> graph and flat observations from the same Blue-observable host information.

Unsafe claim:

> We recover the full network topology from raw Blue observations.

The current graph only uses shared observed subnets, not full router/gateway or
firewall topology.

## Data Source

The encoder uses the following raw Blue observation groups:

- `System info`
- `Interface`
- `Sessions`
- `Processes`
- `User Info`

It does not use hidden simulator true state.

Top-level metadata such as `success` is separated from host observations by
`split_blue_observation`. The current host encoder does not use `success` as a
host feature.

## Host-Level Representation

Current graph level:

- One host = one graph node.
- Node features summarize the raw host dictionary.
- Edge = directed same-subnet relation between two different hosts.

The graph is directed in tensor form because `edge_index` stores both ordered
directions:

- `A -> B`
- `B -> A`

Conceptually this is an undirected same-subnet relation represented as two
directed edges.

## Schema And Cache

`HostFeatureSchema` stores the fixed observation contract:

- host order
- feature order
- dynamic feature indices
- observed subnet categories
- categorical vocabularies

`RawBlueHostEncoder.latest_hosts` stores the latest known raw host information.
This is necessary because CybORG step observations are sparse: a host missing
from one step does not mean the host disappeared.

Important terminology:

- `known` means the host has cached data in `latest_hosts`.
- `known` does not mean the host appeared in the latest sparse step observation.

## Feature Design

The full official-CC2 inspection schema currently has 36 features per host.

Static categorical fields are one-hot encoded:

- `system.Architecture`
- `system.OSType`
- `system.OSDistribution`
- `system.OSVersion`
- `interface.Subnet`

Static/structural boolean feature:

- `interface.has_interface`

Dynamic list fields are summarized with count or flag features instead of
one-hotting whole dictionaries:

- `sessions.count_norm`
- `sessions.blue_count_norm`
- `sessions.red_count_norm`
- `sessions.has_velociraptor_client`
- `sessions.has_privileged_user`
- `processes.count_norm`
- `processes.unique_user_count_norm`
- `processes.has_privileged_user`
- `processes.new_count_norm`
- `activity.is_scan`
- `activity.is_exploit`
- `compromised.is_unknown`
- `compromised.is_user`
- `compromised.is_privileged`
- `user_info.count_norm`
- `user_info.has_default_password`
- `user_info.has_privileged_user`
- `user_info.has_green_agent`
- `user_info.has_password_exposed`

Reason:

- `Sessions`, `Processes`, and `User Info` are lists of dictionaries with
  variable length.
- PPO/GNN inputs need stable dimensions.
- Counts and flags preserve simple risk signals without making the schema grow
  with every PID, path, session id, or username.

## Count Normalization

Counts are normalized as:

```python
min(count / 10, 1.0)
```

This is a benchmark heuristic, not an optimal normalization claim.

Reason:

- keeps count features in the same rough scale as one-hot and boolean features;
- avoids large raw counts dominating the vector;
- keeps the encoder simple and easy to audit.

Risk:

- `cap=10` is hand chosen;
- values above 10 collapse to 1.0;
- if true counts are usually 0, 1, or 2, the signal may be small.

Future benchmark should treat feature scaling as an ablation or threat to
validity.

## Graph Observation

`graph_obs` contains:

- `node_features`: shape `[num_hosts, feature_dim]`
- `adjacency`: shape `[num_hosts, num_hosts]`
- `edge_index`: shape `[2, num_edges]`
- `hostnames`
- `feature_names`
- `dynamic_feature_indices`

For the current full official-CC2 inspection observation:

- `num_hosts = 13`
- `feature_dim = 36`
- `adjacency shape = [13, 13]`
- `edge_index shape = [2, 44]`

The edge rule is:

```text
if two distinct hosts share the same known Interface.Subnet:
    adjacency[src, dst] = 1.0
```

This creates same-subnet components. It does not encode gateway/router links
between different subnets.

## Flat Observation

`flat_obs` is built from the same graph observation:

```text
flat_obs = flatten(node_features) + flatten(adjacency)
```

For the current full official-CC2 inspection observation:

```text
13 * 36 + 13 * 13 = 468 + 169 = 637
```

Reason:

- the flat baseline sees the same host features as the graph encoder;
- the flat baseline also sees equivalent topology information through the
  flattened adjacency matrix;
- this reduces the confound that GNN performs better only because it receives
  topology while flat does not.

## Partial Observation Masking

`partial_obs.py` contains two masking layers:

- diagnostic group masking for `sessions`, `processes`, and `user_info`;
- Stage 1 Bernoulli host visibility masking over the five semantic dynamic
  features, with an explicit `visibility.host_visible` bit.

Diagnostic masks default to `-1.0`; the frozen Stage 1 protocol fills hidden
dynamic features with `0.0` and preserves static features and topology.

Neither mechanism changes host count or graph topology.

## Audit Status

The current audit script checks:

- expected host features computed from raw `latest_hosts`;
- `graph_obs.node_features`;
- same-subnet adjacency;
- `edge_index` equals nonzero adjacency positions;
- `flat_obs` contains exactly flattened node features plus flattened adjacency.

Command:

```bash
uv run python -m scripts.inspection.audit_observation_encoder --steps 5 --max-hosts 13
```

Latest observed status:

```text
PASSED
```

Report path:

- `runs/observation_encoder_audit/report.md`

## Known Limitations

1. Same-subnet graph is not full network topology.

The encoder does not currently represent router, gateway, firewall, ACL, or
inter-subnet reachability. The graph should be described as a host-level
same-subnet graph.

2. `Processes` may be event-like in sparse step observations.

CybORG step observations may include partial process/connection information
rather than a complete process table. Therefore process count features should
be interpreted as observed process evidence, not guaranteed full host process
state.

3. Subnet one-hot categories are seed-specific.

Observed subnet strings such as `10.0.x.x/28` may change across seeds/resets.
Before multi-seed benchmark claims, consider:

- removing subnet one-hot while keeping adjacency;
- mapping subnets to stable zone roles;
- treating subnet encoding as an ablation.

4. `known` is cache-based.

`known=1` indicates that the host has cached data from the reset/latest-known
belief. It is not a current-step visibility flag.

5. Count normalization is heuristic.

`min(count / 10, 1.0)` is suitable for this encoder but should not be presented as
the only valid preprocessing.

## Decisions Applied In Training

- The archived official-CC2 pilot trained on the five
  `SEMANTIC_FEATURE_NAMES`; its flat policy also received dense adjacency.
- Stage 1 replaces raw IP/subnet categories with frozen role, zone, and
  criticality features, then appends the same five semantic dynamic features
  and one visibility bit.
- Full-schema counts and one-hots remain implemented for audit and ablation,
  but they are not the final scalable-protocol policy input.
