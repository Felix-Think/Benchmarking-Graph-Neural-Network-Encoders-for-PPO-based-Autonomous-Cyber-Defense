# Pilot2 Legacy Reference

## Identity

The deleted Stage1/Pilot2 source was restored as a detached Git worktree at:

```text
../Pilot2_Legacy_Reference
```

It is pinned to commit `4e0154a` (`prepare reproducible Pilot 2 benchmark`).
This worktree is a read-only engineering reference. It is not an active
protocol and its old outputs must not be used as evidence for the current
official-CAGE-2 benchmark.

Recreate it if needed:

```bash
git worktree add --detach ../Pilot2_Legacy_Reference 4e0154a
```

## Relevant Legacy Path

The behavior of interest is mainly in:

- `src/acd_benchmark/wrappers/shared/host_obs.py`;
- `src/acd_benchmark/wrappers/pilot2/env.py`;
- `src/acd_benchmark/wrappers/pilot2/observation.py`.

`RawBlueHostEncoder` maintained `latest_hosts`, a persistent compromise state,
and selected-action feedback. Restore changed the cached state to clean;
Remove changed a suspected compromise to unknown. Sparse observations were
merged into the latest-known host cache.

## Important Confound In The Old Order

The old step order was:

1. encode and merge the raw native Blue observation into a persistent cache;
2. update action-conditioned security state;
3. apply the synthetic fixed-budget host-retention mask.

Therefore native telemetry could enter the cache before the wrapper hid that
host from the policy at the same timestep. Later observations could expose
information retained from a synthetically hidden step. This does not prove
that old results were invalid, but it prevents the old implementation from
being copied wholesale as a clean partial-observability treatment.

Other legacy differences also prevent direct result comparison: a custom
16-host scalable scenario, a same-subnet graph construction, a different
action surface, older PPO settings, and old evaluation contracts.

## What Was Ported

Only a compact five-state host security belief was ported:

- unseen;
- clean;
- uncertain;
- user compromise;
- privileged compromise.

The new implementation is
`src/acd_benchmark/wrappers/pilot/belief.py`. It updates from the exact
post-retention policy observation. It can use Blue's selected Restore/Remove
action as internal predictive feedback, but it cannot read action success,
simulator true state, or evaluator metrics.

The full host cache, stale process/session lists, pre-mask telemetry updates,
legacy topology, and legacy environment were deliberately not ported.

## Interpretation Boundary

The restored code supports code-level comparison only. A new bridge result can
test whether observable temporal belief improves learning, but it cannot by
itself prove that temporal belief was the sole cause of the old Stage1/Pilot2
behavior.
