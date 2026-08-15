# Benchmarking GNN Encoders for PPO-based Cyber Defense

Code workspace for the research project **Benchmarking Graph Neural Network
Encoders for PPO-based Autonomous Cyber Defense under Partial Observability**.

## Current Status

The previous benchmark implementation was removed intentionally. A minimal
native CAGE-2 Flat PPO pipeline is now being rebuilt incrementally. The active
pipeline currently includes the BlueTable observation wrapper, a 41-action
catalogue, Flat actor-critic policy, rollout collection, GAE, PPO update,
single-environment training, deterministic evaluation, and checkpoint support.

Historical run artifacts and design notes are retained for comparison. They are
not an active or reproducible pipeline after this reset.

## Retained Foundation

```text
.
|-- CybORG/                  vendored CybORG/CybORG++ simulator
|-- CybORG_UPSTREAM.md       simulator source and patch identity
|-- patches/                 local CybORG++ patch documentation
|-- notes/                   historical research and design notes
|-- runs/                    historical generated results; reference only
|-- scripts/inspection/      simulator inspection utilities
|-- src/acd_benchmark/cyber/ minimal CAGE-2 environment helpers
|-- src/acd_benchmark/rl/    Flat policy, rollout, PPO train and evaluation
|-- tests/                   simulator and PPO regression tests
|-- pyproject.toml           Python dependencies
`-- uv.lock                  locked environment
```

The current code is a learning qualification pipeline, not yet the main encoder
benchmark. GNN comparison, mixed observability, topology generalisation and the
final experiment runner remain intentionally out of scope at this stage.

## Environment

Required Python version: `3.10`.

```bash
uv sync --all-groups
```

The local `CybORG/` directory is installed as an editable dependency. Do not
replace it with a separate PyPI installation because the local copy contains
the selected CybORG++ fixes.

## Simulator Checks

Run a short headless CAGE-2 episode:

```bash
uv run --python 3.10 python -m scripts.inspection.smoke_cyborg_cc2 --steps 5
```

Inspect the native Blue action space and raw reset observation:

```bash
uv run --python 3.10 python -m scripts.inspection.inspect_cc2_spaces
```

Run only the retained simulator and patch tests:

```bash
uv run --python 3.10 pytest tests -q
```

## Development Rule

New modules should be added only after their input, output, ownership, and test
contract are understood. The recommended implementation order is:

1. native CAGE-2 reset/step loop;
2. explicit Blue observation parser;
3. explicit action catalogue and action execution;
4. rollout data structure;
5. minimal Flat policy;
6. PPO loss and update;
7. training and deterministic evaluation;
8. qualification gates;
9. GNN encoder and partial-observation experiments.

Do not reintroduce GNN or mixed observability until the independently written
Flat PPO policy demonstrably learns from native CAGE-2 observations.
