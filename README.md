# Benchmarking GNN Encoders For PPO-based Cyber Defense

Code thi nghiem cho de tai **Benchmarking Graph Neural Network Encoders for
PPO-based Autonomous Cyber Defense under Partial Observability** tren
CybORG/CAGE-2.

## Trang Thai

Protocol hien hanh la Pilot 2-A: frozen 16-host CAGE-2-style scenario, Flat MLP
va GraphSAGE, mixed-q training, fixed-q evaluation tai 100%/75%/50%/25%, 3
seeds va 1M timesteps moi run.

Stage 1 cu duoc giu lam engineering evidence. Pilot 2-A van la qualification,
chua phai bang chung cuoi cung rang GNN tot hon Flat. Topology generalisation bi
block cho den khi graph encoder va simulator reachability dung cung mot nguon
topology.

## Nguyen Tac Benchmark

- Raw Blue observation la nguon dynamic state cua policy.
- Flat va graph nhan cung dynamic/static host features. Pilot 2-A Flat khong
  nhan adjacency; day la practical vector baseline, khong phai strict
  information-parity control.
- Flat, GCN va GraphSAGE dung chung custom PPO core.
- Simulator true state chi duoc doc de tinh evaluation metrics.
- Partial observability chi mask dynamic host state; frozen topology va static
  host descriptors van hien thi.
- Pilot 2-A khong dung action-validity mask va khong force action target visible.
- Policy schema hien tai khong nhan top-level action result; result chi duoc log
  doi xung cho moi encoder. Pilot 2-A khong dung learned normalizer.
- Reward khong duoc dung mot minh de ket luan policy tot.

## Cau Truc Code

```text
configs/
  stage1_16.yaml                 frozen Stage 1 protocol
  pilot2_observation_16.yaml     active Pilot 2-A protocol
  scenarios/cage2_16.yaml        generated 16-host scenario
notes/
  code_reading_guide.md          thu tu doc source
  observation_encoder_design_note.md
  stage1_protocol.md
  pilot2_protocol.md             full Pilot 2-A experiment contract
  topology_coupling_gate.md      blocker before structural generalisation
scripts/
  inspection/                    CybORG and observation inspection/audit
  stage1/                        train, matrix, evaluation, audit, analysis
  pilot2/                        matrix, baselines, summary, protocol audit
src/acd_benchmark/
  cyber/                         CybORG factories, scenarios, actions
  wrappers/
    shared/                      raw encoder, matched views, masking
    stage1/                      scalable observations and environments
    pilot2/                      mixed-q fixed-budget observation process
  rl/
    policies/                   Flat, GCN, GraphSAGE actor-critics
    rollout/                    on-policy experience collection
    ppo/                        GAE, returns, loss, update, evaluation
  evaluation/                    evaluation-only operational metrics
  utils/                         reproducibility artifacts
tests/                            focused regression and protocol tests
runs/
  archive/stage0_official_cage2/ archived pilot evidence
  stage1_scalable_16/            canonical Stage 1 evidence
  pilot2_observation_16/         active Pilot 2-A output
```

Bat dau doc code tai [notes/code_reading_guide.md](notes/code_reading_guide.md).

## Wrapper Map

```text
raw Blue dict
  -> shared/host_obs.py       fixed schema + sparse latest-known cache
  -> shared/graph_obs.py      GraphObservation
       -> shared/flat_obs.py  deterministic vector view

GraphObservation + frozen 16-host topology
  -> shared/partial_obs.py    host visibility mask
  -> stage1/observation.py    static + semantic + visibility schema
  -> stage1/env.py            matched Stage 1 Graph/Flat environments
  -> pilot2/observation.py    exact fixed-budget stratified host mask
  -> pilot2/env.py            mixed-q train and fixed-q evaluation views
```

Thu muc `wrappers/` khong chua PPO. No chi quyet dinh policy duoc thay gi va
du lieu do co dang graph hay vector.

Thu muc `rl/policies/` la diem mo rong encoder. Them mot GNN moi o day; rollout
va PPO mathematics nam rieng de khong bi sao chep hoac thay doi ngoai y muon.

## Cai Dat

Yeu cau Python 3.10 va `uv`:

```bash
uv sync --all-groups
```

CybORG duoc cai editable tu thu muc local `CybORG/`. Dependency duoc pin trong
`pyproject.toml` va `uv.lock`.

## Inspection Va Tests

```bash
uv run python -m scripts.inspection.smoke_cyborg_cc2
uv run python -m scripts.inspection.inspect_cc2_spaces
uv run python -m scripts.inspection.inspect_graph_flat_obs --steps 5
uv run python -m scripts.inspection.audit_observation_encoder --steps 5 --max-hosts 13
uv run pytest tests -q
uv run ruff check src scripts tests
```

## Stage 1

Kiem tra matrix ma khong train:

```bash
uv run python -m scripts.stage1.run_matrix --dry-run
```

Chay/resume frozen matrix:

```bash
uv run python -m scripts.stage1.run_matrix
```

Tai tao summary va audit:

```bash
uv run python -m scripts.stage1.summarize_results
uv run python -m scripts.stage1.audit_protocol
uv run python -m scripts.stage1.analyze_results
```

## Pilot 2-A Observation Qualification

Doc contract truoc khi chay:

```bash
less notes/pilot2_protocol.md
```

Kiem tra dung 6 cells ma khong train:

```bash
uv run --python 3.10 python -m scripts.pilot2.run_matrix --dry-run
```

Chay/resume Pilot 2-A, sau do tu dong chay baselines, summary va audit:

```bash
uv run --python 3.10 python -m scripts.pilot2.run_matrix
```

Ket qua can doc dau tien:

```bash
less runs/pilot2_observation_16/pilot2_protocol_audit.md
less runs/pilot2_observation_16/pilot2_summary.md
```

Audit tu dong fail cho protocol-integrity error. Scientific go/no van can doc
paired uncertainty va ba metric availability, return, privileged-compromise
cung nhau. Moi `report.json` luu `training_q_exposure` de kiem tra so episode va
timestep thuc te tai tung q. q=25% la stress/floor diagnostic, khong phai hard
trainability gate.

Main encoder benchmark va topology Pilot 2-B chua duoc chay. Mo track topology
chi sau khi dat cac dieu kien trong `notes/topology_coupling_gate.md`.

## Ket Qua Canonical

- Stage 1: `runs/stage1_scalable_16/stage1_summary.md`
- Stage 1 protocol audit: `runs/stage1_scalable_16/protocol_audit.md`
- Stage 1 diagnostics: `runs/stage1_scalable_16/stage1_diagnostics.md`

Moi run Stage 1 luu config, checkpoint, raw metrics, environment metadata,
source snapshot va SHA-256 manifest. Khong sua source snapshot trong `runs/`.
Ket qua official-CC2 pilot cu duoc giu read-only trong
`runs/archive/stage0_official_cage2/`; source va scripts cua protocol do khong
con nam trong active code path.

## CybORG Upstream Patches

Hai regression fix cuc bo duoc ghi trong `CybORG_UPSTREAM.md` va bao ve boi:

- `tests/test_cyborg_ssh_decoy_fix.py`
- `tests/test_cyborg_stop_process_fix.py`

Khi viet paper, can noi ro cac run Stage 1 thuoc hai source-patch cohorts nhu
`notes/stage1_protocol.md` da ghi; khong mo ta chung la mot simulator build byte
identical.
