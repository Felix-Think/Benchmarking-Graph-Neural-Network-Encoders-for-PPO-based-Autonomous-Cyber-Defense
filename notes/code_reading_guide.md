# Huong Dan Doc Code

Tai lieu nay chi ra duong code dang duoc protocol cuoi su dung. Doc theo thu tu
duoi day se tranh bi lan vao source snapshot trong `runs/` hoac code CybORG
upstream qua som.

## 1. Cau Noi Vao CybORG

1. `src/acd_benchmark/cyber/import_path.py`
2. `src/acd_benchmark/cyber/cyborg_cc2.py`
3. `src/acd_benchmark/cyber/env_factory.py`
4. `src/acd_benchmark/cyber/action_space.py`

Sau buoc nay can tra loi duoc: scenario nao duoc tao, Red agent nao duoc dung,
va integer action cua policy duoc doi thanh CybORG action nhu the nao.

## 2. Encoder Raw Blue

Doc `src/acd_benchmark/wrappers/shared/host_obs.py` theo cac cum:

1. constants va hai dataclass;
2. cac ham normalize va tach observation;
3. cac ham merge sparse observation;
4. tao category/schema;
5. `RawBlueHostEncoder`;
6. chon feature, adjacency va flatten.

Dung `notes/host_obs_pipeline.md` de giu buc tranh tong the. Dung
`notes/observation_encoder_design_note.md` de kiem tra y nghia khoa hoc va cac
gioi han cua representation.

## 3. Hai Graph/Flat View Dung Chung

1. `src/acd_benchmark/wrappers/shared/graph_obs.py`
2. `src/acd_benchmark/wrappers/shared/flat_obs.py`

`GraphObservation` la representation goc. Flat view chi flatten node features
va adjacency, giu cung thong tin dau vao cho policy vector va graph.

## 4. Partial Observation Va Stage 1

1. `src/acd_benchmark/cyber/scalable_scenario.py`
2. `src/acd_benchmark/wrappers/shared/partial_obs.py`
3. `src/acd_benchmark/wrappers/stage1/observation.py`
4. `src/acd_benchmark/wrappers/stage1/env.py`

`Stage1GraphEnv` them frozen topology va host masking. `Stage1FlatEnv` tiep tuc
la mot view flatten cua cung graph observation. Protocol day du nam trong
`notes/stage1_protocol.md`.

## 5. PPO Va Policy

Doc theo tung trach nhiem trong cac package sau:

1. Policy interface va action layout:
   - `src/acd_benchmark/rl/policies/base.py`
   - `src/acd_benchmark/rl/policies/action_head.py`
2. Tung encoder rieng:
   - `src/acd_benchmark/rl/policies/flat.py`
   - `src/acd_benchmark/rl/policies/graph_base.py`
   - `src/acd_benchmark/rl/policies/gcn.py`
   - `src/acd_benchmark/rl/policies/graphsage.py`
3. Thu thap experience:
   - `src/acd_benchmark/rl/rollout/types.py`
   - `src/acd_benchmark/rl/rollout/collection.py`
4. PPO mathematics va update:
   - `src/acd_benchmark/rl/ppo/config.py`
   - `src/acd_benchmark/rl/ppo/advantages.py`
   - `src/acd_benchmark/rl/ppo/update.py`
   - `src/acd_benchmark/rl/ppo/training.py`
   - `src/acd_benchmark/rl/ppo/evaluation.py`
5. Noi model vao experiment:
   - `src/acd_benchmark/rl/stage1_factory.py`
   - `scripts/stage1/train_ppo.py`

Flat, GCN va GraphSAGE dung chung rollout, GAE, clipped PPO loss, optimizer va
evaluation loop. Chi policy encoder/readout khac nhau.

Khong co facade PPO tong hop. Tests va scripts import truc tiep module so huu
khai niem do, vi du `policies.gcn`, `rollout.collection`, hoac `ppo.update`.
Khi them GAT/GIN/GGNN, tao policy module moi trong `policies/`; khong sua rollout
hoac PPO mathematics neu protocol khong thay doi.

## 6. Metrics Va Reproducibility

1. `src/acd_benchmark/evaluation/operational_metrics.py`
2. `src/acd_benchmark/utils/reproducibility.py`
3. `scripts/stage1/evaluate_checkpoint.py`
4. `scripts/stage1/summarize_results.py`
5. `scripts/stage1/audit_protocol.py`
6. `scripts/stage1/analyze_results.py`

True state chi duoc doc trong evaluation metrics, khong duoc dua vao policy
observation.

## 7. Scripts Ho Tro

- `scripts/inspection/`: smoke, inspect va audit raw observation.
- `scripts/stage1/`: tao scenario, train, orchestration, audit va phan tich.

## Khong Doc Truoc

- `runs/*/source_snapshot/`: ban dong bang cua tung run, khong phai source hien
  tai.
- `runs/archive/stage0_official_cage2/`: evidence cua pilot da dong, khong phai
  active workflow.
- Toan bo `CybORG/`: chi mo file upstream khi can truy vet mot behavior cu the.
- Checkpoint `.pt`, CSV va JSON lon: day la evidence, khong phai implementation.

## Kiem Tra Sau Moi Cum

```bash
uv run pytest tests/test_raw_obs_wrappers.py tests/test_raw_env.py -q
uv run pytest tests/test_stage1_observation.py tests/test_stage1_ppo.py -q
uv run ruff check src scripts tests
```
