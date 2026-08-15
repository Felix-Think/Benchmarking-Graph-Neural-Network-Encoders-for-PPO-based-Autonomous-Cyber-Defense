# Huong Dan Doc Code Pilot

Trang thai: `configs/pilot.yaml` la protocol active duy nhat.

## 1. Cau Noi Vao CybORG

1. `src/acd_benchmark/cyber/import_path.py`
2. `src/acd_benchmark/cyber/cyborg_cc2.py`
3. `src/acd_benchmark/cyber/env_factory.py`

Ba file nay cho biet CybORG local duoc import o dau, Scenario2 nam o dau va
official CAGE-2 duoc khoi tao nhu the nao.

## 2. Kiem Tra Environment

1. `scripts/inspection/smoke_cyborg_cc2.py`
2. `scripts/inspection/inspect_cc2_spaces.py`

Script thu nhat chi kiem tra reset/step. Script thu hai in action space va raw
Blue observation de phuc vu thiet ke protocol moi.

## 3. Ban Va Simulator

1. `CybORG_UPSTREAM.md`
2. `patches/cyborg_plus_plus/README.md`
3. `tests/test_cyborg_plus_plus_patches.py`

Khong thay CybORG local ma khong port va chay lai regression tests.

## 4. Observation, Topology Va Action

1. `src/acd_benchmark/cyber/topology.py`
2. `src/acd_benchmark/cyber/action_space.py`
3. `src/acd_benchmark/wrappers/shared/observation.py`
4. `src/acd_benchmark/wrappers/pilot/masking.py`
5. `src/acd_benchmark/wrappers/pilot/env.py`

Doc theo thu tu tren de thay ro static prior, raw observation, retention mask
va model-specific output duoc tach nhau nhu the nao.

## 5. Model Va PPO

1. `src/acd_benchmark/rl/pilot_factory.py`
2. `src/acd_benchmark/rl/policies/flat.py`
3. `src/acd_benchmark/rl/policies/graphsage.py`
4. `src/acd_benchmark/rl/rollout/collection.py`
5. `src/acd_benchmark/rl/ppo/advantages.py`
6. `src/acd_benchmark/rl/ppo/update.py`
7. `src/acd_benchmark/rl/ppo/training.py`

Flat va GraphSAGE chi khac encoder. Rollout, GAE, objective, optimizer va
hyperparameters la shared code.

## 6. Evaluation Va Orchestration

1. `src/acd_benchmark/evaluation/metrics.py`
2. `src/acd_benchmark/evaluation/baselines.py`
3. `scripts/pilot/audit_protocol.py`
4. `scripts/pilot/train.py`
5. `scripts/pilot/evaluate_baselines.py`
6. `scripts/pilot/evaluate_observation_gate.py`
7. `scripts/pilot/run_matrix.py`
8. `scripts/pilot/summarize_results.py`

True state chi duoc doc trong `evaluation/metrics.py`. Bat ky import nguoc tu
policy/wrapper vao evaluator la vi pham contract.

## 7. Main Benchmark Bi Khoa

`configs/main_benchmark_16.yaml` chi ghi blocker. Main chi duoc active sau khi
Pilot dat va topology generator dieu khien dong thoi simulator connectivity va
graph edges.
