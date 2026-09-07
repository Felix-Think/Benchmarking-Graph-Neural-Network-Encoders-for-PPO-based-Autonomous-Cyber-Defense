# Canonical V1 GraphSAGE Five-Seed Extension

**Status:** Hoàn tất 10/10 policies, 10/10 static gates và 10/10 dynamic audits.
Kết quả được tích hợp tại `results/CANONICAL_V1_ENCODER_MULTI_SEED_RESULT.md`.

## Question

GraphSAGE có ổn định hơn Flat hoặc GCN khi giữ canonical v1, PPO, parameter
budget, reward, action catalogue và `random_host` distribution cố định hay
không? Kết quả có phụ thuộc semantic action mask hay không?

## Conditions

| Branch | Encoder | Action mask | Role |
|---|---|---|---|
| S-O | GraphSAGE | Official CAGE 4 | Native primary |
| S-S | GraphSAGE | Official + semantic traffic mask | Action-validity ablation |

Training seeds: 42, 43, 44, 45 và 46. Mỗi policy train độc lập từ random
initialization. Không warm-start và không chọn checkpoint theo held-out return.

## Fixed Controls

Controls giống ma trận Flat/GCN:

- canonical v1 `action_aware`, `random_host`, episode 400 ticks;
- 100,000 requested simulator ticks;
- PPO learning rate `3e-4`, gamma `0.99`, GAE lambda `0.95`, four epochs,
  minibatch 64, clip 0.2, value coefficient 0.5, entropy coefficient 0.001,
  gradient norm 0.5 và target KL 0.015;
- fixed reward scale 0.01 trong PPO;
- 30 held-out evaluation episodes, seeds 10,000-10,029;
- stochastic evaluation primary, deterministic argmax diagnostic;
- không periodic evaluation, reward shaping hoặc checkpoint selection.

## Parameter Matching

| Encoder | Parameters | Difference from Flat |
|---|---:|---:|
| Flat | 292,339 | 0 |
| GCN, hidden 28, two layers | 291,735 | -604 (-0.21%) |
| GraphSAGE, hidden 28, two layers | 292,995 | +656 (+0.22%) |

Graph construction, ordered node embeddings, context projection and actor/critic
heads are shared with GCN. Only `GCNConv` is replaced by `SAGEConv`.

## Monitoring Rule

Training logs are emitted every 50 PPO updates. No intermediate evaluation is
used. Only one GraphSAGE policy process runs at a time; official and semantic
conditions are not parallelized. Scientific comparison is produced only after
all ten final checkpoints complete their held-out evaluation, static gate and
action/evidence audit.
