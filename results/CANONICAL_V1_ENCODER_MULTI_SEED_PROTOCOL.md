# Canonical V1 Flat/GCN/GraphSAGE Multi-Seed Protocol

**Status:** Hoàn tất 30/30 policies, 30/30 static observation gates và 30/30
dynamic action/evidence audits. Kết quả được ghi tại
`results/CANONICAL_V1_ENCODER_MULTI_SEED_RESULT.md`.

## Research Question

Với cùng canonical v1, PPO, action catalogue, reward và `random_host`
distribution, GCN và GraphSAGE có khác Flat về held-out return hay không? Tác
động có ổn định qua training seeds và phụ thuộc semantic action mask hay không?

## Factorial Design

| Branch | Encoder | Action mask | Role |
|---|---|---|---|
| F-O | Flat | Official CAGE 4 | Native primary |
| G-O | GCN | Official CAGE 4 | Native primary |
| S-O | GraphSAGE | Official CAGE 4 | Native primary |
| F-S | Flat | Official + semantic traffic mask | Action-validity ablation |
| G-S | GCN | Official + semantic traffic mask | Action-validity ablation |
| S-S | GraphSAGE | Official + semantic traffic mask | Action-validity ablation |

Training seeds là 42, 43, 44, 45 và 46. Mỗi policy được train độc lập; không warm-start,
không dùng checkpoint của condition khác và không chọn seed theo return.

## Fixed Controls

- Environment distribution: `random_host`.
- Canonical schema: v1, variant `action_aware`, 1,651 dimensions.
- Episode length: 400 simulator ticks.
- Requested budget: 100,000 simulator ticks; rollout kết thúc trọn episode nên
  actual budget có thể vượt nhẹ và phải được báo cáo.
- PPO: learning rate `3e-4`, gamma `0.99`, GAE lambda `0.95`, four update epochs,
  minibatch 64, clip 0.2, value coefficient 0.5, entropy coefficient 0.001,
  max gradient norm 0.5 và target KL 0.015.
- Reward: official CAGE 4 reward nhân fixed scale 0.01 chỉ bên trong PPO update.
- Action catalogue: padded 242 actions.
- No reward shaping, no periodic evaluation và no checkpoint selection.
- Held-out evaluation: 30 episodes, environment seeds 10,000-10,029.
- Stochastic action seed: 10,000. Stochastic evaluation là primary;
  deterministic argmax là diagnostic.

## Parameter Matching

| Encoder | Configuration | Parameters |
|---|---|---:|
| Flat | encoder hidden 128, latent 128, head hidden 128 | 292,339 |
| GCN | two GCN layers, graph hidden 28, latent 128, head hidden 128 | 291,735 |
| GraphSAGE | two SAGE layers, graph hidden 28, latent 128, head hidden 128 | 292,995 |

Sai lệch lớn nhất so với Flat là 656 parameters, tương đương 0.22%. Không thay
hidden dimension theo reward của từng seed.

## Existing Runs Reused

Hai run sau đã dùng đúng controls trên và được dùng làm seed 42 của nhánh native:

- `runs/canonical_flat_random_pivot_100k/seed_42` cho F-O;
- `runs/canonical_gcn_random_pivot_100k/seed_42` cho G-O.

Tên thư mục cũ dùng chữ `random_pivot`, nhưng trường config thực tế là
`environment_distribution=random_host`. Config cũ không ghi
`canonical_schema` và `semantic_action_mask` vì được tạo trước khi hai CLI field
này được thêm; behavior tương ứng là v1 và semantic mask Off.

## Analysis

Đơn vị độc lập chính là training seed, không phải từng evaluation episode.
Báo cáo:

1. mean và standard deviation của per-seed held-out stochastic return;
2. paired encoder differences theo cùng training seed;
3. encoder-by-mask sensitivity theo training seed;
4. deterministic results chỉ như failure-mode diagnostic;
5. action composition, traffic false-completion rate, normalized entropy,
   valid-action count và observation-use gate;
6. toàn bộ per-seed result, không chỉ seed tốt nhất.

Với năm training seeds, interval và significance test vẫn nhạy với từng seed.
Không dùng 30 environment seeds như 30 independent training replicates.

## Stop Rule

Không thêm canonical v2, reward shaping, Red mixture hoặc encoder mới trong khi
ma trận này đang chạy. Nếu một run lỗi kỹ thuật, sửa lỗi chung rồi chạy lại cùng
seed; không thay hyperparameter riêng cho condition đó.
