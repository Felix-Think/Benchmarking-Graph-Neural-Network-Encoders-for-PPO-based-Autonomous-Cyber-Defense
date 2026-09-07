# Benchmarking GNN Encoders for PPO-based Autonomous Cyber Defense

Workspace nghiên cứu CAGE 4: so sánh encoder trên cùng observation, action scorer,
PPO và ngân sách simulator ticks. Nhánh hiện tại chỉ giữ pipeline mới:
**canonical schema v2, parser v4, structured NodeMLP/GCN**.

Không còn pipeline v1 chạy xen kẽ. Toàn bộ source/kết quả tiền-refactor nằm ở
branch archive/canonical-v1-before-main-2026-09-07, commit 009e34b.
Archive chỉ được commit local, chưa push.

## Bắt đầu đọc

1. [Đọc từng bước để làm chủ code](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/docs/CODE_READING_GUIDE.md>): đi theo observation -> action -> rollout -> PPO.
2. [Phạm vi benchmark hiện tại](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/docs/MAIN_PROTOCOL.md>): ngân sách, seed, claim và giới hạn.
3. [Tra cứu v1 và mapping cũ/mới](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/docs/ARCHIVE.md>): xem lịch sử mà không đổi working tree.
4. [Kiểm chứng trước/sau dọn](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/docs/REFACTOR_VALIDATION.md>): test và tái lập số học.

**NodeMLP không phải Flat MLP flatten toàn graph.** Nó là MLP dùng chung trên từng
node với cùng action head như GCN. Main hiện so sánh có/không message passing.
GraphSAGE cũ có head khác, được giữ trong archive; chưa phải encoder main mới.

## Cấu trúc

```text
src/acd_benchmark/
  cyber/
    cc4_env.py                         Tạo raw CAGE 4 + năm Blue
    cc4_distribution.py                Chọn pivot official/random_host
    observation_schema.py              Feature names, dimensions, serialization
    topology.py                        Public subnet links và mission rules
    canonical_observation.py           Builder và temporal memory hiện tại
    cc4_canonical_env.py                Parse raw Blue obs, step/reset, status
    cc4_structured_action_env.py        Map 242 action slots sang native action
    benchmark_env.py                    Lắp một pipeline dùng chung train/eval
    canonical_observation_condition.py  Full/NoEvidence và diagnostic phụ
    dynamic_evidence_counterfactual.py  Zero/shuffle/time-shift evidence lúc eval
    dynamic_host_observation_mask.py    Output-retention stress test lúc eval
  rl/
    policies/
      actor_critic.py                   Categorical API, mask, sample/argmax
      structured_action.py              Head chung theo host/edge/global
      nodemlp.py                        Encoder không message passing
      gcn.py                            Encoder GCN
      factory.py                        Tạo policy từ config
    rollout/rollout.py                  Multi-tick transitions, buffer
    ppo/advantages.py                  SMDP GAE
    ppo/update.py                      PPO update
    ppo/training.py                    Vòng train và checkpoint
    ppo/evaluation.py                  Greedy/stochastic evaluation
    ppo/init_weight.py                 Khởi tạo trọng số
    tensorboard_logger.py              Scalars, config, histograms
scripts/
  inspection/inspect_benchmark.py       Xem shape/features/actions; không train
  training/train.py                    Một CLI cho cả hai encoder
  evaluation/evaluate.py               Final/best và các diagnostic
  analysis/summarize_main.py           Tổng hợp paired training seeds
  experiments/run_main.zsh             Lịch tuần tự; mặc định chỉ in kế hoạch
tests/                                Regression/unit/integration tests
docs/                                 Hướng dẫn, protocol và archive
results/qualification/                Bằng chứng qualification trước-refactor
runs/                                 Checkpoint/log local; ignored trên main
CybORG/                               Vendor CAGE 4; không cần đọc toàn bộ ngay
```

Các file __init__.py đánh dấu package, không chứa thuật toán.
pyproject.toml định nghĩa package/dependencies; uv.lock khóa toàn bộ môi trường;
.python-version chọn Python3.10; .gitignore giữ cache/checkpoint ngoài commit main.
[CybORG provenance](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/CybORG_UPSTREAM.md>) ghi nguồn và bản vá đang giữ.
Workspace .code-workspace chỉ phục vụ editor.

## Môi trường và kiểm tra nhanh

Dùng thư mục repo này làm working directory. Giữ Python3.10 và uv.lock, không tự
nâng NumPy/Torch để “sửa warning”. Sau clone mới dùng uv sync --locked; trên máy
hiện tại dùng alias uvr của zsh.

```zsh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42
uvr python -m scripts.inspection.inspect_benchmark --seed 123 --steps 10
uvr python -m pytest tests -q
```

Inspection in shape 5185, 242 action slots, feature names và vài random steps.
Tests kiểm tra parser/bộ nhớ, action mapping, policy masks, GAE, checkpoint và
evaluation. Warning Gym/pkg_resources là từ dependency cũ; traceback/failing test
mới là lỗi cần xử lý, không phải mọi warning.

## Một run train

Ví dụ smoke dưới đây chỉ kiểm tra plumbing, không dùng làm kết quả nghiên cứu:

```zsh
uvr python -m scripts.training.train \
  --encoder nodemlp --seed 42 \
  --total-simulator-steps 1000 --episode-steps 100 \
  --episodes-per-update 2 --eval-every-updates 2 \
  --periodic-eval-episodes 1 --eval-episodes 2 \
  --print-every-updates 1 --run-dir runs/main_smoke_nodemlp
```

Đổi encoder thành gcn và **đổi run-dir** để thử GCN. CLI không ghi đè run đã có
config.json. Không có auto-resume; dùng run mới.

Cấu hình main mặc định: 300k ticks, generator steps500, 4 episodes/update,
reward-scale .01, entropy .001, target KL .015, random_host, official mask,
TTL5/20, hidden128, parameter count119881 cho mỗi policy.
PPO, rollout, TensorBoard không bị thay thuật toán trong refactor.

## Chạy main tuần tự sau khi đã đọc và duyệt

```zsh
zsh scripts/experiments/run_main.zsh --plan
zsh scripts/experiments/run_main.zsh --execute
```

--plan không train. --execute chạy NodeMLP rồi GCN, mỗi encoder seeds42–46,
**một policy tại một thời điểm**. Nó gọi uvr qua interactive zsh của bạn.
Dùng RUN_ROOT để chọn thư mục mới khi cần:

```zsh
RUN_ROOT=runs/main_fixed_budget_v2 zsh scripts/experiments/run_main.zsh --plan
```

Không chạy main tự động trong lần dọn này.
Budget thực tế có thể vượt nhẹ vì hoàn tất episode; xem training_summary.json.
Claim là so sánh fixed-budget, không phải đã hội tụ.

## Evaluation và TensorBoard

Train luôn đánh giá final cả stochastic và greedy, xuất ba JSON trong seed_N.
Chạy lại final/best bằng một file output mới:

```zsh
uvr python -m scripts.evaluation.evaluate \
  --run-dir runs/main_fixed_budget/nodemlp --seed 42 \
  --checkpoint final --episodes 30 --eval-seed 10000 \
  --output runs/main_fixed_budget/nodemlp/seed_42/recheck_final.json

uvr tensorboard --logdir runs/main_fixed_budget --port 6006
```

Mở http://localhost:6006 sau khi bạn chạy TensorBoard.
Đổi --checkpoint best để kiểm tra policy được chọn bởi validation.
Stochastic là metric chính; deterministic là diagnostic. Không chọn mode/chọn
checkpoint theo test score rồi báo đó là phép so sánh đã định trước.

Diagnostic từng điều kiện, vẫn dùng CLI evaluate và output riêng:
- --condition no_evidence: xóa sensor features của policy đã train Full.
- --counterfactual shuffle_hosts: đổi host mang evidence.
- --counterfactual time_shift --time-shift-lag 6: làm trễ evidence.
- --retention .5 --mask-process persistent: giữ dynamic features của một nửa host.
- --retention 1/.75/.5/.25 cần gọi thành bốn lệnh với giá trị số riêng.

Host-mask hiện là output stress test: không xóa topology/global/mask, không ngăn
builder cập nhật memory, không phải chỉ bỏ sensor, không phải subnet masking.
Các output bị can thiệp được gắn metadata; không ghi đè evaluation gốc.
NoEvidence train từ đầu dùng --observation-condition no_evidence trong train,
khác hoàn toàn can thiệp observation tại evaluation.

## Đọc output

| File | Mục đích |
|---|---|
| config.json | Schema/parser, PPO, seed, encoder, số tham số, runtime controls |
| checkpoint.pt | Trọng số/optimizer/history cuối ngân sách |
| best_checkpoint.pt | Checkpoint tốt nhất trên periodic validation, nếu có |
| periodic_evaluation.json | Tiến trình validation và lần được chọn best |
| training_summary.json | Ticks/decisions/episodes/updates thực tế |
| evaluation_stochastic.json | Raw return và action counts, mode chính |
| evaluation_deterministic.json | Raw return greedy, diagnostic |
| evaluation_comparison.json | Hai mode trên cùng eval seeds |
| tensorboard/ | Training losses, KL/entropy, returns, validation |

Sau khi đủ hai encoder và cùng bộ training seeds:

```zsh
uvr python -m scripts.analysis.summarize_main \
  --run-root runs/main_fixed_budget \
  --output results/main_fixed_budget_summary.json
```

Script kiểm tra các trường matched trong từng cặp và dùng training seed làm
đơn vị thống kê. Không biến 30 eval episodes của một model thành 30 lần train.

## Điều chưa được kết luận

Observation utility đã có bằng chứng trên NodeMLP năm seeds. Điều đó không
chứng minh GCN sẽ thắng hoặc đã hội tụ. random_host không phải structural topology
generalisation. Các curve và kết quả v1 không được đổi nhãn thành v2.
Xem MAIN_PROTOCOL trước khi bổ sung encoder, partial-observation process hoặc claim mới.
