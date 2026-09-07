# Kiểm chứng refactor 2026-09-07

## Phạm vi

Baseline là commit archive 009e34b, gồm trạng thái v1 và các cải tiến v2/v4 có sẵn.
Không train main trong refactor. Không push.
Mục đích kiểm tra: giữ nguyên hành vi pipeline mới đã có, không chứng minh
lại hội tụ và không chuyển kết quả v1 thành v2.

## Những file giữ nguyên

So với 009e34b, nội dung các file sau không thay đổi:
- src/acd_benchmark/rl/ppo/advantages.py
- src/acd_benchmark/rl/ppo/update.py
- src/acd_benchmark/rl/ppo/training.py
- src/acd_benchmark/rl/ppo/evaluation.py
- src/acd_benchmark/rl/ppo/init_weight.py
- src/acd_benchmark/rl/rollout/rollout.py
- src/acd_benchmark/rl/tensorboard_logger.py
- CybORG/ toàn bộ vendor, pyproject.toml, uv.lock
- cyber/cc4_env.py và cyber/cc4_distribution.py

Categorical API và structured heads được giữ hành vi, bỏ các class/nhánh không
còn dùng. Hai encoder được tách file, giữ state_dict keys cho checkpoint hiện tại.

## Số học trước/sau

Trên Python3.10/Torch2.2 CPU của máy hiện tại, đặt PYTHONHASHSEED=42,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, Torch1 thread.
Environment random_host, seed123, episode_steps80, schema v2/parser v4,
TTL5/20. Random valid action stream seed456 trong60 simulator ticks.

Observation/mask/reward trajectory SHA256 trước và sau đều:

    4aba38d91df18cab7ae808921010ee474619632735d11085b3b86fa1cc2bb295

Policy init seed42; NodeMLP/GCN hidden128, mỗi model119881 tham số.
Forward logits và values có cùng hash trước/sau:

    NodeMLP: 40501c6118fe1de2c8e934d84fad6c1caf99e303c9bdb96e86386be7ba21a615
    GCN:     e97e7048aeef6d8ba220ebe071a8a31ed7072720320b679a7faf09aa3a743d31

Lượt train ngắn80 ticks danh nghĩa,2 episodes/update,1 epoch,minibatch64,
entropy .001,target KL .015. Mỗi policy chạy tuần tự.
Hai episode thực tế dài79 ticks/episode. Return trước/sau:

    NodeMLP: [-650.0, -210.0]
    GCN:     [-546.0, -379.0]

Trọng số sau PPO có cùng SHA256 trước/sau:

    NodeMLP: 928f3a667c5589bfe70d3ebfd5a37560276192ea1f146ff6e7e397bdce4364f8
    GCN:     81762871b97a4e5dd176f748d51a6c6137e5e272a98bd27a514c84f176f64e46

Script tùy chọn để kiểm lại trên runtime đã pin:

    export PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    uvr python tests/refactor_parity.py

Không đòi bitwise equality giữa CPU/GPU hoặc khác platform/dependency.
Mẫu kiểm tra này không phải chứng minh toàn bộ trajectory300k luôn khớp.

## Tests và CLI

- Bộ test trước refactor:82 passed.
- Bộ test main sau loại coverage v1/head cũ và thêm integration:62 passed.
- Coverage mới: mọi observation condition chạy được GCN, kể cả graph rỗng;
  GAE duration/truncation; round-trip schema; persistent host budget;
  periodic eval giữ RNG; train/load final+best cho cả NodeMLP/GCN;
  greedy/stochastic JSON và event TensorBoard.
- CLI main --plan in10 jobs tuần tự, không khởi chạy training.
- CLI evaluate mới load checkpoint v4 NodeMLP300k seed42 cũ:
  một episode seed10000 mỗi mode, deterministic=-729, stochastic=-531.
  Output ở /tmp/acd_refactor_old_checkpoint_v4_eval.json; không ghi đè run cũ.
- Gym/pkg_resources/pyparsing warnings vẫn đến từ dependencies.

## Khác biệt có chủ đích cần biết

1. GCN nhận graph không cạnh ở zero/action-state-only diagnostics: tạo tensor
   edge_index rỗng thay vì torch.cat(list rỗng) gây lỗi.
   Forward/training có graph bình thường khớp baseline.
2. Host-output-mask dùng schema hiện tại thay vì chỉ v1. Thuật toán chọn host giữ
   nguyên, nhưng tập dynamic features là của schema hiện tại. Cần đo curve lại.
3. Một CLI train/eval thay các CLI pilot. Train mặc định dùng cấu hình qualification
   hiện hành; có chặn ghi đè run và evaluation output.
4. Loại các class Flat/GraphSAGE/head cũ khỏi đường chạy main. Không port GraphSAGE
   sang head mới trong thao tác dọn code.
5. Giữ nguyên bằng chứng lịch sử trong archive và các report qualification.
   Chỉ thay tiêu chuẩn chấp nhận nghiên cứu thành fixed-budget theo quyết định người dùng.

Đây là refactor được kiểm chứng, không phải một đợt tuning mới.
