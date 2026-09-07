# Phạm vi main sau tinh gọn

2026-09-07: người dùng chọn **so sánh cùng ngân sách**, không đòi hỏi mỗi encoder
đã hội tụ. Không sửa Proposal.docx/Experiment.docx trong lần dọn này.

| Thành phần | Cấu hình hiện tại |
|---|---|
| Environment | CAGE 4 vendored hiện tại, random_host |
| Blue | 5 agents; một actor-critic dùng chung, memory riêng |
| Observation | schema v2; parser raw_blue_observation_v4_relational_events |
| Input / action | 5185 float32 / 242 structured slots |
| Encoder | NodeMLP hidden128; GCN 2 layers hidden128; embedding64 |
| Tham số | 119881 cho mỗi policy |
| Head | cùng host/edge/global, context/head hidden128 |
| Memory | passive TTL5, Analyse TTL20, action_aware |
| Mask | official; semantic chỉ là ablation |
| Reward | native team reward; .01 trước GAE; báo cáo raw return |
| PPO | lr .0003, gamma .99, lambda .95, 4 epochs, minibatch64 |
| PPO tiếp | clip .2, value coef .5, entropy .001, grad clip .5, target KL .015 |
| Budget | 300000 simulator ticks danh nghĩa, hoàn thành episode |
| Episode | generator steps500, rollout hiện có499 ticks |
| Update | 4 episodes liên tiếp, update cuối có thể ít hơn |
| Training seeds | 42–46, từng policy chạy tuần tự |
| Validation | mỗi25 updates, stochastic,10 episodes, seeds20000 trở đi |
| Final eval | greedy và stochastic,30 episodes, seeds10000 trở đi |
| Primary | stochastic raw team return, final checkpoint |
| Secondary | best-validation checkpoint, ghi nhãn riêng |
| Runtime | CPU, Torch/OMP/MKL1 thread; PYTHONHASHSEED trước process |
| Đơn vị thống kê | training seed, không phải evaluation episode |

Runner đặt PYTHONHASHSEED bằng training seed. Hai encoder trong một cặp có cùng
giá trị. Không gọi là tái lập nguyên run cũ nếu hash seed/config/checkpoint không khớp.

300000 ticks thường thành300398 ticks sau602 episodes. Báo cáo budget thực từ
training_summary.json. Cùng ticks không bảo đảm cùng decisions do action durations.
Checkpoint cuối ngân sách và best-validation không được trộn trong cùng bảng.

## Bằng chứng đang có

[Qualification trước-refactor](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/results/qualification/OBSERVATION_LEARNING_QUALIFICATION_STATUS.md>)
ghi Full vượt NoEvidence trung bình +862.77 raw return trên năm seeds của NodeMLP.
Đó không phải kết quả GCN main.
Report lịch sử còn ghi “blocked by plateau”: thuộc tiêu chuẩn hội tụ cũ.
Quyết định fixed-budget hiện tại **không biến plateau chưa đạt thành đã hội tụ**.

## Giới hạn cần giữ rõ

- NodeMLP không phải MLP flatten toàn graph. Nếu paper cần Flat-global thì thêm
  baseline riêng sau khi duyệt thiết kế, không đổi nhãn kết quả NodeMLP.
- random_host thay discovery pivot, không phải held-out physical topology.
- Native q=1 vẫn partial observation. q=1 chỉ không thêm masking của ta.
- q=1/.75/.5/.25 hiện là zero-shot output-retention stress test; không train mixed-q.
- Host-mask bỏ dynamic features của host bị ẩn, gồm decoy/Analyse state.
  Global/matrices/official mask giữ nguyên. Không phải sensor-only masking.
- Persistent host masking không phải persistent subnet masking.
- Host-mask đã chuyển sang schema hiện tại và test; curve v1 phải đo lại,
  không được coi thành kết quả v2. Memory builder vẫn cập nhật dù đầu ra bị che.
- Full/NoEvidence train từ đầu khác can thiệp zero/shuffle/time-shift lúc eval.
- GraphSAGE v1 có head khác. Structured GraphSAGE cần port/validation riêng sau
  khi người dùng duyệt; không được gắn kết quả cũ vào benchmark mới.
- Không thêm messages, five independent policies, reward shaping, cross-scale.
- Core chưa lưu đầy đủ RNG state; không có bảo đảm exact resume.

## Trình tự sau khi đọc và duyệt

1. Chạy inspection và tests.
2. Xem lịch: zsh scripts/experiments/run_main.zsh --plan.
3. Duyệt nhãn NodeMLP/GCN, budget, checkpoint và claim fixed-budget.
4. Dùng --execute để chạy tuần tự. Lần dọn code này không khởi chạy main.
5. Tổng hợp final stochastic return theo paired training seeds.
6. Chạy diagnostic/retention với cùng checkpoint, eval/action/mask seeds.
7. Phân tích trước khi mở rộng encoder hoặc claim partial/generalisation.

Mọi thay đổi schema/parser, reward, catalog hoặc head cần run mới; không ghi đè cũ.
