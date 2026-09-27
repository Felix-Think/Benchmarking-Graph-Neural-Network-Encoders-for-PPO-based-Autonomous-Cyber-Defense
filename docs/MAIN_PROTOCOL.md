# Main Benchmark: 500k CPU + CUDA

Cấu hình thực thi duy nhất:
`configs/experiments/encoder_retention_500k_cuda.json`.
Protocol chi tiết: [ENCODER_RETENTION_PROTOCOL.md](ENCODER_RETENTION_PROTOCOL.md).

| Thành phần | Cấu hình |
|---|---|
| Encoders | NodeMLP, GCN, mean GraphSAGE; cùng structured action head |
| Parameters | 119881 / 119881 / 119817 |
| Environment | CAGE 4, random_host; 5 Blue agents dùng chung policy |
| Observation | Canonical v2, parser v4; 5185 features, 242 action slots |
| Training regimes | Full q=1; mixed-q uniform {1,.75,.5,.25} |
| Masking | Persistent fixed-budget host masking sau memory update |
| Reward/action mask | Native reward; reward scale .01 trước GAE; official mask |
| Training seeds | 42,43,44; 18 runs |
| Budget | 500000 simulator ticks danh nghĩa mỗi policy |
| Milestones | 100k,200k,300k,400k,500k; không restart giữa các mốc |
| Rollout | 2 CPU workers, 4 episodes/update, generator steps500 |
| Learner | CUDA FP32; TF32 off; không AMP |
| Evaluation | CPU copy đồng bộ; stochastic chính, greedy phụ; tách riêng từng q |
| Validation | Seeds20000; 10 episodes/q ở milestone; periodic q=1 mỗi25 updates |
| Final test | Seeds10000; 30 episodes/q; final checkpoint là kết quả chính |
| Thống kê | Mean/sample std giữa 3 training seeds; báo cáo missing |
| Compute | Wall/CPU time, peak RAM, peak CUDA allocated/reserved; cùng hardware |

Hyperparameters PPO, schema, reward, mask và parameter budget không thay đổi.
So sánh cùng ngân sách không đồng nghĩa mỗi encoder đã hội tụ ở 500k.
Số ticks thực tế cuối run dự kiến500497; dùng số trong training_summary.json.

## Giới Hạn

- NodeMLP là MLP dùng chung trên từng node, không phải flatten toàn graph.
- random_host đổi discovery pivot, không phải held-out physical topology.
- q=1 vẫn là partial observation native. Host masking không phải sensor-only:
  topology/global/action mask giữ nguyên; memory vẫn cập nhật.
- Qualification Full/NoEvidence trước đây trên NodeMLP không chứng minh mọi GNN
  học dynamic evidence hay sẽ vượt NodeMLP.
- TTR chỉ tính các lần phục hồi hoàn tất; đọc cùng unrecovered/recovery rate.
- Chưa có exact-resume CLI đã được kiểm chứng. Không nạp weights rồi gọi là resume.
- Không gộp timing M4 cũ với CUDA mới; không tuning theo final-test scores.
- Giữ kết quả và source snapshots lịch sử trong runs/results; không đổi nhãn
  kết quả300k thành500k.

## Thực Hiện

1. Chạy tests, sau đó smoke_cuda trên NVIDIA GPU thực.
2. Xem kế hoạch bằng scripts.experiments.run_encoder_matrix (mặc định không train).
3. Chỉ thêm --foreground hoặc --launch khi sẵn sàng chạy; dùng run-root mới.
4. Báo cáo riêng encoder/regime/q, checkpoint selection và hardware metadata.

Lần dọn cấu hình này không khởi chạy training.
