# Tra cứu lịch sử v1

- Nhánh lưu: archive/canonical-v1-before-main-2026-09-07.
- Commit: 009e34bf8a876a74ea2d050f4885fa496d3f7341.
- Gồm source trước dọn, v1 và các bước v2/v4 có sẵn, configs, kết quả,
  checkpoint, log, TensorBoard trong runs/.
- Đây là snapshot tiền-refactor có chứa v1, không phải thời điểm “v1 thuần”.
- Nhánh làm việc: main-benchmark.
- Chỉ Git local; chưa push remote.

## Đọc không đổi working tree

    git show archive/canonical-v1-before-main-2026-09-07:src/acd_benchmark/cyber/canonical_observation.py
    git show archive/canonical-v1-before-main-2026-09-07:ARCHIVE_README.md

Nếu cần duyệt cả cây trong editor, có thể tạo worktree khi cần:

    git worktree add ../canonical-v1-reference archive/canonical-v1-before-main-2026-09-07

Lệnh này chưa được thực hiện. Không cần checkout đè lên main.
Không load checkpoint v1 với parser v4 chỉ vì tensor shape khớp.

## Dữ liệu local

runs/ vẫn còn trên máy nhưng không track trong main. Snapshot archive vẫn track
checkpoint/log cũ. Report cũ còn ở runs/_archive_reports_pre_main_2026-09-07/.
Sáu report bằng chứng được sao nguyên vào results/qualification/.

Không dùng git clean -fdx: sẽ xóa cả dữ liệu ignored và môi trường Python.
Không xóa archive khi chưa có backup khác.

## Mapping code

| Cũ | Mới |
|---|---|
| canonical_observation.py của v1 | Archive |
| canonical_observation_v2.py: builder | canonical_observation.py |
| Constants/schema lẫn builder | observation_schema.py |
| Topology/mission helpers trong v1 | topology.py |
| StructuredNodeMLP trong structured_action.py | policies/nodemlp.py |
| StructuredGCN trong structured_action.py | policies/gcn.py |
| 6 train CLI phụ thuộc nhau | scripts/training/train.py |
| Các eval CLI theo encoder | scripts/evaluation/evaluate.py |
| Flat/GCN/GraphSAGE với head cũ | Archive |
| Pilot audits/experiment scripts cũ | Archive |
| PPO/rollout/TensorBoard | Giữ nguyên |

Chữ v1 trong pivot_salt và host_edge_global_v1 không phải observation v1.
