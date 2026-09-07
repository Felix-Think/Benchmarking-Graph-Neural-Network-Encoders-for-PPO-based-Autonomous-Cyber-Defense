# Kết luận Feature-Group Gates - Canonical V1, Seed 42

## Phạm vi

- Checkpoint: checkpoint cuối của qualification run 30k simulator steps.
- Evaluation: stochastic, 30 held-out episodes, environment seeds 10000-10029.
- Tất cả gate dùng cùng action-sampling seed 10000 để cho phép so sánh paired.
- Khoảng tin cậy: paired bootstrap 95% với 20.000 lần resample.

## Kết quả chính

| Điều kiện | Mean return | Std |
|---|---:|---:|
| Full canonical observation | -834.53 | 271.06 |
| Static only | -854.00 | 355.09 |
| Zero all, giữ action mask | -3751.90 | 556.14 |
| Mission phase only | -3646.87 | 652.76 |
| Node identity only | -1176.73 | 374.22 |
| Structure only | -2734.50 | 541.24 |
| Topology/static group | -937.07 | 325.75 |
| Topology/static không adjacency | -997.77 | 406.85 |
| Full observation, bỏ action mask | -1322.43 | 487.36 |

## Các phép so sánh có ý nghĩa

- Node identity so với zero-all: +2575.17, CI [2373.73, 2781.57].
- Structure so với zero-all: +1017.40, CI [728.30, 1307.10].
- Topology/static group so với zero-all: +2814.83, CI [2615.86, 3016.10].
- Adjacency khi đã có identity và membership: +60.70,
  CI [-92.17, 218.17].
- Mission phase khi đã có topology/static: -15.87,
  CI [-169.70, 147.00].
- Dynamic evidence khi đã có static context: +19.47,
  CI [-137.20, 170.53].
- Action mask khi có full observation: +487.90,
  CI [277.97, 700.77].

## Diễn giải

1. Policy không chỉ học action mask hoặc output bias. Khi observation bị xóa,
   return giảm gần 3.000 điểm so với full observation.
2. Tín hiệu mạnh nhất là static node identity/type/existence/control theo slot.
   Đây chưa phải bằng chứng policy đang suy luận trên graph edges.
3. Host membership và structural adjacency có ích khi dùng chung như một nhóm,
   nhưng adjacency riêng phần thêm vào identity + membership chưa có delta dương
   rõ ràng.
4. Mission phase không có đóng góp dương quan sát được cho checkpoint này.
5. Dynamic process/connection/analysis evidence không cải thiện so với
   static-only. Canonical V1 đã learnable, nhưng observation-use gate cho claim
   partial observability vẫn chưa đạt.
6. Action mask có đóng góp rõ ràng. Tuy nhiên, gate bỏ mask làm policy chọn
   `[Padding]` và `[Invalid]` khoảng 68,5% decision, nên delta này một phần phản
   ánh action-wrapper contract, không chỉ là năng lực suy luận của policy.

## Quyết định

- Không train checkpoint seed 42 lâu hơn với cùng protocol: khả năng cao chỉ
  củng cố static-slot strategy hiện tại.
- Chưa bắt đầu main GNN benchmark. Hiện chưa có evidence rằng adjacency hoặc
  dynamic partial observation là nguồn thông tin mà policy cần dùng.
- Có thể bắt đầu xây GNN ở mức implementation/prototype, nhưng không nên chạy
  benchmark dài hoặc dùng kết quả để claim GNN/partial-observation superiority.
- Bước nghiên cứu kế tiếp nên là observation-use qualification: tạo train/test
  distribution mà static slot policy không đủ, trong khi giữ reward, PPO core,
  action mask và action space giống nhau. Sau đó lặp lại full/static/dynamic gates
  trên ít nhất hai training seeds trước khi chạy GNN main.

## Giới hạn

Đây là kết quả của một training seed. Gate tại evaluation cho biết nhóm feature
nào đủ để checkpoint hoạt động, nhưng không chứng minh quan hệ nhân quả duy nhất
vì các nhóm feature có thể tương tác phi tuyến.
