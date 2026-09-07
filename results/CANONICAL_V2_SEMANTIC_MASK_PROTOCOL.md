# Protocol: Canonical V2 and Semantic Action Mask

## Mục tiêu

Hai thay đổi được coi là hai can thiệp độc lập. Không được gộp chúng thành một
"bản sửa môi trường" duy nhất, vì khi reward thay đổi ta sẽ không biết nguyên
nhân đến từ observation hay action validity.

1. **Canonical V2** thay đổi observation nhưng giữ nguyên action catalogue và
   official action mask.
2. **Semantic action mask** thay đổi tập action hợp lệ nhưng giữ nguyên
   observation schema.

Reward, PPO, episode length, training distribution, evaluation seeds và model
budget phải được giữ cố định giữa các condition.

## Canonical V2

V2 chỉ đọc event đã được automatic `Monitor` chuyển sang các buffer
`old_process_creation` và `old_network_connections`. Nó không đọc event hiện
đang nằm trong buffer chưa quan sát và không dùng Red/Green identity.

Passive alert hiện tại giữ các thuộc tính quan sát được sau:

- số process-creation events;
- process event có `pid` hay không;
- process event có danh sách `Connections` hay không;
- số network-connection events;
- connection event có remote address, remote port và `pid` hay không.

Event count được cap tại 5 rồi chuẩn hóa về `[0, 1]` để một burst lớn không làm
thay đổi scale của input. Cap này là một giả định thiết kế và phải được ghi lại;
không được diễn giải nó như số event chính xác khi count lớn hơn 5.

Passive memory và Analyse history là hai trạng thái riêng:

- passive memory chỉ được refresh bởi process/connection alert và lưu snapshot
  alert mới nhất, không lấy maximum với alert cũ;
- Analyse memory chỉ được refresh khi `Analyse` hoàn thành thành công và lưu kết
  quả Analyse gần nhất;
- mỗi memory có tuổi, linear decay và TTL riêng;
- `Restore` thành công xóa cả hai memory trên target host;
- TTL được đo bằng simulator ticks, không phải PPO decision count.

Giá trị mặc định để qualification là 20 ticks. Đây chưa phải hyperparameter đã
được chứng minh tối ưu; sensitivity chính thức nên kiểm tra ít nhất 10/20/40
sau khi pipeline học ổn.

## Semantic Action Mask

Official CAGE 4 mask chủ yếu kiểm tra action có tồn tại trong catalogue. Nó vẫn
cho phép `AllowTrafficZone` khi edge chưa bị block và cho phép lặp lại
`BlockTrafficZone` trên edge đã block; hai trường hợp này trả `FALSE` trong
`ControlTraffic.py`.

Semantic mask bắt đầu từ official mask rồi áp dụng thêm state machine chỉ dựa
trên lịch sử action Blue thành công:

- edge chưa block: cho `BlockTrafficZone`, mask `AllowTrafficZone`;
- edge đã block: mask `BlockTrafficZone`, cho `AllowTrafficZone`;
- sau Allow thành công, edge quay về trạng thái chưa block.

Mask là directional theo đúng cặp `(from_subnet, to_subnet)`. Nó không đọc
firewall true state. Cùng implementation được dùng cho Flat và GCN tại lớp
environment trước khi policy nhận `action_mask`.

## Ma Trận 2x2

| Condition | Canonical schema | Semantic mask | Câu hỏi |
|---|---|---:|---|
| A | v1 | Off | Baseline hiện tại |
| B | v2 | Off | V2 observation có cải thiện học từ evidence không? |
| C | v1 | On | Loại action dư thừa có cải thiện policy không? |
| D | v2 | On | Hai can thiệp có bổ sung cho nhau không? |

Không dùng riêng D để kết luận. Phải có B và C để quy attribution.

## Qualification Trước 100k

Chạy A/B/C/D ở 20k simulator steps, cùng một training seed để phát hiện lỗi kỹ
thuật. Evaluation dùng cùng held-out environment seeds và stochastic action
seeds. Chỉ đưa lên 100k và nhiều training seeds nếu:

- mọi observation hữu hạn và nằm trong `[0, 1]`;
- v2 current alerts có variation và passive memory không bão hòa gần 100%;
- Analyse memory chỉ thay đổi sau Analyse thành công;
- semantic mask không tạo empty action set;
- số redundant Allow/Block completion giảm về 0;
- PPO rollout/update hoàn thành với action replay vẫn hợp lệ;
- performance không suy giảm nghiêm trọng do lỗi representation.

Sau qualification, chạy ít nhất 3 training seeds cho A/B/C/D. Báo cáo mean,
standard deviation, paired seed differences, action composition, action failure
rate, evidence-conditioned action rate và return. Không thêm reward shaping vào
thí nghiệm này.

Semantic mask làm thay đổi trần entropy vì số action hợp lệ giảm. Do đó phải
báo cáo thêm `valid_action_count_mean` và `normalized_entropy =
entropy/log(valid_action_count)`. PPO objective vẫn dùng entropy thô và cùng
`entropy_coef`; hai metric mới chỉ dùng để chẩn đoán, không thay đổi training.

## Encoder Comparison

Ma trận v1/v2 phía trên cũng phải khớp model budget. Nếu giữ
`encoder_hidden_dim=128` cho cả hai schema, Flat v2 có 379,891 tham số trong khi
Flat v1 chỉ có 292,339; khi đó hiệu ứng schema bị trộn với model capacity. Kết
quả chính của qualification vì vậy dùng:

- Flat v1: `encoder_hidden_dim=128`, 292,339 tham số;
- Flat v2: `encoder_hidden_dim=92`, 291,187 tham số.

Độ lệch còn lại là 1,152 tham số, tương đương 0.39% so với Flat v1. Hai run v2
width 128 được giữ làm chẩn đoán phụ, không dùng cho contrast chính.

Chỉ so Flat với GCN sau khi chọn condition observation/mask bằng protocol trên.
Tại bước đó phải tính lại parameter count và match model budget trong chính
schema được chọn. Nếu giữ nguyên hidden dimensions mà parameter count khác
đáng kể, kết quả không được gọi là parameter-matched.

Với hidden dimensions mặc định còn lại là 128 và action catalogue 242:

- Flat v1: 292,339 parameters;
- GCN v1 với `gcn_hidden_dim=28`: 291,735 parameters;
- Flat v2: 379,891 parameters;
- GCN v2 với `gcn_hidden_dim=40`: 380,811 parameters.

Do đó nếu schema v2 được chọn cho encoder benchmark, cặp Flat v2 width 128 và
GCN v2 hidden 40 có chênh lệch khoảng 0.24%. Đây là parameter matching giữa
encoder, không phải cấu hình dùng cho ma trận schema v1/v2 và không phải tuning
theo reward.

## Lệnh Kiểm Tra Độc Lập

```bash
uvr python -m scripts.analysis.validate_canonical_v2_interventions \
  --seed 123 \
  --episode-steps 20 \
  --passive-memory-ttl 20 \
  --analysis-memory-ttl 20
```

Validation này kiểm tra hai invariant:

- bật/tắt semantic mask không làm thay đổi observation;
- đổi v1/v2 không làm thay đổi action mask khi mask setting được giữ nguyên.

## Trạng Thái Xác Minh Kỹ Thuật

Tại thời điểm hoàn tất implementation:

- validation 2x2 đạt toàn bộ năm independence checks;
- observation dimension là 1,651 cho v1 và 2,335 cho v2;
- semantic mask ban đầu loại toàn bộ Allow chưa có block tương ứng nhưng giữ
  nguyên Block hợp lệ;
- 39 test CAGE 4/canonical/PPO/GCN/audit đã pass;
- Flat và GCN đều hoàn thành `collect_rollout -> GAE -> PPO update` với v2 và
  semantic mask;
- smoke Flat 9 simulator ticks tạo checkpoint và chạy audit v2 thành công;
- trong smoke audit, 6 traffic actions hoàn tất có 0 `FALSE` completion.

Các con số smoke không phải performance evidence và không được đưa vào bảng kết
quả nghiên cứu. Qualification A/B/C/D ở 20k, seed 42 đã hoàn tất; kết quả và
giới hạn diễn giải nằm trong `CANONICAL_2X2_RANDOM_HOST_20K_RESULT.md`.
