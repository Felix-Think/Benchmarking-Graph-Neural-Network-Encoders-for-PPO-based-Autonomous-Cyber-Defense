# Audit Dynamic Evidence, Action Timing and Reward

## Phạm vi

Audit sử dụng checkpoint cuối của matched Flat và GCN 100k trên
`random_host`, cùng 30 held-out environment seeds 10000-10029 và stochastic
action seed 10000. Mỗi trace tái lập chính xác từng episode return và toàn bộ
action counts trong evaluation gốc.

Đơn vị suy luận thống kê là episode, không phải simulator tick hay policy
decision. Khoảng tin cậy là paired bootstrap 95% trên 30 environment seeds.
Chỉ có một PPO training seed (42), vì vậy audit không ước lượng được training
variance.

Audit ghi ba mốc riêng:

1. evidence có trong canonical observation trước action;
2. policy decision và thời điểm action hoàn thành;
3. reward sau simulator step, tách `BlueRewardMachine` và `action_cost`.

Reward sau action chỉ là liên hệ quan sát. Năm Blue agents cùng tác động lên một
team reward, Red/Green đồng thời hành động, và nhiều Blue action kéo dài nhiều
tick; do đó không diễn giải nó thành causal action effect.

## 1. Event channel có hoạt động đúng không?

Có, ở mức binary event transport hiện tại.

| Chỉ số | Flat trajectory | GCN trajectory |
|---|---:|---:|
| Evidence host-ticks | 9,390 | 10,500 |
| Process events | 2,582 | 2,878 |
| Connection events | 6,113 | 6,820 |
| Successful Analyse markers | 1,449 | 1,813 |
| Unobserved `current_*` buffer items | 0 | 0 |

Tất cả passive events được canonical đọc từ `old_process_creation` hoặc
`old_network_connections` sau automatic Monitor. Không thấy lỗi kiểu CAGE 2
trước đây, nơi process indicator bị bật thường trực do đọc process state thay
vì process event.

Tuy nhiên representation vẫn làm mất thông tin:

- process/connection events bị nén thành một bit cho mỗi host;
- benign Green false-positive events và Red-related events không được phân
  biệt bằng các thuộc tính Blue quan sát được;
- `analysis_result_now` chỉ cho biết `Analyse` thành công, không chứng minh host
  compromised;
- `analysis_memory` là action-history marker, không phải security evidence.

Vì vậy tên gọi và cách gộp feature hiện tại có nguy cơ khiến ta overclaim rằng
policy đang học threat evidence.

## 2. Evidence có đến kịp policy decision không?

| Chỉ số episode-mean | Flat | GCN |
|---|---:|---:|
| Decisions có current evidence | 19.31% | 20.99% |
| Decisions có memory evidence | 95.74% | 95.04% |
| Event xảy ra khi agent decision-ready | 38.62% | 44.83% |
| Event có decision kế tiếp trong 1 tick | 58.40% | 64.19% |
| Cùng host được phản hồi trong 10 ticks | 19.12% | 22.45% |
| Median latency đến cùng-host action | 29 ticks | 27 ticks |

Action duration giải thích vì sao không phải event nào cũng trùng đúng một
decision. Tuy nhiên gần như mọi event có decision kế tiếp trong tối đa năm
ticks, trong khi median same-host response vẫn là 27-29 ticks. Do đó chậm phản
hồi không chỉ do action đang `IN_PROGRESS`; policy thường chọn host/action khác.

Memory hiện diện ở khoảng 95% decisions. Binary memory được giữ đến khi
`Restore` thành công và age sau cùng bão hòa ở 1.0. Đây là dấu hiệu memory quá
sticky, làm `memory present` gần như không còn khả năng phân biệt state.

## 3. Policy có chọn đúng host có evidence không?

Trong các host-targeted decisions khi current evidence tồn tại:

| Chỉ số | Flat | GCN |
|---|---:|---:|
| Target thật sự có current evidence | 10.99% | 18.86% |
| Uniform-host chance | 8.98% | 9.06% |
| Episode-mean excess over chance | +2.00 pp | +8.72 pp |
| Bootstrap CI của excess | [0.88, 3.12] pp | [6.25, 11.07] pp |

GCN có bằng chứng căn chỉnh current evidence với target host tốt hơn Flat.
Paired GCN-minus-Flat excess là +6.71 pp, CI [3.95, 9.27] pp.

Nhưng kết quả memory khác hẳn:

- Flat memory-target alignment thấp hơn uniform chance 10.45 pp.
- GCN memory-target alignment thấp hơn uniform chance 5.12 pp.

Vì vậy GCN định vị event mới tốt hơn, nhưng không ưu tiên các host đang mang
sticky memory theo cách đơn giản. Điều này phù hợp với khả năng memory đang tác
động lên action type hoặc global strategy hơn là tạo targeted response.

## 4. Policy nhạy với feature nào?

Total variation (TV) đo độ thay đổi của toàn bộ masked action distribution khi
chỉ zero một feature group trên cùng observation. TV=0 nghĩa là distribution
không đổi; TV=1 là khác hoàn toàn.

| Feature bị zero khi feature hiện diện | Flat TV | GCN TV |
|---|---:|---:|
| Passive process/connection event now | 0.0078 | 0.0050 |
| Analyse marker/file density now | 0.0088 | 0.0016 |
| Passive process/connection memory | 0.0592 | 0.1255 |
| Analyse/file-density memory | 0.0072 | 0.0127 |
| `observed_now` | 0.0099 | 0.0059 |
| Evidence age | 0.0198 | 0.0699 |
| Decoy state | 0.0072 | 0.0196 |
| Toàn bộ dynamic block | 0.0754 | 0.1931 |

GCN nhạy với dynamic representation mạnh hơn Flat, nhưng phần lớn độ nhạy đến
từ passive-event memory và evidence age. GCN thậm chí ít nhạy hơn Flat với
current passive events và current Analyse result.

Kết luận chính xác là:

> GCN đã học phụ thuộc vào temporal belief representation, đặc biệt là sticky
> alert memory và age; chưa có bằng chứng nó khai thác mạnh event mới tại thời
> điểm event xuất hiện.

Đây giải thích vì sao GCN có thể có input sensitivity cao nhưng vẫn không vượt
strict full-versus-static return gate.

## 5. Action và reward liên hệ như thế nào?

| Reward trung bình mỗi episode | Flat | GCN | GCN - Flat |
|---|---:|---:|---:|
| Total | -762.83 | -1034.33 | -271.50 |
| BlueRewardMachine | -448.93 | -758.80 | -309.87 |
| Action cost | -313.90 | -275.53 | +38.37 |

GCN dùng ít `Restore` hơn và tiết kiệm khoảng 38.37 action-cost points mỗi
episode. Tuy nhiên mission-outcome penalty tệ hơn khoảng 309.87, nên tổng return
vẫn thấp hơn 271.50.

Mọi `Restore` được chọn có cost -1 và duration năm ticks. Tổng Restore selections
khớp chính xác tổng action-cost magnitude:

- Flat: 9,417 Restore, action cost -9,417;
- GCN: 8,266 Restore, action cost -8,266.

Do đó Flat tốt hơn không phải vì action-cost thấp; nó trả nhiều Restore cost hơn
nhưng tránh được nhiều Green failure/Red Impact penalty hơn.

### Redundant traffic actions

| AllowTrafficZone | Flat | GCN |
|---|---:|---:|
| Selected | 1,697 | 4,484 |
| Returned `FALSE` | 1,671 (98.47%) | 4,390 (97.90%) |

`AllowTrafficZone` trả `FALSE` khi edge chưa bị block. Official action mask chỉ
mask host/subnet không tồn tại; nó không mask action không hợp lệ theo firewall
state. GCN dành 24.13% decisions cho `AllowTrafficZone`, phần lớn là no-op. Đây
là một action-validity confound rõ ràng, không phải bằng chứng GNN encoder yếu.

### Reward ngay sau evidence-aligned action

Flat có next-5 reward trung bình -9.92 khi target có current evidence và -10.27
khi current evidence ở host khác. GCN tương ứng là -14.67 và -14.16. Không có
tín hiệu nhất quán rằng targeted response cải thiện reward ngay lập tức.

Không được diễn giải các số này là targeted action vô ích: evidence thường xuất
hiện khi threat/activity đã cao, nên reward xấu có reverse causality; action type,
mission phase, host criticality và bốn Blue agents khác cũng chưa được kiểm soát.

## 6. Kết luận audit

1. Canonical event transport không bị lỗi always-on và không thấy hidden
   unobserved-event leakage trong trace này.
2. Current evidence có đến observation, nhưng chỉ trùng khoảng 19-21% policy
   decisions; memory là cần thiết để nối action duration với decision tiếp theo.
3. Memory hiện tại quá persistent và gần bão hòa.
4. GCN căn chỉnh target với current evidence tốt hơn Flat, nhưng action
   distribution của nó gần như không nhạy với chính current-event bits.
5. Độ nhạy GCN chủ yếu đến từ passive memory và evidence age; độ nhạy đó chưa
   chuyển thành reward tốt hơn.
6. GCN thất bại chủ yếu ở mission outcome, không phải Restore action cost.
7. Redundant `AllowTrafficZone` là blocker của action interface và ảnh hưởng GCN
   mạnh hơn Flat.

## 7. Quyết định trước bước train tiếp theo

Không nên chỉ train thêm seed hoặc tăng budget trên canonical v1 hiện tại. Hai
can thiệp nên được triển khai và qualification riêng:

1. **Canonical v2 evidence semantics**
   - tách passive alert memory khỏi Analyse/action history;
   - giữ các thuộc tính event Blue thực sự thấy thay vì một bit duy nhất;
   - dùng last-seen age/decay hoặc TTL cho passive memory thay vì sticky binary;
   - không dùng hidden Red identity để gắn nhãn event trong policy input.
2. **Matched semantic action validity**
   - mask `AllowTrafficZone` khi edge không bị block;
   - mask `BlockTrafficZone` khi edge đã bị block;
   - áp dụng cùng một mask cho Flat và mọi GNN;
   - báo cáo redundant/invalid action rate như metric kiểm soát.

Reward shaping chưa phải thay đổi đầu tiên. Reward hiện tại đã cho thấy rõ
mission penalty và Restore cost; sửa reward trước sẽ che khuất hai lỗi
representation/action interface vừa xác định.

Sau mỗi can thiệp cần chạy lại ba qualification gates:

- current-event và passive-memory feature-group sensitivity;
- target alignment excess over uniform-host chance;
- full-versus-static held-out return gate.

Chỉ khi các gate này qua trên nhiều training seeds mới dùng kết quả để claim GNN
khai thác partial observation tốt hơn Flat.

## Artifacts

- `results/audit_dynamic_evidence_flat_random_pivot_100k_summary.json`
- `results/audit_dynamic_evidence_gcn_random_pivot_100k_summary.json`
- `results/audit_dynamic_evidence_matched_100k_comparison.json`
- `results/audit_dynamic_evidence_*_decisions.csv.gz`
- `results/audit_dynamic_evidence_*_events.csv.gz`
- `results/audit_dynamic_evidence_*_ticks.csv.gz`
