# Kết quả qualification canonical 2x2 trên `random_host`

## Phạm vi

Đây là qualification kỹ thuật ở 20k simulator ticks, chưa phải thí nghiệm chính
và chưa phải bằng chứng Flat/GCN. Mục tiêu là tách hai can thiệp:

| Condition | Schema | Semantic mask | Encoder width | Parameters |
|---|---|---:|---:|---:|
| A | v1 | Off | 128 | 292,339 |
| B | v2 | Off | 92 | 291,187 |
| C | v1 | On | 128 | 292,339 |
| D | v2 | On | 92 | 291,187 |

Mọi condition dùng cùng Flat-PPO, training seed 42, distribution `random_host`,
20,000 requested ticks, episode length 400, reward gốc nhân fixed scale 0.01,
action catalogue, PPO hyperparameters và 30 held-out evaluation seeds
10,000-10,029. Stochastic evaluation là kết quả chính; deterministic evaluation
chỉ là diagnostic. Script kiểm tra xác nhận tất cả controlled config fields bằng
nhau. Chênh lệch parameter budget lớn nhất là 0.39%.

## Kết quả return

Return cao hơn, tức ít âm hơn, là tốt hơn.

| Condition | Stochastic mean +/- SD | Deterministic mean +/- SD | Mean decisions/episode |
|---|---:|---:|---:|
| A | -754.97 +/- 252.76 | -3491.67 +/- 780.73 | 605.33 |
| B | -1434.50 +/- 509.21 | -4162.00 +/- 904.50 | 761.63 |
| C | -1032.97 +/- 366.17 | -3430.40 +/- 1013.73 | 573.30 |
| D | -1054.47 +/- 515.67 | -3648.83 +/- 817.78 | 616.73 |

Ở training seed 42, A là policy stochastic tốt nhất. D không vượt A. Vì vậy kết
quả này không hỗ trợ thay canonical v1 bằng v2 trước encoder benchmark.

## Paired contrasts

Các contrast dưới đây được tính theo cùng 30 environment seeds. Bootstrap 95%
CI phản ánh variation giữa held-out episodes của bốn policy đã train; nó không
phản ánh training-seed variance.

| Contrast | Mean return difference | Paired bootstrap 95% CI | Diễn giải ở seed 42 |
|---|---:|---:|---|
| B - A | -679.53 | [-872.00, -491.33] | v2 làm kém đi khi mask Off |
| C - A | -278.00 | [-441.87, -126.90] | mask làm kém đi trên v1 |
| D - B | +380.03 | [+102.90, +651.83] | mask cứu một phần v2 |
| D - C | -21.50 | [-252.70, +198.57] | v1/v2 gần như chưa tách được khi mask On |
| V2 main effect | -350.52 | [-510.12, -189.33] | v2 bất lợi trung bình ở qualification này |
| Mask main effect | +51.02 | [-131.17, +226.00] | không có hiệu ứng chung rõ ràng |
| Interaction | +658.03 | [+385.73, +930.34] | tác dụng mask phụ thuộc mạnh vào schema |

Không được diễn giải CI này như độ chắc chắn qua nhiều lần train PPO. Chỉ có một
training seed nên mọi kết luận performance hiện tại vẫn là qualification.

## Audit cơ chế action mask

Audit dùng ba held-out episodes, stochastic action seed 10,000, cùng checkpoint
với evaluation. `FALSE completion` là action traffic được simulator hoàn tất
nhưng báo thất bại.

| Condition | Completed traffic actions | FALSE completions | FALSE rate |
|---|---:|---:|---:|
| A | 271 | 247 | 91.14% |
| B | 621 | 589 | 94.85% |
| C | 92 | 0 | 0.00% |
| D | 59 | 0 | 0.00% |

Semantic mask đạt đúng mục tiêu cơ chế: loại các cặp Allow/Block dư thừa mà
không tạo empty action set. Tuy nhiên, C kém A về stochastic return. Vì vậy tính
hợp lệ của action và chất lượng policy là hai câu hỏi khác nhau. Không được dùng
việc false rate về 0 để claim mask cải thiện reward trong mọi schema.

Action composition cũng thay đổi mạnh. Trong 30 evaluation episodes, B dùng
`AllowTrafficZone` cho 29.95% decisions; D chỉ còn 1.72%, đồng thời tăng
`Analyse` từ 15.53% lên 23.60% và `Remove` từ 15.17% lên 26.75%. Điều này giải
thích vì sao mask có thể cứu v2: nó loại một attractor action sai và phân phối
probability mass sang các hành động phòng thủ khác. Trên v1, policy A vốn có
return tốt dù vẫn chọn traffic action sai, nên thay đổi action distribution ở C
không tự động tạo policy tốt hơn trong 20k ticks.

## Dynamic evidence diagnostic

Audit ba episode cho mean total-variation giữa policy đầy đủ và policy được zero
toàn bộ dynamic features:

| Condition | Mean TV | Argmax-change rate | Same-host response within 10 ticks |
|---|---:|---:|---:|
| A | 0.0345 | 0.00% | 16.85% |
| B | 0.0182 | 2.15% | 18.01% |
| C | 0.0515 | 1.57% | 20.03% |
| D | 0.0169 | 0.77% | 25.02% |

Các policy có phản ứng với dynamic features ở mức probability distribution,
nhưng ảnh hưởng còn nhỏ và hiếm khi đổi argmax. Canonical v2 chưa chứng minh
được lợi thế khai thác dynamic evidence. D có response rate cao hơn B, nhưng ba
episode không đủ để biến chênh lệch này thành claim.

## Kết luận quyết định

1. Semantic action mask được xác minh đúng về cơ chế và nên được giữ như một
   control option của benchmark, nhưng chưa được mặc định hóa chỉ dựa trên return.
2. Canonical v2 hiện chưa vượt qualification. Không dùng v2 làm schema chính và
   chưa chạy Flat/GCN 100k trên v2.
3. A, tức canonical v1 + official mask, vẫn là baseline learning tốt nhất ở
   training seed 42. Cần giữ A làm reference cho bước tiếp theo.
4. Interaction lớn cho thấy không được tune observation và action mask cùng lúc
   rồi quy toàn bộ improvement cho GNN.
5. Trước main benchmark, cần ít nhất ba training seeds cho các condition còn
   cạnh tranh. Nếu mục tiêu gần nhất là xây GNN, hướng tiết kiệm và sạch nhất là
   dùng canonical v1, chạy Flat/GCN parameter-matched với cả official mask và
   semantic-mask ablation; không tiếp tục mở rộng v2 cho đến khi có lý do cụ thể
   từ evidence audit.

## Tệp tái lập

- `scripts/analysis/compare_canonical_2x2.py`: kiểm tra config và tính paired
  contrasts.
- `canonical_2x2_random_host_20k_comparison.json`: toàn bộ episode returns,
  bootstrap intervals, action counts và PPO diagnostics.
- `audit_2x2_*_summary.json`: action completion và dynamic-evidence audit cho
  từng condition.
