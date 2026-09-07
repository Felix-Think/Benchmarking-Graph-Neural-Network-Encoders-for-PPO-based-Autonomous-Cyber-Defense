# Kết luận GCN Dynamic-Evidence Qualification - Seed 42

## Protocol

- Canonical V1 `action_aware`, CAGE 4, episode length 400.
- GCN: 2 lớp, 28 chiều ẩn, ordered-node concatenation.
- PPO budget yêu cầu: 100.000 simulator steps; thực chạy: 100.149.
- 251 PPO updates, 164.252 policy decisions.
- Số tham số GCN: 291.735; Flat reference: 292.339.
- Evaluation chính: stochastic, 30 held-out episodes, seeds 10000-10029.
- Tất cả gates dùng cùng action-sampling seed 10000.
- Khoảng tin cậy: paired bootstrap 95%, 20.000 resamples.

## Hiệu năng tổng thể

| Evaluation | Mean return | Std |
|---|---:|---:|
| GCN stochastic | -867.60 | 317.07 |
| GCN deterministic | -4163.60 | 891.58 |
| Flat stochastic reference, 30k | -834.53 | 271.06 |

GCN đã học được policy hữu ích, nhưng deterministic argmax vẫn collapse. GCN và
Flat stochastic không khác rõ trên các held-out seeds: GCN - Flat = -33,07,
CI [-191,70; 121,23]. Đây không phải comparison công bằng cuối cùng vì GCN được
train 100k còn Flat reference chỉ được train 30k.

## Feature-group gates

| Gate | Mean return | Std |
|---|---:|---:|
| Full | -867.60 | 317.07 |
| Static only | -864.90 | 302.96 |
| Zero all | -3674.67 | 744.18 |
| Mission only | -3478.00 | 606.96 |
| Node identity only | -1110.87 | 414.51 |
| Structure only | -3516.00 | 627.55 |
| Topology only | -1256.40 | 407.77 |
| Topology không adjacency | -1408.40 | 519.87 |
| Topology + mission | -1255.67 | 420.31 |
| Full không action mask | -1469.93 | 509.58 |

## Paired effects

- Dynamic evidence khi đã có static context: -2,70,
  CI [-130,47; 127,17].
- Node identity khi so với zero observation: +2563,80,
  CI [2204,06; 2943,13].
- Structure khi so với zero observation: +158,67,
  CI [-68,27; 391,60].
- Adjacency khi đã có identity và membership: +152,00,
  CI [-73,23; 382,33].
- Mission khi đã có topology/static: +0,73,
  CI [-209,47; 210,60].
- Communication-policy, firewall và non-mission globals khi thêm vào
  topology + mission: +390,77, CI [190,63; 590,67].
- Action mask khi có full observation: +602,33,
  CI [412,23; 795,93].

## Diễn giải

1. GCN có khả năng nhận dynamic features về mặt kiến trúc, nhưng checkpoint 100k
   chưa chuyển chúng thành cải thiện return. Full và static-only gần như bằng
   nhau, nên chưa có evidence về dynamic-evidence exploitation.
2. Full và static-only có action-type total-variation 0,068. `Analyse` giảm từ
   14,8% xuống 8,0% khi xóa dynamic features, nhưng trajectories sau action đầu
   sẽ phân kỳ. Đây chỉ là behavioral sensitivity diagnostic, chưa chứng minh
   dynamic evidence mang lại utility.
3. Nguồn thông tin chính vẫn là node identity/type/existence/control theo slot.
4. Adjacency có delta dương nhỏ nhưng CI chứa 0; chưa thể claim GCN sử dụng graph
   edges hiệu quả.
5. Action mask quan trọng, nhưng khi bỏ mask policy chọn `[Padding]` hoặc
   `[Invalid]` khoảng 77,6% decisions. Delta mask một phần phản ánh fixed-action
   wrapper contract.
6. GCN dùng ordered-node concatenation để khớp fixed action head. Kết quả này chỉ
   áp dụng cho same-scale qualification, không chứng minh cross-scale
   generalisation.

## So sánh mức sử dụng feature với Flat

- Dynamic-use interaction:
  `(GCN full - GCN static) - (Flat full - Flat static) = -22,17`,
  CI [-193,70; 166,60].
- Adjacency-use interaction: +91,30, CI [-156,90; 343,70].

Không có interaction nào khác 0. Hiện chưa có bằng chứng GCN khai thác dynamic
evidence hoặc adjacency tốt hơn Flat.

## Quyết định

- Không tiếp tục tăng training budget hoặc fine-tune GCN trên protocol hiện tại.
- Không kết luận GCN kém về khả năng khai thác dynamic evidence nói chung; chỉ
  kết luận task hiện tại không buộc checkpoint này sử dụng evidence đó.
- Bước tiếp theo là tối ưu train/test environment distribution để static-slot
  strategy không đủ, trong khi giữ canonical schema, reward, PPO core, action
  mask và parameter budget không đổi.
- Sau khi environment qualification đạt, train lại Flat và GCN với cùng budget,
  ít nhất ba training seeds, rồi lặp lại full/static/edge gates.

## Giới hạn

Đây là một training seed. Evaluation-time gates đo sufficiency và utility của
feature groups cho checkpoint này, không chứng minh quan hệ nhân quả duy nhất.
