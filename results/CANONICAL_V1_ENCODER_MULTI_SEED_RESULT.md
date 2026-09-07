# Kết quả Canonical V1 Flat/GCN/GraphSAGE với 5 Training Seeds

## 1. Phạm vi

Thí nghiệm so sánh Flat, GCN và GraphSAGE trên cùng canonical v1, PPO, reward,
action catalogue và `random_host` distribution. Official CAGE 4 action mask là
cấu hình native và kết quả chính. Semantic traffic mask là ablation độc lập,
được áp dụng đối xứng cho cả ba encoder.

Mỗi condition có năm policy train độc lập với seed 42-46. Mỗi policy được đánh
giá stochastic trên 30 held-out environment seeds 10,000-10,029. Đơn vị lặp
độc lập là training seed, không phải từng evaluation episode.

## 2. Kiểm soát thực nghiệm

- Canonical schema: v1, variant `action_aware`, 1,651 dimensions.
- Distribution: `random_host`; episode dài 400 simulator ticks.
- Budget yêu cầu: 100,000 ticks; cả 30 policies thực tế chạy 100,149 ticks do
  rollout kết thúc trọn episode.
- PPO, reward scale, action catalogue và evaluation seeds giống nhau giữa sáu
  conditions.
- Flat: 292,339 parameters; GCN: 291,735; GraphSAGE: 292,995. Sai lệch lớn nhất
  so với Flat là 0.22%.
- Graph construction, ordered node embeddings, context projection và PPO heads
  của GCN/GraphSAGE giống nhau; chỉ thay `GCNConv` bằng `SAGEConv`.
- Không reward shaping, periodic evaluation, checkpoint selection hay chọn seed
  theo return.
- Config audit tự động: **PASS**.

## 3. Held-Out Stochastic Return

Return càng cao, tức càng ít âm, càng tốt.

| Condition | S42 | S43 | S44 | S45 | S46 | Mean | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-O: Flat + official | -762.83 | -919.70 | -730.87 | -827.87 | -1087.50 | -865.75 | 143.44 |
| G-O: GCN + official | -1034.33 | -612.97 | -1101.20 | -784.43 | -801.17 | -866.82 | 199.08 |
| S-O: GraphSAGE + official | -723.80 | -1248.77 | -1015.50 | -786.37 | -967.30 | -948.35 | 207.23 |
| F-S: Flat + semantic | -640.03 | -838.00 | -645.00 | -639.30 | -919.07 | **-736.28** | **133.00** |
| G-S: GCN + semantic | -683.80 | -647.47 | -580.03 | -884.70 | -1102.80 | -779.76 | 213.29 |
| S-S: GraphSAGE + semantic | -735.40 | -678.80 | -624.77 | -866.57 | -969.60 | -775.03 | 141.11 |

Trong native configuration, Flat và GCN gần như hòa về mean; GraphSAGE thấp
hơn khoảng 82 điểm và không ổn định hơn. Với semantic mask, Flat có mean tốt
nhất và SD thấp nhất; GraphSAGE gần hòa GCN về mean nhưng ổn định hơn GCN.

Không encoder nào thắng theo mọi seed. Seed variance lớn hơn phần lớn mean
encoder differences, nên không được chọn một seed đẹp để làm đại diện.

## 4. Paired Contrasts

| Contrast | Mean delta | Student-t 95% CI | Thắng/hòa/thua |
|---|---:|---:|---:|
| GCN - Flat, official | -1.07 | [-388.19, +386.05] | 3/0/2 |
| GraphSAGE - Flat, official | -82.59 | [-340.73, +175.55] | 3/0/2 |
| GraphSAGE - GCN, official | -81.53 | [-521.57, +358.51] | 2/0/3 |
| GCN - Flat, semantic | -43.48 | [-264.70, +177.74] | 2/0/3 |
| GraphSAGE - Flat, semantic | -38.75 | [-216.03, +138.54] | 2/0/3 |
| GraphSAGE - GCN, semantic | +4.73 | [-90.65, +100.12] | 2/0/3 |
| Semantic - official, Flat | +129.47 | [+69.80, +189.15] | 5/0/0 |
| Semantic - official, GCN | +87.06 | [-333.47, +507.59] | 2/0/3 |
| Semantic - official, GraphSAGE | +173.32 | [-185.43, +532.07] | 2/0/3 |

Không pairwise encoder contrast nào có interval loại trừ 0. GraphSAGE-vs-Flat
main effect trung bình qua hai masks là -60.67 với CI [-141.43, +20.09]. Kết
quả nghiêng về Flat nhưng chưa đủ để kết luận GraphSAGE kém hơn một cách tổng
quát.

Semantic mask cải thiện Flat ở 5/5 seed. Với hai graph encoders, mean mask
effect dương chủ yếu do một vài seed cải thiện rất mạnh và không nhất quán về
dấu. Semantic mask vì vậy vẫn là ablation, không phải thành phần mặc định của
GNN.

## 5. Action-Mask Audit

Audit completion dùng ba held-out episodes trên mỗi training seed, tổng cộng 15
episodes cho mỗi condition.

| Condition | Traffic completions | `FALSE` | False rate | Traffic share | Restore share |
|---|---:|---:|---:|---:|---:|
| F-O | 2,061 | 1,883 | 91.36% | 22.61% | 39.26% |
| G-O | 2,113 | 1,968 | 93.14% | 23.64% | 39.69% |
| S-O | 1,619 | 1,509 | 93.21% | 17.63% | 34.73% |
| F-S | 313 | 0 | 0.00% | 3.73% | 41.53% |
| G-S | 156 | 0 | 0.00% | 2.14% | 43.11% |
| S-S | 228 | 0 | 0.00% | 2.74% | 43.02% |

Official mask tạo cùng traffic-action failure mode cho cả ba encoder. Semantic
mask loại bỏ failure này, nhưng GraphSAGE-semantic dành khoảng 80.2% action cho
`Remove` và `Restore`, cao hơn Flat/GCN semantic. Đây là action-composition
diagnostic, chưa phải bằng chứng nhân quả cho return.

## 6. Dynamic-Observation Gates

`Full - static-only` so sánh canonical đầy đủ với cùng checkpoint khi toàn bộ
dynamic node features bị đặt về 0.

| Condition | Mean delta | Student-t 95% CI | Full tốt hơn theo seed |
|---|---:|---:|---:|
| F-O | +66.93 | [+13.07, +120.78] | 5/5 |
| G-O | +54.05 | [-86.17, +194.27] | 3/5 |
| S-O | +111.69 | [-115.01, +338.38] | 4/5 |
| F-S | +49.84 | [-2.06, +101.74] | 4/5 |
| G-S | +115.50 | [+24.68, +206.32] | 5/5 |
| S-S | +59.57 | [-80.52, +199.66] | 4/5 |

Flat-official và GCN-semantic có utility lặp lại rõ nhất khi giữ dynamic
features. GraphSAGE có positive mean gate ở cả hai masks nhưng interval rộng và
một seed âm trong mỗi condition.

Counterfactual sensitivity khi xóa dynamic node features:

| Condition | Mean total variation | Argmax change rate |
|---|---:|---:|
| F-O | 0.092 | 17.82% |
| G-O | 0.158 | 35.89% |
| S-O | **0.228** | **49.82%** |
| F-S | 0.091 | 14.04% |
| G-S | 0.203 | 37.83% |
| S-S | 0.151 | 40.91% |

GraphSAGE-official nhạy nhất với dynamic input nhưng có return thấp nhất trong
native conditions. Do đó, sensitivity không đồng nghĩa với useful information
use. Agent có thể phản ứng mạnh nhưng phản ứng sai, không ổn định hoặc không
chuyển được representation thành action có reward tốt.

## 7. PPO Diagnostics

Last-25-update condition means: `approx_kl` nằm trong 0.0095-0.0107, value loss
trong 0.0130-0.0223 và pre-clipping gradient norm trong 1.40-1.53. Không có dấu
hiệu critic nổ riêng ở GraphSAGE. Target-KL early stopping vẫn xuất hiện thường
xuyên ở cả ba encoder, nên đây là shared training-regime sensitivity chứ không
phải lời giải thích riêng cho GraphSAGE.

Deterministic argmax mean return nằm từ -3299 đến -3755 và thấp hơn stochastic
ở mọi condition. Stochastic evaluation tiếp tục là primary; deterministic chỉ
là argmax-pathology diagnostic.

## 8. Kiểm thử và Tái lập

GraphSAGE được kiểm thử về parameter matching, action-mask validity, dynamic
evidence response và complete PPO rollout. Training seed 46 được chạy tuần tự,
một policy tại một thời điểm, để giới hạn tải CPU; model/protocol parameters
không thay đổi.

F-O seed 44 trước đó từng gặp CAGE 4 `Sessions` lookup bug trong evaluation.
Local patch chỉ bỏ qua malformed observation entry và evaluation được chạy lại
từ đúng final checkpoint; không policy nào bị retrain do lỗi này.

## 9. Kết luận và Quyết định

1. Canonical v1 giữ lại cho bước partial-observation benchmark.
2. Flat, GCN và GraphSAGE đều được parameter-matched và có đủ năm training
   seeds; bộ ba encoder đã đủ cho partial-observation pilot tiếp theo.
3. Chưa có bằng chứng GCN hoặc GraphSAGE vượt Flat trong full-observation
   qualification. Đây là null result hợp lệ, không phải lý do sửa reward hay
   chọn seed có lợi cho GNN.
4. Official mask tiếp tục là primary/native; semantic mask là ablation riêng.
5. Flat-semantic hiện có mean và seed SD tốt nhất, nhưng không được dùng làm
   bằng chứng cho encoder superiority.
6. GraphSAGE dynamic sensitivity cao không chuyển thành stable return. Partial
   observation cần đo robustness curve và degradation, không chỉ raw return.
7. Không tuning encoder trên held-out seeds 10,000-10,029. Mọi tuning tiếp theo
   phải dùng validation seeds/distribution riêng.

Machine-readable output: `results/canonical_v1_encoder_multi_seed.json`.
Protocol: `results/CANONICAL_V1_ENCODER_MULTI_SEED_PROTOCOL.md` và
`results/CANONICAL_V1_GRAPHSAGE_EXTENSION_PROTOCOL.md`.
