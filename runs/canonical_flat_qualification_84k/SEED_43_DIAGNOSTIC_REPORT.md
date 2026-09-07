# Chẩn đoán seed 43 và nhiễu chọn checkpoint

## Kết luận ngắn

Seed 43 không cho thấy PPO bị lỗi số học riêng biệt. Policy đã học một phân phối
hành động có stochastic return tốt, nhưng deterministic argmax gần như biến thành
một policy tĩnh: bốn Blue agent vùng mạng liên tục `Restore` một host cố định và
Blue agent 4 gần như liên tục `AllowTrafficZone`. Evidence động có làm thay đổi
xác suất hành động, nhưng hầu như không đủ để đổi hành động có xác suất cao nhất.

Checkpoint selection bằng 10 episode cũng có nhiễu đáng kể và đã làm checkpoint
update 80 trông tốt hơn held-out evaluation. Tuy nhiên, selection noise không phải
lời giải thích duy nhất: chính checkpoint được chọn thực sự thất bại ở dynamic
observation-use gate trên 100 held-out episodes.

## 1. Checkpoint selection có nhiễu không?

Có.

- Có 21 checkpoint candidates, nhưng mỗi checkpoint chỉ được đánh giá trên 10
  episode seeds `20000-20009`.
- Update 80 được chọn với mean `-2908.30`; runner-up update 90 là `-3093.70`.
- Chênh lệch quan sát được là `+185.40`, nhưng paired bootstrap 95% CI là
  `[-127.80, 448.80]`. Khoảng này chứa 0 nên 10 episode chưa phân biệt chắc chắn
  update 80 với update 90.
- Khi resample mười periodic episodes, update 80 vẫn thắng khoảng `71.85%` số
  lần. Vì vậy không thể nói update 80 chỉ được chọn do một episode ngoại lệ.
- Tuy nhiên, có 12/21 checkpoints từng trở thành winner trong bootstrap. Việc
  lấy maximum qua nhiều lần đánh giá vẫn tạo winner-selection bias.
- Periodic mean của update 80 là `-2908.30`, nhưng held-out mean trên 100 episodes
  là `-3393.34`: periodic set đánh giá tốt hơn `485.04` reward.

Resample từ 100 held-out returns của chính update 80 cho thấy:

- Standard error ước tính với 10 episodes là khoảng `289.99` reward.
- 95% khoảng biến động của mean 10 episodes là `[-3961.70, -2820.20]`.
- Xác suất một mẫu 10 episodes đạt mean ít nhất `-2908.30` là khoảng `4.85%`.

Mức `4.85%` không quá nhỏ khi ta liên tục chọn giá trị cao nhất qua 21 checkpoint
evaluations. Không được nhân trực tiếp xác suất này thành 21 phép thử độc lập vì
các checkpoints và episode returns có tương quan.

**Diễn giải chính xác:** periodic set 10 seeds không đại diện đủ tốt cho held-out
seed distribution. Leave-one-out vẫn luôn chọn update 80 trong chính mười seeds
đó, nên vấn đề không phải một seed đơn lẻ mà là kích thước và độ đại diện của cả
selection set.

## 2. Seed 43 có dùng evidence động không?

Ở deterministic behavior level: gần như không.

- Full canonical minus static-only mean return: `+0.21`.
- Paired bootstrap 95% CI: `[-2.94, 3.92]`.
- 89/100 episodes có return giống hệt nhau.
- Full canonical chỉ tốt hơn static-only ở 2 episodes; static-only tốt hơn ở 9.

Trên 10 deterministic full-observation trajectories gồm 7,190 policy decisions:

- Xóa evidence động làm policy-distribution thay đổi với mean total-variation
  distance `0.0857`.
- Latent representation thay đổi tương đối khoảng `0.2597` theo L2 norm.
- Nhưng greedy action chỉ đổi ở khoảng `0.0695%` decisions, tức 5/7,190.
- Xóa toàn bộ observation làm greedy action đổi khoảng `78.90%` decisions.

Như vậy network không hoàn toàn bỏ qua observation. Nó chủ yếu dùng static
canonical information và action mask để xác định một hành vi cố định theo agent;
dynamic evidence chỉ điều chỉnh probability nhưng không thay đổi action ranking.

Action pattern trên mười trajectories:

- `Restore`: 3,200 decisions.
- `AllowTrafficZone`: 3,988 decisions.
- `BlockTrafficZone`: 2 decisions.
- Không có `Remove`.

Đối chứng cho thấy dynamic evidence có ý nghĩa hành vi ở các seed PASS:

- Seed 42: full-versus-static greedy action change rate `7.51%`; observation-use
  delta `+359.28`.
- Seed 44: change rate `8.54%`; observation-use delta `+280.35`.
- Seed 43: change rate `0.07%`; observation-use delta `+0.21`.

Do đó observation-use gate không thiếu độ nhạy. Seed 43 thực sự hội tụ vào một
deterministic action-ranking khác hai seed PASS.

## 3. Stochastic policy nói gì?

Best seed 43 có stochastic return `-1131.01`, tốt hơn deterministic return
`-3393.34`. Đây không phải random policy: PPO sampling từ distribution đã học.

Static-only stochastic evaluation đạt `-1192.26`. Full minus static-only là
`+61.25`, nhưng paired bootstrap 95% CI `[-61.91, 184.07]`; full tốt hơn ở 49
episodes và static-only tốt hơn ở 51. Vì vậy chưa có bằng chứng reward-level rằng
dynamic evidence giúp stochastic policy của seed 43, dù logits có phản ứng với
evidence.

Kết quả stochastic tốt chủ yếu cho thấy distribution vẫn chứa nhiều hành động
hữu ích. Deterministic argmax chọn lặp lại một mode không tốt và không đại diện
cho expected behavior khi sampling.

## 4. PPO có bị hỏng ở seed 43 không?

Không có bằng chứng về một lỗi optimizer riêng cho seed 43.

| Metric | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Training return, 20 episodes đầu | -3194.70 | -2842.15 | -3085.05 |
| Training return, 20 episodes cuối | -1306.80 | -1251.50 | -1047.45 |
| Entropy, 20 updates cuối | 3.3973 | 3.2320 | 3.2336 |
| Mean approximate KL | 0.010107 | 0.010239 | 0.010253 |
| Early-stop rate | 95.73% | 95.73% | 97.16% |
| Mean gradient norm trước clipping | 1.6579 | 1.5405 | 1.6668 |

Seed 43 có training improvement, KL và entropy cùng thang với hai seed còn lại.
Early stopping và gradient clipping xảy ra thường xuyên ở cả ba seed; đây là đặc
tính chung của PPO configuration hiện tại, không giải thích riêng failure seed 43.

## 5. Nguyên nhân hợp lý nhất

Evidence hiện tại phù hợp nhất với chuỗi nguyên nhân sau:

1. PPO sampling trong training học được một broad action distribution có reward
   tương đối tốt.
2. Với initialization và trajectory của seed 43, actor logits tạo một mode tĩnh:
   `Restore` cho agent 0-3 và `AllowTrafficZone` cho agent 4.
3. Dynamic evidence thay đổi logits nhưng không đủ đảo top-1 action.
4. Mười periodic seeds tình cờ đánh giá mode này tốt hơn held-out population.
5. Max-selection qua 21 evaluations giữ update 80 làm best checkpoint.

Đây là kết luận có căn cứ mạnh hơn hai khẳng định cực đoan sau:

- Không đủ bằng chứng để nói canonical observation không học được.
- Không đủ bằng chứng để nói seed 43 thất bại chỉ vì checkpoint-selection noise.

## 6. Protocol v1 đề xuất

Không thay reward, observation hay PPO hyperparameters trước khi kiểm tra protocol
selection sạch hơn.

1. Vẫn có thể chạy periodic evaluation 10 episodes để theo dõi, nhưng không dùng
   trực tiếp maximum của 10 episodes làm final checkpoint.
2. Lưu model state ở mọi periodic checkpoint hoặc ít nhất mọi 20 updates. Hiện
   chỉ giữ best và final nên không thể re-evaluate update 90/110/210 trên 100
   held-out episodes.
3. Dùng two-stage selection:
   - Stage A: 10 monitoring seeds để shortlist ba checkpoints.
   - Stage B: 30-50 selection seeds độc lập để chọn một checkpoint trong top ba.
4. Giữ nguyên 100 held-out seeds `10000-10099`; tuyệt đối không dùng chúng để
   shortlist, chọn checkpoint hoặc tuning.
5. Báo cáo deterministic là primary và stochastic là secondary như protocol đã
   định trước. Không đổi primary metric hậu nghiệm vì seed 43 stochastic tốt.
6. Observation-use gate chỉ dùng để qualification sau khi chọn model; không dùng
   gate để chọn checkpoint vì như vậy sẽ ưu tiên artificial policy sensitivity.
7. Chạy protocol v1 trên ít nhất năm training seeds mới trước benchmark Flat-GNN.

Một cấu hình cân bằng chi phí là đánh giá mỗi 20 updates thay vì mỗi 10 updates,
shortlist bằng 10 episodes, rồi đánh giá top ba bằng 30 selection episodes. Tổng
chi phí selection chỉ tăng vừa phải nhưng giảm mạnh việc chọn đỉnh từ mean 10
episodes.

## Giới hạn chẩn đoán

- Intermediate model states không được lưu, nên không thể chứng minh update 90,
  110 hoặc 210 sẽ generalize tốt hơn update 80.
- Bootstrap checkpoint ranking chỉ điều kiện trên mười periodic seeds đã quan sát;
  nó không biết true environment-seed distribution.
- Static-only và zero-all là counterfactual observations, có thể nằm ngoài training
  distribution. Vì vậy cần đọc cả logits, action changes và reward gate cùng nhau.

Raw diagnostics nằm trong `seed_42/seed_diagnostics.json`,
`seed_43/seed_diagnostics.json` và `seed_44/seed_diagnostics.json`.
