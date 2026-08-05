# Giao thức Pilot 2: kiểm định quá trình quan sát trên mạng 16 host

## 1. Mục tiêu và phạm vi kết luận

Pilot 2-A kiểm tra liệu pipeline PPO hiện tại có đủ sạch để so sánh một bộ mã hóa vector phẳng (Flat) với GraphSAGE khi lượng thông tin động quan sát được thay đổi.

Pilot này được phép kết luận về:

- tính đúng đắn của cơ chế che quan sát theo ngân sách cố định;
- khả năng học của Flat và GraphSAGE trên cùng PPO, reward, action space và ngân sách huấn luyện;
- đường đáp ứng theo tỷ lệ giữ lại quan sát (q);
- quan hệ giữa reward, mission availability và mức độ host bị chiếm quyền;
- chi phí tính toán sơ bộ.

Pilot này chưa được phép kết luận về:

- khả năng khái quát hóa sang topology chưa từng thấy;
- khả năng khái quát hóa khác kích thước mạng;
- encoder GNN nào tốt nhất;
- lợi ích riêng của message passing tách khỏi mọi topology prior;
- độ bền trước một loại quá trình quan sát khác với quá trình đã dùng khi train.

Lý do chặn kết luận topology: graph hiện tại dùng cho GNN là graph đầu vào được dựng thủ công. Nó chưa phải cùng một nguồn sự thật với router, subnet và Network Access Control List (NACL, danh sách kiểm soát truy cập mạng) quyết định reachability trong CybORG. Thay edge đầu vào mà không thay reachability của simulator chỉ tạo nhiễu biểu diễn, không phải thay đổi topology thật.

## 2. Quan hệ với Pilot 1

Pilot 1 được giữ nguyên như bằng chứng kiểm tra kỹ thuật và khả năng huấn luyện ban đầu. Không ghi đè kết quả cũ.

Pilot 2 thay đổi bốn điểm:

1. Mỗi policy được train trên hỗn hợp bốn mức (q), thay vì train một policy riêng cho từng mức.
2. Số host nhìn thấy tại mỗi timestep là cố định chính xác, thay vì lấy mẫu Bernoulli khiến số host dao động.
3. Không ép host vừa được Blue chọn làm target phải hiện ra trong quan sát sau đó.
4. Flat không nhận adjacency; kết luận phải được viết là so sánh gói biểu diễn topology-aware với baseline vector thực dụng, không phải so sánh hoàn toàn đồng nhất thông tin đầu vào.

## 3. Ma trận chạy

| Thành phần | Giá trị |
|---|---|
| Scenario | CAGE 2, 16 host, Red Meander |
| Encoder | Flat, GraphSAGE |
| Seed huấn luyện | 0, 1, 2 |
| Số run PPO | 2 encoder x 3 seed = 6 |
| Ngân sách mỗi run | 1.000.000 timestep |
| Độ dài episode tối đa | 100 timestep |
| Mức giữ lại quan sát | 1.00, 0.75, 0.50, 0.25 |
| Chọn (q) khi train | Đồng xác suất, một lần ở đầu mỗi episode |
| Đánh giá checkpoint | 20 episode riêng cho mỗi (q) |
| Đánh giá cuối | 100 episode riêng cho mỗi (q), dùng seed chưa dùng khi theo dõi train |

Tất cả encoder dùng chung PPO core, reward, action catalogue, số timestep, lịch đánh giá và environment seed. Đây là các biến được kiểm soát.

## 4. Định nghĩa chính xác về mức quan sát

### 4.1 Đơn vị bị che

Đơn vị tính ngân sách là các host slot đã biết và có thể tác động, không gồm router. Với scenario hiện tại, mẫu số là 16 host.

Static feature (đặc trưng tĩnh) như role, subnet, criticality và graph topology vẫn được giữ lại. Chỉ dynamic feature (đặc trưng động) như activity, trạng thái compromise và cờ visibility bị điều khiển bởi mask.

Pilot 2-A không học thêm normalizer từ dữ liệu. Các count feature dùng phép scale xác định sẵn trong observation encoder và checkpoint ghi `normalizer_statistics: null`. Nếu main benchmark thêm learned normalizer, statistics chỉ được fit trên training observation và phải giống nhau giữa các encoder.

Vì vậy, (q) phải được diễn giải là tỷ lệ giữ lại thông tin động theo host, không phải phần trăm toàn bộ true state của CybORG mà agent nhìn thấy.

### 4.2 Ngân sách toàn cục

Với (N=16), số host được quan sát tại mỗi timestep là:

\[
K(q)=\operatorname{round\_half\_up}(qN)
\]

Kết quả bắt buộc:

| (q) | Host nhìn thấy | Host bị che |
|---:|---:|---:|
| 1.00 | 16 | 0 |
| 0.75 | 12 | 4 |
| 0.50 | 8 | 8 |
| 0.25 | 4 | 12 |

### 4.3 Phân bổ theo subnet

Ngân sách (K) được chia cho các subnet theo phương pháp largest remainder (phần dư lớn nhất):

1. Tính quota lý tưởng theo tỷ lệ số host của từng subnet.
2. Lấy phần nguyên cho từng subnet.
3. Phân các slot còn lại cho subnet có phần dư lớn nhất.
4. Nếu bằng nhau, dùng thứ tự phá hòa được sinh từ seed và giữ cố định trong episode.
5. Trong mỗi subnet, lấy mẫu host không hoàn lại ở từng timestep.

Cơ chế này giữ đúng ngân sách toàn cục và tránh trường hợp một subnet nhỏ thường xuyên biến mất chỉ do dao động Bernoulli.

### 4.4 Chống rò rỉ thông tin

- Không buộc target của action hiện tại hoặc trước đó phải visible.
- Không đưa hidden state dùng tính metric vào observation của policy.
- Bộ sinh (q) và bộ sinh mask dùng hai luồng số ngẫu nhiên tách biệt.
- Evaluation dùng (q) cố định, không dùng hỗn hợp (q).
- Mỗi model và seed dùng cùng chuỗi environment seed tại từng (q).

## 5. Hợp đồng action

Pilot 2 không dùng action masking, tức cơ chế loại trước các action bị cho là không hợp lệ. Cả Flat và GraphSAGE nhận cùng action catalogue cố định.

CybORG tự xử lý action được chọn và trả về trạng thái thành công, thất bại hoặc chưa xác định. Schema policy hiện tại chỉ chứa feature theo host nên trường kết quả action ở cấp observation không được đưa vào Flat hoặc GraphSAGE; nó chỉ được ghi log để chẩn đoán và không được dùng để sửa logits.

Observation encoder có một belief update dùng chung dựa trên action đã chọn: Restore đặt belief compromise về clean, còn Remove có thể đưa belief về unknown; update này không đọc kết quả action. Đây là quy tắc wrapper phụ thuộc môi trường, được giữ giống hệt cho hai encoder và phải được báo cáo khi diễn giải hiệu năng tuyệt đối.

Quyết định này loại action masking khỏi nguồn gây nhiễu. Nếu thêm action masking trong benchmark chính, nó phải giống hệt nhau giữa mọi encoder và cần một ablation riêng để đo ảnh hưởng của nó.

## 6. Hợp đồng mô hình

### Flat

Flat nhận các node feature theo thứ tự host cố định rồi nối thành vector. Pilot 2 không nối adjacency vào vector Flat.

### GraphSAGE

GraphSAGE nhận node feature và edge index. Readout hiện tại nối embedding theo thứ tự node và đưa vào action head kích thước cố định.

Readout này đủ cho Pilot 2-A vì kích thước mạng và thứ tự host không đổi. Nó chưa đủ để chứng minh permutation invariance (tính bất biến khi đổi thứ tự node), topology generalisation hoặc cross-scale generalisation. Các claim đó cần shared node-conditioned action scorer, tức bộ chấm điểm action dùng chung theo từng node.

## 7. Evaluation và metric

### 7.1 Metric chính

Mission availability là metric chính: tỷ lệ thời gian dịch vụ quan trọng còn hoạt động.

Không dùng cumulative reward làm metric duy nhất vì reward có thể không đồng biến với mục tiêu vận hành. Nếu reward tăng nhưng availability giảm hoặc số host đặc quyền bị chiếm tăng, kết quả phải được báo cáo như một bất đồng giữa các metric.

### 7.2 Metric phụ và chẩn đoán

- Episode return: tổng reward một episode.
- Mean privileged compromised hosts: số host trung bình bị Red chiếm quyền đặc quyền; thấp hơn là tốt hơn.
- Mean Red presence hosts: số host trung bình có Red session đang hoạt động.
- Detection latency hiện chỉ là proxy phản ứng: thời gian từ khi Red hiện diện trên host đến lần đầu Blue chọn Analyse/Remove/Restore nhắm vào host đó. Nó không chứng minh sensor đã phát hiện đúng và chỉ được dùng để chẩn đoán, kèm số quan sát bị right-censored, tức sự kiện chưa xảy ra trước khi episode kết thúc. Recovery latency cũng là metric chẩn đoán.
- Action success rate among known: tỷ lệ action thành công trong các action có kết quả xác định.
- Inference latency, training throughput và peak memory: độ trễ suy luận, tốc độ train và bộ nhớ đỉnh.

### 7.3 Đường suy giảm theo (q)

Với model (m), định nghĩa:

\[
D_m(q)=A_m(q)-A_m(1)
\]

trong đó (A_m(q)) là mission availability ở mức (q). Giá trị gần 0 hơn nghĩa là model suy giảm ít hơn so với chính nó tại q=1.00, tức giữ lại toàn bộ native Blue observation chứ không truy cập true state.

So sánh độ bền GraphSAGE với Flat bằng:

\[
C(q)=D_{GraphSAGE}(q)-D_{Flat}(q)
\]

(C(q)>0) nghĩa là GraphSAGE mất ít availability hơn Flat. Tuy nhiên, vẫn phải xem chênh lệch tuyệt đối (A_{GraphSAGE}(q)-A_{Flat}(q)); một model có thể suy giảm ít chỉ vì hiệu năng ban đầu đã thấp.

Area Under Curve (AUC, diện tích dưới đường đáp ứng) được xuất như mô tả phụ. Nó không thay thế các điểm (q) riêng và không phải tiêu chí xác nhận hypothesis chính trong Pilot 2.

## 8. Baseline và quy tắc qua cổng

Ba baseline không học được đánh giá bằng cùng final evaluation seed:

- Sleep: luôn không hành động.
- Monitor: luôn giám sát.
- Random: chọn action đồng xác suất từ action catalogue.

### Cổng kỹ thuật bắt buộc

Pilot chỉ hợp lệ khi:

- đủ 6 report và đủ timestep;
- loss và các đại lượng tối ưu hóa PPO hữu hạn; `mean_completed_episode_return` có thể chưa xác định ở rollout chưa kết thúc episode và không bị coi là lỗi học;
- artifact ghi đủ số episode và timestep train thực tế tại từng (q), tổng timestep bằng đúng ngân sách và cả bốn mức đều đã xuất hiện;
- mỗi episode đánh giá có đúng 16/12/8/4 host nhìn thấy theo (q);
- chuỗi environment seed giống nhau giữa model và giữa các (q);
- có checkpoint, log, final evaluation và artifact manifest;
- cấu hình xác nhận không action mask, không target-visibility leak và topology claim đang bị chặn.
- cấu hình xác nhận action result bị loại khỏi policy input một cách đối xứng và Pilot 2-A không dùng learned normalizer.

Script audit tự động thất bại nếu một điều kiện trên sai.

### Cổng khoa học cần đọc kết quả

- Ở (q=1.00): cả hai model phải thể hiện học được so với baseline không học trên tổ hợp availability, return và privileged-compromise burden.
- Ở (q=0.50): ưu tiên thấy tín hiệu học rõ; nếu không, cần kiểm tra mask severity, reward và observation semantics trước benchmark chính.
- Ở (q=0.25): chỉ dùng làm stress/floor diagnostic. Thất bại ở mức này không tự động bác pipeline.
- Không quyết định chỉ bằng chênh lệch mean. Cần xem paired episode results, độ bất định giữa seed và bất đồng metric.

Audit chỉ xuất các margin so với baseline và yêu cầu người nghiên cứu xác nhận cổng khoa học. Nó không tự gắn nhãn thắng/thua bằng một ngưỡng tùy ý.

## 9. Thứ tự chạy ngày mai

### 9.1 Kiểm tra unit test

```bash
uv run --python 3.10 pytest tests/test_stage1_observation.py tests/test_pilot2_observation.py tests/test_operational_metrics.py -q
```

Mục đích: xác nhận cơ chế mask, chống target leak, Flat/GNN parity ở phần dữ liệu chung và metric logger.

### 9.2 Xem trước đúng 6 lệnh

```bash
uv run --python 3.10 python -m scripts.pilot2.run_matrix --dry-run
```

Mục đích: kiểm tra matrix, đường dẫn output và tham số mà chưa train.

### 9.3 Smoke run ngắn một cell

```bash
uv run --python 3.10 python -m scripts.stage1.train_ppo \
  --config configs/pilot2_observation_16.yaml \
  --model flat \
  --observation-level mixed \
  --seed 0 \
  --total-timesteps 4096 \
  --eval-frequency 4096 \
  --eval-episodes 2 \
  --final-eval-episodes 2 \
  --num-envs 2 \
  --rollout-steps-per-env 128 \
  --ppo-epochs 1 \
  --output-root runs/pilot2_smoke
```

Mục đích: kiểm tra end-to-end train, fixed-q evaluation, CSV/JSON/report và checkpoint. Xóa hoặc giữ thư mục này tùy nhu cầu; nó không thuộc kết quả Pilot 2 chính.

### 9.4 Chạy ma trận chính

```bash
uv run --python 3.10 python -m scripts.pilot2.run_matrix
```

Runner chạy tuần tự và có thể tiếp tục: cell đã có report đủ 1.000.000 timestep sẽ được bỏ qua. Sau 6 run, nó tự chạy baseline, tổng hợp và audit.

### 9.5 Đọc kết quả

Các file chính trong `runs/pilot2_observation_16/`:

- `pilot2_protocol_audit.md`: kết quả cổng kỹ thuật và margin so baseline;
- `report.json` của từng run: gồm `training_q_exposure` để audit số episode/timestep thực tế tại mỗi (q);
- `pilot2_summary.md`: bảng mean theo model và (q);
- `pilot2_paired_robustness_contrasts.csv`: contrast suy giảm GraphSAGE so với Flat theo seed;
- `pilot2_per_seed_fixed_q.csv`: kết quả gốc theo model, seed và (q);
- `baselines/baseline_per_episode.csv`: episode-level baseline để phân tích ghép cặp.

## 10. Điều kiện mở Pilot 2-B về topology

Chỉ bắt đầu track topology khi hoàn thành cả bốn việc:

1. Graph edge được dẫn xuất từ cùng subnet/router/NACL đang điều khiển CybORG reachability.
2. Thay topology phải làm thay connectivity, reachability và lateral-movement path trong simulator, không chỉ đổi hostname hoặc IP.
3. Train topology và held-out same-scale topology được sinh từ các instance khác nhau và không rò seed.
4. Action head được kiểm tra permutation và không phụ thuộc cố định vào số node nếu muốn chạy cross-scale.

Trước khi đạt các điều kiện này, không dùng cụm “OOD topology” hoặc “topology generalisation” cho kết quả Pilot 2-A.
