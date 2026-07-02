# Benchmarking GNN Encoders for PPO-based Autonomous Cyber Defense

Repository này chứa code thí nghiệm cho đề tài:

`Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability`

Mục tiêu hiện tại chưa phải là chạy benchmark đầy đủ. Giai đoạn này là **sanity pilot**: xác nhận CybORG/CAGE2 chạy ổn, quan sát và action space được inspect rõ ràng, sau đó mới xây wrapper flat/graph, PPO, metric logger, và benchmark chính.

## Mục Tiêu Khoa Học

Dự án muốn kiểm tra một câu hỏi hẹp và sạch:

> Khi cùng dùng PPO trong môi trường autonomous cyber defense dưới partial observability, graph neural network encoder có đem lại lợi ích rõ hơn flat encoder hay không?

Điểm quan trọng là phải kiểm soát các confound:

- Action masking có thể làm agent tốt hơn, không nhất thiết do GNN.
- Reward shaping có thể thay đổi bài toán, không nhất thiết do encoder.
- Graph observation wrapper có thể đưa thêm thông tin topology hoặc state nếu thiết kế không cẩn thận.
- Metric cumulative reward có thể không phản ánh đúng mission outcome.
- Generalization topology cần được benchmark riêng, không được claim nếu chưa test.

Vì vậy codebase được tách thành các lớp rõ ràng: cyber environment, wrappers, models, RL, evaluation, configs.

## Trạng Thái Hiện Tại

Đã có:

- CybORG vendored local trong `CybORG/`.
- Python 3.10 và dependency pin bằng `uv`.
- Smoke test cho CC2/Scenario2 không dùng renderer.
- Script inspect action space và initial observation.
- Cấu trúc thư mục cho benchmark mở rộng.
- Test tối thiểu kiểm tra import và CybORG step.

Chưa có:

- PPO training thật.
- Flat observation wrapper hoàn chỉnh.
- Graph observation wrapper hoàn chỉnh.
- Partial-observation masking thật.
- Action masking benchmark.
- Evaluation metrics đầy đủ như availability, compromised hosts, invalid action rate.

Các file placeholder hiện tại tồn tại để giữ ranh giới kiến trúc, không phải đã implement xong logic benchmark.

## Yêu Cầu Môi Trường

Khuyến nghị:

- Python `3.10`
- `uv`
- macOS hoặc Linux

Project đã khóa Python trong `.python-version`:

```text
3.10
```

Dependency chính nằm trong `pyproject.toml` và `uv.lock`.

Các dependency quan trọng:

- `gym==0.23.1`: CybORG hiện dùng Gym cũ.
- `numpy==1.23.5`: giữ tương thích với Gym/CybORG cũ.
- `torch==2.3.1`
- `torch-geometric==2.5.3`
- `gymnasium==0.29.1`: có mặt để hỗ trợ ecosystem mới hơn, nhưng CybORG sanity pilot vẫn dùng `gym`.
- `cyborg`: được cài dạng editable từ thư mục local `CybORG/`.

## Cài Đặt

Từ root của repository code:

```bash
uv sync
```

Nếu sau này cần thử Stable-Baselines3 adapter:

```bash
uv sync --extra sb3
```

Hiện tại SB3 chưa nằm trong pipeline chính vì SB3/Gymnasium adapter có thể tạo thêm confound. PPO chính của benchmark nên được kiểm soát trực tiếp trước.

## Các Lệnh Kiểm Tra Hiện Tại

### 1. Smoke test CybORG CC2

```bash
uv run --python 3.10 python scripts/smoke_cyborg_cc2.py --steps 5
```

Mục đích:

- Kiểm tra CybORG import được.
- Kiểm tra `Scenario2.yaml` load được.
- Kiểm tra RedMeanderAgent hoạt động.
- Kiểm tra BlueReactRemoveAgent chọn action được.
- Kiểm tra `cyborg.step("Blue", action)` chạy được.
- Không dùng `render()`.

Kết quả mong đợi:

- In ra `hosts=16`.
- In từng bước gồm Blue action, Red action, reward.
- Không crash.

Ví dụ:

```text
hosts=16
step=0 blue=Monitor red=DiscoverRemoteSystems ... reward=0.0
...
total_reward=0.0
```

Nếu thấy warning của Gym như:

```text
Gym has been unmaintained since 2022...
```

thì đây là warning dự kiến. Ta đang cố ý dùng `gym==0.23.1` để tương thích CybORG.

### 2. Inspect CC2 action space và observation

```bash
uv run --python 3.10 python scripts/inspect_cc2_spaces.py
```

Mục đích:

- In danh sách host trong CC2.
- In action space native của Blue agent.
- In key của initial observation.
- Cho biết observation ban đầu có bao nhiêu entry.

Lệnh này rất quan trọng trước khi viết wrapper. Ta không nên viết flat/graph observation theo cảm giác. Phải nhìn action space và observation thật trước.

Kết quả mong đợi:

- Danh sách 16 host gồm `Defender`, `Enterprise*`, `Op_*`, `User*`, router.
- Action space có action như `Sleep`, `Monitor`, `Analyse`, `Remove`, `Restore`, `Misinform`, decoy actions.
- Initial observation thường có key `success` và các host Blue quan sát được.

### 3. Chạy test

```bash
uv run --python 3.10 pytest tests -q
```

Mục đích:

- Kiểm tra package `acd_benchmark` import được.
- Kiểm tra CC2 environment reset và step được một lần.
- Phát hiện sớm lỗi import path, dependency, hoặc thay đổi CybORG làm hỏng smoke path.

Kết quả mong đợi:

```text
2 passed
```

Warning từ Gym vẫn là warning dự kiến.

## Không Nên Chạy Gì

Không dùng file này cho sanity pilot:

```bash
uv run --python 3.10 python CybORG/demo.py
```

Lý do: `CybORG/demo.py` của upstream gọi `cyborg.render()`, nhưng bản CybORG hiện tại có:

```python
raise NotImplementedError("Rendering functionality is not currently available")
```

Do đó demo upstream sẽ crash ở renderer. Benchmark hiện tại chạy headless, không cần GUI.

## Cấu Trúc Thư Mục

```text
.
├── CybORG/
├── configs/
├── runs/
├── scripts/
├── src/
│   └── acd_benchmark/
├── tests/
├── CybORG_UPSTREAM.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── main.py
```

## Root Files

### `README.md`

Tài liệu vận hành chính của repository. File này giải thích mục tiêu dự án, cách setup, cách chạy script, và ý nghĩa từng thư mục/file.

### `pyproject.toml`

File cấu hình project Python:

- Khóa Python `>=3.10, <3.11`.
- Pin dependency chính.
- Khai báo optional dependency `sb3`.
- Khai báo package layout `src/`.
- Cấu hình pytest để test thấy `src/`.
- Trỏ `cyborg` tới local editable source `CybORG/`.

### `uv.lock`

Lock file của `uv`. File này giúp tái lập dependency đúng version trên máy khác. Nên commit file này.

### `.python-version`

Khai báo Python version cho project:

```text
3.10
```

### `.gitignore`

Chặn các file sinh tự động:

- `.venv/`
- `.uv-cache/`
- `__pycache__/`
- `.pytest_cache/`
- `*.egg-info`
- `.DS_Store`
- experiment outputs như `runs/*`, `artifacts/`, `checkpoints/`, `logs/`

### `CybORG_UPSTREAM.md`

Ghi lại nguồn CybORG đã vendor:

- Upstream repository.
- Commit hash đã clone/vendor.

File này quan trọng cho reproducibility. Khi viết paper hoặc artifact, ta cần biết chính xác CybORG đến từ commit nào.

### `main.py`

File mặc định do `uv init` tạo. Hiện chưa dùng cho benchmark. Về sau có thể xóa hoặc chuyển thành entry point nếu cần.

### `Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability.code-workspace`

File workspace cho VS Code. Không ảnh hưởng logic benchmark.

## `CybORG/`

Đây là CybORG source được vendor vào repo.

Vai trò:

- Cung cấp cyber-defense environment.
- Cung cấp CAGE/CC2 scenario file.
- Cung cấp agent heuristic như `RedMeanderAgent`, `BlueReactRemoveAgent`.
- Cung cấp action, observation, state, scenario parser.

Nguyên tắc:

- Hạn chế sửa trực tiếp `CybORG/`.
- Nếu cần sửa, phải ghi rõ đó là thay đổi benchmark hay bugfix.
- Không nhầm `CybORG/demo.py` với script benchmark. Demo upstream dùng renderer và không phù hợp sanity pilot hiện tại.

Một số file đáng chú ý:

- `CybORG/setup.py`: setup package CybORG upstream.
- `CybORG/Requirements.txt`: dependency gốc của CybORG.
- `CybORG/CybORG/env.py`: class môi trường chính, hiện `render()` chưa khả dụng.
- `CybORG/CybORG/Simulator/Scenarios/scenario_files/Scenario2.yaml`: scenario CC2 hiện dùng trong smoke test.
- `CybORG/CybORG/Agents/`: heuristic agents và wrappers upstream.

## `configs/`

Chứa cấu hình YAML cho experiment. Mục tiêu là tách setting khỏi code để sau này chạy nhiều thí nghiệm có thể đổi config thay vì sửa Python.

### `configs/env/`

Cấu hình môi trường cyber.

#### `configs/env/cc2.yaml`

Cấu hình CC2 hiện tại:

- `name: cc2`
- dùng CybORG backend
- scenario `Scenario2.yaml`
- Blue agent name là `Blue`
- Red agent hiện là `RedMeanderAgent`
- `render: false`

File này chưa được script training đọc tự động. Nó là chuẩn cấu hình cho bước tiếp theo.

### `configs/obs/`

Cấu hình observation và partial observability.

#### `configs/obs/full.yaml`

Observation đầy đủ:

- `observed_dynamic_host_ratio: 1.0`
- không mask static topology
- có mask bits

#### `configs/obs/po_50.yaml`

Partial observation 50%:

- chỉ giữ 50% dynamic host features theo thiết kế sau này.
- static topology vẫn giữ.

#### `configs/obs/po_25.yaml`

Partial observation 25%:

- mức che quan sát mạnh hơn.
- dùng để kiểm tra robustness dưới partial observability.

Lưu ý: các file này mới là config định hướng. Logic mask thật sẽ nằm trong `src/acd_benchmark/wrappers/partial_obs.py`.

### `configs/model/`

Cấu hình model/encoder.

#### `configs/model/ppo_flat.yaml`

PPO với flat MLP encoder. Đây là baseline chính.

#### `configs/model/ppo_gcn.yaml`

PPO với GCN encoder.

#### `configs/model/ppo_graphsage.yaml`

PPO với GraphSAGE encoder.

#### `configs/model/ppo_gat.yaml`

PPO với GAT encoder.

#### `configs/model/ppo_gin.yaml`

PPO với GIN encoder.

Các file này chưa có model implementation thật. Chúng xác định tên encoder, hidden size, số layer, pooling để sau này train script đọc vào.

### `configs/train/`

Cấu hình training run.

#### `configs/train/sanity.yaml`

Cấu hình sanity run:

- env config: `configs/env/cc2.yaml`
- obs config: `configs/obs/full.yaml`
- model config: `configs/model/ppo_flat.yaml`
- seed: `0`
- total steps: `100000`
- eval episodes: `5`
- log dir: `runs/sanity`

File này sẽ dùng sau khi PPO và wrapper được implement.

## `scripts/`

Chứa entry points chạy trực tiếp từ terminal.

### `scripts/_bootstrap.py`

Helper để script chạy trực tiếp vẫn import được package trong `src/`.

Lý do cần file này: khi chạy:

```bash
python scripts/smoke_cyborg_cc2.py
```

Python không tự thêm `src/` vào import path. `_bootstrap.py` thêm `src/` vào `sys.path` để script chạy được mà không cần người dùng nhớ `PYTHONPATH=src`.

### `scripts/smoke_cyborg_cc2.py`

Smoke test headless cho CybORG CC2.

Chạy:

```bash
uv run --python 3.10 python scripts/smoke_cyborg_cc2.py --steps 5
```

File này dùng:

- `make_cc2_env()` từ `src/acd_benchmark/cyber/env_factory.py`
- `BlueReactRemoveAgent`
- `RedMeanderAgent` mặc định trong factory

Mục đích là kiểm tra môi trường chạy được trước khi viết RL.

### `scripts/inspect_cc2_spaces.py`

Script inspect environment interface.

Chạy:

```bash
uv run --python 3.10 python scripts/inspect_cc2_spaces.py
```

Mục đích là nhìn rõ:

- host list
- native action space
- initial observation keys
- số entry trong initial observation

Đây là input trực tiếp cho thiết kế `flat_obs.py`, `graph_obs.py`, `partial_obs.py`, và `action_mask.py`.

### `scripts/train.py`

Placeholder cho training entry point.

Hiện tại nếu chạy sẽ báo:

```text
NotImplementedError
```

Điều này là cố ý. Chưa nên viết training loop trước khi observation wrapper được validate.

### `scripts/evaluate.py`

Placeholder cho evaluation entry point.

Sau này file này sẽ load trained agent, chạy evaluation episodes, và ghi metric.

## `src/acd_benchmark/`

Package chính của benchmark.

Nguyên tắc thiết kế:

- `cyber/` biết CybORG.
- `wrappers/` biết cách chuyển observation/action giữa CybORG và learning model.
- `models/` biết neural network.
- `rl/` biết PPO nhưng không nên biết chi tiết CybORG.
- `evaluation/` biết metric và evaluator.
- `utils/` chứa helper dùng chung.

### `src/acd_benchmark/__init__.py`

Khai báo package và version:

```python
__version__ = "0.1.0"
```

## `src/acd_benchmark/cyber/`

Module cyber-domain và CybORG adapter.

### `cyber/import_path.py`

Đảm bảo vendored CybORG được import đúng khi chạy từ repo root.

Vấn đề đã gặp: nếu Python nhìn thấy folder `CybORG/` ở root trước, nó có thể hiểu sai package `CybORG`. Helper này thêm đúng vendor root vào `sys.path`.

### `cyber/cyborg_cc2.py`

Chứa helper xác định đường dẫn CybORG và `Scenario2.yaml`.

Các hàm chính:

- `get_cyborg_package_dir()`
- `get_cc2_scenario_path()`

### `cyber/env_factory.py`

Factory tạo CC2 environment hiện tại.

Hàm chính:

```python
make_cc2_env()
```

Hiện tại factory tạo:

- `FileReaderScenarioGenerator` từ `Scenario2.yaml`
- `RedMeanderAgent`
- `CybORG(..., "sim", agents={"Red": red_agent})`

### `cyber/action_space.py`

Placeholder cho action-space inspection/adaptation.

Sau này đây là nơi xử lý các vấn đề như:

- mapping action native sang discrete action id
- valid action handling
- action mask nếu dùng
- log invalid action rate

Không nên nhét action-space logic vào PPO.

## `src/acd_benchmark/wrappers/`

Module chuyển đổi interface giữa CybORG và model/RL.

### `wrappers/flat_obs.py`

Nơi implement flat observation cho MLP baseline.

Mục tiêu sau này: tạo vector cố định từ cùng nguồn thông tin với graph observation.

### `wrappers/graph_obs.py`

Nơi implement graph observation cho GNN encoders.

Mục tiêu sau này: chuyển observation thành graph tensors như node features, edge index, masks.

### `wrappers/partial_obs.py`

Nơi implement partial-observation masking.

Thiết kế dự kiến:

- mask dynamic host features
- không mask static topology
- thêm mask bits để model biết feature nào bị che
- log effective observed ratio

### `wrappers/action_mask.py`

Nơi implement action masking nếu sanity pilot cho thấy cần.

Lưu ý khoa học: action masking là confound lớn. Nếu dùng, phải dùng giống nhau cho flat và GNN, và phải log rõ.

## `src/acd_benchmark/models/`

Module neural network.

### `models/flat_encoder.py`

Nơi implement MLP/flat encoder baseline.

### `models/gnn_encoders.py`

Nơi implement các encoder:

- GCN
- GraphSAGE
- GAT
- GIN

### `models/actor_critic.py`

Nơi implement actor-critic heads dùng chung cho flat và graph encoders.

Nguyên tắc: nếu có thể, actor/critic head nên giống nhau giữa flat và GNN để khác biệt chính nằm ở encoder.

## `src/acd_benchmark/rl/`

Module reinforcement learning.

### `rl/ppo.py`

Nơi implement PPO.

Nguyên tắc: PPO không nên phụ thuộc trực tiếp vào CybORG. PPO chỉ nên nhận observation đã được wrapper xử lý.

### `rl/rollout.py`

Nơi lưu rollout/trajectory.

Sau này sẽ chứa buffer cho:

- observations
- actions
- rewards
- dones
- log probabilities
- values
- advantages

### `rl/losses.py`

Nơi implement loss functions của PPO:

- policy loss
- value loss
- entropy bonus
- clipped objective

## `src/acd_benchmark/evaluation/`

Module đánh giá.

### `evaluation/metrics.py`

Hiện có dataclass:

```python
EpisodeMetrics
```

Metric tối thiểu hiện tại:

- total reward
- steps
- done

Sau này cần mở rộng thêm:

- mission availability
- compromised hosts
- critical asset impact
- invalid action rate
- recovery latency
- action distribution
- wall-clock time
- steps per second

### `evaluation/evaluator.py`

Placeholder cho evaluation loop.

Sau này sẽ chạy trained policy qua nhiều episodes/seeds và ghi metric.

## `src/acd_benchmark/utils/`

Helper dùng chung.

### `utils/seed.py`

Hàm:

```python
seed_everything(seed)
```

Seed Python, NumPy, và PyTorch.

### `utils/logging.py`

Hàm:

```python
get_logger(name)
```

Tạo logger chuẩn cho project.

## `tests/`

Test tối thiểu để bắt lỗi sớm.

### `tests/test_imports.py`

Kiểm tra package `acd_benchmark` import được và version đúng.

### `tests/test_cyborg_smoke.py`

Kiểm tra:

- tạo CC2 env được
- reset Blue được
- BlueReactRemoveAgent chọn action được
- step một lần được
- host count là 16

Đây không phải benchmark test. Đây là health check cho environment.

## `runs/`

Thư mục chứa output thí nghiệm local.

Git ignore toàn bộ nội dung trong `runs/`, trừ `runs/.gitkeep`.

Sau này có thể chứa:

- training logs
- TensorBoard logs
- CSV metrics
- model checkpoints tạm
- evaluation outputs

Không nên commit output lớn vào Git.

## Quy Trình Phát Triển Khuyến Nghị

Thứ tự tiếp theo nên là:

1. Dùng `inspect_cc2_spaces.py` để hiểu rõ observation/action space.
2. Implement flat observation wrapper.
3. Implement graph observation wrapper.
4. Viết test đảm bảo flat và graph wrapper dùng cùng nguồn thông tin.
5. Implement action mapping cơ bản.
6. Chạy random/heuristic baseline logging.
7. Implement PPO-flat sanity.
8. Chỉ khi PPO-flat học được, thêm GNN encoder.
9. Sau pilot mới quyết định partial-observation levels, action mask, reward shaping, và metric mở rộng.

Không nên viết full PPO+GNN benchmark trước khi wrapper được validate. Nếu wrapper sai, toàn bộ kết quả sau đó sẽ không đáng tin.

## Ghi Chú Về Docker

Chưa cần Docker cho sanity pilot.

Hiện tại `uv` local đã đủ để:

- cài dependency
- chạy CybORG
- chạy smoke test
- chạy pytest

Docker nên làm sau khi:

- wrapper ổn
- PPO-flat smoke chạy được
- cấu hình thí nghiệm chính rõ hơn

Docker sẽ hữu ích cho reproducibility artifact, nhưng thêm Docker quá sớm sẽ làm tăng chi phí debug.

## Git Checklist Trước Khi Commit

Nên commit:

- `README.md`
- `.gitignore`
- `.python-version`
- `pyproject.toml`
- `uv.lock`
- `CybORG_UPSTREAM.md`
- `CybORG/`
- `configs/`
- `scripts/`
- `src/`
- `tests/`
- `runs/.gitkeep`

Không nên commit:

- `.venv/`
- `.uv-cache/`
- `.pytest_cache/`
- `__pycache__/`
- `*.egg-info`
- `runs/` output
- checkpoint/log lớn

