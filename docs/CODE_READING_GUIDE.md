# Lộ trình đọc code main

Mục tiêu: tự giải thích được một observation đi từ Blue tới PPO, và một action
đi từ policy trở lại simulator. Không cần đọc v1. Bản đang dùng là
**schema v2 + parser v4**, không phải hai pipeline song song.

Cách học: mỗi bước đọc các hàm được chỉ ra, lấy ví dụ của một Blue, rồi tự trả lời
câu hỏi kiểm tra. Chưa trả lời được thì dừng ở bước đó. Chưa cần đọc hết vendor,
test hay lịch sử pilot.

## 1. Xem sơ đồ lắp môi trường

File: [src/acd_benchmark/cyber/benchmark_env.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/benchmark_env.py>).
Đọc make_benchmark_env() để biết ba tầng:

    CybORG + BlueFixedActionWrapper
      -> CC4CanonicalFlatEnv
      -> CC4StructuredActionEnv
      -> CanonicalObservationCondition
      -> PPO

Condition(full) không bỏ thông tin. Chữ Flat trong tên wrapper nói rằng nó trả ra
vector tuần tự; cả NodeMLP và GCN nhận vector này rồi tách thành node/matrix.
Nó không đồng nghĩa Flat policy.

Tự trả lời: tầng nào xây observation, tầng nào đổi action ID, tầng nào xóa feature?

## 2. Khởi tạo simulator và năm Blue

File: [src/acd_benchmark/cyber/cc4_env.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/cc4_env.py>).
Đọc BLUE_AGENTS và make_cc4_env().

SleepAgent là controller Blue dự phòng. Khi rollout gửi action PPO qua step(),
đó là action Blue thực thi. Hiện tại **một mạng dùng chung cho năm Blue**,
không phải năm mạng độc lập. Observation và bộ nhớ của từng Blue vẫn riêng.

Tự trả lời: vì sao SleepAgent không ngăn Blue học bằng PPO?

## 3. Hiểu distribution đã đọc

File: [src/acd_benchmark/cyber/cc4_distribution.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/cc4_distribution.py>).
Thứ tự: create_scenario -> _scenario_fingerprint -> _sample_pivot_assignments
-> _apply_pivot_assignments.

random_host đổi host mang thông tin khám phá sang subnet khác. Không đổi physical
links; không tự tạo benchmark structural generalisation (khái quát cấu trúc mạng).
pivot_salt là salt bộ chọn pivot, không phải phiên bản canonical.

Tự trả lời: điều gì thay đổi và điều gì giữ nguyên so với official?

## 4. Chốt hợp đồng observation trước thuật toán

File: [src/acd_benchmark/cyber/observation_schema.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/observation_schema.py>).
Đọc SUBNETS, AGENT_SUBNETS, NODE_FEATURES, GLOBAL_FEATURES,
CanonicalObservation.flat_view(), deserialize_flat_view().

| Khối | Shape | Ý nghĩa |
|---|---|---|
| Node features | 57 x 79 | 9 subnet slots + tối đa 48 host slots cục bộ |
| Membership | 48 x 9 | Host thuộc subnet nào |
| Structural | 9 x 9 | Liên kết subnet giả định biết |
| Communication | 9 x 9 | Chính sách giao tiếp theo mission phase |
| Firewall | 9 x 9 | Trạng thái từ phản hồi hành động đã quan sát |
| Global | 7 | 3 mission phase + 4 trạng thái success |

Tổng: 57*79 + 48*9 + 3*9*9 + 7 = **5185**.
57 là số slot có padding, không có nghĩa mọi Blue nhìn thấy 57 entity thật.
exists phân biệt thực thể và padding. Không có compromise true state trong vector.

Tự trả lời: vector đoạn nào chứa một host? Vì sao NoEvidence phải giữ exists?

## 5. Topology prior và mission phase

File: [src/acd_benchmark/cyber/topology.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/topology.py>).
Đọc mission_phase, structural_subnet_adjacency, desired_communication_adjacency.

Đây là cấu trúc/quy tắc cấp sẵn, không phải sensor evidence (tín hiệu cảm biến).
Thông tin này có thể giúp policy ngay cả khi evidence bằng 0.

Tự trả lời: firewall state và structural adjacency khác nhau như thế nào?

## 6. Builder: từ observation hiện tại sang bộ nhớ

File: [src/acd_benchmark/cyber/canonical_observation.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/canonical_observation.py>).
Thứ tự:
1. ObservedHostEventDetails.normalized_values: số lượng/thuộc tính event từ parser.
2. HostCompressedBuilder.__init__, reset: khởi tạo và xóa memory riêng của Blue.
3. _node_index, _mark_host_exists: gắn hostname vào slot.
4. update: reset feature hiện tại, tăng tuổi memory, đọc observation mới.
5. _update_node, _advance_memories: passive alert và Analyse history tách biệt.
6. _apply_successful_action, _clear_host_memory: phản hồi action cập nhật belief.
7. _snapshot: tạo sáu khối output.

now chỉ nói về tick hiện tại; memory là thông tin quá khứ còn hiệu lực.
TTL là thời hạn giữ; decay là giảm cường độ theo tuổi.
Main dùng passive TTL 5 ticks, Analyse TTL 20 ticks.
Process có PID không đồng nghĩa độc hại hoặc host compromised.

Tự làm ví dụ: tick 10 có connection, tick 11 không có, tick 15 không có.
Viết now/memory/age theo _snapshot; đừng chỉ đoán theo tên feature.

## 7. Wrapper: thông tin đến từ đâu và action chạy ra sao?

File: [src/acd_benchmark/cyber/cc4_canonical_env.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/cc4_canonical_env.py>).
Thứ tự: reset -> step -> _observed_host_event_details -> _refresh_visible_ip_maps
-> _build_step_canonical.

Phải hiểu:
- Input là raw Blue observation, không phải simulator compromise state.
- Inventory ở reset không tính là passive alert.
- Process feedback của DeployDecoy thành công được loại khỏi passive evidence.
- Connection vẫn có thể được giữ khi observation đó có DeployDecoy feedback.
- Memory IP/subnet riêng từng Blue, reset mỗi episode.
- decision_ready chỉ lúc policy được chọn action; khác simulator tick.
- _normalise_episode_end giữ quy ước hiện tại: horizon -> truncated, không terminated.
  Đây là giả định bootstrap của benchmark, không phải kết luận phổ quát về finite-horizon MDP.

Tiếp theo [src/acd_benchmark/cyber/cc4_structured_action_env.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/cc4_structured_action_env.py>).
Đọc structured_action_keys -> _build_mapping -> _transform_infos -> step.
242 slots = 4*48 action host + 2*24 action zone + 2 action global.
Đổi ID không tạo thêm native action. Padding và action không hợp lệ được mask.
host_edge_global_v1 là phiên bản action catalog, không phải canonical v1.

Quan sát mà không train:

    uvr python -m scripts.inspection.inspect_benchmark --seed 123 --steps 10

Tự trả lời: Properties có phải bằng chứng process độc hại không?
Action mask có nằm trong node features không?

## 8. Policy chung rồi mới encoder

File: [src/acd_benchmark/rl/policies/actor_critic.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/policies/actor_critic.py>).
get_action_and_value áp mask lên logits, sample/argmax, trả log-prob/value.
Stochastic lấy mẫu từ phân phối đã học, không phải uniform random.

File: [src/acd_benchmark/rl/policies/structured_action.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/policies/structured_action.py>).
Theo _unpack -> encode_nodes -> pooling -> host/edge/global heads -> critic.
Đây là bộ chấm điểm action chung khi đổi encoder.

Sau đó [src/acd_benchmark/rl/policies/nodemlp.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/policies/nodemlp.py>) và
[src/acd_benchmark/rl/policies/gcn.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/policies/gcn.py>).
NodeMLP dùng cùng MLP trên từng node. GCN truyền thông tin qua cạnh membership/
structural. Cấu hình main mặc định có 119881 tham số cho mỗi policy.
Communication/firewall không trực tiếp làm message-passing edges; chúng đi qua edge head.

**NodeMLP không phải Flat MLP flatten toàn graph.** So sánh hiện tại đo message
passing với cùng structured head. Không đổi tên thành “GCN thắng mọi Flat baseline”.
GraphSAGE cũ có head khác và nằm trong archive, chưa vào ma trận main này.

## 9. Rollout: simulator tick khác PPO decision

File: [src/acd_benchmark/rl/rollout/rollout.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/rollout/rollout.py>).
Đọc PendingTransition, RolloutBuffer.add/build, collect_rollout.

Action nhiều ticks tạo một transition có duration. Reward được cộng có discount
khi action chạy. Chỉ Blue rảnh mới sample action. Memory cập nhật mỗi tick.
Team return lấy một lần, không cộng năm Blue làm score gấp năm.

Tự trả lời: vì sao 499 simulator ticks không nhất thiết bằng 499 PPO decisions?

## 10. GAE rồi PPO update

Files: [src/acd_benchmark/rl/ppo/advantages.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/ppo/advantages.py>) và
[src/acd_benchmark/rl/ppo/update.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/ppo/update.py>).

Scaled reward -> delta -> advantage -> return.
Bootstrap discount là gamma**duration; trace discount là (gamma*lambda)**duration.
Terminated ngắt bootstrap; truncated ngắt trace nhưng vẫn bootstrap.
Normalize advantage nằm trong update. Reward scale .01 áp trước GAE; evaluation
vẫn báo reward gốc.

Tự trả lời: old_log_prob lấy lúc nào? Vì sao update dùng action mask đã lưu?

## 11. Training, validation, checkpoint và TensorBoard

Files: [src/acd_benchmark/rl/ppo/training.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/ppo/training.py>),
[scripts/training/train.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/scripts/training/train.py>), [src/acd_benchmark/rl/tensorboard_logger.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/tensorboard_logger.py>).

Đọc train_policy trước, rồi run_training và callback on_update.
4 episode liên tiếp tạo một update. Validation mỗi 25 updates.
best_checkpoint.pt chọn theo validation stochastic, không theo final test.
checkpoint.pt là cuối ngân sách, dùng cho fixed-budget mặc định.
Core chưa lưu đầy đủ RNG nên không có bảo đảm exact resume.

Tự trả lời: periodic evaluation có đổi RNG training không? Budget thực lưu ở đâu?

## 12. Evaluation và các điều kiện phụ

Files: [src/acd_benchmark/rl/ppo/evaluation.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/rl/ppo/evaluation.py>),
[scripts/evaluation/evaluate.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/scripts/evaluation/evaluate.py>), [scripts/analysis/summarize_main.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/scripts/analysis/summarize_main.py>).

Đọc evaluate_policy_modes, evaluate_policy_with_preserved_rng.
Hai mode dùng cùng environment seeds nhưng chọn action khác nhau.
Training seed mới là đơn vị độc lập; 30 eval episodes không thay 5 training seeds.

Chỉ sau main flow mới đọc:
- [src/acd_benchmark/cyber/canonical_observation_condition.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/canonical_observation_condition.py>): năm điều kiện thông tin.
- [src/acd_benchmark/cyber/dynamic_evidence_counterfactual.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/dynamic_evidence_counterfactual.py>): zero, shuffle host, time shift.
- [src/acd_benchmark/cyber/dynamic_host_observation_mask.py](</Users/felix/Documents/Cybersecurity_Research_Topic/Topic_1/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/Code/Benchmarking Graph Neural Network Encoders for PPO-based Autonomous Cyber Defense under Partial Observability/src/acd_benchmark/cyber/dynamic_host_observation_mask.py>): giữ tỷ lệ host ở đầu ra.

NoEvidence train từ đầu khác zero evidence lúc eval.
Host mask đầu ra không xóa memory bên trong builder, không phải sensor-loss thực.
Zero/action-state-only còn mất exists/controlled, tác động pooling/routing; không
được diễn giải là chỉ xóa sensor. Full/NoEvidence vẫn giữ các feature cấu trúc đó.

## Tự kiểm tra đã nắm được chưa

Tự lần một ví dụ:

    Blue thấy connection -> parser -> memory -> node tensor
    -> encoder -> host score -> mask -> action ID -> native action
    -> nhiều ticks -> reward -> GAE -> PPO update

Đọc test theo vấn đề cần kiểm tra, không cần đọc hết lịch sử v1.
