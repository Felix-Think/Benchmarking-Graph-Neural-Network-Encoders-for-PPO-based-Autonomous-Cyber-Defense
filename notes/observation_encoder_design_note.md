# Observation Encoder Design Note

Trang thai: **freeze cho Pilot v1**.

## Schema

- Version: `raw_blue_current_step_v1`.
- Shape: 16 nodes x 26 features.
- Static block: 15 features ve actionable/router, zone, role, OS, criticality
  va reset-time default-credential risk.
- Raw dynamic block: 10 features ve native evidence, process/connection/file/
  session anomalies, scan, exploit, malware va privileged-session evidence.
- Missingness block: 1 `wrapper_retained` bit.
- Normalization: chi fixed count caps; khong learned normalizer.

## Topology

- Version: `official_cc2_link_diagram_v1`.
- 13 hosts + 3 routers.
- Simulator link diagram duoc chuyen thanh bidirectional edges.
- Khong self-loop tu wrapper; GraphSAGE van co root transform cua layer.
- Khong dung dynamic true state de tao edge.

## Explicit exclusions

- Khong persistent compromise cache.
- Khong action-conditioned belief.
- Khong selected action, target hoac success result trong policy input.
- Khong target-forced visibility.
- Khong evaluator-only state trong observation.
- Flat khong nhan adjacency.

Schema manifest duoc ghi vao moi run. Thay doi feature, edge rule hoac
missingness semantics bat buoc tang version va chay lai Pilot.
