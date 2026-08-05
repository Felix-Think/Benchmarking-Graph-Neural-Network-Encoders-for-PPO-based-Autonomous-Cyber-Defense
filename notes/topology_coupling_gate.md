# Cổng đồng bộ topology với CybORG

## Phát hiện hiện tại

`build_static_topology()` dựng graph host-level cho encoder bằng các clique trong subnet và một số gateway host thủ công. CybORG lại quyết định kết nối thật qua router, subnet và NACL trong scenario.

Do đó, thay graph encoder hiện tại chưa đồng nghĩa với thay môi trường mà Red và Blue đang tương tác. Đây là một confound, tức nguồn gây nhiễu khiến hiệu năng thay đổi nhưng không xác định được nguyên nhân.

## Điều kiện nghiệm thu

- Có một nguồn mô tả topology duy nhất sinh cả scenario CybORG và edge index của GNN.
- Test tự động chứng minh edge bị xóa/thêm làm thay reachable hosts trong simulator.
- Topology train/test cùng 16 host nhưng khác connectivity, shortest paths và lateral-movement routes.
- Hostname, IP và seed không được dùng như dấu hiệu nhận diện tập train/test.
- Mỗi model dùng cùng topology, Red seed, mask seed và evaluation seed.
- Chỉ gọi test set là OOD khi cả một generator family hoặc distribution parameter được giữ lại; nếu chỉ giữ lại instance, dùng tên held-out same-scale topology.

Pilot 2-B bị chặn cho đến khi các điều kiện này có test và audit artifact.
