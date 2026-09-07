# CybORG CAGE 4 hiện tại

Vendor CAGE 4 được thêm bằng subtree ở commit 02fb82d, từ upstream 8c3c50c.
Nguồn: https://github.com/cage-challenge/cage-challenge-4
Không thay cây vendor hoặc cập nhật upstream trong đợt tinh gọn 2026-09-07.

So với cây import 02fb82d, snapshot 009e34b chỉ khác hai file vendor:

1. CybORG/CybORG/Simulator/SimulationController.py:
   _find_observed_session tìm đúng session qua nhiều observation, không giả định
   host observation đầu tiên luôn có trường Sessions. Regression:
   tests/test_cc4_session_reassignment.py.
2. CybORG/setup.py: cấu hình find_packages, package data và version.txt để cài
   package local. Các thay đổi này đã có trước refactor.

Kiểm tra provenance local:

    git diff --stat 02fb82d 009e34b -- CybORG
    git diff 009e34b -- CybORG
    uvr python -m pytest tests/test_cc4_session_reassignment.py -q

Lệnh diff thứ hai phải rỗng ở bản main sau dọn. Không tuyên bố vendor này chứa
các bản vá Restore/CybORG++ CAGE 2 cũ: tài liệu đó thuộc lịch sử và nằm trong
branch archive. Các đường dẫn test CAGE 2 cũ không còn áp dụng.

Reward/actions/durations của CAGE 4 không bị chỉnh sửa trong refactor này.
