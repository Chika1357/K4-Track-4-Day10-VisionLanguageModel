# Tiến độ Lab Day 10

Nguyễn Tuấn Thành — 2A202602640. Môi trường chính: RTX 6000 Ada, GPU vật lý 1 (cập nhật theo yêu cầu mới).

- [x] Kết nối SSH và xác thực Kaggle trên máy trường.
- [x] Đọc công cụ gốc; lập PLAN.md và commit kế hoạch `eeb8bce`.
- [x] Baseline Windows: `68 passed, 1 skipped`.
- [x] Dừng job Gemma DDP theo quyền người dùng cấp; GPU 2 đã về 2 MiB / 0%.
- [x] Kiểm thử chuyển nhãn trên CPU: `7 passed`.
- [x] Làm mới SSH portal: kết nối thành công vào `research02` bằng phiên mới.
- [x] Gửi SIGTERM đến job cũ của `research02` trên GPU 1 (PID 3944915) theo quyền người dùng cấp.
- [x] Kiểm thử CPU chuyển nhãn/mask/collate/reward và kiểm tra evidence: `25 passed`.
- [x] Hoàn thiện notebook gọi mã `lab/`: 28 cell, 17 code cell kiểm tra cú pháp; chưa chạy các cell GPU.
- [x] Sau sửa notebook, bộ test gốc vẫn `68 passed, 1 skipped`; scripts/tests/configs gốc không đổi.
- [x] Viết các phase điều phối E0–E4, resume, save/load projector, phân tích/ảnh/báo cáo; chưa kiểm chứng phần GPU.
- [x] Khôi phục Full access và kết nối SSH vào research02 với session mới.
- [x] Dừng script tự chạy fold `run_ct_folds.sh` và job con của nó; GPU 1 đã về 2 MiB / 0%.
- [x] Cài các pin đề vào venv riêng; bổ sung torchvision 0.29.1 và matplotlib 3.10.9.
- [x] Bộ test gốc trên Linux trước tải dữ liệu: `68 passed, 1 skipped`.
- [x] Tải dataset trên server, xác minh 2.009 ảnh / 4.737 box, không trùng phiên; vẽ 6 ảnh nhãn.
- [x] Sau tải, test dữ liệu thật: `69 passed`; test bổ sung: `25 passed`.
- [x] Commit checkpoint môi trường/mã CPU: `95be016`.
- [x] Đồng bộ audit, 6 ảnh nhãn và metadata split về máy local; ảnh nhãn đã kiểm tra trực quan.
- [x] Commit checkpoint dữ liệu (xem lịch sử Git, thông điệp `feat: verify deterministic dataset splits and label examples`).
- [x] E0 chạy đủ 311 ảnh test trên GPU 1; `missing_predictions=0`, F1=0.087527, parse_valid_rate=0.845659.
- [x] Đồng bộ evidence, model audit, ví dụ output thô và raw predictions E0 về local.
- [x] Commit baseline E0 và tài liệu bàn giao (RUN_GPU.md).
- [x] Kiểm chứng 220 token, mask/collate/gradient, commit checkpoint encoding: E1/E2/E3 đều 220 vision token, 312 masked/250 supervised, kết thúc `<|im_end|>`; module LoRA 252/252/348, gradient đúng nhóm, không tham số đóng băng nhận gradient.
- [x] E1: 300 bước, F1 test 0.2913 (CI95 0.2535–0.3300), parse_valid 0.9968, val F1 0.1910; chấm lại độc lập khớp.
- [ ] Chạy E2/E3 (tuần tự trên GPU 1) và commit bằng chứng SFT.
- [ ] Chạy E4 hai rank và commit ablation.
- [ ] Hoàn thiện báo cáo, minh họa, nghiệm thu và commit hồ sơ.
- [ ] E5 bonus: reward tests, GRPO và evidence.
- [ ] Xác minh repo public và chuẩn bị nộp LMS.

## Ghi chú tiếp tục

Phiên Codex dừng sau E0 theo yêu cầu người dùng; Claude Code tiếp tục từ checkpoint encoding qua bridge SSH cục bộ (`work/gpu_bridge.py`, không lưu mật khẩu).

Checkpoint chỉ được đánh dấu hoàn thành khi có kết quả chạy thật và sản phẩm kiểm chứng.
