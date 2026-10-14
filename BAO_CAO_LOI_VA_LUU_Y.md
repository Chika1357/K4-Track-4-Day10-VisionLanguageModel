# Lỗi và lưu ý triển khai

## Môi trường và tài nguyên

- Người dùng đã chuyển GPU được cấp từ số 2 sang số 1. Các run lab dùng `CUDA_VISIBLE_DEVICES=1`; bên trong tiến trình dùng `cuda:0`.
- Trước triển khai GPU 2 có job Gemma 12B DDP. Người dùng cho phép dừng; gửi SIGTERM đến PID 3852451. Torchrun kết thúc job DDP tương ứng, giải phóng GPU 0 và 2. Lab chỉ dùng GPU 2.
- Sau khi làm mới portal, SSH đã vào tài khoản `research02`; thư mục dự án mới là `/home/research02/day10-vlm`. Đã gửi chuỗi lệnh tạo venv, tải repo và cài phụ thuộc qua tmux; cần xác minh kết quả thực tế khi kết nối lại.
- Job cũ trên GPU 1 là `research02/tb` PID 3944915; đã gửi SIGTERM theo quyền người dùng cấp. Không gửi tín hiệu dừng tới job GPU 0/2 trong lần chuyển GPU này.
- Lỗi `NO_RECENT_AUTH` đã được giải quyết bằng làm mới portal. Sau đó cấu hình quyền của phiên công cụ đổi sang sandbox restricted: SSH bị chặn qua `.sbx-denybin/ssh.bat`; `git add` không tạo được `.git/index.lock`. Đây là giới hạn công cụ local, cần khôi phục quyền để tiếp tục GPU/commit, không phải lỗi mật khẩu hay mã lab.
- Người dùng đã khôi phục Full access. SSH với session mới thành công; công cụ gốc trên Linux đạt `68 passed, 1 skipped` trước tải dữ liệu.
- Job CT tự chạy fold tiếp theo do script `run_ct_folds.sh` (PID 3889360). Đã dừng đúng script và job con PID 3959661 theo quyền được cấp; GPU 1 xác minh còn 2 MiB, 0% utilization. GPU 0/2 tiếp tục chạy job riêng.
- Venv của research02 ban đầu thiếu torchvision. Các pin của đề cài thành công, kéo torch 2.14.1; đã bổ sung torchvision 0.29.1 tương ứng, matplotlib 3.10.9. Không sửa requirements-colab.txt, ghi pin môi trường Linux riêng ở requirements-gpu.txt.
- Gọi trực tiếp `.venv/bin/python` không tự thêm `.venv/bin` vào PATH. Lần đầu prepare_data.py không tìm thấy CLI kaggle. Đã sửa wrapper `lab.run.run_command` để cấp PATH của venv cho subprocess; giữ nguyên prepare_data.py.
- Baseline Windows có cảnh báo phụ thuộc requests nhưng đạt `68 passed, 1 skipped`. Chưa dùng môi trường Windows để train.

## Hai khác biệt quan trọng giữa ví dụ LMS và công cụ gốc

- LoRA ViT E3 cần cả `attn.proj`, ngoài `qkv`, `linear_fc1`, `linear_fc2`; thiếu `proj` sẽ không đủ 96 module ViT / 348 tổng.
- `tests/test_configs_and_docs.py` yêu cầu REPORT.md còn `<điền`, còn checker yêu cầu xóa placeholder. Lưu baseline test trước khi điền; báo cáo đầy đủ phải ưu tiên yêu cầu nghiệm thu, công bố test template mâu thuẫn và giữ nguyên công cụ gốc.

## Kiểm chứng quantization trên model thật

Lần nạp E0 đầu tiên bị audit dừng trước inference: Transformers 5.19.0 vẫn chuyển các layer `model.visual.*` thành Linear4bit nếu skip list chỉ có `visual`/`lm_head` như ví dụ. Matcher của phiên bản này so prefix/regex từ đầu tên đầy đủ, không tìm substring ở giữa. Đã bổ sung `model.visual` vào skip list trong mã tự viết, kiểm thử matcher của thư viện thực tế và giữ audit kiểu layer để xác nhận ViT/projector/lm_head không bị lượng tử hóa. Chưa sinh prediction từ lần nạp sai này.

Venv kế thừa gói hệ thống ban đầu dùng Jinja2 3.0.3; chat template của Transformers yêu cầu >=3.1.0. Đã chuyển venv sang không kế thừa site-packages, cài Jinja2 3.1.6, PyYAML 6.0.3 và các dependency còn thiếu vào venv riêng. `pip check` xác minh dependency sau sửa; không nâng cấp Python/gói hệ thống.

## Nguyên tắc số liệu

Không dùng số F1/VRAM/phút mẫu trong đề làm kết quả. Chỉ điền báo cáo từ evidence và log thực tế; công bố rõ GPU RTX 6000 Ada thay Colab T4.
