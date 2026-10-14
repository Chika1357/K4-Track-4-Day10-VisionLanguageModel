# Chạy lab trên GPU trường

Nguyễn Tuấn Thành — 2A202602640. GPU vật lý 1, tài khoản research02.

## Môi trường

Tại thư mục gốc trên server, tạo venv riêng và cài phụ thuộc. Giữ nguyên scripts/tests/configs của đề.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
export CUDA_VISIBLE_DEVICES=1
```

Pin thư viện của đề đã cài thành công trên server. Runtime được ghi vào report/runs/environment.json.
Không chạy đồng thời nhiều job GPU của lab. Venv không thay môi trường dùng chung.

## Các checkpoint bắt buộc

Chạy tuần tự, kiểm tra sản phẩm rồi commit mã/evidence/hình liên quan sau từng checkpoint.
Không commit data/, work/, credential hoặc trọng số. Các checkpoint đang thực hiện được ghi trong CHECKLIST.md.

```bash
# Cấp Kaggle key trong bộ nhớ shell; không nhập key trực tiếp vào lệnh có lịch sử.
read -rp 'Kaggle username: ' KAGGLE_USERNAME
read -rsp 'Kaggle API key: ' KAGGLE_KEY
printf '\n'
export KAGGLE_USERNAME KAGGLE_KEY
bash run_gpu.sh prepare
unset KAGGLE_USERNAME KAGGLE_KEY

bash run_gpu.sh e0
bash run_gpu.sh encoding
bash run_gpu.sh e1
bash run_gpu.sh e2
bash run_gpu.sh e3
bash run_gpu.sh e4_rank4
bash run_gpu.sh e4_rank64
```

`prepare`: tải bằng công cụ gốc, audit, 6 ảnh nhãn, pytest trước/sau dữ liệu.
`encoding`: smoke test thật cho E1/E2/E3, token/mask/module/dtype/gradient.
`e1/e2/e3/e4_*`: train đủ 300 optimizer updates, fresh reload rồi val và test; chấm bằng score.py gốc.

Chọn setup nền E4 bằng F1 trên đầy đủ 276 ảnh val, giữ seed/LR/effective batch/steps/processor/prompt cố định.
Rank 4/64 đặt alpha=2r. Phần train resumable từ work/<name>/step_*/resume.pt nếu mã/cấu hình không đổi.
Inference `--resume` chỉ bỏ qua ID đã ghi, giữ output thô. Lưu lại run cũ nếu sửa lỗi phải chạy lại.

Model revision được ghim ở report/runs/model_revision.json, không tự đổi trong các thí nghiệm.
VRAM dùng bytes/1e9; thời gian train đo có đồng bộ CUDA. Số đo RTX không được trình bày như số đo T4.

## Bonus E5

Sau E0–E4, reward đã có 8 ca unit test và wrapper completions được kiểm tra theo API TRL 1.9.2.

```bash
.venv/bin/python -m pytest -q student_tests/test_reward.py student_tests/test_grpo_reward_adapter.py
.venv/bin/python -m lab.grpo
.venv/bin/python -m lab.infer --revision <revision_da_ghim> --name E5 \
  --checkpoint work/e5/final --split test --output work/e5.jsonl \
  --evidence report/evidence/e5.json
```

E5 lấy adapter/projector nền chọn bằng val; 60 bước, 4 generations, LR 5e-6, beta=0, không vLLM.
E5 không phải bằng chứng thay thế các thí nghiệm bắt buộc; báo cáo reward hacking bằng output thật.

## Báo cáo và nghiệm thu

Chỉ build report khi đủ E0–E4; chương trình từ chối bằng chứng thiếu hoặc trộn dataset version.

```bash
# Server chạy mã nhưng không có Git repo: build trước, đồng bộ artifact về repo local.
.venv/bin/python -m lab.finalize --build-only

# Trong Git repo dùng để nộp, có data/splits/meta.json không track:
python scripts/check_submission.py
git diff c2f3997 -- scripts tests configs
git status --short
git rev-list --count HEAD
```

Nghiệm thu phải **0 FAIL, 0 WARN**, đủ hai file E4, ít nhất 6 hình dự đoán và đáp án báo cáo có số liệu thật.
Test template gốc yêu cầu placeholder trong REPORT.md; báo cáo hoàn chỉnh làm test đó fail do mâu thuẫn đề.
Giữ nguyên test, công bố ngoại lệ và kiểm tra các test còn lại; xem BAO_CAO_LOI_VA_LUU_Y.md.
