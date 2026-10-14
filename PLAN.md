# Kế hoạch thực hiện Lab Day 10 — Qwen3-VL-4B Road Damage Grounding

**Sinh viên:** Nguyễn Tuấn Thành  
**Mã số:** 2A202602640  
**Môi trường chạy chính:** tài khoản `research02`, máy `bcseserver1`, GPU vật lý số **1**, NVIDIA RTX 6000 Ada khoảng 48 GB VRAM (cập nhật theo phiên portal mới).  
**Ưu tiên:** hoàn thành đầy đủ bài bắt buộc với kết quả chạy thật; sau đó thực hiện E5 nếu môi trường hỗ trợ. Không đặt thời hạn chạy cứng.  
**Trạng thái tài liệu:** kế hoạch trước triển khai; các số F1, VRAM, thời gian và tham số huấn luyện chưa được đo.

## 1. Mục tiêu và điều kiện hoàn thành

Hiện thực pipeline tái lập được: tải dữ liệu → chia theo phiên quay → chuyển nhãn → chạy E0 → kiểm tra mask loss → huấn luyện E1/E2/E3 → chạy E4 hai giá trị → chấm bằng công cụ gốc → phân tích lỗi → hoàn thiện hồ sơ nộp bài.

| Hạng mục | Điều kiện hoàn thành |
|---|---|
| Dữ liệu | 2.009 ảnh, 4.737 box; split đúng bảng đề; không trùng phiên giữa các split |
| E0 | Dự đoán đủ 311 ảnh test bằng mô hình gốc; có `report/evidence/e0.json` |
| E1–E3 | Mỗi setup có 300 bước tối ưu thành công, checkpoint, log, metadata đo thực tế và bằng chứng trên test |
| E1 | `parse_valid_rate >= 0.95`; `F1(E1) - F1(E0) >= 0.10` theo rubric |
| E4 | Chọn biến rank, chạy cả `r=4` và `r=64`; giữ các yếu tố còn lại cố định |
| Bằng chứng | `score_version=1`, `split=test`, `missing_predictions=0`, `n_images=311`; cùng `dataset_version` |
| Báo cáo | Đúng họ tên, mã số; đầy đủ bảng kết quả và 10 câu trả lời bằng số liệu thật; không còn `<điền>` |
| Hình minh họa | Ít nhất 6 ảnh có GT và dự đoán, chú thích đúng/sai và loại lỗi |
| Git | Ít nhất 5 commit theo các bước có sản phẩm thực tế; không track dữ liệu, khóa hay trọng số |
| Nghiệm thu | `python scripts/check_submission.py` đạt **0 FAIL, 0 WARN** tại bản dùng để kiểm tra cuối |
| E5 bonus | Hàm reward có kiểm thử, GRPO chạy thật, có evidence và phân tích so với SFT; không thay thế phần bắt buộc |

Đạt điều kiện hình thức không đồng nghĩa đạt đủ điểm chất lượng. Nếu E1 chưa đạt ngưỡng rubric, phải điều tra nguyên nhân và ghi rõ kết quả; không sửa evidence, scorer hoặc dự đoán bằng nhãn test để vượt ngưỡng.

## 2. Căn cứ và hiện trạng đã xác minh

Kế hoạch căn cứ vào `README.md`, `GUIDE.md`, `RUBRIC.md`, `REPORT.md`, `configs/e0.yaml–e5.yaml`, mã nguồn các công cụ gốc và mô tả LMS người dùng cung cấp.

- SSH qua alias `bcse-kit` đã thành công; `nvidia-smi` nhìn thấy 3 GPU RTX 6000 Ada.
- Người dùng ban đầu cấp GPU 2, sau đó chuyển sang GPU 1; mọi run lab từ thời điểm cập nhật dùng GPU 1. Kiểm tra job đang chạy trước khi train; người dùng đã cho phép dừng job chiếm GPU được cấp.
- Xác thực Kaggle trực tiếp từ máy trường thành công: key thật trả HTTP 200; key sai đối chứng trả HTTP 401. Đã đọc được danh sách file dataset; **chưa tải dataset**.
- Đã chạy `python -m pytest -q` trên máy Windows hiện tại: **68 passed, 1 skipped**. Test dữ liệu thật cần biến `D10_RAW`; cài đặt Python hiện tại có cảnh báo phụ thuộc `requests`, sẽ dùng môi trường Linux riêng khi triển khai.
- Repo hiện có một commit gốc `c2f3997`; remote `origin` là `https://github.com/Chika1357/K4-Track-4-Day10-VisionLanguageModel.git`. Cần xác minh quyền push và trạng thái Public ở bước chuẩn bị nộp.
- Chưa kiểm chứng môi trường Python/PyTorch trên server, truy cập Hugging Face, dung lượng ổ đĩa, phiên bản thư viện ghim và hiệu năng thực tế.
- Đề mô tả Colab T4 16 GB. Theo yêu cầu người dùng, thực hiện trên RTX 6000 Ada và công bố rõ khác biệt phần cứng trong báo cáo. Việc giảng viên chấp nhận môi trường thay thế vẫn chưa được xác nhận.

## 3. Nguyên tắc triển khai

1. Giữ nguyên `scripts/`, `tests/` và các cấu hình gốc `configs/e0.yaml–e5.yaml`. Mã tự viết nằm trong thư mục mới `lab/`; test bổ sung nằm ngoài `tests/`.
2. Dùng Linux GPU trường làm nơi train. Máy Windows dùng chỉnh sửa mã, theo dõi, nhận bằng chứng và đóng gói bài.
3. Mỗi lần chạy chỉ dùng GPU vật lý số 1 bằng `CUDA_VISIBLE_DEVICES=1`. Bên trong tiến trình, GPU này trở thành `cuda:0`; ánh xạ model vào thiết bị 0 đã được giới hạn.
4. E1, E2, E3 đều khởi tạo độc lập từ cùng model gốc, không tiếp tục huấn luyện adapter của setup trước.
5. Split seed giữ nguyên `day10`; training seed theo cấu hình là `42`. Ghi rõ chúng phục vụ hai mục đích khác nhau.
6. Gỡ lỗi và lựa chọn checkpoint/cấu hình trên `val`; không dùng nhãn test cho huấn luyện hoặc chọn siêu tham số. Test là đánh giá cuối sau khi chốt mỗi setup.
7. F1, CI95, tham số, VRAM và thời gian đều lấy từ chạy thật. Các số mẫu trong LMS là ví dụ định dạng, không được chép làm kết quả.
8. Không đưa credential vào notebook, Git, báo cáo hoặc log. Cấp credential khi chạy qua môi trường tiến trình, tránh ghi vào lịch sử shell.
9. Không cài đặt hay nâng cấp vào môi trường Python dùng chung trên server. Dùng virtual environment riêng trong thư mục dự án.
10. Ghi lại commit mã nguồn, revision model, cấu hình hiệu lực và phiên bản thư viện cho từng run; checkpoint dài được lưu trên đĩa server.

## 4. Thiết kế file và sản phẩm

Các file dưới đây là **dự kiến sẽ tạo**, chưa phải mã đã triển khai.

```text
K4-Track-4-Day10-VisionLanguageModel/
├── PLAN.md                         # Kế hoạch này
├── CHECKLIST.md                    # Trạng thái, lệnh chạy, đường dẫn sản phẩm
├── BAO_CAO_LOI_VA_LUU_Y.md         # Lỗi thực tế, cách sửa, khác biệt so với T4
├── REPORT.md                       # Báo cáo cuối, giữ 10 câu theo mẫu đề
├── lab_day10.ipynb                 # Điền đủ các phần, gọi mã trong lab/
├── lab/
│   ├── __init__.py
│   ├── data.py                     # Chuyển nhãn, dataset, encode, collate
│   ├── modeling.py                 # Nạp model, chọn LoRA, save/load projector
│   ├── train.py                    # SFT, checkpoint, resume, đo tài nguyên
│   ├── infer.py                    # Suy luận val/test, ghi output thô JSONL
│   ├── inspect.py                  # Kiểm tra split, cây module, token, gradient
│   ├── visualize.py                # Vẽ GT/prediction và biểu đồ
│   ├── summarize.py                # Tổng hợp evidence thành bảng báo cáo
│   ├── reward.py                   # Hàm reward E5
│   └── grpo.py                     # Chạy E5 sau phần bắt buộc
├── student_tests/                  # Test mã tự viết, không sửa tests/ gốc
├── report/
│   ├── evidence/                   # JSON được sinh bởi scorer gốc
│   ├── figures/                    # Ảnh mẫu, dự đoán, loss và so sánh
│   └── runs/                       # Log đã rà soát credential, cấu hình, metadata
├── requirements-gpu.txt            # Phụ thuộc bổ sung cho Linux, khi cần
├── configs/                        # Giữ nguyên đề
├── scripts/                        # Giữ nguyên đề
├── tests/                          # Giữ nguyên đề
├── data/                           # Không track Git
└── work/                           # Model cache/checkpoint/prediction; không track Git
```

Trong `work/<run_name>/` lưu riêng: cấu hình hiệu lực, `run.json`, log loss, adapter, projector, optimizer/scheduler/scaler, RNG state và số bước. Chỉ xuất bản các log/metadata cần thiết đã rà soát vào `report/runs/`.

Notebook giữ các tiêu đề `## 0.` đến `## 8.` mà bộ test gốc kiểm tra. Có thể bổ sung mục E5 và hướng dẫn Linux/Colab; tránh nhúng output chứa credential hoặc lượng log lớn.

## 5. Bước 0 — Chuẩn bị máy trường và kiểm tra công cụ

### Công việc

1. Đăng nhập `bcse-kit`; kiểm tra tài khoản, hostname, GPU 1, tiến trình đang dùng GPU, dung lượng đĩa và thư mục làm việc.
2. Xác định tiến trình trên GPU 1 đang chạy gì và thuộc ai. Người dùng đã cho phép dừng job chiếm GPU; đã gửi SIGTERM đến job `research02/tb` PID 3944915 sau khi xác minh, kiểm tra lại trạng thái trước khi train.
3. Tạo thư mục dự án riêng, chuyển mã hiện tại sang server với checksum/commit tương ứng. Không đưa dữ liệu cá nhân hoặc credential vào gói mã.
4. Tạo virtual environment; kiểm tra Python, PyTorch, torchvision, CUDA driver và bitsandbytes. Driver báo CUDA 13.0 không có nghĩa phải cài PyTorch CUDA 13.0; chọn bản tương thích và xác minh bằng chạy tensor CUDA.
5. Thử cài đúng các phiên bản trong `requirements-colab.txt`. Nếu pin không tồn tại hoặc không tương thích, ghi lỗi và phương án riêng trong `requirements-gpu.txt`, giữ file đề gốc và kiểm chứng lại API.
6. Chạy toàn bộ test gốc trên Linux **trước khi sửa REPORT.md**. Chạy `pip check`, kiểm tra import và phép tính CUDA cơ bản.
7. Kiểm tra truy cập dataset Kaggle và tải model Hugging Face. Ghim revision model dùng chung cho mọi setup.
8. Dùng `tmux` nếu có để job không phụ thuộc phiên SSH; chuyển log ra file. Nếu thiếu `tmux`, dùng cách chạy nền có PID/log rõ ràng và kiểm tra tiến trình.

### Lệnh định hướng trên server

```bash
export CUDA_VISIBLE_DEVICES=1
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
python -m pytest -q
python -m pip check
```

### Điều kiện chuyển bước

- Môi trường server chạy được công cụ gốc, PyTorch truy cập CUDA và chỉ thấy GPU được cấp.
- Có log phiên bản Python/PyTorch/CUDA runtime/driver và GPU UUID.
- Bộ test ban đầu đạt `68 passed, 1 skipped`, hoặc có giải thích cụ thể nếu trạng thái dữ liệu khiến số skipped thay đổi.
- Không có xung đột tài nguyên chưa xử lý trên GPU 1.

## 6. Bước 1 — Dữ liệu và split chống rò rỉ

### Công việc

1. Cấp `KAGGLE_USERNAME` và `KAGGLE_KEY` vào môi trường tiến trình, chạy công cụ gốc:

   ```bash
   python scripts/prepare_data.py --kaggle
   ```

2. Giữ bản raw để kiểm thử; xác định thư mục thực sự chứa `images/` và `labels-YOLO/`, gán biến `D10_RAW` rồi chạy lại test dữ liệu thật.
3. Đếm lại ảnh, box và phân bố lớp từ JSONL, đối chiếu `meta.json`.
4. Kiểm tra mỗi `image_id` duy nhất, mỗi ảnh/nhãn đọc được, không có phiên quay xuất hiện ở hai split và tổng số box khớp.
5. Kiểm tra `dataset_version` bắt đầu bằng `d10-`. Bản dữ liệu chuẩn hiện được test/doc tham chiếu là `d10-21e4070301`; lấy giá trị thực tế do script sinh, không tự ghi giá trị này.
6. Vẽ ít nhất 6 ảnh train với box YOLO gốc, đủ cả ba lớp và mẫu khó/box nhỏ.

| Split | Ảnh | pothole | crack | manhole | Tổng box |
|---|---:|---:|---:|---:|---:|
| train | 1.422 | 1.074 | 1.750 | 691 | 3.515 |
| val | 276 | 66 | 470 | 103 | 639 |
| test | 311 | 121 | 299 | 163 | 583 |
| Tổng | 2.009 | 1.261 | 2.519 | 957 | 4.737 |

### Bằng chứng và chuyển bước

- `data/splits/{train,val,test}.jsonl`, `data/splits/meta.json`.
- `report/runs/data_audit.json`: thống kê, kiểm tra trùng phiên, phiên bản dữ liệu.
- `report/figures/labels_*.png`: ảnh kiểm tra nhãn.
- Mọi số khớp bảng; nếu khác, điều tra phiên bản dataset và nhãn, không sửa thuật toán split để ép kết quả.

**Commit dự kiến #1:** chuẩn bị môi trường, kiểm chứng dữ liệu và hiển thị nhãn; chỉ commit mã, tài liệu, thống kê nhỏ và hình minh họa.

## 7. Bước 2 — Grounding JSON và baseline E0

### Hiện thực nhãn và prompt

- YOLO `[xc,yc,w,h]` → `[x1,y1,x2,y2]`, nhân 1000, làm tròn và kẹp `[0,1000]`.
- Bảo đảm `x2>x1`, `y2>y1`, kể cả box sát biên 1000; nếu cần sửa box suy biến do làm tròn thì ghi lại số lượng. Không bỏ nhãn âm thầm.
- Sắp thứ tự `(y1,x1)`, JSON gọn, không chứa văn bản thừa; ảnh không có đối tượng trả `[]`.
- Dùng prompt chuẩn người dùng cung cấp cho E0 và các setup SFT, giữ cùng ba lớp và cùng thang tọa độ.
- Kiểm tra chuyển ngược về pixel, sai số làm tròn và box gốc; test các trường hợp rỗng, nhiều box, sát biên.

### Nạp model và suy luận E0

1. Dùng `Qwen/Qwen3-VL-4B-Instruct`, NF4 4-bit, double quant, compute fp16, skip `visual` và `lm_head`.
2. Dùng model revision và processor revision đã ghim. Giới hạn thiết bị bằng biến môi trường; không để model phân tán sang GPU 0/1 của trường.
3. Kiểm chứng loại layer/dtype thực tế: LLM lượng tử hóa; ViT, projector và lm_head không bị chuyển thành 4-bit.
4. Inference dùng `eval()`, `inference_mode()`, left padding, `do_sample=False`, `max_new_tokens=256`, batch ban đầu 4.
5. Thử trên mẫu val để kiểm tra đường dẫn ảnh, xử lý ảnh và cắt token prompt theo chiều dài input đã pad. Sau khi chốt, chạy đủ 311 ảnh test.
6. Lưu từng dòng `{"image_id": ..., "raw": ...}` vào `work/e0.jsonl`. Output thô được giữ nguyên; không sửa JSON hay box bằng ground truth.
7. Chấm bằng scorer gốc:

   ```bash
   python scripts/score.py --pred work/e0.jsonl --split test --coord norm1000 \
     --name E0 --out report/evidence/e0.json
   ```

### Bằng chứng và chuyển bước

- `e0.json`: đúng schema, đủ 311 ảnh, không thiếu dự đoán.
- Log thời gian inference, VRAM và 5 output thô đại diện.
- Ghi parse-valid theo scorer và quan sát định dạng JSON thuần riêng: scorer có thể chấp nhận markdown hoặc phần JSON nằm trong văn bản.
- Không giả định E0 có F1 thấp hơn 0.10; báo đúng số đo dù khác ví dụ LMS.

**Commit dự kiến #2:** chuyển nhãn, inference baseline và evidence E0.

## 8. Bước 3 — Chat template, encode, collate và mask loss

### Công việc

1. Encode user message gồm ảnh + prompt; encode chuỗi đầy đủ gồm user + assistant JSON.
2. Assert prompt token là tiền tố của full token; lấy độ dài tiền tố thực tế, không hard-code vị trí bắt đầu đáp án.
3. Clone `input_ids` thành labels; mask toàn bộ prefix bằng `-100`; giữ JSON và token kết thúc assistant.
4. Train collate dùng right padding cho `input_ids`, attention mask và labels; padding labels là `-100`, attention mask là 0.
5. Không hard-code pad token; lấy từ tokenizer. Giữ `mm_token_type_ids` và các trường processor yêu cầu nếu processor phiên bản thực tế trả về chúng.
6. `pixel_values` và `image_grid_thw` nối theo chiều 0, không pad như văn bản.
7. Với ảnh 640×360, kiểm tra số token ảnh bằng `prod(image_grid_thw) / spatial_merge_size²` và đối chiếu token ảnh trong input; mục tiêu theo đề là 220.
8. Decode token `labels != -100` với special token còn hiển thị; phải thấy đúng đáp án JSON và `<|im_end|>`.
9. Chạy một forward/backward trên GPU bằng mẫu train; kiểm tra loss hữu hạn, gradient đến LoRA và không đến phần đang đóng băng.

### Test bổ sung có giá trị

- Nhãn rỗng, nhãn nhiều box, sát biên và round-trip tọa độ.
- Prefix alignment, phần đáp án không rỗng, padding bị mask.
- Batch nhiều ảnh giữ đúng số grid và tensor ảnh.
- Kiểm tra tích hợp processor/token thực tế thực hiện trên server, không biến unit test CPU thành bài tải model lớn.

**Bằng chứng:** log token/decoded answer, kiểm tra mask và forward/backward; lưu ở `report/runs/encoding_checks.json`.

**Commit dự kiến #3:** dataset, encode/collate, mask loss và kiểm tra.

## 9. Bước 4 — Chọn module LoRA và kiểm tra kiến trúc

### Thiết kế setup

| Setup | LLM | ViT | Projector |
|---|---|---|---|
| E1 | LoRA 252 module | Đóng băng | Đóng băng |
| E2 | LoRA 252 module | Đóng băng | Mở full, float32 |
| E3 | LoRA 252 module | LoRA 96 module | Mở full, float32 |

LLM: 36 lớp × 7 module attention/MLP: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

ViT E3: 24 block × 4 module: `attn.qkv`, **`attn.proj`**, `mlp.linear_fc1`, `mlp.linear_fc2`. Đoạn ví dụ LMS chỉ thêm ba suffix sẽ thiếu `attn.proj`; cần áp dụng danh sách đúng từ `GUIDE.md` và cây module thực tế.

### Công việc

1. In cây module thực tế; dùng regex có phạm vi LLM/ViT cụ thể thay vì khớp suffix rộng dễ chạm nhầm projector.
2. Freeze base model; bật input gradients và checkpointing; LoRA `r=16`, alpha 32, dropout 0.05, bias none.
3. Với E2/E3, mở đủ cả `visual.merger` và `visual.deepstack_merger_list`, đưa tham số về float32, bảo đảm optimizer nhận đủ tham số đó.
4. Đếm module trước và sau PEFT: E1/E2 là 252, E3 là 348; assert không adapter nào bị chèn nhầm vào merger.
5. Đếm tham số theo khối có ranh giới rõ, không cộng projector hai lần khi nó nằm trong visual. Với trọng số lượng tử hóa, đối chiếu logical parameter count với shape/config gốc thay vì chỉ lấy kích thước storage nén.
6. Đếm riêng trainable LoRA LLM, LoRA ViT và projector; số liệu báo cáo lấy từ model thực tế.
7. Kiểm tra gradient ở từng khối theo setup; forward dưới autocast phải tương thích projector float32.

### Điều kiện chuyển bước

Đúng số module, đúng dtype, đúng khối cập nhật; E2 có thêm khoảng 109M tham số projector so với E1, xác nhận bằng số đếm thật. Nếu khác cây module dự kiến, điều tra model revision/thư viện trước khi chạy dài.

## 10. Bước 5 — Huấn luyện SFT E1, E2, E3

### Cấu hình chung cố định

| Thuộc tính | Giá trị |
|---|---|
| Dataset | Toàn bộ 1.422 ảnh train |
| Seed | 42 |
| Quantization | NF4 4-bit; visual và lm_head không lượng tử hóa |
| Precision | fp16 autocast; projector trainable float32 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Optimizer | AdamW, learning rate `2e-4`, weight decay 0 |
| Lịch learning rate | Warmup 5% = 15 bước, sau đó cosine |
| Micro batch | 2 ảnh |
| Gradient accumulation | 4; effective batch = 8 |
| Max steps | 300 **optimizer updates thành công**, không phải 300 micro-batch |
| Gradient clip | Norm 1.0 sau unscale |
| Checkpointing | `use_reentrant=False`; `use_cache=False` khi train |
| Checkpoint | Mỗi 100 bước và khi kết thúc |

### Vòng lặp và đo lường

1. Chạy smoke test trên train, kiểm tra loss/gradient/memory; reset model và RNG trước run chính thức.
2. DataLoader shuffle tái lập; khi hết một lượt dữ liệu, tiếp tục cho tới đủ 300 update.
3. Chia loss theo số bước tích lũy; dùng GradScaler fp16; unscale → clip → optimizer step → scaler update → scheduler step theo update thành công.
4. Nếu loss/gradient NaN/Inf hoặc scaler bỏ qua update, ghi nhận rõ; không đếm update bị bỏ qua là bước huấn luyện thành công.
5. Log loss, learning rate, bước, elapsed time; theo dõi xu hướng giảm và các bất thường, không yêu cầu loss giảm đơn điệu từng bước.
6. Đồng bộ CUDA quanh phép đo thời gian; reset peak memory sau nạp/setup model, trước train. Đo `torch.cuda.max_memory_allocated`, ghi rõ GB theo mẫu số dùng; có thể ghi thêm reserved memory và số đo `nvidia-smi` riêng.
7. Lưu adapter và projector ở cùng bước. Resume cần optimizer, scheduler, scaler, RNG và trạng thái data order, không chỉ adapter.
8. Log môi trường, effective config, model/code revision và `dataset_version` vào `run.json`.

### Lưu/nạp projector bắt buộc

- `PEFT.save_pretrained()` không đủ để giữ projector được mở full bằng `requires_grad=True`.
- Lưu riêng toàn bộ tham số trainable của `visual.merger` và `visual.deepstack_merger_list` vào `work/<run_name>/projector.pt`.
- Khi nạp, assert đủ key, đúng shape/dtype và không thừa key; không dùng `strict=False` rồi bỏ qua lỗi projector.
- Kiểm tra fresh reload trên cùng mẫu val: output/loss nhất quán trong sai số số học, tham số projector đúng checkpoint.
- Checkpoint full phục vụ resume được giữ ở server, không commit lên GitHub.

### Kiểm chứng trên val

- Chọn trước tập 40 ảnh val cố định để debug; khi đánh giá lựa chọn setup, dùng đầy đủ 276 ảnh val.
- Ghi val F1, parse-valid và loss để chốt run. Không đổi prompt riêng cho setup nhằm làm đẹp kết quả so sánh.
- Nếu E1 có dấu hiệu không học: kiểm tra labels mask, tọa độ, LoRA targets, loss reduction, optimizer và save/reload trước khi thay đổi hyperparameter.
- Nếu phải đổi learning rate/batch để chạy ổn định, ghi rõ override; các setup đối chứng phải dùng cùng quy tắc thay đổi để giữ so sánh có ý nghĩa.

**Sản phẩm:** checkpoint E1/E2/E3, training logs, `run.json`, val evidence và log kiểm tra reload.

## 11. Bước 6 — Suy luận và chấm E1–E3

1. Nạp fresh base model → adapter → projector đúng setup. Bật lại cấu hình inference phù hợp, `eval()` và không gradient.
2. Cùng processor, prompt, coordinate scale, revision và generation config với E0.
3. Chạy đủ 311 ảnh test cho mỗi setup đã chốt; lưu output thô vào `work/e1.jsonl`, `work/e2.jsonl`, `work/e3.jsonl`.
4. Kiểm tra trùng ID, ID ngoài split, thiếu ID và số dòng trước khi chấm; các dấu hiệu này là lỗi pipeline cần sửa, không bỏ qua cảnh báo scorer.
5. Sinh evidence bằng công cụ gốc, metadata lấy từ `run.json`. Ví dụ cấu trúc lệnh, phần trong dấu góc phải thay bằng số đo thật:

   ```bash
   python scripts/score.py --pred work/e1.jsonl --split test --coord norm1000 \
     --name E1 --out report/evidence/e1.json \
     --meta trainable_params=<so_do_thuc> \
     --meta peak_vram_gb=<so_do_thuc> \
     --meta train_minutes=<so_do_thuc>
   ```

6. Lập bảng F1, precision/recall, CI95, FP/ảnh, F1 từng lớp, trainable params, VRAM và phút.
7. Kiểm tra ngưỡng E1 của rubric bằng evidence gốc. Không dùng kết quả test để lựa chọn checkpoint tiếp theo. Nếu cần sửa lỗi kỹ thuật sau test, ghi run cũ/run mới và lý do; công bố việc đã đánh giá lại.

**Commit dự kiến #4:** train/inference SFT, kiểm tra save/reload, evidence E1–E3 và bảng kết quả ban đầu.

## 12. Bước 7 — E4: ablation rank 4 và 64

### Quyết định

Chọn **rank** làm biến duy nhất. Chọn setup nền trong E1–E3 bằng **F1 trên đầy đủ val**, nếu hòa ưu tiên setup nhẹ hơn. Ghi quyết định trước khi chạy test E4; báo cáo thêm kết quả test của setup được chọn nhưng không dùng test để tối ưu lựa chọn này.

### Hai run đối chứng

| Run | Rank | Alpha | Tỷ lệ alpha/r | Các thuộc tính khác |
|---|---:|---:|---:|---|
| E4 rank 4 | 4 | 8 | 2 | Như setup nền đã chọn |
| E4 rank 64 | 64 | 128 | 2 | Như setup nền đã chọn |

Alpha thay đổi theo quy tắc `alpha=2*r` của đề, giữ hệ số LoRA scaling cố định; không coi alpha là một biến được tối ưu riêng. Nếu setup nền có LoRA ViT, áp dụng rank đồng nhất cho các targets như cấu hình nền và ghi rõ phạm vi.

### Công việc và kiểm chứng

1. Khởi tạo hai run từ cùng model gốc, không tiếp tục adapter rank 16.
2. Giữ seed, train split, thứ tự lấy dữ liệu, batch, gradient accumulation, learning rate, 300 steps, dropout, scheduler, projector policy và inference config.
3. Lưu effective config và tự động so diff giữa hai run; chỉ rank/alpha phụ thuộc rank và tên/đường dẫn run được khác.
4. Chạy val để kiểm tra kỹ thuật; sau khi chốt, suy luận test và sinh:
   - `report/evidence/e4_rank_4.json`.
   - `report/evidence/e4_rank_64.json`.
5. Báo F1, CI95, trainable params, VRAM, thời gian và F1 từng lớp cho cả hai.
6. CI95 chồng lấn được dùng để mô tả mức bất định, không tự coi overlap là phép kiểm định ý nghĩa thống kê. Nếu muốn kết luận chênh lệch rõ hơn, thêm paired bootstrap chênh lệch F1 theo cùng ảnh ở mã phân tích riêng; không sửa scorer và không ghi đè CI gốc.

**Commit dự kiến #5:** ablation hai rank, evidence, biểu đồ và phân tích Câu 10.

## 13. Bước 8 — Phân tích lỗi, minh họa và báo cáo

### Phân tích có số liệu

- Thống kê lỗi định dạng, FP, FN, nhầm lớp, định vị kém và box nhỏ; so sánh theo lớp thay vì chỉ micro F1.
- Giữ output thô để phân biệt lỗi model sinh với lỗi parser/định dạng.
- Minh họa ít nhất 6 ảnh test đại diện sau khi chốt chạy: đúng, bỏ sót, FP, nhầm lớp, box lệch; ghi cách chọn ảnh để tránh chỉ chọn mẫu đẹp.
- Vẽ GT và prediction bằng hai màu, legend và nhãn; đưa tất cả về pixel khi vẽ.
- Để trả lời Câu 3, dùng output đã có và scorer gốc chấm thêm với thang sai, lưu ở `report/runs/coordinate_diagnostic.json`; evidence chính vẫn giữ `coord=norm1000`. Ghi đây là phép phân tích sau đánh giá, không dùng để chọn model.
- Vẽ loss curve từ log thật; không chỉnh dữ liệu log để tạo đường đẹp.

### Nội dung 10 câu REPORT.md

| Câu | Bằng chứng dùng để trả lời |
|---|---|
| 1 | Cây module, số tham số theo khối, số tham số cập nhật E1/E2/E3 |
| 2 | E0 parse-valid, F1, output thô và hai lỗi thực sự quan sát được |
| 3 | Chuyển nhãn, round-trip pixel, so sánh chấm đúng/sai thang |
| 4 | Decode answer-only, số token bị mask, forward/backward kiểm chứng |
| 5 | Kiểm tra loại layer 4-bit, dtype projector và save/reload |
| 6 | Split audit, số ảnh/box mỗi lớp, phiên không giao nhau |
| 7 | Bảng E1/E2/E3: F1, CI95, VRAM, thời gian, trade-off |
| 8 | CI95 thật trên 311 ảnh; giới hạn bootstrap theo ảnh và tương quan frame cùng phiên |
| 9 | F1 từng lớp và ít nhất 6 ảnh minh họa lỗi có chú thích |
| 10 | E4 hai rank, effective config diff, kết quả và mức bất định |

Điền họ tên **Nguyễn Tuấn Thành**, mã số **2A202602640**. Thêm môi trường RTX 6000 Ada/GPU 1, revision model, phiên bản thư viện, ngoại lệ đã xử lý, seed và cách đo tài nguyên. Không diễn giải tốc độ trên RTX 6000 Ada là tốc độ Colab T4.

**Sản phẩm:** `REPORT.md`, `CHECKLIST.md`, `BAO_CAO_LOI_VA_LUU_Y.md`, hình và bảng tổng hợp bằng chứng.

## 14. Bước 9 — E5 bonus sau phần bắt buộc

E5 thực hiện sau khi E0–E4 ổn định. Nếu API TRL ghim trong đề không cài/chạy được, ghi rõ lỗi và kiểm chứng phương án tương thích trước; không để phần bonus làm mất checkpoint hoặc kết quả bắt buộc.

### Reward và kiểm thử

1. Dùng `scripts.score.parse_output` và `iou_xyxy`, không viết lại định nghĩa chấm điểm.
2. JSON hỏng → 0; GT và pred đều rỗng → 1.
3. Match greedy cùng lớp, IoU >= 0.5, mỗi box chỉ match một lần.
4. Reward `0.2 + 0.8 * sum_matched_iou / max(N_gt,N_pred,1)` cho output parse được.
5. Box của ground truth đầu vào reward phải được chuyển từ YOLO sang cùng thang 0–1000; parser trả prediction có key `bbox`.
6. Test tối thiểu: hoàn hảo 1.0; JSON hỏng 0.0; sai lớp 0.2; thêm một box thừa bên cạnh một GT matched hoàn hảo 0.6; cùng rỗng 1.0. Bổ sung test duplicate matching và giant box để xác nhận không match nhiều lần.

### Chạy và đánh giá

- Khởi tạo từ setup SFT chọn theo val, nạp đủ adapter **và projector** nếu có.
- Dùng `trl.GRPOTrainer`; xác minh API thực tế của phiên bản được cài trước khi hiện thực.
- `num_generations=4`, `max_completion_length=128`, `max_steps=60`, `learning_rate=5e-6`, `beta=0`, `use_vllm=False` theo cấu hình/guidance đề.
- Train bằng train split; ghi phần thưởng, loss và tỷ lệ output rỗng/giant box, kiểm tra batch phù hợp số generations.
- Kiểm tra save/reload rồi val; chốt run, suy luận test và sinh `report/evidence/e5.json` bằng scorer gốc.
- So sánh F1/CI95 với SFT cùng checkpoint nền; phân tích reward hacking bằng output thật. Không mặc định reward cao sẽ làm F1 tăng.

**Commit bonus:** hàm reward, test độc lập, GRPO, evidence và phân tích.

## 15. Bước 10 — Kiểm thử cuối và đóng gói

### Kiểm tra kỹ thuật

1. Chạy test mã tự viết ở `student_tests/` và kiểm tra tích hợp cần thiết trên server.
2. Đối chiếu hash/nội dung `scripts/`, `tests/`, `configs/` với commit gốc; không để các công cụ này bị sửa.
3. Kiểm tra schema và đầy đủ ID cho mọi evidence; E4 có hai file dù checker gốc chỉ yêu cầu tối thiểu một.
4. Kiểm tra các điều kiện rubric ngoài checker: E1 parse-valid/F1 improvement, đủ minh họa, số bước train, bảng và 10 câu trả lời đúng bằng chứng.
5. Quét credential, đường dẫn nhạy cảm và file cấm trong mã, notebook output và các file sắp commit. Dữ liệu, `.env`, adapter, optimizer, trọng số đều nằm ngoài Git.
6. Đồng bộ evidence, hình, log chọn lọc và báo cáo về Windows. Đồng bộ cả `data/splits/meta.json` vào vùng dữ liệu không track để checker local đối chiếu phiên bản, tránh WARN do thiếu metadata.
7. Giữ raw predictions/checkpoint/log đầy đủ trên server để có thể tái kiểm chứng dù không đưa chúng vào Git.

### Mâu thuẫn test template cần công bố

Test gốc `test_report_template_has_student_markers_and_ten_questions` trong `tests/test_configs_and_docs.py` assert `"<điền" in REPORT.md`, còn `scripts/check_submission.py` báo FAIL nếu REPORT.md còn `<điền`.

Do đó, **không thể đồng thời để toàn bộ test template gốc pass và REPORT.md cuối hoàn chỉnh**, khi giữ nguyên hai công cụ này. Cách xử lý:

- Chạy và lưu full pytest trước khi điền báo cáo: đã xác minh baseline `68 passed, 1 skipped` ở Windows; lặp lại trên server và sau khi có raw data.
- Khi báo cáo hoàn chỉnh, chạy lại full pytest, ghi rõ nếu test template duy nhất này thất bại vì yêu cầu trái nhau. Các test còn lại và test mã tự viết phải pass; nếu có lỗi khác, xử lý thật.
- Không để lại placeholder để lách checker, không sửa test gốc, không trình bày full pytest cuối là pass khi nó fail.
- Ghi vào `BAO_CAO_LOI_VA_LUU_Y.md` và nhờ giảng viên xác nhận cách đánh giá test template nếu cần.
- Tiêu chí nộp bài chính vẫn là báo cáo hoàn chỉnh, công cụ gốc nguyên vẹn và `check_submission.py` đạt 0 FAIL, 0 WARN.

### Lệnh nghiệm thu

```bash
python -m pytest -q
python -m pytest -q student_tests
python scripts/check_submission.py
git status --short
git rev-list --count HEAD
git diff c2f3997 -- scripts tests configs
git ls-files
```

Các lệnh cho `student_tests` chỉ chạy sau khi thư mục đã có test. Trên Windows bật UTF-8 nếu console cần. Rà soát danh sách file thực tế trước khi stage, tránh dùng `git add -A` không kiểm tra.

**Commit cuối dự kiến #6:** báo cáo, minh họa, checklist và log nghiệm thu. Số commit gốc không thay thế các commit thực hiện từng bước có nội dung thật.

## 16. Bước 11 — GitHub và LMS

1. Xác minh remote là repo cá nhân dùng để nộp, quyền push và branch chính thực tế; không mặc định đổi chủ repo.
2. Chuẩn bị commit cuối và kết quả nghiệm thu để xem lại trước khi công bố.
3. Khi đến bước xuất bản, thực hiện push theo phạm vi người dùng cho phép; xác minh GitHub hiển thị đúng HEAD và trạng thái Public.
4. Tạo hướng dẫn tái chạy và đường dẫn repo nộp bài. Nộp lên LMS khi người dùng yêu cầu và có công cụ/truy cập phù hợp; không tuyên bố đã nộp chỉ vì repo hoàn chỉnh.
5. Chuẩn bị ghi chú vấn đáp: kiến trúc, mask loss, float32 projector, save/reload, split session, E4 và CI95.

## 17. Thứ tự chạy, tài nguyên và thời gian

### Thứ tự thực hiện

```text
Môi trường + test gốc
  → Dataset + audit + 6 ảnh nhãn
  → Grounding + E0
  → Encode/collate + mask + gradient smoke test
  → Setup LoRA + save/reload checks
  → E1 → E2 → E3, mỗi run có val và test đã chốt
  → Chọn nền E4 theo val → rank 4 → rank 64
  → Phân tích lỗi + báo cáo bắt buộc + nghiệm thu
  → E5 bonus + cập nhật báo cáo, nếu chạy được
  → Kiểm tra cuối + đóng gói → GitHub/LMS
```

Chỉ có một GPU được cấp nên các run GPU chạy tuần tự. Có thể dùng CPU chuẩn bị hình/bảng trong lúc train nếu không làm mất tài nguyên của job chính; kế hoạch này không yêu cầu chạy nhiều agent.

### Khối lượng phải chạy

- E0: một run inference test 311 ảnh.
- E1/E2/E3: ba run × 300 optimizer updates.
- E4 rank: hai run × 300 optimizer updates.
- Tổng bắt buộc: **1.500 optimizer updates SFT**, 6 bộ prediction test = **1.866 lượt ảnh test**, chưa gồm val/smoke test/chạy lại do lỗi.
- Bonus E5: khoảng 60 bước GRPO và thêm 311 lượt ảnh test; mỗi ảnh train có nhiều generation nên chi phí không tương đương SFT.

Không lấy ước lượng 20–30 phút/run trên T4 làm thời gian cam kết trên server. Sau smoke test, đo tốc độ update thực tế cho từng setup và tốc độ inference trên val rồi cập nhật ETA; E3/projector/rank64 có chi phí khác nhau. Tải model, cài thư viện và xử lý lỗi có thể chiếm phần lớn thời gian đầu.

## 18. Phương án xử lý lỗi

| Vấn đề | Hành động | Bằng chứng cần ghi |
|---|---|---|
| Gateway `NO_RECENT_AUTH` | Làm mới phiên portal/Cloudflare theo trường; reconnect đúng alias | Trạng thái kết nối, không ghi mật khẩu |
| GPU 1 còn job đang chạy | Xác định chủ/PID/mục đích và dừng đúng job theo quyền đã được cấp | GPU UUID, memory/process snapshot |
| Pin thư viện không cài được | Giữ file gốc, xác minh phiên bản khả dụng/API, tạo file phụ thuộc Linux riêng | Lỗi cài đặt, version thay thế, test tương thích |
| Import torchvision/CUDA/bitsandbytes lỗi | Ghép bản torch/torchvision tương thích, test CUDA và quantized forward trong venv | `pip check`, version và kết quả smoke test |
| Dataset/split khác đề | Kiểm tra dataset version, raw root, nhãn và tên file; không sửa split script | Audit chênh lệch và nguyên nhân |
| 220 vision tokens không khớp | Kiểm tra kích thước ảnh, processor resize/config, merge size và revision | Image grid, token count, config processor |
| LoRA module count sai | So cây module; E3 có cả `attn.proj`; tránh regex bắt nhầm merger | Danh sách matched module và count |
| CUDA OOM | Xem job khác và peak memory; batch 1/accum 8 giữ effective batch 8; ghi override chung | Memory snapshot, cấu hình hiệu lực |
| NaN/Inf | Dừng run, kiểm tra dtype projector, labels, scaler và gradient; giảm lr có kiểm soát nếu cần | Bước lỗi, loss/gradient, thay đổi so cấu hình gốc |
| Reload làm mất kết quả E2/E3 | Kiểm tra projector checkpoint cùng bước, đủ key và dtype | So tham số/output trước-sau reload |
| Parse-valid thấp | Kiểm tra EOS, mask, raw output và truncation; debug trên val | Output mẫu, token length, generation config |
| F1 thấp dù parse-valid cao | Kiểm tra scale, box ordering, labels và checkpoint | Round-trip, ảnh GT/pred, val metrics |
| Mất SSH | Reattach tmux, kiểm tra PID/log/checkpoint; resume từ trạng thái đã lưu | Bước cuối, số update, resume state |
| CI rộng/overlap | Báo mức bất định, không kết luận mạnh hơn bằng chứng | CI95 gốc, paired analysis nếu thực hiện |
| Full pytest cuối fail test placeholder | Công bố mâu thuẫn test template và checker, giữ nguyên đề | Baseline test log và final test log |

## 19. Checklist điều phối

- [x] Đã nhận họ tên, mã số và quyền dùng GPU 1 (thay GPU 2 theo yêu cầu mới).
- [x] Đã xác minh SSH và Kaggle từ server.
- [x] Đã chạy baseline test tại Windows: 68 passed, 1 skipped.
- [x] Đã đọc cấu hình, scorer, checker và test gốc để viết kế hoạch.
- [ ] Thiết lập venv/server, kiểm tra GPU 1 và full test Linux.
- [ ] Tải dataset, audit split, test dữ liệu thật và vẽ nhãn.
- [ ] Hoàn thiện grounding và evidence E0.
- [ ] Hoàn thiện mask loss/collate, kiểm chứng 220 token và gradient.
- [ ] Kiểm tra 252/348 module, dtype và save/reload projector.
- [ ] Chạy E1, E2, E3 đủ 300 update và chấm test.
- [ ] Chạy E4 rank 4/64 và phân tích CI95.
- [ ] Hoàn thiện report, minh họa, checklist và báo cáo lỗi.
- [ ] Thực hiện E5 sau phần bắt buộc nếu môi trường hỗ trợ.
- [ ] Nghiệm thu 0 FAIL, 0 WARN; công bố riêng mâu thuẫn test template.
- [ ] Kiểm tra commit/file cấm, chuẩn bị và xuất bản repo đúng phạm vi cho phép.
- [ ] Chuẩn bị URL và hồ sơ nộp LMS.

**Điểm bắt đầu khi triển khai:** Bước 0 trên GPU trường. Tại thời điểm viết kế hoạch, chỉ đã kiểm tra kết nối/xác thực, đọc đề và chạy baseline test; chưa tải dataset, cài môi trường GPU hay train mô hình.
