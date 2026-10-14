Họ tên: Nguyễn Tuấn Thành — Mã số: 2A202602640
`dataset_version`: d10-21e4070301

> **Trạng thái báo cáo (bản tạm, 2026-10-14):** đã hoàn thành chuẩn bị dữ liệu, E0 zero-shot, kiểm chứng encoding/mask/gradient và E1. E4 rank 64, phân tích lỗi và ảnh minh họa **chưa xong**; các ô và câu hỏi phụ thuộc kết quả đó ghi `<điền sau khi chạy ...>` và sẽ cập nhật bằng số đo thật.
>
> Môi trường: NVIDIA RTX 6000 Ada (GPU vật lý số 1, máy `bcseserver1`), **không phải Colab T4**; VRAM/thời gian không so trực tiếp với số mẫu trên T4. Model `Qwen/Qwen3-VL-4B-Instruct` ghim revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, NF4 4-bit cho LLM, giữ ViT/projector/`lm_head` không lượng tử hoá. Python 3.10.12, torch 2.14.1 (CUDA 13.0), transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 3.911 (suy luận) | 3.78 (suy luận 311 ảnh) | 0.0875 | 0.0620–0.1143 | 0.936 | 0.1088 | 0.0353 | 0.1790 |
| E1 | LoRA LLM (r=16) | 33,030,144 | 6.685 (train) | 8.16 (train 300 bước) | 0.2913 | 0.2535–0.3300 | 1.238 | 0.2049 | 0.2212 | 0.5405 |
| E2 | + projector | 142,135,296 | 8.408 (train) | 8.24 (train 300 bước) | 0.3270 | 0.2903–0.3694 | 1.119 | 0.1717 | 0.2667 | 0.5857 |
| E3 | + LoRA ViT | 148,426,752 | 8.595 (train) | 9.75 (train 300 bước) | 0.4342 | 0.3953–0.4790 | 0.897 | 0.3519 | 0.3553 | 0.6715 |
| E4 | rank LoRA: r=4 và r=64 (alpha=2r) | | <điền sau khi chạy E4> | | | | | | | |

E1: `parse_valid_rate=0.9968` (≥ 0.95), F1 tăng +0.2038 so với E0 (≥ 0.10); val F1=0.1910 (CI95 0.1576–0.2226); suy luận test 4.98 phút, 4.04 GB. VRAM đỉnh đo bằng `torch.cuda.max_memory_allocated` (bytes/1e9). E2: `parse_valid_rate=0.9968`, val F1=0.2920 (CI95 0.2530–0.3293); khác biệt E2−E1 trên test +0.0357 nhưng CI95 hai setup chồng lấp. E3: `parse_valid_rate=1.0000`, val F1=0.3626 (CI95 0.3224–0.4045), 2 bước fp16 overflow bị bỏ (không tính vào 300 update), được chọn làm nền E4 theo val. Số tham số lấy từ preflight trên model thật (`report/runs/preflight_e{1,2,3}.json`), chưa phải số của run huấn luyện. E0: `parse_valid_rate=0.8457`, `missing_predictions=0`, `n_images=311`, P=0.1208, R=0.0686 (TP=40, FP=291, FN=583), mean IoU của cặp khớp 0.600; batch 4, greedy, `max_new_tokens=256`.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm trên model đã nạp (`report/runs/e0_model_audit.json`, đếm tham số logic, không phải số byte 4-bit):

| Khối | Tham số |
|---|---|
| ViT (không gồm projector) | 306,242,560 |
| Projector (merger, gồm deepstack mergers) | 109,105,152 |
| LLM + heads | 4,022,468,096 |
| Tổng | 4,437,815,808 |

Phần cập nhật, kiểm chứng bằng một bước forward/backward thật cho mỗi setup:

- **E1:** chỉ LoRA r=16 trên `q/k/v/o_proj` và `gate/up/down_proj` của 36 layer LLM: 252 module, 33,030,144 tham số. Projector và ViT đóng băng; gradient chỉ xuất hiện ở LoRA LLM (tổng norm 50.25).
- **E2:** E1 + mở toàn bộ projector (109,105,152 tham số, float32): tổng 142,135,296. Gradient projector có norm 5.90.
- **E3:** E2 + LoRA r=16 trên `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2` của 24 block ViT (96 module, 6,291,456 tham số): 348 module, tổng 148,426,752. Gradient ViT LoRA có norm 15.45.

Ở cả ba setup không có tham số đóng băng nào nhận gradient (`frozen_parameters_with_grad=[]`). Nếu chỉ dùng `qkv/linear_fc1/linear_fc2` như ví dụ LMS sẽ thiếu `attn.proj` và không đủ 96 module ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate = 263/311 = 0.8457`. 48 output không parse được: 43 bị cắt tại `max_new_tokens=256` vì model lặp vô hạn các box `crack` gần như giống nhau, 5 có lỗi cú pháp (ví dụ `[287, 478, 357, 500)}`).

Theo số ảnh có lỗi (`report/runs/e0_examples.json`):

1. **Bỏ sót GT:** 290/311 ảnh có ít nhất một box GT không được khớp; recall chỉ 0.0686 (543/583 box bị bỏ sót). 16 ảnh model trả `[]`.
2. **Dự đoán sai (FP):** 219/311 ảnh có FP, tổng 291 FP (0.936 FP/ảnh). Crack tệ nhất: 202 FP, chỉ 9 TP (F1 0.0353).

Ví dụ output thô (`vlcsnap-2025-02-18-23h07m47s847`): model sinh một cột box crack cách đều 40 đơn vị rồi bị cắt giữa chừng, nên vừa invalid JSON vừa bỏ sót GT:

```
[{"label": "crack", "bbox_2d": [350, 370, 400, 400]}, {"label": "crack", "bbox_2d": [350, 330, 400, 360]}, {"label": "crack", "bbox_2d": [350, 290, 400, 320]}, ... {"label": "crack", "bbox_2d": [350, 90, 4
```

Khi khớp được thì box khá đúng vị trí (mean IoU 0.600). Vấn đề chính là chọn đối tượng, định nghĩa lớp và định dạng, không phải hệ tọa độ.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Qwen3-VL xuất `bbox_2d` theo thang tương đối 0–1000, độc lập kích thước ảnh. Ảnh dataset là 640×360 pixel. Nếu hiểu các số đó là pixel, một box như `[770, 640, 843, 686]` nằm ngoài khung 640×360. Các box còn trong khung bị co về góc trên-trái (trục x lệch hệ số 1000/640, trục y lệch 1000/360). IoU với GT gần như bằng 0.

Kiểm chứng: chấm lại **cùng file** `work/e0.jsonl` bằng `scripts/score.py` gốc.

- `--coord norm1000`: F1 = 0.0875 (CI95 0.0620–0.1143), khớp `report/evidence/e0.json`.
- `--coord pixel`: F1 = **0.000**, TP=0, FP=331, FN=583, mean IoU 0 (`report/runs/e0_coordinate_diagnostic.json`).

`parse_valid_rate` không đổi (0.8457) vì chỉ phép đổi tọa độ khác nhau. Phía huấn luyện, nhãn YOLO được đổi sang 0–1000 (`lab/data.py`, có unit test), nên model học đúng thang nó xuất ra. Sau khi có model nền E4 sẽ chạy lại chẩn đoán này trên output đã fine-tune.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Số đo thật trên ảnh train đầu tiên (preflight E1–E3): chuỗi dài 562 token, gồm 220 token ảnh (grid 1×22×40, merge 2×2 → 11×20). 312 token bị mask `-100` (system/user prompt + token ảnh + template). 250 token được tính loss, đúng bằng câu trả lời JSON cộng `<|im_end|>`; giải mã các token có nhãn ra đúng JSON GT, kết thúc bằng `<|im_end|>`. Loss forward đầu tiên 1.386.

Chỉ tính loss trên trả lời vì mục tiêu là p(trả lời | ảnh, prompt). Nếu tính cả prompt và token ảnh:

- **Prompt cố định** giống hệt mọi mẫu, nên model sẽ học thuộc prompt. Phần loss này nhanh chóng gần 0 nhưng vẫn chiếm 312/562 = 56% số token trong mẫu này, làm loãng tín hiệu của box.
- **Token ảnh** là placeholder `<|image_pad|>` được thay bằng embedding thị giác. Dự đoán placeholder kế tiếp là nhiệm vụ vô nghĩa, có thể kéo LoRA về phía học lặp token thay vì grounding.
- **`<|im_end|>`:** nếu không giám sát token này, model không học cách dừng. Đây chính là lỗi lặp cho đến 256 token đã thấy ở E0.

Đây là lập luận cộng với số đo encoding; tôi chưa chạy thí nghiệm đối chứng "loss trên toàn chuỗi".

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Khi lượng tử hoá, bitsandbytes thay `nn.Linear` bằng `Linear4bit`. Trọng số lúc đó được lưu dạng NF4 đóng gói (uint8) kèm hằng số scale, không còn là tensor float có `requires_grad`, nên không thể cập nhật trực tiếp toàn bộ trọng số, chỉ có thể gắn LoRA lên trên. Projector nằm trong `model.visual` (merger), nên nếu ViT bị lượng tử hoá thì projector cũng bị, và E2/E3 không mở được projector thật sự.

Lỗi gặp thực tế: với transformers 5.19.0, `llm_int8_skip_modules=["visual","lm_head"]` như ví dụ **không đủ**. Matcher so prefix/regex từ đầu tên đầy đủ, không tìm chuỗi con, nên các layer `model.visual.*` vẫn thành `Linear4bit`. Audit kiểu layer phát hiện lỗi này trước khi chạy suy luận. Cách xử lý: thêm `model.visual` vào skip list (`lab/modeling.py`), test matcher thật (`student_tests/test_quantization_scope.py`), và giữ audit `visual_and_lm_head_not_4bit=true` mỗi lần nạp.

Projector ở float32 vì nó được huấn luyện toàn bộ, không qua LoRA. Với fp16 (10 bit mantissa), cập nhật AdamW cỡ lr×update = 2e-4×O(1) lên trọng số có độ lớn ~1e-1–1 dễ bị làm tròn mất hoặc tràn. GradScaler cũng không unscale được gradient fp16 của tham số fp16. Giữ master weight float32 và tính toán trong autocast fp16. Khi lưu/nạp lại, kiểm tra hash projector để chắc chắn trọng số không bị đổi kiểu.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh được trích từ video dashcam (`vlcsnap-*`), nên các frame liền nhau trong cùng phiên quay gần như trùng nhau: cùng ổ gà, cùng nắp cống, chỉ dịch vài pixel. Chia ngẫu nhiên sẽ đưa frame anh em vào cả train và test, model chỉ cần nhớ cảnh. F1 khi đó đo khả năng nhớ, không đo khả năng tổng quát sang đoạn đường mới.

Dataset có 18 phiên, chia theo phiên, `session_leakage=false`. 2,009 ảnh, 4,737 box, đã xác minh tất cả ảnh (`report/runs/data_audit.json`):

| Split | Ảnh | pothole | crack | manhole | Tổng box |
|---|---|---|---|---|---|
| train | 1,422 | 1,074 | 1,750 | 691 | 3,515 |
| val | 276 | 66 | 470 | 103 | 639 |
| test | 311 | 121 | 299 | 163 | 583 |

Phân bố lớp khác nhau rõ giữa các split (val chỉ có 66 pothole, nhưng 470 crack). Đây là cái giá của split theo phiên và là lý do chỉ dùng val để chọn setup E4.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

<điền sau khi chạy E1–E3: so F1, CI95, VRAM đỉnh, phút huấn luyện, dùng cùng seed 42, lr 2e-4, batch 2 × grad_accum 4, 300 optimizer step>

Đã biết trước khi huấn luyện: E2 tăng số tham số huấn luyện gấp 4.3 lần so với E1 (142.1M so với 33.0M), E3 chỉ thêm 6.3M so với E2.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

Mới có E0: CI95 bootstrap 0.0620–0.1143, rộng 0.052 quanh F1 = 0.0875 (khoảng ±30% tương đối). <điền sau khi chạy E1–E4: so độ rộng CI và mức chồng lấp giữa các setup>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở E0, crack kém nhất (F1 0.0353: 9 TP, 202 FP, 290 FN), dù là lớp nhiều box nhất trong test (299). Model gốc tách vết nứt dài thành chuỗi box nhỏ lặp lại, không khớp cách gán nhãn. Ảnh nhãn GT: `report/figures/labels_*.png`. <điền sau khi chạy setup nền: phân tích theo lớp và ít nhất 6 ảnh dự đoán đúng/sai>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Biến đã chọn: rank LoRA r ∈ {4, 64}, alpha = 2r. Setup nền là setup tốt nhất trên val trong E1–E3; giữ nguyên seed, lr, batch hiệu dụng, số bước, processor và prompt. <điền sau khi chạy E4>

## Nhìn lại

<điền đầy đủ sau khi hoàn tất. Hiện đã ghi nhận:>

- Ví dụ skip list `["visual", "lm_head"]` âm thầm lượng tử hoá ViT/projector với transformers 5.19.0. Công cụ chấm không bắt lỗi này, chỉ audit kiểu layer mới phát hiện.
- Ví dụ LoRA ViT thiếu `attn.proj`.
- Venv kế thừa Jinja2 3.0.3 của hệ thống, làm chat template lỗi.
- Test template `tests/test_configs_and_docs.py` yêu cầu REPORT.md còn placeholder, mâu thuẫn với checker nộp bài. Giữ nguyên test, sẽ công bố ngoại lệ khi hoàn thiện báo cáo.
