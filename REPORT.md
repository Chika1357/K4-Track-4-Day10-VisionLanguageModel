Họ tên: Nguyễn Tuấn Thành — Mã số: 2A202602640
`dataset_version`: d10-21e4070301

> **Trạng thái báo cáo (2026-10-14):** đã chạy đủ E0–E4 (E4: rank 4 và 64 trên nền E3) và chấm trên 311 ảnh test bằng `score.py` gốc. E5 (bonus GRPO) **chưa chạy**. Mọi số liệu lấy từ `report/evidence/*.json`; chấm lại độc lập khớp.
>
> Môi trường: NVIDIA RTX 6000 Ada (GPU vật lý số 1, máy `bcseserver1`), **không phải Colab T4**; VRAM/thời gian không so trực tiếp với số mẫu trên T4. Model `Qwen/Qwen3-VL-4B-Instruct` ghim revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, NF4 4-bit cho LLM, giữ ViT/projector/`lm_head` không lượng tử hoá. Python 3.10.12, torch 2.14.1 (CUDA 13.0), transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 3.911 (suy luận) | 3.78 (suy luận 311 ảnh) | 0.0875 | 0.0620–0.1143 | 0.936 | 0.1088 | 0.0353 | 0.1790 |
| E1 | LoRA LLM (r=16) | 33,030,144 | 6.685 (train) | 8.16 (train 300 bước) | 0.2913 | 0.2535–0.3300 | 1.238 | 0.2049 | 0.2212 | 0.5405 |
| E2 | + projector | 142,135,296 | 8.408 (train) | 8.24 (train 300 bước) | 0.3270 | 0.2903–0.3694 | 1.119 | 0.1717 | 0.2667 | 0.5857 |
| E3 | + LoRA ViT | 148,426,752 | 8.595 (train) | 9.75 (train 300 bước) | 0.4342 | 0.3953–0.4790 | 0.897 | 0.3519 | 0.3553 | 0.6715 |
| E4 r=4 | E3 với rank 4 (alpha=8) | 118,935,552 | 8.123 (train) | 9.80 | 0.4076 | 0.3650–0.4499 | 0.965 | 0.2857 | 0.3401 | 0.6502 |
| E4 r=64 | E3 với rank 64 (alpha=128) | 266,391,552 | 10.571 (train) | 10.11 | 0.4419 | 0.3993–0.4868 | 0.971 | 0.2870 | 0.3872 | 0.6806 |

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

`parse_valid_rate` không đổi (0.8457) vì chỉ phép đổi tọa độ khác nhau. Phía huấn luyện, nhãn YOLO được đổi sang 0–1000 (`lab/data.py`, có unit test), nên model học đúng thang nó xuất ra. Chẩn đoán tương tự trên output của E3 (`report/runs/coordinate_diagnostic.json`): đọc đúng norm1000 cho F1 = 0.4342, đọc như pixel cho F1 = 0.0000 (TP=0), nên đây là yếu tố quyết định đúng/sai chứ không phải chi tiết nhỏ.

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

| | Tham số huấn luyện | VRAM đỉnh (GB) | Phút train | F1 test | CI95 | F1 val |
|---|---|---|---|---|---|---|
| E1 | 33.0M | 6.685 | 8.16 | 0.2913 | 0.2535–0.3300 | 0.1910 |
| E2 | 142.1M | 8.408 | 8.24 | 0.3270 | 0.2903–0.3694 | 0.2920 |
| E3 | 148.4M | 8.595 | 9.75 | 0.4342 | 0.3953–0.4790 | 0.3626 |

- **E1 → E2 (mở projector):** F1 test +0.0358 (val +0.1010). Phải trả 4.3 lần số tham số huấn luyện và +1.72 GB VRAM, thời gian gần như không đổi (+0.07 phút). CI95 của E1 và E2 chồng lấp rộng, nên trên test chỉ có thể nói "có xu hướng tốt hơn", chưa phải khác biệt chắc chắn. Val cùng chiều và lớn hơn, nhưng val và test là hai tập khác nhau.
- **E2 → E3 (thêm LoRA ViT):** F1 test +0.1072 (val +0.0705), chỉ thêm 6.3M tham số, +0.19 GB và +1.52 phút. CI95 của E2 (cận trên 0.3694) nằm dưới cận dưới của E3 (0.3953), không chồng lấp. Đây là bước đáng giá nhất trong ba bước: chi phí rất nhỏ, cải thiện rõ ở cả ba lớp (pothole 0.172 → 0.352, crack 0.267 → 0.355, manhole 0.586 → 0.672) và FP/ảnh giảm 1.119 → 0.897.
- **Giới hạn:** một seed (42) cho mỗi setup. Thời gian đo trên RTX 6000 Ada, không so được với T4. E3 bỏ 2 bước fp16 overflow (xem `BAO_CAO_LOI_VA_LUU_Y.md`), nên thứ tự dữ liệu lệch 2 batch so với E1/E2.

Kết luận: với 300 bước, mở LoRA trên ViT đáng làm; mở projector rẻ về thời gian nhưng tốn bộ nhớ và lợi ích trên test chưa chắc chắn.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

Độ rộng CI95 bootstrap (1.000 lần, theo ảnh) của F1:

| E0 | E1 | E2 | E3 | E4 r=4 | E4 r=64 |
|---|---|---|---|---|---|
| 0.052 | 0.077 | 0.079 | 0.084 | 0.085 | 0.088 |

Ở F1 ≈ 0.4 (E3), nửa độ rộng là ±0.042, tức khoảng ±10% tương đối; càng tốt thì khoảng càng rộng tuyệt đối. Hệ quả cho so sánh:

- Chênh lệch lớn tách được: E0 → E1 (+0.204) và E2 → E3 (+0.107) có CI không chồng lấp.
- Chênh lệch nhỏ không tách được: E1 → E2 (+0.036), E3 → E4 r=64 (+0.008), r=4 so với r=64 (+0.034) đều chồng lấp.
- CI theo ảnh có thể **lạc quan**: các frame liền kề cùng phiên (test chỉ gồm một số phiên) tương quan nhau, nên 311 ảnh không phải 311 mẫu độc lập. CI chồng lấp chỉ mô tả bất định, không phải phép kiểm định chênh lệch.
- Chỉ có một seed và 583 box test (121 pothole), nên F1 từng lớp càng nhiễu hơn F1 tổng.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Theo lớp trên test, model nền E3 (`report/evidence/e3.json`):

| Lớp | Box test | Diện tích box trung vị (% ảnh) | P | R | F1 | TP / FP / FN |
|---|---|---|---|---|---|---|
| pothole | 121 | 0.42 | 0.400 | 0.314 | **0.352** | 38 / 57 / 83 |
| crack | 299 | 1.98 | 0.350 | 0.361 | 0.355 | 108 / 201 / 191 |
| manhole | 163 | 0.67 | 0.816 | 0.571 | 0.672 | 93 / 21 / 70 |

- **Kém nhất:** pothole (0.352) và crack (0.355) gần như bằng nhau, đều chỉ bằng nửa manhole. Pothole kém vì **ít mẫu** (1.074 box train, chỉ 66 box ở val) và **box nhỏ** (trung vị 0.42% ảnh, cỡ 30×30 px trên ảnh 640×360), nên lệch vài pixel là IoU < 0.5. Recall 0.314: 83/121 box bị bỏ sót.
- **Crack** có nhiều mẫu nhất (1.750 box train) nhưng 201 FP và 191 FN. Giả thuyết (chưa kiểm chứng bằng nhãn): vết nứt dài, mảnh, ranh giới không rõ nên model và người gán nhãn chia/gộp vết nứt khác nhau; ảnh `prediction_e3_3.png` cho thấy box crack dự đoán ngắn hơn GT nên IoU dưới 0.5.
- **Manhole** tốt nhất: hình dạng rõ, tương phản mạnh, precision 0.816.
- Tính ổn định: giữa E3 và E4 F1 pothole dao động 0.352 → 0.286 → 0.287 trong khi F1 tổng đổi ít, cho thấy F1 từng lớp trên 121 box nhiễu.

Ảnh minh họa (GT xanh, dự đoán cam; 6 ảnh thật từ test với E3, `report/figures/prediction_e3_{1..6}.png`, manifest `selection_e3.json` ghi TP/FP/FN):

| Ảnh | ID | Loại | TP/FP/FN |
|---|---|---|---|
| 1 | vlcsnap-00202 | đúng | 1/0/0 |
| 2 | vlcsnap-00200 | FP (manhole đặt sai chỗ, manhole GT ở mép phải bị bỏ sót) | 1/1/1 |
| 3 | vlcsnap-00201 | bỏ sót (crack dự đoán quá ngắn, manhole nhỏ ở xa bị bỏ) | 0/1/2 |
| 4 | vlcsnap-00203 | đại diện | 0/1/1 |
| 5 | vlcsnap-00204 | đúng | 1/0/0 |
| 6 | vlcsnap-00205 | đại diện | 1/2/1 |

Toàn test có 106 ảnh đúng hoàn toàn, 167 ảnh có FP, 187 ảnh bỏ sót, 0 output sai JSON. Ảnh 4 và 6 tôi chưa xem trực quan; mô tả chỉ dựa trên TP/FP/FN. Biểu đồ: `f1_comparison.png`, `training_loss.png`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

**Biến đổi:** rank LoRA r ∈ {4, 64} với alpha = 2r (scale cố định = 2), trên nền **E3** (setup có F1 val cao nhất trong E1–E3: 0.3626). Giữ nguyên seed 42, lr 2e-4, batch 2 × grad_accum 4, 300 bước, processor, prompt và cách suy luận. Rank 16 chính là E3.

| Rank | Tham số train | VRAM (GB) | Phút | F1 test | CI95 | F1 val |
|---|---|---|---|---|---|---|
| 4 | 118.9M | 8.123 | 9.80 | 0.4076 | 0.3650–0.4499 | 0.3320 |
| 16 (E3) | 148.4M | 8.595 | 9.75 | 0.4342 | 0.3953–0.4790 | 0.3626 |
| 64 | 266.4M | 10.571 | 10.11 | 0.4419 | 0.3993–0.4868 | 0.3545 |

**Kết quả:** trên test, F1 tăng đơn điệu theo rank (+0.034 từ r=4 lên r=64), nhưng **trên val thì r=16 tốt hơn r=64** (0.3626 so với 0.3545). Hai tập cho thứ tự khác nhau ở cặp r=16/r=64.

**Có đáng tin không?** Chỉ ở mức thận trọng:
- Mọi CI95 chồng lấp mạnh; r=4 so với r=64 chênh +0.034 nằm trong độ bất định.
- Một seed, và thứ tự r=16/r=64 đảo giữa val và test, tức chênh lệch r=16 so với r=64 nằm trong nhiễu.
- Tổng tham số huấn luyện của r=4 vẫn là 118.9M vì phần projector (109.1M) cố định; rank chỉ đổi phần LoRA.
- Rank 64 tốn thêm 2.0 GB VRAM so với r=16 mà không có lợi ích chắc chắn.

Kết luận hợp lý: giữ r=16 (E3) vì cân bằng tốt nhất. Chưa đủ bằng chứng để nói rank cao hơn thực sự tốt hơn; cần nhiều seed.

## Nhìn lại

**Lỗi mà công cụ chấm không bắt được** (đều do tôi phát hiện nhờ audit/kiểm tra riêng):
- Skip list `["visual", "lm_head"]` trong ví dụ đề âm thầm lượng tử hoá ViT/projector với transformers 5.19.0; scorer chỉ nhận prediction nên không thấy. Chỉ audit kiểu layer mới phát hiện (`visual_and_lm_head_not_4bit=true`).
- LoRA ViT thiếu `attn.proj` sẽ thiếu module (không đủ 96/348) mà vẫn chạy được và vẫn cho F1.
- E3 dừng vì gradient fp16 tràn ở scale 65536. Bước tràn được bỏ và không tính vào 300 update (E3 bỏ 2, mỗi run E4 bỏ 3); chi tiết trong `BAO_CAO_LOI_VA_LUU_Y.md`.
- Venv kế thừa Jinja2 3.0.3 làm chat template lỗi.
- E0: 43/311 output bị cắt vì model lặp box đến hết `max_new_tokens`; `parse_valid_rate` gộp lỗi bị cắt và lỗi cú pháp.

**Điều chưa hiểu / chưa kiểm chứng:**
- Vì sao rank cao làm giảm F1 pothole (0.352 → 0.287) trong khi F1 tổng tăng; có thể chỉ là nhiễu trên 121 box.
- Nhãn crack có nhất quán không (một vết hay nhiều đoạn); mới là giả thuyết từ một ảnh, cần xem thêm nhãn GT.
- Chưa có thí nghiệm đối chứng "loss trên cả chuỗi" cho Câu 4.
- Mới có một seed; chưa đo phương sai giữa các lần chạy.

**Điều kiện thí nghiệm:** RTX 6000 Ada chứ không phải T4. Server dùng chung; job của người dùng khác làm hỏng hai lần chạy (E2 lần đầu, E3 lần đầu) do OOM; các run lỗi được giữ lại và chạy lại từ đầu với cùng cấu hình.

**Ngoại lệ kiểm thử:** `tests/test_configs_and_docs.py` yêu cầu REPORT.md còn chuỗi giữ chỗ của mẫu (dấu `<` rồi chữ "điền"), mâu thuẫn với checker nộp bài (yêu cầu xoá). Giữ nguyên test gốc; báo cáo hoàn chỉnh làm test đó fail, đã nêu trong `BAO_CAO_LOI_VA_LUU_Y.md`.
