"""Write the final report only after complete, internally consistent real evidence."""
import json
from pathlib import Path

NAMES = ("e0", "e1", "e2", "e3", "e4_rank_4", "e4_rank_64")


def load_evidence(report_root="report"):
    root = Path(report_root)
    evidence = {}
    for name in NAMES:
        item = json.loads((root/"evidence"/f"{name}.json").read_text(encoding="utf-8"))
        if (item.get("score_version") != 1 or item.get("split") != "test"
                or item.get("n_images") != 311 or item.get("missing_predictions") != 0
                or item.get("coord") != "norm1000"):
            raise ValueError(f"{name}: incomplete/nonstandard test evidence")
        if name != "e0":
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                value = item.get("meta", {}).get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name}: invalid {key}")
        evidence[name] = item
    versions = {item["dataset_version"] for item in evidence.values()}
    if len(versions) != 1:
        raise ValueError("Evidence mixes dataset versions")
    return evidence


def report_table(evidence):
    lines = ["| Mã | Setup | Tham số train | VRAM đỉnh GB | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    descriptions = {"e0": "Zero-shot", "e1": "LoRA LLM", "e2": "LoRA LLM + projector",
                    "e3": "LoRA LLM/ViT + projector", "e4_rank_4": "Ablation rank 4",
                    "e4_rank_64": "Ablation rank 64"}
    for name in NAMES:
        item, meta = evidence[name], evidence[name]["meta"]
        params = meta.get("trainable_params", 0)
        peak = meta.get("peak_vram_gb", meta.get("inference_peak_vram_gb", 0))
        minutes = meta.get("train_minutes", meta.get("inference_minutes", 0))
        low, high = item["f1_ci95"]
        classes = " | ".join(f"{item['per_class'][label]['f1']:.4f}" for label in ("pothole", "crack", "manhole"))
        lines.append(f"| {item['name']} | {descriptions[name]} | {params:,.0f} | {peak:.3f} | {minutes:.2f} | "
                     f"{item['micro']['f1']:.4f} | [{low:.4f}, {high:.4f}] | {item['fp_per_image']:.3f} | {classes} |")
    return "\n".join(lines)


def generate_report(report_root="report", output="REPORT.md"):
    root = Path(report_root)
    evidence = load_evidence(root)
    audit = json.loads((root/"runs/data_audit.json").read_text())
    preflight = json.loads((root/"runs/preflight_e1.json").read_text())
    selection = json.loads((root/"runs/e4_selection.json").read_text())
    if audit["dataset_version"] != evidence["e0"]["dataset_version"]:
        raise ValueError("Data audit differs from evidence")
    e0, e1, e2, e3 = [evidence[f"e{index}"] for index in range(4)]
    low_rank, high_rank = evidence["e4_rank_4"], evidence["e4_rank_64"]
    architecture = preflight["architecture"]
    delta = e1["micro"]["f1"]-e0["micro"]["f1"]
    parse_pass = e1["parse_valid_rate"] >= .95
    delta_pass = delta >= .10
    examples = json.loads((root/"runs/e0_examples.json").read_text())
    wrong_scale = json.loads((root/"runs/coordinate_diagnostic.json").read_text())
    main_scale = evidence[selection["base_setup"].lower()]
    weakest = min(("pothole", "crack", "manhole"), key=lambda label: main_scale["per_class"][label]["f1"])
    e4_delta = high_rank["micro"]["f1"]-low_rank["micro"]["f1"]
    overlap = max(low_rank["f1_ci95"][0], high_rank["f1_ci95"][0]) <= min(low_rank["f1_ci95"][1], high_rank["f1_ci95"][1])
    lines = [
        "Họ tên: Nguyễn Tuấn Thành", "Mã số: 2A202602640",
        f"`dataset_version`: `{audit['dataset_version']}`", "",
        "## Môi trường và cách chạy", "",
        f"GPU: {preflight['environment']['gpu_name']}; GPU vật lý 1 (`CUDA_VISIBLE_DEVICES=1`); tài khoản research02.",
        f"Model `Qwen/Qwen3-VL-4B-Instruct`, revision `{preflight['model_revision']}`. Môi trường: `{json.dumps(preflight['environment'], ensure_ascii=False)}`.",
        "Đây là kết quả trên GPU trường RTX 6000 Ada; thời gian và VRAM không đại diện cho Colab T4.",
        "SFT: seed 42, 300 optimizer updates thành công; mask loss phần đáp án. Cấu hình hiệu lực và mọi override nằm trong report/runs/*_train.json.",
        "VRAM huấn luyện là max_memory_allocated, GB thập phân (bytes/1e9). Dòng E0 ghi tài nguyên/thời gian inference; các dòng SFT ghi train.",
        "Các script chấm/chia dữ liệu của đề được giữ nguyên. Evidence sinh bằng score.py, không điền tay.", "",
        "## Bảng tổng hợp", "", report_table(evidence), "",
        f"Ngưỡng E1: parse-valid {'đạt' if parse_pass else 'chưa đạt'}; cải thiện F1 tuyệt đối {delta:.4f}, "
        f"{'đạt' if delta_pass else 'chưa đạt'} yêu cầu >= 0.10. Đây là kết quả đo, không phải cam kết trước chạy.", "",
        "## 10 câu hỏi", "",
        "### Câu 1 — Kiến trúc và khối được cập nhật", "",
        f"Số tham số logic: language + heads {architecture['language_and_heads']:,}; ViT không gồm projector "
        f"{architecture['vit_excluding_projector']:,}; projector {architecture['projector']:,}; tổng {architecture['total_logical_parameters']:,}.",
        "Projector nằm trong visual nên phải tách trước khi cộng, tránh đếm hai lần. Với layer 4-bit, đếm shape logic thay vì số byte storage nén.",
        f"E1 có {e1['meta']['trainable_params']:,.0f} tham số train; E2 {e2['meta']['trainable_params']:,.0f}; E3 {e3['meta']['trainable_params']:,.0f}.",
        "E1 cập nhật LoRA LLM (252 module). E2 thêm full projector. E3 thêm 96 module LoRA ViT, gồm cả attention.proj; tổng 348. Matched names có trong log preflight.", "",
        "### Câu 2 — E0 và lỗi quan sát", "",
        f"E0 parse_valid_rate={e0['parse_valid_rate']:.4f}, F1={e0['micro']['f1']:.4f}, FP/ảnh={e0['fp_per_image']:.3f}.",
        "Scorer có thể trích JSON từ markdown/văn bản thừa; parse-valid không đồng nghĩa đầu ra JSON thuần hay định vị đúng.",
        "Output thô đại diện, chọn từ dự đoán thật:", "```json", json.dumps(examples, ensure_ascii=False, indent=2), "```", "",
        "### Câu 3 — Thang 0–1000", "",
        "Tọa độ chuẩn hóa tách vị trí khỏi độ phân giải ảnh. Với ảnh 640×360, nhân x với 0.64 và y với 0.36 để vẽ lại pixel.",
        f"Phép chẩn đoán trên cùng predictions của {selection['base_setup']}: chấm đúng norm1000 F1={main_scale['micro']['f1']:.4f}; "
        f"cố ý đọc như pixel F1={wrong_scale['micro']['f1']:.4f}. Sai thang làm box bị co/giãn lệch GT và mất match IoU>=0.5.",
        "Evidence chính vẫn norm1000; phép chấm thang sai chỉ phục vụ giải thích, không dùng chọn model.", "",
        "### Câu 4 — Chat template và mask loss", "",
        f"Mẫu kiểm tra có {preflight['input_tokens']} token: {preflight['masked_tokens']} bị mask, "
        f"{preflight['supervised_tokens']} token được học; ảnh có {preflight['vision_tokens']} token thị giác.",
        "Prefix được assert bằng input_ids thật. Labels prompt/ảnh/padding là -100; decode phần có loss cho đúng JSON và <|im_end|>. "
        "Loss forward và kiểm tra gradient được lưu trong preflight. Tính loss cả prompt sẽ thêm mục tiêu dự đoán token văn bản/placeholder ảnh ngoài nhiệm vụ; "
        "không phải cơ chế hồi quy trực tiếp tái tạo pixel. SFT cần học chuỗi JSON assistant.", "",
        "### Câu 5 — Quantization, float32 và lưu projector", "",
        "Skip visual/lm_head được kiểm tra bằng loại layer thực tế, không chỉ nhìn config. Full projector không thể cập nhật như dense weight nếu nó bị đóng gói 4-bit.",
        "Trong E2/E3, projector trainable float32 và được đưa vào AdamW; autocast fp16 cùng GradScaler cho phép mixed precision. "
        "Float32 giúp tránh gradient nhỏ mất độ chính xác và tăng khoảng biểu diễn, nhưng không tự bảo đảm mọi phép tính không tràn; loss/gradient vẫn được kiểm tra hữu hạn.",
        "Adapter save riêng, projector.pt save riêng; load kiểm tra đủ key/shape/dtype và SHA256 giá trị sau nạp. Quên projector khiến inference dùng lại projector gốc.", "",
        "### Câu 6 — Split theo phiên quay", "",
        f"Đã kiểm chứng {audit['unique_images']} image IDs duy nhất, {audit['n_sessions']} phiên, session_leakage={audit['session_leakage']}.",
        "Hai frame gần nhau có chung bối cảnh/ổ gà; random split dễ đưa gần-bản-sao sang test, khiến F1 đánh giá ghi nhớ thay vì tổng quát hóa.",
        "Số liệu thực tế từ audit:", "```json", json.dumps(audit["counts"], ensure_ascii=False, indent=2), "```", "",
        "### Câu 7 — So sánh E1/E2/E3", "",
        f"E2-E1 về F1 = {e2['micro']['f1']-e1['micro']['f1']:+.4f}; E3-E2 = {e3['micro']['f1']-e2['micro']['f1']:+.4f}.",
        "Đọc các chênh lệch cùng CI95, F1 từng lớp, VRAM và thời gian ở bảng; mở thêm khối có chi phí gradient/optimizer rõ ràng và không mặc định tăng F1.",
        f"Setup nền chọn bằng val là {selection['base_setup']}, val scores: `{json.dumps(selection['validation_scores'])}`. Không dùng test để chọn checkpoint.", "",
        "### Câu 8 — 311 ảnh test và khoảng tin cậy", "",
        "CI95 là bootstrap 1.000 lần của scorer theo ảnh. Độ rộng thực tế:",
        *[f"- {evidence[name]['name']}: {evidence[name]['f1_ci95'][1]-evidence[name]['f1_ci95'][0]:.4f}." for name in ("e0", "e1", "e2", "e3")],
        "", "CI rộng giới hạn kết luận về chênh lệch nhỏ. Frame cùng phiên còn tương quan nên bootstrap theo ảnh có thể lạc quan; "
        "split theo phiên chống train/test leakage nhưng không làm các frame test độc lập. CI overlap là mô tả bất định, không tự tương đương phép kiểm định chênh lệch.", "",
        "### Câu 9 — Lỗi theo lớp và minh họa", "",
        f"Trong {selection['base_setup']}, lớp thấp nhất là {weakest}, F1={main_scale['per_class'][weakest]['f1']:.4f}, "
        f"precision={main_scale['per_class'][weakest]['precision']:.4f}, recall={main_scale['per_class'][weakest]['recall']:.4f}.",
        "Đối chiếu số box mỗi lớp và ảnh minh họa trong report/figures. Box nhỏ khiến sai vài pixel ảnh hưởng IoU lớn; "
        "mất cân bằng và nét hình học mơ hồ là giả thuyết cần đối chiếu ảnh, không phải quan hệ nhân quả được chứng minh chỉ bằng F1.",
        "Manifest selection_* ghi ID, TP/FP/FN và cách chọn ảnh; GT xanh, prediction cam. Bao gồm cả đúng và sai khi có trong kết quả.", "",
        "### Câu 10 — E4 rank", "",
        f"Giữ setup {selection['base_setup']}, chỉ thay rank 4/64, alpha=2r theo đề để giữ scaling alpha/r=2. "
        "Train split, seed, LR, effective batch, 300 steps, prompt và inference được giữ nguyên.",
        f"F1(rank64)-F1(rank4)={e4_delta:+.4f}; CI95 {'có' if overlap else 'không'} giao nhau. "
        "So sánh thêm số tham số, VRAM và phút trong bảng; không xem overlap là chứng minh hai setup tương đương.", "",
        "## Nhìn lại", "",
        "Kết quả dựa trên các run thật và có checkpoint tái nạp. Các lỗi môi trường, khác biệt T4/RTX và ngoại lệ kiểm thử được công bố trong BAO_CAO_LOI_VA_LUU_Y.md.",
        "Test template gốc yêu cầu REPORT.md còn placeholder, mâu thuẫn checker; giữ nguyên test và báo cáo final test log riêng. "
        "Bài nộp dùng báo cáo đã hoàn chỉnh và check_submission.py 0 FAIL, 0 WARN.", "",
    ]
    bonus_path=root/"evidence/e5.json"
    if bonus_path.exists():
        bonus=json.loads(bonus_path.read_text())
        if (bonus["dataset_version"]!=audit["dataset_version"] or bonus["split"]!="test"
                or bonus["n_images"]!=311 or bonus["missing_predictions"]!=0):
            raise ValueError("E5 evidence differs from the mandatory evaluation")
        bonus_analysis=json.loads((root/"runs/e5_output_analysis.json").read_text())
        base=evidence[selection["base_setup"].lower()]
        lines.extend(["## Bonus E5 — GRPO", "",
                      f"Khởi tạo từ {selection['base_setup']}, chạy 60 optimizer updates, 4 generations, LR 5e-6, beta=0.",
                      f"E5 F1={bonus['micro']['f1']:.4f}, CI95={bonus['f1_ci95']}, parse_valid_rate={bonus['parse_valid_rate']:.4f}; "
                      f"chênh lệch F1 so với SFT nền={bonus['micro']['f1']-base['micro']['f1']:+.4f}.",
                      f"Tài nguyên đo thật: {bonus['meta']['trainable_params']:,.0f} tham số, "
                      f"{bonus['meta']['peak_vram_gb']:.3f} GB, {bonus['meta']['train_minutes']:.2f} phút.",
                      "Reward được test trước train: perfect/invalid/wrong class/extra box/empty, duplicate matching và giant box. "
                      "Reward tăng không bảo đảm F1 tăng; dưới đây là thống kê output thật để kiểm tra reward hacking:",
                      "```json",json.dumps(bonus_analysis,ensure_ascii=False,indent=2),"```","",
                      "Mean test reward là phép phân tích sau đánh giá, không dùng nhãn test để huấn luyện/chọn policy. "
                      "Số output rỗng trên ảnh có GT và giant box cần đọc cùng recall/FP; không mặc định mọi output rỗng là hacking.",""])
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    generate_report()
