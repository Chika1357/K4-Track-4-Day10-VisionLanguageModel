"""Select representative successes/errors and annotate actual test predictions."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, parse_output


def font(size=14):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def visualize(prediction_file, run_name, data_root="data", destination="report/figures"):
    entries = read_jsonl(Path(data_root)/"splits/test.jsonl")
    predictions = {r["image_id"]: r["raw"] for r in read_jsonl(prediction_file)}
    options = {"success": [], "false_positive": [], "missed": [], "invalid_json": []}
    diagnostics = {}
    for row in entries:
        raw = predictions[row["image_id"]]
        parsed = parse_output(raw)
        per, _ = _match(_gt_boxes(row), parsed or [])
        tp, fp, fn = [sum(v[index] for v in per.values()) for index in range(3)]
        diagnostics[row["image_id"]] = {"tp": tp, "fp": fp, "fn": fn,
                                         "parse_valid": parsed is not None}
        if parsed is None:
            options["invalid_json"].append(row)
        if tp > 0 and fp == 0 and fn == 0:
            options["success"].append(row)
        if fp:
            options["false_positive"].append(row)
        if fn:
            options["missed"].append(row)
    selected, seen = [], set()
    for category in options:
        for row in options[category]:
            if row["image_id"] not in seen:
                selected.append((category, row))
                seen.add(row["image_id"])
                break
    for row in entries:
        if len(selected) >= 6:
            break
        if row["image_id"] not in seen:
            selected.append(("additional_representative", row))
            seen.add(row["image_id"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (category, row) in enumerate(selected):
        with Image.open(Path(data_root)/row["image_path"]) as source:
            image = source.convert("RGB")
        canvas = Image.new("RGB", (640, 440), "#15202b")
        canvas.paste(image, (0, 80))
        draw = ImageDraw.Draw(canvas)
        counts = diagnostics[row["image_id"]]
        draw.text((10, 7), f"{run_name}: {category} | TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}",
                  font=font(15), fill="white")
        draw.text((10, 30), row["image_id"], font=font(12), fill="white")
        draw.text((10, 53), "GREEN = ground truth; ORANGE = model prediction (norm1000 -> pixels)",
                  font=font(12), fill="white")
        for color, boxes in (("#3dff84", _gt_boxes(row)),
                             ("#ff9b3d", parse_output(predictions[row["image_id"]]) or [])):
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                rect = [x1*.64, y1*.36+80, x2*.64, y2*.36+80]
                draw.rectangle(rect, outline=color, width=2)
                draw.text((rect[0], max(80, rect[1]-14)), box["label"], font=font(12), fill=color)
        filename = f"prediction_{run_name.lower()}_{index+1}.png"
        canvas.save(destination/filename)
        manifest.append({"filename": filename, "image_id": row["image_id"],
                         "category": category, **counts})
    (destination/f"selection_{run_name.lower()}.json").write_text(json.dumps({
        "selection_policy": "first unique example per error category, then remaining test order",
        "examples": manifest, "category_totals": {k: len(v) for k,v in options.items()}}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    visualize(args.pred, args.name)
