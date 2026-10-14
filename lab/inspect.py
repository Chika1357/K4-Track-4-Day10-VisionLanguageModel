"""Audit the unchanged dataset preparer and inspect model encoding."""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from lab.data import yolo_to_box
from scripts.labcommon import CLASSES, read_jsonl

EXPECTED = {
    "train": {"images": 1422, "boxes": {"pothole": 1074, "crack": 1750, "manhole": 691}},
    "val": {"images": 276, "boxes": {"pothole": 66, "crack": 470, "manhole": 103}},
    "test": {"images": 311, "boxes": {"pothole": 121, "crack": 299, "manhole": 163}},
}
COLORS = {"pothole": "#ff5656", "crack": "#ffcf40", "manhole": "#40d7ff"}


def audit_dataset(data_root="data", report_root="report"):
    root, report = Path(data_root), Path(report_root)
    meta = json.loads((root / "splits/meta.json").read_text())
    assert meta["dataset_version"].startswith("d10-")
    sessions, seen, counts, splits = {}, set(), {}, {}
    for split in EXPECTED:
        records = read_jsonl(root / "splits" / f"{split}.jsonl")
        splits[split] = records
        boxes = Counter({label: 0 for label in CLASSES})
        for row in records:
            assert row["image_id"] not in seen, "Duplicate image ID"
            seen.add(row["image_id"])
            if row["session"] in sessions:
                assert sessions[row["session"]] == split, "Session leakage"
            sessions[row["session"]] = split
            with Image.open(root / row["image_path"]) as image:
                assert image.size == (row["width"], row["height"]) == (640, 360)
                image.verify()
            boxes.update(o["label"] for o in row["objects"])
        counts[split] = {"images": len(records), "boxes": dict(boxes)}
    assert counts == EXPECTED, (counts, EXPECTED)
    assert counts == meta["counts"]
    result = {"dataset_version": meta["dataset_version"], "counts": counts,
              "n_sessions": len(sessions), "session_leakage": False,
              "unique_images": len(seen), "all_images_verified": True}
    (report / "runs").mkdir(parents=True, exist_ok=True)
    (report / "runs/data_audit.json").write_text(json.dumps(result, indent=2))
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected, selected_ids = [], set()
    for label in CLASSES:
        matches = [r for r in splits["train"]
                   if any(o["label"] == label for o in r["objects"])]
        for row in matches:
            if row["image_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["image_id"])
                if sum(any(o["label"] == label for o in r["objects"])
                       for r in selected) >= 2:
                    break
    for row in splits["train"]:
        if len(selected) >= 6:
            break
        if row["image_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["image_id"])
    for i, row in enumerate(selected[:6]):
        with Image.open(root / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"]
            pixel = [box[0] * 0.64, box[1] * 0.36, box[2] * 0.64, box[3] * 0.36]
            draw.rectangle(pixel, outline=COLORS[obj["label"]], width=2)
            draw.text((pixel[0], max(0, pixel[1] - 12)), obj["label"], fill=COLORS[obj["label"]])
        image.save(figures / f"labels_{i+1}_{row['image_id']}.png")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--report-root", default="report")
    args = parser.parse_args()
    audit_dataset(args.data_root, args.report_root)
