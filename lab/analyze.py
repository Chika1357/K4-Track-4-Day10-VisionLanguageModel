"""Post-evaluation diagnostics; never modify prediction files or the scorer."""
import json
from collections import Counter
from pathlib import Path

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, main as score_main, parse_output


def analyze_e0(data_root="data", report_root="report"):
    report = Path(report_root)
    records = read_jsonl(Path(data_root)/"splits/test.jsonl")
    predictions = {row["image_id"]: row["raw"] for row in read_jsonl("work/e0.jsonl")}
    if set(predictions) != {row["image_id"] for row in records}:
        raise ValueError("E0 predictions do not match test IDs")
    counts = Counter({"invalid_json": 0, "extraneous_wrapper": 0,
                      "false_positive": 0, "missed_ground_truth": 0})
    candidates = {}
    for record in records:
        raw = predictions[record["image_id"]]
        parsed = parse_output(raw)
        failures = []
        if parsed is None:
            failures.append("invalid_json")
        else:
            try:
                json.loads(raw.strip())
            except (ValueError, TypeError):
                failures.append("extraneous_wrapper")
        per, _ = _match(_gt_boxes(record), parsed or [])
        if sum(value[1] for value in per.values()):
            failures.append("false_positive")
        if sum(value[2] for value in per.values()):
            failures.append("missed_ground_truth")
        counts.update(failures)
        for category in failures:
            candidates.setdefault(category, {"image_id": record["image_id"], "raw": raw,
                                               "observed_categories": failures})
    examples = list({row["image_id"]: row for row in candidates.values()}.values())
    for record in records:
        if len(examples) >= 5:
            break
        if record["image_id"] not in {row["image_id"] for row in examples}:
            examples.append({"image_id": record["image_id"], "raw": predictions[record["image_id"]]})
    result = {"n_images": len(records), "error_images_per_category": dict(counts),
              "most_frequent_observed_categories": [name for name, count in counts.most_common() if count > 0][:2],
              "selection_policy": "first example per observed category, then test order",
              "examples": examples}
    (report/"runs").mkdir(parents=True, exist_ok=True)
    (report/"runs/e0_examples.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def diagnose_coordinates(base_setup, report_root="report"):
    score_main(["--pred", f"work/{base_setup.lower()}.jsonl", "--name", "coordinate_scale_diagnostic",
                "--split", "test", "--coord", "pixel",
                "--out", str(Path(report_root)/"runs/coordinate_diagnostic.json")])


if __name__ == "__main__":
    analyze_e0()
    selection = json.loads(Path("report/runs/e4_selection.json").read_text())
    diagnose_coordinates(selection["base_setup"])
