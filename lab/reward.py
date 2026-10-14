"""Verifiable class-aware IoU reward; matches the assignment definition."""
from scripts.score import iou_xyxy, parse_output


def compute_reward(completion, gt_boxes):
    """gt_boxes are [{label, bbox}], in the same norm1000 space as predictions."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    for label in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        truth = [g["bbox"] for g in gt_boxes if g["label"] == label]
        guessed = [p["bbox"] for p in pred if p["label"] == label]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(truth)
                        for j, b in enumerate(guessed)), reverse=True)
        used_truth, used_pred = set(), set()
        for overlap, i, j in pairs:
            if overlap < 0.5:
                break
            if i in used_truth or j in used_pred:
                continue
            used_truth.add(i)
            used_pred.add(j)
            total_iou += overlap
    return 0.2 + 0.8*total_iou/max(len(gt_boxes), len(pred), 1)
