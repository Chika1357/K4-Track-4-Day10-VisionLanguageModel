"""Grounding labels and answer-only multimodal training batches."""
import json
from pathlib import Path

from PIL import Image

PROMPT = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)


def yolo_to_box(label, yolo):
    xc, yc, width, height = yolo
    values = [(xc - width / 2) * 1000, (yc - height / 2) * 1000,
              (xc + width / 2) * 1000, (yc + height / 2) * 1000]
    x1, y1, x2, y2 = [int(round(max(0, min(1000, v)))) for v in values]
    # A one-unit box also remains valid at the right/bottom boundary.
    if x2 <= x1:
        x1 = min(x1, 999)
        x2 = x1 + 1
    if y2 <= y1:
        y1 = min(y1, 999)
        y2 = y1 + 1
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def user_messages(image, prompt=PROMPT):
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def encode_example(image, prompt, answer, processor):
    import torch

    user = user_messages(image, prompt)
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template(user, add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(
        user + [{"role": "assistant", "content": answer}],
        add_generation_prompt=False, **kwargs)
    n_prompt = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n_prompt], prefix["input_ids"][0]):
        raise ValueError("Assistant training sequence does not share the prompt prefix")
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if not (labels != -100).any():
        raise ValueError("No supervised answer tokens")
    result = dict(full)
    result["labels"] = labels
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        if key in result:
            result[key] = result[key][0]
    return result


class GroundingDataset:
    def __init__(self, records, processor, data_root="data"):
        self.records = records
        self.processor = processor
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.data_root / record["image_path"]) as source:
            image = source.convert("RGB")
        return encode_example(image, PROMPT, objects_to_answer(record["objects"]), self.processor)


class GroundingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        pads = {"input_ids": self.pad_token_id, "attention_mask": 0,
                "labels": -100, "mm_token_type_ids": 0}
        result = {}
        for key, pad in pads.items():
            if all(key in row for row in batch):
                result[key] = pad_sequence([row[key] for row in batch],
                                          batch_first=True, padding_value=pad)
        for key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat([row[key] for row in batch], dim=0)
        return result
