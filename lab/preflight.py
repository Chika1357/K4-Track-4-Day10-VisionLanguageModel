"""Run real processor/quantization/gradient checks before long experiments."""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from PIL import Image

from lab.data import GroundingCollator, encode_example, objects_to_answer
from lab.modeling import apply_setup, architecture_audit, is_projector, load_base, load_processor
from scripts.labcommon import read_jsonl


def check(revision, setup="E1", data_root="data", report="report/runs"):
    import torch

    processor = load_processor(revision)
    model = load_base(revision)
    architecture = architecture_audit(model)
    model, setup_stats = apply_setup(model, setup)
    record = read_jsonl(Path(data_root)/"splits/train.jsonl")[0]
    from lab.data import PROMPT
    with Image.open(Path(data_root)/record["image_path"]) as source:
        encoded = encode_example(source.convert("RGB"), PROMPT,
                                 objects_to_answer(record["objects"]), processor)
    supervised = encoded["labels"] != -100
    decoded = processor.tokenizer.decode(encoded["input_ids"][supervised], skip_special_tokens=False)
    assert decoded.strip().endswith("<|im_end|>"), decoded
    merge = processor.image_processor.merge_size
    visual_tokens = int(encoded["image_grid_thw"].prod().item()) // merge**2
    assert visual_tokens == 220, (encoded["image_grid_thw"].tolist(), merge, visual_tokens)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    batch = GroundingCollator(processor.tokenizer.pad_token_id)([encoded])
    batch = {k: v.to("cuda:0") for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss
    assert torch.isfinite(loss), "Nonfinite preflight loss"
    loss.backward()
    groups = {"llm_lora": [], "vit_lora": [], "projector": []}
    frozen_with_grad = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_with_grad.append(name)
            continue
        if parameter.grad is None:
            raise AssertionError(f"Trainable parameter has no gradient: {name}")
        assert torch.isfinite(parameter.grad).all(), name
        group = "projector" if is_projector(name) else (
            "vit_lora" if "visual" in name else "llm_lora")
        groups[group].append(float(parameter.grad.float().norm().item()))
    assert not frozen_with_grad
    assert sum(groups["llm_lora"]) > 0
    assert bool(groups["projector"]) == (setup in {"E2", "E3"})
    assert bool(groups["vit_lora"]) == (setup == "E3")
    result = {"setup": setup, "model_revision": revision, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "image_grid_thw": encoded["image_grid_thw"].tolist(), "vision_tokens": visual_tokens,
              "input_tokens": len(encoded["input_ids"]), "masked_tokens": int((~supervised).sum()),
              "supervised_tokens": int(supervised.sum()), "decoded_answer_and_eos": decoded,
              "forward_loss": loss.item(), "frozen_parameters_with_grad": frozen_with_grad,
              "gradient_norm_sums": {k: sum(v) for k,v in groups.items()},
              "architecture": architecture, "setup_stats": setup_stats,
              "environment": {"python": platform.python_version(), "torch_cuda_runtime": torch.version.cuda,
                              "gpu_name": torch.cuda.get_device_name(0)}}
    for package in ("torch", "torchvision", "transformers", "peft", "trl", "accelerate", "bitsandbytes"):
        result["environment"][package] = importlib.metadata.version(package)
    destination = Path(report)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/f"preflight_{setup.lower()}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k,v in result.items() if k != "setup_stats"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--setup", choices=["E1", "E2", "E3"], default="E1")
    args = parser.parse_args()
    check(args.revision, args.setup)
