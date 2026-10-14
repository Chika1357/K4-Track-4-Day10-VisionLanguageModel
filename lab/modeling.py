"""NF4 loading, scoped adapters and full-precision projector checkpoints."""
import hashlib
import json
import math
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
SKIP_QUANTIZATION_MODULES = ("visual", "model.visual", "lm_head")
LLM_TARGET = (r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
              r"|mlp\.(?:gate_proj|up_proj|down_proj))")
VIT_TARGET = (r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)"
              r"|mlp\.(?:linear_fc1|linear_fc2))")


def is_projector(name):
    return bool(re.search(r"(?:^|\.)visual\.(?:merger|deepstack_merger_list)(?:\.|$)", name))


def load_base(revision, model_id=MODEL_ID):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not revision:
        raise ValueError("A pinned model revision is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the single allocated physical GPU")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(SKIP_QUANTIZATION_MODULES))
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, revision=revision, quantization_config=quantization,
        dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")


def load_processor(revision, model_id=MODEL_ID):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad token")
    return processor


def architecture_audit(model):
    import bitsandbytes as bnb

    totals = {"projector": 0, "vit_excluding_projector": 0, "language_and_heads": 0}
    for name, parameter in model.named_parameters():
        # Params4bit stores packed values; quant_state.shape is the logical tensor.
        state = getattr(parameter, "quant_state", None)
        count = math.prod(state.shape) if state is not None else parameter.numel()
        group = "projector" if is_projector(name) else (
            "vit_excluding_projector" if ".visual." in name else "language_and_heads")
        totals[group] += count
    forbidden = [name for name, module in model.named_modules()
                 if isinstance(module, bnb.nn.Linear4bit)
                 and ("visual" in name or name.endswith("lm_head"))]
    if forbidden:
        raise RuntimeError(f"Visual/lm_head layers were quantized: {forbidden[:5]}")
    totals["total_logical_parameters"] = sum(totals.values())
    totals["visual_and_lm_head_not_4bit"] = True
    return totals


def apply_setup(model, setup, rank=16, alpha=None):
    import torch
    from peft import LoraConfig, get_peft_model

    if setup not in {"E1", "E2", "E3"}:
        raise ValueError(f"Unknown setup {setup}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()
    target = f"(?:{LLM_TARGET}|{VIT_TARGET})" if setup == "E3" else LLM_TARGET
    matched = [name for name, _ in model.named_modules() if re.fullmatch(target, name)]
    expected = 348 if setup == "E3" else 252
    if len(matched) != expected:
        raise RuntimeError(f"{setup}: expected {expected} target modules, got {len(matched)}")
    config = LoraConfig(r=rank, lora_alpha=2*rank if alpha is None else alpha,
                        lora_dropout=0.05, target_modules=target, bias="none",
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    if setup in {"E2", "E3"}:
        for name, parameter in model.named_parameters():
            if is_projector(name):
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad = True
    unexpected = [name for name, p in model.named_parameters()
                  if p.requires_grad and "lora_" not in name and not is_projector(name)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:5]}")
    stats = {"matched_modules": len(matched), "matched_module_names": matched,
             "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "trainable_projector_params": sum(p.numel() for n, p in model.named_parameters()
                                                if p.requires_grad and is_projector(n)),
             "trainable_vit_lora_params": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and "visual" in n and "lora_" in n),
             "trainable_llm_lora_params": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and "language_model" in n and "lora_" in n)}
    if setup == "E1" and stats["trainable_projector_params"] != 0:
        raise RuntimeError("E1 projector must remain frozen")
    return model, stats


def tensor_fingerprint(parameters):
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_weights(model, directory, setup):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    result = {"adapter": "adapter", "projector": None}
    if setup in {"E2", "E3"}:
        parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                      if is_projector(name)}
        if not parameters:
            raise RuntimeError("Projector parameters missing from checkpoint")
        torch.save(parameters, directory / "projector.pt")
        result["projector"] = "projector.pt"
        result["projector_sha256"] = tensor_fingerprint(parameters)
    (directory / "weights_manifest.json").write_text(json.dumps(result, indent=2))
    return result


def load_trained(revision, directory, trainable=False):
    import torch
    from peft import PeftModel

    directory = Path(directory)
    manifest = json.loads((directory / "weights_manifest.json").read_text())
    model = PeftModel.from_pretrained(load_base(revision), directory / manifest["adapter"],
                                     is_trainable=trainable)
    if manifest.get("projector"):
        weights = torch.load(directory / manifest["projector"], map_location="cpu", weights_only=True)
        expected = {n: p for n, p in model.named_parameters() if is_projector(n)}
        if set(weights) != set(expected):
            raise RuntimeError("Checkpoint projector keys differ from model")
        if tensor_fingerprint(weights) != manifest["projector_sha256"]:
            raise RuntimeError("Projector checkpoint checksum mismatch")
        with torch.no_grad():
            for name, value in weights.items():
                parameter = expected[name]
                if parameter.shape != value.shape or value.dtype != torch.float32:
                    raise RuntimeError(f"Invalid projector shape/dtype: {name}")
                parameter.data = parameter.data.to(torch.float32)
                parameter.copy_(value.to(parameter.device))
                parameter.requires_grad = trainable
        actual_hash = tensor_fingerprint({n: p for n, p in model.named_parameters() if is_projector(n)})
        if actual_hash != manifest["projector_sha256"]:
            raise RuntimeError("Reloaded projector values do not match checkpoint")
    return model
