"""Reproducible 300-update SFT with full optimizer and projector checkpoints."""
import argparse
import json
import os
import random
import time
from pathlib import Path

from lab.data import GroundingCollator, GroundingDataset
from lab.modeling import apply_setup, architecture_audit, load_base, load_processor, save_weights
from lab.provenance import code_manifest, environment
from scripts.labcommon import read_jsonl


def train(args):
    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup, set_seed

    set_seed(args.seed)
    source_hashes = code_manifest()
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.revision)
    model = load_base(args.revision)
    architecture = architecture_audit(model)
    model, stats = apply_setup(model, args.setup, args.rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    records = read_jsonl(Path(args.data_root) / "splits/train.jsonl")
    dataset = GroundingDataset(records, processor, args.data_root)
    # Order is explicitly checkpointed; no nondeterministic loader worker state.
    order_generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(dataset), generator=order_generator).tolist()
    data_cursor = 0
    collator = GroundingCollator(processor.tokenizer.pad_token_id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=round(args.max_steps*.05), num_training_steps=args.max_steps)
    scaler = torch.amp.GradScaler("cuda")
    step, previous_seconds, previous_peak = 0, 0.0, 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["code_manifest"] != source_hashes:
            raise ValueError("Resume code/config fingerprints differ from the saved run")
        for key in ("revision", "setup", "rank", "seed", "lr", "batch_size", "grad_accum", "max_steps"):
            if state["config"][key] != getattr(args, key):
                raise ValueError(f"Resume config differs for {key}")
        expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(expected) != set(state["trainable_state"]):
            raise RuntimeError("Resume trainable parameter keys differ")
        with torch.no_grad():
            for name, value in state["trainable_state"].items():
                expected[name].copy_(value.to(expected[name].device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        order_generator.set_state(state["order_rng"])
        order, data_cursor, step = state["order"], state["data_cursor"], state["step"]
        previous_seconds, previous_peak = state["train_seconds"], state["peak_vram_gb"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metadata = vars(args).copy()
    metadata.update({k: v for k, v in stats.items() if k != "matched_module_names"})
    metadata.update({"architecture": architecture, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "environment": environment(), "code_manifest": source_hashes,
                     "vram_units": "decimal GB (bytes / 1e9)", "training_steps": step,
                     "dataset_version": json.loads((Path(args.data_root)/"splits/meta.json").read_text())["dataset_version"]})
    (directory / "target_modules.json").write_text(json.dumps(stats["matched_module_names"], indent=2))

    def checkpoint(final=False):
        torch.cuda.synchronize()
        elapsed = previous_seconds + time.perf_counter()-start
        peak = max(previous_peak, torch.cuda.max_memory_allocated()/1e9)
        destination = directory / ("final" if final else f"step_{step:04d}")
        save_weights(model, destination, args.setup)
        state = {"step": step, "config": vars(args).copy(), "code_manifest": source_hashes,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "train_seconds": elapsed, "peak_vram_gb": peak,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "order_rng": order_generator.get_state(), "order": order, "data_cursor": data_cursor,
                 "trainable_state": {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}}
        torch.save(state, destination / "resume.pt")
        metadata.update({"training_steps": step, "train_minutes": elapsed/60,
                         "peak_vram_gb": peak, "checkpoint": str(destination), "complete": final})
        (directory / "run.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"event": "checkpoint", "step": step, "path": str(destination)}), flush=True)

    optimizer.zero_grad(set_to_none=True)
    with (directory / "train.jsonl").open("a" if args.resume else "w") as log:
        while step < args.max_steps:
            losses = []
            for _ in range(args.grad_accum):
                indices = []
                while len(indices) < args.batch_size:
                    if data_cursor == len(order):
                        order = torch.randperm(len(dataset), generator=order_generator).tolist()
                        data_cursor = 0
                    indices.append(order[data_cursor])
                    data_cursor += 1
                batch = collator([dataset[index] for index in indices])
                batch = {key: value.to("cuda:0") for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    raw_loss = model(**batch).loss
                    loss = raw_loss/args.grad_accum
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"Nonfinite loss before update {step+1}")
                losses.append(raw_loss.detach().item())
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient norm before update {step+1}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < old_scale:
                raise FloatingPointError("GradScaler skipped update; investigate before resuming")
            scheduler.step()
            step += 1
            row = {"step": step, "loss": sum(losses)/len(losses),
                   "lr": scheduler.get_last_lr()[0], "gradient_norm": norm.item(),
                   "elapsed_seconds": previous_seconds+time.perf_counter()-start}
            log.write(json.dumps(row)+"\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(json.dumps(row), flush=True)
            if step % 100 == 0 and step != args.max_steps:
                checkpoint()
        checkpoint(final=True)
    return metadata


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=["E1", "E2", "E3"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--resume")
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
