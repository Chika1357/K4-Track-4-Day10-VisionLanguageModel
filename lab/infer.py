"""Write untouched generations and score them with the assignment scorer."""
import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from lab.data import user_messages
from lab.modeling import load_base, load_processor, load_trained
from scripts.labcommon import read_jsonl


def generate_records(model, processor, records, data_root, output, batch_size=4,
                     max_new_tokens=256, resume=False):
    import torch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed = {r["image_id"] for r in records}
    complete = set()
    if resume and output.exists():
        for row in read_jsonl(output):
            if row["image_id"] in complete or row["image_id"] not in allowed:
                raise ValueError("Invalid IDs in inference resume file")
            complete.add(row["image_id"])
    pending = [r for r in records if r["image_id"] not in complete]
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with output.open("a" if resume else "w", encoding="utf-8") as stream:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            images = []
            for row in batch:
                with Image.open(Path(data_root) / row["image_path"]) as source:
                    images.append(source.convert("RGB"))
            messages = [user_messages(image) for image in images]
            encoded = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True)
            encoded = encoded.to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            answers = processor.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, answers):
                stream.write(json.dumps({"image_id": row["image_id"], "raw": raw},
                                        ensure_ascii=False) + "\n")
            stream.flush()
            torch.cuda.synchronize()
            done = len(complete) + offset + len(batch)
            print(json.dumps({"event": "inference", "done": done, "total": len(records),
                              "elapsed_seconds": round(time.perf_counter()-start, 2)}), flush=True)
    torch.cuda.synchronize()
    result = {"images_generated_this_invocation": len(pending),
              "inference_minutes": (time.perf_counter()-start)/60,
              "inference_peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
              "batch_size": batch_size, "max_new_tokens": max_new_tokens, "do_sample": False}
    actual = read_jsonl(output)
    if len(actual) != len(records) or {r["image_id"] for r in actual} != allowed:
        raise RuntimeError("Inference did not produce exactly one prediction per image")
    return result


def run_inference(args):
    from scripts.score import main as score_main

    processor = load_processor(args.revision)
    model = load_trained(args.revision, args.checkpoint) if args.checkpoint else load_base(args.revision)
    records = read_jsonl(Path(args.data_root) / "splits" / f"{args.split}.jsonl")
    result = generate_records(model, processor, records, args.data_root, args.output,
                              args.batch_size, args.max_new_tokens, args.resume)
    meta = {"model_revision": args.revision, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.checkpoint:
        run = json.loads((Path(args.checkpoint).parent / "run.json").read_text())
        meta = {key: run[key] for key in ("trainable_params", "peak_vram_gb", "train_minutes")}
        meta.update({"physical_gpu": run["physical_gpu"], "model_revision": args.revision,
                     "training_steps": run["training_steps"], "rank": run["rank"],
                     "base_setup": run["setup"]})
    meta.update(result)
    command = ["--pred", args.output, "--splits-dir", str(Path(args.data_root)/"splits"),
               "--split", args.split, "--name", args.name, "--out", args.evidence]
    for key, value in meta.items():
        command += ["--meta", f"{key}={value}"]
    score_main(command)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    return ap


if __name__ == "__main__":
    run_inference(parser().parse_args())
