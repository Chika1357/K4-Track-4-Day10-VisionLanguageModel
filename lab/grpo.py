"""Optional 60-step GRPO, using the verified TRL 1.9.2 VLM interface."""
import json
import os
from pathlib import Path
import time

from lab.data import PROMPT, yolo_to_box
from lab.modeling import is_projector, load_processor, load_trained, save_weights
from lab.provenance import code_manifest, environment
from lab.reward import compute_reward
from scripts.labcommon import read_jsonl


def reward_function(completions, gt_boxes, **kwargs):
    answers=[]
    for completion in completions:
        if isinstance(completion,str):
            answers.append(completion)
        else:
            content=completion[0]["content"]
            if isinstance(content,list):
                content="".join(part.get("text","") for part in content if part.get("type")=="text")
            answers.append(content)
    if len(answers)!=len(gt_boxes):
        raise ValueError("Rewards and expanded ground truths are misaligned")
    return [compute_reward(answer,truth) for answer,truth in zip(answers,gt_boxes)]


def train_grpo():
    import torch
    from datasets import Dataset, Image as DatasetImage
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    selection=json.loads(Path("report/runs/e4_selection.json").read_text())
    setup=selection["base_setup"]
    baseline=json.loads(Path(f"work/{setup.lower()}/run.json").read_text())
    rev=baseline["revision"]
    model=load_trained(rev,Path(f"work/{setup.lower()}/final"),trainable=True)
    processor=load_processor(rev)
    rows=[]
    for record in read_jsonl("data/splits/train.jsonl"):
        truth=[]
        for obj in record["objects"]:
            box=yolo_to_box(obj["label"],obj["yolo"])
            truth.append({"label":box["label"],"bbox":box["bbox_2d"]})
        rows.append({"prompt":[{"role":"user","content":[{"type":"image"},
                      {"type":"text","text":PROMPT}]}],
                     "image":str((Path("data")/record["image_path"]).resolve()),
                     "gt_boxes":truth})
    dataset=Dataset.from_list(rows).cast_column("image",DatasetImage())
    directory=Path("work/e5")
    directory.mkdir(parents=True,exist_ok=True)
    set_seed(42)
    config=GRPOConfig(output_dir=str(directory/"trainer"),max_steps=60,learning_rate=5e-6,
                      per_device_train_batch_size=4,gradient_accumulation_steps=1,
                      num_generations=4,generation_batch_size=4,max_completion_length=128,
                      beta=0,use_vllm=False,fp16=True,bf16=False,report_to="none",
                      logging_steps=1,save_steps=20,save_total_limit=3,
                      gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant":False},
                      remove_unused_columns=False,seed=42,log_completions=False)
    class ProjectorCallback(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            save_weights(kwargs["model"],directory/f"step_{state.global_step:04d}",setup)
    trainer=GRPOTrainer(model=model,args=config,train_dataset=dataset,
                        reward_funcs=reward_function,processing_class=processor,
                        callbacks=[ProjectorCallback()])
    # Preserve the SFT policy even if trainer initialization adjusts PEFT flags.
    for name,parameter in trainer.model.named_parameters():
        parameter.requires_grad=("lora_" in name or (is_projector(name) and setup in {"E2","E3"}))
        if is_projector(name) and parameter.requires_grad:
            parameter.data=parameter.data.to(torch.float32)
    trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable!=baseline["trainable_params"]:
        raise RuntimeError("GRPO trainable parameter policy differs from SFT checkpoint")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start=time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed=(time.perf_counter()-start)/60
    if trainer.state.global_step!=60:
        raise RuntimeError("GRPO did not complete 60 optimizer steps")
    manifest=save_weights(trainer.model,directory/"final",setup)
    metadata={"setup":setup,"name":"E5","rank":baseline["rank"],"revision":rev,
              "training_steps":trainer.state.global_step,"trainable_params":trainable,
              "peak_vram_gb":torch.cuda.max_memory_allocated()/1e9,"train_minutes":elapsed,
              "physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"dataset_version":baseline["dataset_version"],
              "environment":environment(),"code_manifest":code_manifest(),"complete":True,
              "grpo":{"num_generations":4,"max_completion_length":128,"learning_rate":5e-6,"beta":0},
              "weights_manifest":manifest}
    (directory/"run.json").write_text(json.dumps(metadata,indent=2))
    Path("report/runs/e5_train.json").write_text(json.dumps(metadata,indent=2))
    Path("report/runs/e5_grpo_history.json").write_text(json.dumps(trainer.state.log_history,indent=2))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1")
    train_grpo()
