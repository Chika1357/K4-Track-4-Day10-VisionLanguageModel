"""Run one verified lab checkpoint at a time; keeps evidence and training separate."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from lab.inspect import audit_dataset
from lab.modeling import MODEL_ID


def run_command(*command, env=None):
    environment = dict(os.environ if env is None else env)
    # prepare_data.py invokes the Kaggle executable by name. A venv Python alone
    # does not activate its bin directory in PATH for that child process.
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    subprocess.run([str(part) for part in command], check=True, env=environment)


def revision():
    destination = Path("report/runs/model_revision.json")
    if destination.exists():
        return json.loads(destination.read_text())["revision"]
    from huggingface_hub import HfApi
    info = HfApi().model_info(MODEL_ID)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"model_id": MODEL_ID, "revision": info.sha}, indent=2))
    return info.sha


def choose_e4_setup():
    destination = Path("report/runs/e4_selection.json")
    if destination.exists():
        return json.loads(destination.read_text())["base_setup"]
    values = []
    for index in (1, 2, 3):
        name = f"E{index}"
        evidence = json.loads(Path(f"report/runs/{name.lower()}_val.json").read_text())
        assert evidence["split"] == "val" and evidence["n_images"] == 276
        values.append((evidence["micro"]["f1"], -index, name))
    selected = max(values)[2]
    destination.write_text(json.dumps({"base_setup": selected, "selection_split": "val",
                                       "selection_metric": "micro.f1", "tie_break": "simpler setup",
                                       "validation_scores": {name: score for score, _, name in values}}, indent=2))
    return selected


def infer(name, rev, checkpoint=None, split="test"):
    output = f"work/{name.lower()}_{split}.jsonl" if split == "val" else f"work/{name.lower()}.jsonl"
    evidence = (f"report/runs/{name.lower()}_val.json" if split == "val"
                else f"report/evidence/{name.lower()}.json")
    command = [sys.executable, "-m", "lab.infer", "--name", name,
               "--revision", rev, "--split", split, "--output", output,
               "--evidence", evidence, "--resume"]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    run_command(*command)


def train_and_score(name, setup, rank, rev, learning_rate=2e-4, batch_size=2, grad_accum=4):
    directory = Path("work") / name.lower()
    final = directory / "final"
    training = json.loads((directory / "run.json").read_text()) if (directory / "run.json").exists() else {}
    if not training.get("complete"):
        command = [sys.executable, "-m", "lab.train", "--setup", setup,
                   "--run-dir", str(directory), "--revision", rev, "--rank", str(rank),
                   "--lr", str(learning_rate), "--batch-size", str(batch_size),
                   "--grad-accum", str(grad_accum)]
        resumes = sorted(directory.glob("step_*/resume.pt"))
        if resumes:
            command += ["--resume", str(resumes[-1])]
        run_command(*command)
    training = json.loads((directory / "run.json").read_text())
    assert training["complete"] and training["training_steps"] == 300
    assert training["setup"] == setup and training["rank"] == rank
    assert training["revision"] == rev
    assert training["lr"] == learning_rate and training["batch_size"] == batch_size
    assert training["grad_accum"] == grad_accum and training["seed"] == 42
    infer(name, rev, final, "val")
    infer(name, rev, final, "test")
    destination = Path("report/runs") / f"{name.lower()}_train.json"
    destination.write_text(json.dumps(training, indent=2))


def execute(phase):
    Path("report/runs").mkdir(parents=True, exist_ok=True)
    if phase == "prepare":
        run_command(sys.executable, "-m", "pytest", "-q")
        run_command(sys.executable, "scripts/prepare_data.py", "--kaggle")
        audit_dataset()
        from scripts.prepare_data import _find_root
        environment = os.environ.copy()
        environment["D10_RAW"] = str(_find_root("work/raw").resolve())
        run_command(sys.executable, "-m", "pytest", "-q", env=environment)
        run_command(sys.executable, "-m", "pytest", "-q", "student_tests")
    elif phase == "e0":
        infer("E0", revision())
    elif phase == "encoding":
        rev = revision()
        for setup in ("E1", "E2", "E3"):
            run_command(sys.executable, "-m", "lab.preflight", "--revision", rev, "--setup", setup)
    elif phase in {"e1", "e2", "e3"}:
        train_and_score(phase.upper(), phase.upper(), 16, revision())
    elif phase in {"e4_rank4", "e4_rank64"}:
        setup = choose_e4_setup()
        # Every other training parameter must equal the selected SFT setup.
        baseline = json.loads(Path(f"work/{setup.lower()}/run.json").read_text())
        rank = 4 if phase == "e4_rank4" else 64
        train_and_score(f"E4_rank_{rank}", setup, rank, revision(),
                        baseline["lr"], baseline["batch_size"], baseline["grad_accum"])
    else:
        raise ValueError(phase)
    marker = {"phase": phase, "status": "verified", "needs_git_commit": True}
    Path(f"report/runs/checkpoint_{phase}.json").write_text(json.dumps(marker, indent=2))
    print(json.dumps({"event": "checkpoint_ready_for_commit", **marker}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=[
        "prepare", "e0", "encoding", "e1", "e2", "e3", "e4_rank4", "e4_rank64"])
    args = parser.parse_args()
    if args.phase != "prepare" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This lab was allocated physical GPU 1; set CUDA_VISIBLE_DEVICES=1")
    execute(args.phase)
