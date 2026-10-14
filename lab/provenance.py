"""Record versions and code fingerprints without reading environment credentials."""
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def code_manifest():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/"lab").glob("*.py")) + sorted((root/"configs").glob("*.yaml"))
    return {str(path.relative_to(root)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in files}


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    return result
