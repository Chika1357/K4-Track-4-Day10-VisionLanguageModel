#!/usr/bin/env bash
# Run a single checkpoint in the allocated server account.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_gpu.sh {prepare|e0|encoding|e1|e2|e3|e4_rank4|e4_rank64}"
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Create the project's .venv and install requirements-colab.txt first."
    exit 2
fi
mkdir -p work/logs
# Credentials are supplied in the environment and are not echoed or persisted.
.venv/bin/python -m lab.run --phase "$1" 2>&1 | tee "work/logs/$1.log"
