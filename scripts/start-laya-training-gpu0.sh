#!/usr/bin/env bash
# GPU 0 only. Run manually; leaves other GPUs and inference services untouched.
set -euo pipefail
script_dir="$(dirname -- "$(readlink -f -- "$0")")"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec python3 "$script_dir/train-laya-sentiment.py" --help
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is unavailable. Run this script on the GPU host or in a CUDA-enabled container.' >&2
    exit 1
fi
# Select nvidia-smi GPU 0 by UUID; CUDA device ordering cannot silently pick another card.
gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python="$script_dir/../.venv-laya/bin/python"
if [[ ! -x "$python" ]]; then
    python3 -m venv "$script_dir/../.venv-laya"
fi
if ! "$python" -c 'import importlib.metadata as m; import torch, laya; assert m.version("laya") == "0.3.20"' >/dev/null 2>&1; then
    "$python" -m pip install -r "$script_dir/requirements-laya-training.txt"
fi
printf 'Training on GPU 0: %s\n' "$gpu_uuid"
nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
exec "$python" "$script_dir/train-laya-sentiment.py" "$@" --device cuda:0
