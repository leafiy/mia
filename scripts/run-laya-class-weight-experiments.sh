#!/usr/bin/env bash
# Train one or more --class-weights variants on one nvidia-smi GPU chosen by index (default 1, the 3080),
# so a busy GPU 0 is left alone. Usage: ./run-laya-class-weight-experiments.sh balanced none sqrt
# Environment:
#   LAYA_GPU_INDEX      nvidia-smi index of the card to use (default 1)
#   LAYA_DATA_DIR       dataset directory (default laya-sentiment-data-cap100-natural)
#   LAYA_OUTPUT_PREFIX  output directory prefix; the variant name is appended after "-cw-"
#                       (default laya-sentiment-model-cap100-natural)
# Each variant trains from the pinned base into a NEW directory <prefix>-cw-<variant>, then evaluates
# that dataset's unseen holdout and the three Chinese CSVs into that directory. Retained model
# directories are never written.
set -euo pipefail
script_dir="$(dirname -- "$(readlink -f -- "$0")")"
cd "$script_dir/.."  # repo root: data dirs and outputs are relative to it
if [[ $# -eq 0 ]]; then
    printf '%s\n' 'Usage: run-laya-class-weight-experiments.sh <balanced|none|sqrt>...' >&2
    exit 1
fi
gpu_index="${LAYA_GPU_INDEX:-1}"
gpu_uuid="$(nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python="$script_dir/../.venv-laya/bin/python"
data_dir="${LAYA_DATA_DIR:-laya-sentiment-data-cap100-natural}"
prefix="${LAYA_OUTPUT_PREFIX:-laya-sentiment-model-cap100-natural}"
logs="laya-sentiment-experiment-logs"
mkdir -p "$logs"
for variant in "$@"; do
    output="$prefix-cw-$variant"
    log="$logs/${prefix#laya-sentiment-model-}-cw-$variant.log"
    {
        printf 'Training on GPU %s: %s\n' "$gpu_index" "$gpu_uuid"
        printf 'Data: %s\n' "$data_dir"
        nvidia-smi --id="$gpu_index" --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        "$python" "$script_dir/train-laya-sentiment.py" \
            --data-dir "$data_dir" \
            --output "$output" \
            --model convaiinnovations/laya-multilingual \
            --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 \
            --epochs 3 --batch-size 16 --grad-accum 16 --seed 42 \
            --class-weights "$variant" \
            --device cuda:0
        "$python" "$script_dir/test-laya-sentiment.py" \
            --model "$output/final" \
            --data "$data_dir/unseen-test.jsonl" \
            --manifest "$data_dir/manifest.json" \
            --device cuda:0 --batch-size 64 \
            --output "$output/unseen-test-metrics.json" > /dev/null
        "$python" "$script_dir/evaluate-laya-chinese-csv.py" \
            --model "$output/final" \
            --train-data "$data_dir/train.jsonl" \
            --device cuda:0 --batch-size 64 \
            --output "$output/chinese-csv-evaluation.json" > /dev/null
        # checkpoint-latest holds only the last epoch's weights (no optimizer state); final/ supersedes it.
        [[ -d "$output/final" ]] && rm -rf "$output/checkpoint-latest"
        printf 'DONE %s\n' "$variant"
    } 2>&1 | tee "$log"
done
