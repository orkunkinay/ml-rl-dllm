#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -euo pipefail

# Evaluation configuration. Edit DATASETS to select the benchmarks to run.
# Supported: gsm8k, mbpp, xsum
DATASETS="gsm8k"
MAX_NEW_TOKENS=256
CONFIG_PATH="configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_mixture.yaml"
RUN_PATH=""
CHECKPOINTS="last"
TEMPERATURES="1.0"
SAMPLING_MODE="bernoulli-argmax"
SEEDS="42,43,44"
RESULTS_ROOT="eval_results"
VENV_DIR="${VENV_DIR:-$HOME/msc_project/ml-rl-dllm/.venv}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

validate_datasets() {
    local dataset
    local normalized
    local -a selected
    IFS=',' read -ra selected <<< "$DATASETS"
    if [[ ${#selected[@]} -eq 0 ]]; then
        echo "DATASETS must name at least one dataset." >&2
        exit 2
    fi
    for dataset in "${selected[@]}"; do
        normalized="${dataset//[[:space:]]/}"
        case "$normalized" in
            gsm8k|mbpp|xsum) ;;
            *)
                echo "Unknown dataset '$dataset'. Supported: gsm8k, mbpp, xsum." >&2
                exit 2
                ;;
        esac
    done
}

validate_datasets

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found at $VENV_DIR." >&2
    echo "Set VENV_DIR to the environment containing the evaluation dependencies." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

if [[ -z "$RUN_PATH" ]]; then
    RUN_PATH="$(
        python - <<'PY'
from pathlib import Path

for root in (Path("runs"), Path("outputs")):
    if not root.exists():
        continue
    checkpoints = list(root.glob("**/checkpoint-*/model.safetensors"))
    if checkpoints:
        print(max(checkpoints, key=lambda path: path.stat().st_mtime).parent.parent)
        break
PY
    )"
fi

if [[ -z "$RUN_PATH" || ! -d "$RUN_PATH" ]]; then
    echo "Run directory not found: ${RUN_PATH:-<none>}." >&2
    echo "Set RUN_PATH near the top of this script to a directory containing checkpoint-* directories." >&2
    exit 1
fi

RESULTS_DIR="${RESULTS_ROOT}/$(date +%Y%m%d_%H%M%S)_${DATASETS//,/_}"
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Refusing to overwrite existing results directory: $RESULTS_DIR" >&2
    exit 1
fi

echo "===== EVAL CONFIG ====="
echo "run_path: $RUN_PATH"
echo "datasets: $DATASETS"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "results_dir: $RESULTS_DIR"
echo "======================="

python -m eval.pipeline "$RUN_PATH" "$CONFIG_PATH" \
    --checkpoints "$CHECKPOINTS" \
    --datasets "$DATASETS" \
    --temperatures "$TEMPERATURES" \
    --sampling_mode "$SAMPLING_MODE" \
    --seeds "$SEEDS" \
    --gen_length "$MAX_NEW_TOKENS" \
    --save_path "$RESULTS_DIR" \
    --resume auto \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
