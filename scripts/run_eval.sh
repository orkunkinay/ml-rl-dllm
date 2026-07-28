#!/usr/bin/env bash
#SBATCH --job-name=llada_eval
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_eval_%j.out
#SBATCH --error=logs/llada_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200_3g.71gb:1
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
PROJECT_DIR="${PROJECT_DIR:-}"

if [[ -z "$PROJECT_DIR" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    else
        PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
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

if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ ! -f "$HOME/.hf_token" ]]; then
        echo "HF_TOKEN is unset and $HOME/.hf_token does not exist." >&2
        exit 1
    fi
    export HF_TOKEN="$(<"$HOME/.hf_token")"
fi
if [[ -f "$HOME/.wandb_api_key" && -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY="$(<"$HOME/.wandb_api_key")"
fi

if command -v module >/dev/null 2>&1; then
    module add cuda
fi

echo "===== SOURCE INFO ====="
echo "project_dir: $PROJECT_DIR"
echo "venv_dir: $VENV_DIR"
echo "python: $(command -v python)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
python - <<'PY'
from pathlib import Path

import eval.sampler as sampler_module
from eval.sampler import CustomDistributedSampler

sampler = CustomDistributedSampler(range(1), shuffle=False)
if sampler.num_replicas != 1 or sampler.rank != 0:
    raise RuntimeError(
        "Sampler preflight expected a single replica with rank 0, "
        f"got {sampler.num_replicas=} and {sampler.rank=}."
    )

print(f"sampler_source: {Path(sampler_module.__file__).resolve()}")
print("sampler_single_process: ok")
PY
echo "======================="

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

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}_${DATASETS//,/_}"
mkdir -p "$RESULTS_ROOT"
if ! mkdir "$RESULTS_DIR"; then
    echo "Refusing to overwrite existing results directory: $RESULTS_DIR" >&2
    exit 1
fi

echo "===== NODE / GPU INFO ====="
hostname
nvidia-smi
python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print("GPU memory GB:", props.total_memory / 1024**3)
PY
echo "==========================="

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
