#!/bin/bash
#SBATCH --job-name=llada_eval_trainer_output_hf
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_eval_trainer_output_hf_%j.out
#SBATCH --error=logs/llada_eval_trainer_output_hf_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200_3g.71gb:1

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    else
        PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
fi
cd "$PROJECT_DIR"

if [[ ! -f pyproject.toml || ! -f eval/pipeline.py ]]; then
    echo "Project checkout not found at $PROJECT_DIR." >&2
    echo "Submit from the repository root or set PROJECT_DIR explicitly." >&2
    exit 1
fi

VENV_DIR="${VENV_DIR:-$HOME/msc_project/ml-rl-dllm/.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found at $VENV_DIR." >&2
    echo "Set VENV_DIR to the environment containing the eval dependencies." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || true)}"
if [[ -f ~/.wandb_api_key ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_api_key)}"
fi

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

if command -v module >/dev/null 2>&1; then
    module add cuda
fi

echo "===== SOURCE INFO ====="
echo "project_dir: $PROJECT_DIR"
echo "venv_dir: $VENV_DIR"
echo "python: $(command -v python)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "======================="

# Checkpoints for this run live on the Hugging Face Hub, not on local disk:
# https://huggingface.co/orkunkinay/ml-rl-dllm/tree/main/checkpoints/trainer_output
HF_REPO="${HF_REPO:-orkunkinay/ml-rl-dllm}"
HF_SUBPATH="${HF_SUBPATH:-checkpoints/trainer_output}"
RUN_PATH="${RUN_PATH:-hf://${HF_REPO}/${HF_SUBPATH}}"

CONFIG_PATH="${CONFIG_PATH:-configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_trainer_output.yaml}"
SAVE_PATH="${SAVE_PATH:-eval_results/trainer_output_hf}"
# "all" evaluates every checkpoint found under $HF_SUBPATH; can be overridden
# to a comma-separated list, e.g. CHECKPOINTS=500,5000,11500 or CHECKPOINTS=last
CHECKPOINTS="${CHECKPOINTS:-all}"
DATASETS="${DATASETS:-gsm8k}"
TEMPERATURES="${TEMPERATURES:-1.0}"
SAMPLING_MODE="${SAMPLING_MODE:-bernoulli-argmax}"
# Single seed, as requested -- override with SEEDS=42,43,44 for multi-seed runs.
SEEDS="${SEEDS:-42}"

echo "===== NODE / GPU INFO ====="
hostname
nvidia-smi
echo "==========================="

echo "===== EVAL CONFIG ====="
echo "run_path: $RUN_PATH"
echo "config_path: $CONFIG_PATH"
echo "save_path: $SAVE_PATH"
echo "checkpoints: $CHECKPOINTS"
echo "datasets: $DATASETS"
echo "temperatures: $TEMPERATURES"
echo "sampling_mode: $SAMPLING_MODE"
echo "seeds: $SEEDS"
echo "======================="

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("GPU memory GB:", props.total_memory / 1024**3)
else:
    raise RuntimeError("CUDA unavailable")
PY

python -m eval.pipeline "$RUN_PATH" "$CONFIG_PATH" \
    --checkpoints "$CHECKPOINTS" \
    --datasets "$DATASETS" \
    --temperatures "$TEMPERATURES" \
    --sampling_mode "$SAMPLING_MODE" \
    --seeds "$SEEDS" \
    --save_path "$SAVE_PATH" \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log

echo "===== RESULTS ====="
echo "Per-run generations: $SAVE_PATH/"
echo "Aggregated CSV:      $SAVE_PATH/detailed_results.csv"
echo "Summary CSV:         $SAVE_PATH/summary_statistics.csv"
echo "==================="
