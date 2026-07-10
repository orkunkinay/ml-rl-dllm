#!/bin/bash
#SBATCH --job-name=llada_eval_lbl256
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_eval_lbl256_%j.out
#SBATCH --error=logs/llada_eval_lbl256_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200_3g.71gb:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/msc_project/ml-rl-dllm}"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONNOUSERSITE=1

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token)}"
if [[ -f ~/.wandb_api_key ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_api_key)}"
fi

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

if command -v module >/dev/null 2>&1; then
    module add cuda
fi

CONFIG_PATH="${CONFIG_PATH:-configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_mixture.yaml}"
RUN_PATH="${RUN_PATH:-runs/paper_llada_bl256_alpha_0.0_seed_123}"
SAVE_PATH="${SAVE_PATH:-eval_results}"
CHECKPOINTS="${CHECKPOINTS:-last}"
DATASETS="${DATASETS:-gsm8k}"
TEMPERATURES="${TEMPERATURES:-1.0}"
SAMPLING_MODE="${SAMPLING_MODE:-bernoulli-argmax}"
SEEDS="${SEEDS:-42,43,44}"

echo "===== NODE / GPU INFO ====="
hostname
nvidia-smi
echo "==========================="

echo "===== EVAL CONFIG ====="
echo "config_path: $CONFIG_PATH"
echo "run_path: $RUN_PATH"
echo "save_path: $SAVE_PATH"
echo "checkpoints: $CHECKPOINTS"
echo "datasets: $DATASETS"
echo "temperatures: $TEMPERATURES"
echo "sampling_mode: $SAMPLING_MODE"
echo "seeds: $SEEDS"
echo "======================="

if [[ ! -d "$RUN_PATH" ]]; then
    echo "Run directory not found: $RUN_PATH" >&2
    echo "Set RUN_PATH=/path/to/the/training/run if run_train_lbl256.sh wrote elsewhere." >&2
    exit 1
fi

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
