#!/bin/bash
#SBATCH --job-name=llada_diff_sweep
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/llada_diff_sweep_%j.out
#SBATCH --error=logs/llada_diff_sweep_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G

set -euo pipefail

# Usage: sbatch run_train_sweep_diff.sh <config_path>
REPO_DIR="${REPO_DIR:-$HOME/msc_project/ml-rl-dllm-timing-test}"
CONFIG_PATH="${1:-}"
if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: no config path given. Usage: sbatch run_train_sweep_diff.sh <config_path>"
    exit 1
fi

cd "$REPO_DIR"
echo "repo_dir: $(pwd)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config path does not exist after cd to $(pwd): $CONFIG_PATH"
    echo "Available sweep_diff configs:"
    find configs/experiment_configs/sweep_diff -maxdepth 1 -type f -name '*.yaml' -print 2>/dev/null | sort || true
    exit 2
fi

VENV_DIR="${VENV_DIR:-$(dirname "$REPO_DIR")/ml-rl-dllm}"
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1

export HF_TOKEN=$(cat ~/.hf_token)
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

module add cuda

echo "===== NODE / GPU INFO ====="
hostname
nvidia-smi
echo "==========================="

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

echo "===== TRAINING CONFIG SUMMARY ====="
CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["CONFIG_PATH"])

with config_path.open("r") as f:
    cfg = yaml.safe_load(f)

for key in [
    "num_generations",
    "per_device_train_batch_size",
    "generation_batch_size",
    "block_length",
    "run_name",
]:
    print(f"{key}: {cfg.get(key)}")

print("config_path:", config_path)
PY
echo "==================================="

python -m train.train \
    --config "$CONFIG_PATH" \
    --overwrite \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
