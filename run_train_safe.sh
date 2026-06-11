#!/bin/bash
#SBATCH --job-name=llada_train
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --output=logs/llada_train_%j.out
#SBATCH --error=logs/llada_train_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G

cd ~/msc_project/ml-rl-dllm
source .venv/bin/activate
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
python - <<'PY'
from pathlib import Path
import yaml

config_path = Path("configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture_safe.yaml")

with config_path.open("r") as f:
    cfg = yaml.safe_load(f)

for key in [
    "num_generations",
    "per_device_train_batch_size",
    "generation_batch_size",
]:
    print(f"{key}: {cfg.get(key)}")

print("config_path:", config_path)
PY
echo "==================================="

python -m train.train \
    --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture_safe.yaml \
    --overwrite \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
