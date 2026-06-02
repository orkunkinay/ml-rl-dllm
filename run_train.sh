#!/bin/bash
#SBATCH --job-name=llada_train
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_train_%j.out
#SBATCH --error=logs/llada_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:3g.71gb:1

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

python -m train.train \
    --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture_safe.yaml \
    --resume auto \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
