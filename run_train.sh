#!/bin/bash
#SBATCH --job-name=llada_train
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_train_%j.out
#SBATCH --error=logs/llada_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd ~/msc_project/ml-rl-dllm
source .venv/bin/activate
export PYTHONNOUSERSITE=1

export HF_TOKEN=$(cat ~/.hf_token)
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

module add cuda

python -m train.train \
    --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --resume auto \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
