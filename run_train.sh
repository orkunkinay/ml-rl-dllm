#!/bin/bash
#SBATCH --job-name=llada_train
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_train_%j.out
#SBATCH --error=logs/llada_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd ~/msc_project/repo-name
source .venv/bin/activate

module add cuda

python -m train.train \
    --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
