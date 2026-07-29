#!/usr/bin/env bash
#SBATCH --job-name=llada_eval_every_500
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_eval_every_500_%j.out
#SBATCH --error=logs/llada_eval_every_500_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200_1g.18gb:1
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Evaluate checkpoint-500, checkpoint-1000, etc., plus the final numbered
# checkpoint. This intentionally differs from run_eval.sh, which evaluates
# only the latest checkpoint with three seeds.
set -euo pipefail

DATASETS="${DATASETS:-gsm8k}"
N_TEST="${N_TEST:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_mixture.yaml}"
RUN_PATH="${RUN_PATH:-runs/trainer_output}"
TEMPERATURES="${TEMPERATURES:-1.0}"
SAMPLING_MODE="${SAMPLING_MODE:-bernoulli-argmax}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-eval_results}"
VENV_DIR="${VENV_DIR:-$HOME/msc_project/ml-rl-dllm/.venv}"
PROJECT_DIR="${PROJECT_DIR:-}"

if [[ -z "$PROJECT_DIR" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    else
        PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
fi
cd "$PROJECT_DIR"

validate_datasets() {
    local dataset normalized
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

if [[ ! -d "$RUN_PATH" ]]; then
    echo "Run directory not found: $RUN_PATH." >&2
    echo "Set RUN_PATH to a directory containing checkpoint-* directories." >&2
    exit 1
fi

CHECKPOINTS="$(python - "$RUN_PATH" <<'PY'
from pathlib import Path
import sys

run_path = Path(sys.argv[1])
checkpoint_steps = sorted(
    int(checkpoint.name.removeprefix("checkpoint-"))
    for checkpoint in run_path.glob("checkpoint-*")
    if checkpoint.is_dir()
    and checkpoint.name.removeprefix("checkpoint-").isdigit()
    and (checkpoint / "model.safetensors").is_file()
)
if not checkpoint_steps:
    raise SystemExit(f"No numbered checkpoints found in {run_path}")

selected_steps = [step for step in checkpoint_steps if step % 500 == 0]
if checkpoint_steps[-1] not in selected_steps:
    selected_steps.append(checkpoint_steps[-1])

print(",".join(map(str, selected_steps)))
PY
)"

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}_${DATASETS//,/_}_every_500"
mkdir -p "$RESULTS_ROOT"
if ! mkdir "$RESULTS_DIR"; then
    echo "Refusing to overwrite existing results directory: $RESULTS_DIR" >&2
    exit 1
fi

echo "===== SOURCE INFO ====="
echo "project_dir: $PROJECT_DIR"
echo "venv_dir: $VENV_DIR"
echo "python: $(command -v python)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "======================="

echo "===== EVAL CONFIG ====="
echo "run_path: $RUN_PATH"
echo "checkpoints: $CHECKPOINTS"
echo "datasets: $DATASETS"
echo "n_test: $N_TEST"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "seed: $SEED"
echo "results_dir: $RESULTS_DIR"
echo "======================="

python -m eval.pipeline "$RUN_PATH" "$CONFIG_PATH" \
    --checkpoints "$CHECKPOINTS" \
    --datasets "$DATASETS" \
    --n_test "$N_TEST" \
    --temperatures "$TEMPERATURES" \
    --sampling_mode "$SAMPLING_MODE" \
    --seeds "$SEED" \
    --gen_length "$MAX_NEW_TOKENS" \
    --save_path "$RESULTS_DIR" \
    --resume auto \
    --log_memory \
    --memory_log_interval 50 \
    --reset_memory_peak_each_log
