#!/bin/bash
#SBATCH --job-name=llada_eval_lbl256
#SBATCH --partition=Teaching
#SBATCH --output=logs/llada_eval_lbl256_%j.out
#SBATCH --error=logs/llada_eval_lbl256_%j.err
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

if [[ ! -f pyproject.toml || ! -f eval/eval.py ]]; then
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

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token)}"
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

CONFIG_PATH="${CONFIG_PATH:-configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_mixture.yaml}"
RUN_PATH="${RUN_PATH:-}"
SAVE_PATH="${SAVE_PATH:-eval_results}"
CHECKPOINTS="${CHECKPOINTS:-last}"
DATASETS="${DATASETS:-gsm8k}"
TEMPERATURES="${TEMPERATURES:-1.0}"
SAMPLING_MODE="${SAMPLING_MODE:-bernoulli-argmax}"
SEEDS="${SEEDS:-42,43,44}"

if [[ -z "$RUN_PATH" ]]; then
    RUN_PATH="$(
        python - <<'PY'
from pathlib import Path

preferred = [
    Path("runs/trainer_output"),
    Path("runs/paper_llada_bl256_alpha_0.0_seed_123"),
    Path("runs/paper_llada_bl256_alpha_0_seed_123"),
    Path("outputs/my_experiment"),
]

def checkpoint_files(run_dir: Path):
    return list(run_dir.glob("checkpoint-*/model.safetensors"))

for run_dir in preferred:
    if checkpoint_files(run_dir):
        print(run_dir)
        raise SystemExit

candidates = []
for root in (Path("runs"), Path("outputs")):
    if not root.exists():
        continue
    for model_file in root.glob("**/checkpoint-*/model.safetensors"):
        run_dir = model_file.parent.parent
        try:
            mtime = model_file.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, run_dir))

if candidates:
    print(max(candidates)[1])
PY
    )"
fi

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
    echo "Run directory not found." >&2
    echo "Set RUN_PATH=/path/to/the/training/run, for example RUN_PATH=outputs/my_experiment." >&2
    echo "Searched default checkpoint roots under runs/ and outputs/." >&2
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
