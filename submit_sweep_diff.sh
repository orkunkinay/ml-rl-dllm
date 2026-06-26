#!/bin/bash
# Generates yaml variants of the BL256 config for a batch-size sweep and
# submits one sbatch job per valid combination.
#
# Run from the cluster checkout:
#   cd ~/msc_project/ml-rl-dllm-timing-test
#
# Full sweep:
#   DRY_RUN=1 ./submit_sweep_diff.sh
#   ./submit_sweep_diff.sh
#
# Rerun only configs that failed with Python errors in a previous log dir:
#   DRY_RUN=1 RERUN_FROM_LOG_DIR=logs/last_timing_sweep_3520013_3520066 ./submit_sweep_diff.sh
#   RERUN_FROM_LOG_DIR=logs/last_timing_sweep_3520013_3520066 ./submit_sweep_diff.sh
#
# Rerun an explicit list. Each non-comment line may be a full config path,
# bl256_gbs64_bs8_ng4.yaml, or gbs64_bs8_ng4:
#   RERUN_CONFIGS_FILE=failed_configs.txt ./submit_sweep_diff.sh
#
# Constraint (asserted in train/train.py): per_device_train_batch_size
# must be divisible by num_generations, so invalid combos are skipped.
# Constraint (asserted by TRL GRPOConfig): generation_batch_size must be
# divisible by the global batch size. These jobs run one process/GPU, so the
# global batch size is per_device_train_batch_size.
set -euo pipefail

cd "$(dirname "$0")"

BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL256_mixture.yaml
SWEEP_DIR=configs/experiment_configs/sweep_diff
mkdir -p "$SWEEP_DIR" logs

GENERATION_BATCH_SIZES=(4 8 16 32 64 128)
TRAIN_BATCH_SIZES=(2 4 8 16 32)
NUM_GENERATIONS=(2 4 8)

DRY_RUN="${DRY_RUN:-0}"
RERUN_FROM_LOG_DIR="${RERUN_FROM_LOG_DIR:-}"
RERUN_CONFIGS_FILE="${RERUN_CONFIGS_FILE:-}"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: base config not found: $BASE_CONFIG"
  exit 1
fi

requested_configs=()
add_requested_config() {
  local item="$1"
  item="${item%%#*}"
  item="${item#"${item%%[![:space:]]*}"}"
  item="${item%"${item##*[![:space:]]}"}"
  [ -z "$item" ] && return 0

  if [[ "$item" == gbs*_bs*_ng* ]]; then
    item="$SWEEP_DIR/bl256_${item}.yaml"
  elif [[ "$item" == bl256_gbs*_bs*_ng* ]]; then
    [[ "$item" == *.yaml ]] || item="${item}.yaml"
    item="$SWEEP_DIR/$item"
  fi

  requested_configs+=("$item")
}

if [ -n "$RERUN_CONFIGS_FILE" ]; then
  if [ ! -f "$RERUN_CONFIGS_FILE" ]; then
    echo "ERROR: RERUN_CONFIGS_FILE not found: $RERUN_CONFIGS_FILE"
    exit 1
  fi
  while IFS= read -r line; do
    add_requested_config "$line"
  done < "$RERUN_CONFIGS_FILE"
fi

if [ -n "$RERUN_FROM_LOG_DIR" ]; then
  if [ ! -d "$RERUN_FROM_LOG_DIR" ]; then
    echo "ERROR: RERUN_FROM_LOG_DIR not found: $RERUN_FROM_LOG_DIR"
    exit 1
  fi
  while IFS= read -r config_path; do
    add_requested_config "$config_path"
  done < <(python3 - "$RERUN_FROM_LOG_DIR" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
config_re = re.compile(r"configs/experiment_configs/sweep_diff/[A-Za-z0-9_./-]+\.yaml")
tag_re = re.compile(r"llada_diff_sweep_(gbs\d+_bs\d+_ng\d+)_\d+\.err$")
failure_markers = (
    "Traceback",
    "FileNotFoundError",
    "ValueError",
    "RuntimeError",
    "OutOfMemoryError",
    "CUDA out of memory",
    "out of memory",
    "OOM",
)

configs: set[str] = set()
for err_path in sorted(log_dir.glob("*.err")):
    text = err_path.read_text(errors="replace")
    if not any(marker in text for marker in failure_markers):
        continue

    matches = config_re.findall(text)
    if matches:
        configs.update(matches)
        continue

    tag_match = tag_re.match(err_path.name)
    if tag_match:
        configs.add(f"configs/experiment_configs/sweep_diff/bl256_{tag_match.group(1)}.yaml")

for config in sorted(configs):
    print(config)
PY
)
fi

config_requested() {
  local cfg="$1"
  local requested
  [ "${#requested_configs[@]}" -eq 0 ] && return 0
  for requested in "${requested_configs[@]}"; do
    [ "$cfg" = "$requested" ] && return 0
  done
  return 1
}

submitted=0
skipped=0
generated=0
filtered=0

if [ "${#requested_configs[@]}" -gt 0 ]; then
  echo "Restricting submission to ${#requested_configs[@]} requested config(s):"
  printf '  %s\n' "${requested_configs[@]}"
fi

for gbs in "${GENERATION_BATCH_SIZES[@]}"; do
  for bs in "${TRAIN_BATCH_SIZES[@]}"; do
    for ng in "${NUM_GENERATIONS[@]}"; do
      if (( bs % ng != 0 )); then
        echo "SKIP  gbs=${gbs} bs=${bs} ng=${ng} (bs not divisible by ng)"
        skipped=$((skipped + 1))
        continue
      fi

      if (( gbs % bs != 0 )); then
        echo "SKIP  gbs=${gbs} bs=${bs} ng=${ng} (gbs not divisible by bs)"
        skipped=$((skipped + 1))
        continue
      fi

      tag="gbs${gbs}_bs${bs}_ng${ng}"
      cfg="$SWEEP_DIR/bl256_${tag}.yaml"

      sed \
        -e "s/^generation_batch_size:.*/generation_batch_size: ${gbs}/" \
        -e "s/^per_device_train_batch_size:.*/per_device_train_batch_size: ${bs}/" \
        -e "s/^num_generations:.*/num_generations: ${ng}/" \
        "$BASE_CONFIG" > "$cfg"

      # Unique run_name so concurrent jobs don't share (and --overwrite
      # doesn't delete) each other's run directory under runs/.
      printf '\nrun_name: sweep_bl256_%s\n' "$tag" >> "$cfg"
      generated=$((generated + 1))

      if ! config_requested "$cfg"; then
        filtered=$((filtered + 1))
        continue
      fi

      if [ "$DRY_RUN" = "1" ]; then
        echo "DRY   would submit ${tag} -> ${cfg}"
      else
        sbatch \
          --job-name="llada_diff_sweep_${tag}" \
          --output="logs/llada_diff_sweep_${tag}_%j.out" \
          --error="logs/llada_diff_sweep_${tag}_%j.err" \
          run_train_sweep_diff.sh "$cfg"
        echo "SUBMIT ${tag} -> ${cfg}"
      fi
      submitted=$((submitted + 1))
    done
  done
done

echo "---"
echo "Generated configs: ${generated}"
echo "Submitted: ${submitted}, filtered: ${filtered}, skipped (invalid combos): ${skipped}"
