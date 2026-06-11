#!/bin/bash
# Generates yaml variants of the safe config for a batch-size sweep and
# submits one sbatch job per valid combination.
#
# Constraint (asserted in train/train.py): per_device_train_batch_size
# must be divisible by num_generations, so invalid combos are skipped.
set -euo pipefail

cd "$(dirname "$0")"

BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture_safe.yaml
SWEEP_DIR=configs/experiment_configs/sweep
mkdir -p "$SWEEP_DIR" logs

GENERATION_BATCH_SIZES=(4 8 16 32)
TRAIN_BATCH_SIZES=(2 4 8)
NUM_GENERATIONS=(2 4 8)

DRY_RUN="${DRY_RUN:-0}"
submitted=0
skipped=0

for gbs in "${GENERATION_BATCH_SIZES[@]}"; do
  for bs in "${TRAIN_BATCH_SIZES[@]}"; do
    for ng in "${NUM_GENERATIONS[@]}"; do
      if (( bs % ng != 0 )); then
        echo "SKIP  gbs=${gbs} bs=${bs} ng=${ng} (bs not divisible by ng)"
        skipped=$((skipped + 1))
        continue
      fi

      tag="gbs${gbs}_bs${bs}_ng${ng}"
      cfg="$SWEEP_DIR/safe_${tag}.yaml"

      sed \
        -e "s/^generation_batch_size:.*/generation_batch_size: ${gbs}/" \
        -e "s/^per_device_train_batch_size:.*/per_device_train_batch_size: ${bs}/" \
        -e "s/^num_generations:.*/num_generations: ${ng}/" \
        "$BASE_CONFIG" > "$cfg"

      # Unique run_name so concurrent jobs don't share (and --overwrite
      # doesn't delete) each other's run directory under runs/.
      printf '\nrun_name: sweep_safe_%s\n' "$tag" >> "$cfg"

      if [ "$DRY_RUN" = "1" ]; then
        echo "DRY   would submit ${tag} -> ${cfg}"
      else
        sbatch \
          --job-name="llada_${tag}" \
          --output="logs/llada_${tag}_%j.out" \
          --error="logs/llada_${tag}_%j.err" \
          run_train_sweep.sh "$cfg"
        echo "SUBMIT ${tag} -> ${cfg}"
      fi
      submitted=$((submitted + 1))
    done
  done
done

echo "---"
echo "Submitted: ${submitted}, skipped (invalid combos): ${skipped}"
