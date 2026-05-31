# Cluster Resume Changes

This repository keeps the original diffusion-language-model policy training and
sampling logic intact. The cluster changes add run-state files, sidecar
checkpoints, progress reporting, and resumable evaluation around the existing
method.

## Training

Fresh local training writes to a deterministic run directory:

```bash
python -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
```

Example run layout:

```text
runs/paper_llada_bl32_alpha_0.3_seed_1/
  config.yaml
  checkpoint-100/
  checkpoint-200/
  checkpoint-best/
  checkpoints/
    checkpoint_latest.pt
    checkpoint_step_100.pt
    checkpoint_step_200.pt
  logs/
  metrics.jsonl
  progress.json
  outputs/
```

Resume latest checkpoint:

```bash
python -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --resume auto
```

Resume an exact sidecar:

```bash
python -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --resume runs/paper_llada_bl32_alpha_0.3_seed_1/checkpoints/checkpoint_latest.pt
```

The existing TRL/Hugging Face checkpoint directories remain the authoritative
training resume source. The new `checkpoints/checkpoint_latest.pt` sidecar stores
policy weights, optimizer state when available from the callback, scheduler
state, RNG state, config snapshot, latest metrics, epoch, global step, alpha,
seed, git commit, hostname, and SLURM identifiers.

If the cluster sends `SIGTERM` or `SIGINT`, training writes:

```text
checkpoint-emergency-<step>/
checkpoints/checkpoint_emergency_<step>.pt
checkpoints/checkpoint_latest.pt
progress.json status = interrupted
```

## Evaluation

Fresh evaluation:

```bash
python -m eval.eval \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --policy_path runs/paper_llada_bl32_alpha_0.3_seed_1/checkpoint-1000/model.safetensors \
  --dataset gsm8k \
  --output_dir results/eval_policy_alpha_0.3_gsm8k_seed_1
```

Resume evaluation:

```bash
python -m eval.eval \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --policy_path runs/paper_llada_bl32_alpha_0.3_seed_1/checkpoint-1000/model.safetensors \
  --dataset gsm8k \
  --output_dir results/eval_policy_alpha_0.3_gsm8k_seed_1 \
  --resume auto
```

Evaluation now writes incremental `*_generations.jsonl` rows after each completed
sample. On resume it loads completed `sample_id`s and skips fully completed
batches, avoiding duplicate rows. At the end it still writes the legacy
`*_generations.json` file for existing aggregation/parsing code.

## Sweeps

Resume or skip completed sweep work:

```bash
python -m eval.pipeline \
  ./runs/paper_llada_bl32_alpha_0.3_seed_1 \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --checkpoints last \
  --datasets all \
  --seeds 42,43,44 \
  --resume auto
```

`eval.pipeline` skips an evaluation when `progress.json` says `completed` and a
non-empty result file exists. Otherwise it invokes `eval.eval --resume auto`.

## Progress And Logs

Training:

```text
runs/.../progress.json
runs/.../metrics.jsonl
```

Evaluation:

```text
results/.../progress.json
results/.../*_generations.jsonl
```

Use `--disable_tqdm` for less noisy cluster logs. Evaluation also accepts
`--tqdm_position`.

## Limitations

Exact mid-dataloader-position restoration is delegated to the underlying
TRL/Hugging Face trainer state. The sidecar stores RNG state and points back to
the HF checkpoint directory, so resume is exact to the extent supported by the
trainer. Evaluation resume is sample-level for single-process jobs; multi-GPU
evaluation still gathers final results as before, while the incremental JSONL is
most useful for the single-process cluster jobs described here.
