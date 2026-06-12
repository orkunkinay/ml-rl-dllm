# Why the Sweep Results Look the Way They Do

**Sweep:** `generation_batch_size (gbs) ∈ {4, 8, 16, 32}`, `per_device_train_batch_size (bs) ∈ {2, 4, 8}`, `num_generations (ng) ∈ {2, 4, 8}`, 24 valid combos, 20-minute wall time each, one H200 MIG 3g slice (~69.75 GB visible). Companion analysis: `hyperparameter_log_eda.ipynb`.

**Outcome:** 21 runs trained until the time limit; 3 runs (`gbs=4, bs=8, ng=2/4/8`) failed at config validation before training.

---

## 0. The one-sentence summary

Almost everything in these results is explained by a single fact: **in this trainer, one "generation round" processes `gbs` full sequences at once through hundreds of diffusion forward passes, and that round dominates both memory and time** — `bs` and `ng` only change how that generated data is sliced into optimizer steps, and the reward numbers are too early and too noisy to rank configs.

---

## 1. Raw results

| run | ng | bs | gbs | peak reserved (GB) | sec/it | steps done | total steps | best train reward |
|---|---|---|---|---|---|---|---|---|
| gbs4_bs2_ng2 | 2 | 2 | 4 | 20.95 | 16.25 | 44 | 29,946 | 0.225 |
| gbs4_bs4_ng2 | 2 | 4 | 4 | 20.95 | 15.91 | 44 | 14,973 | 0.275 |
| gbs4_bs4_ng4 | 4 | 4 | 4 | 20.92 | 11.79 | 70 | 29,946 | 0.250 |
| gbs4_bs8_* | 2/4/8 | 8 | 4 | — | — | failed | — | — |
| gbs8_bs2_ng2 | 2 | 2 | 8 | 26.90 | 29.72 | 30 | 29,946 | 0.229 |
| gbs8_bs4_ng2 | 2 | 4 | 8 | 26.90 | 31.32 | 24 | 14,973 | **0.350** |
| gbs8_bs4_ng4 | 4 | 4 | 8 | 26.90 | 29.85 | 30 | 29,946 | 0.338 |
| gbs8_bs8_ng2 | 2 | 8 | 8 | 26.40 | 30.00 | 24 | 7,487 | 0.250 |
| gbs8_bs8_ng4 | 4 | 8 | 8 | 26.90 | 28.91 | 30 | 14,973 | 0.313 |
| gbs8_bs8_ng8 | 8 | 8 | 8 | 26.88 | 21.88 | 34 | 29,946 | 0.288 |
| gbs16_bs2_ng2 | 2 | 2 | 16 | 38.58 | 69.66 | 12 | 29,946 | 0.313 |
| gbs16_bs4_ng2 | 2 | 4 | 16 | 38.58 | 65.44 | 10 | 14,973 | 0.313 |
| gbs16_bs4_ng4 | 4 | 4 | 16 | 37.60 | 60.29 | 14 | 29,946 | 0.125 |
| gbs16_bs8_ng2 | 2 | 8 | 16 | 38.58 | 63.01 | 12 | 7,487 | 0.313 |
| gbs16_bs8_ng4 | 4 | 8 | 16 | 38.59 | 59.72 | 14 | 14,973 | 0.208 |
| gbs16_bs8_ng8 | 8 | 8 | 16 | 38.59 | 56.32 | 12 | 29,946 | 0.156 |
| gbs32_bs2_ng2 | 2 | 2 | 32 | 62.21 | 136.27 | 6 | 29,946 | 0.344 |
| gbs32_bs4_ng2 | 2 | 4 | 32 | 62.21 | 137.90 | 4 | 14,973 | 0.344 |
| gbs32_bs4_ng4 | 4 | 4 | 32 | 62.21 | 137.01 | 6 | 29,946 | 0.198 |
| gbs32_bs8_ng2 | 2 | 8 | 32 | 62.21 | 138.02 | 4 | 7,487 | 0.344 |
| gbs32_bs8_ng4 | 4 | 8 | 32 | 62.21 | 137.31 | 6 | 14,973 | 0.198 |
| gbs32_bs8_ng8 | 8 | 8 | 32 | 62.21 | 128.28 | 6 | 29,946 | 0.052 |

A useful sanity check that also explains the runtime structure: the total step count follows
`total_steps = N_prompts × ng × num_iterations / bs` with `N_prompts ≈ 14,973` (GSM8K+MATH train mixture) and `num_iterations = 2`. Every row fits this formula exactly, which confirms the trainer's data flow described below.

---

## 2. Why the `gbs=4, bs=8` runs failed

These are **argument-validation failures, not OOMs**: `generation_batch_size (4) must be divisible by` the global train batch size (8). The trainer fills each generation round with whole train batches, so the generation round must contain at least one full train batch (`gbs % bs == 0` on a single GPU). `gbs=4` cannot cover a train batch of 8.

This is the only constraint our `submit_sweep.sh` did *not* filter for (it only enforced `bs % ng == 0` from `train/train.py:101`). Adding `(( gbs % bs == 0 ))` to the submit loop would have skipped these 3 jobs.

---

## 3. Memory: a clean linear law in `gbs`, and nothing else matters

Observed mean peak reserved memory:

| gbs | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| peak GB | 20.94 | 26.81 | 38.42 | 62.21 |

This fits **`peak ≈ 15 GB + ~1.47 GB × gbs`** almost perfectly (each doubling of `gbs` adds exactly the per-sequence increment: +5.9, +11.6, +23.8 GB).

**Why the intercept (~15–16 GB):** the LLaDA-8B weights in bf16 are ~16 GB. That is the fixed cost every run pays.

**Why ~1.5 GB per generated sequence:** diffusion generation runs full-sequence forward passes (prompt 200 + completion 256 ≈ 456 tokens, no KV cache — every step re-processes the whole sequence) with batch dimension `gbs`. The dominant per-sequence tensors are the vocabulary logits (LLaDA's vocab is ~126k, so one logits tensor is ~456 × 126k × 4 bytes ≈ 230 MB per sequence in float32, per live buffer) plus the per-step confidence/sampling buffers the policy remasking keeps (`sampling_inputs`, `sampling_masks`, per-timestep log-likelihoods in `loglikelihood_dtype: float32`). A handful of such live buffers per sequence gets you to ~1.5 GB.

**Why `bs` and `ng` have no visible effect:** the training forward/backward over `bs ≤ 8` sequences is *smaller* than the generation peak over `gbs ≥ 8` sequences, so the logged peak is always set during generation. `ng` only changes how many of the `gbs` rows are duplicates of the same prompt — the tensor shapes are identical. (The tiny dips, e.g. 26.40 for `gbs8_bs8_ng2` or 37.60 for `gbs16_bs4_ng4`, come from padding: batches whose sampled prompts happened to be shorter.)

**Practical ceiling:** at `gbs=32` the run sits at 62.2/69.75 GB = **89% of the MIG slice**. The linear law predicts `gbs=40` ≈ 74 GB → OOM. `gbs=32` is the practical maximum on this slice, with no headroom for longer prompts (`max_prompt_length` increases would break it).

---

## 4. Runtime: also linear in `gbs`, flat in `bs` — and why that's expected

Observed mean sec/it: `gbs=4 → 14.7`, `gbs=8 → 28.6`, `gbs=16 → 62.4`, `gbs=32 → 135.8` — i.e. **sec/it ≈ 4.3 × gbs**, essentially independent of `bs`.

**Why linear in `gbs` — two stacked reasons:**

1. *Compute saturation.* Each generation round runs ~256 full-sequence model forwards (256 completion tokens unmasked one per step, 8 blocks of `block_length=32`) over `gbs` sequences. A 3g MIG slice has a fraction of the H200's SMs and is already saturated at `gbs=4`, so the per-sequence generation time is constant (~8.5 s/seq across the entire sweep) — there is no free parallelism left to absorb a bigger batch.
2. *No amortization.* Even on a saturated GPU, a large round could be harmless if it fed proportionally more optimizer steps (upstream TRL spreads one round over `gbs/bs × num_iterations` steps). But this fork's buffering (`train/trainer.py:475`) regenerates whenever `global_step % num_iterations == 0`, so **every round feeds exactly `num_iterations=2` steps regardless of `gbs`**. Per-step cost is therefore `gen(gbs)/2 ≈ gbs × 8.5/2 ≈ 4.3 × gbs` seconds — exactly the observed law. (The `gbs=32, bs=2` runs completing only 4–6 uniform ~137 s steps, rather than ~80+ mostly-light steps, rules out TRL-style amortization empirically.)

A corollary worth knowing: because the buffered inputs are never sliced into `bs`-row mini-batches, each optimizer step appears to train on the **full `gbs`-row batch**; `per_device_train_batch_size` only enters the step-count bookkeeping (steps/epoch ∝ 1/bs via the sampler arithmetic). That is *why* `bs` is invisible in both memory and sec/it.

**Why flat in `bs`:** the optimizer step itself (one forward+backward over `bs` sequences) costs on the order of 1–2 model forwards, versus ~256 forwards for generation. It's noise. **But `bs` is not irrelevant** — at fixed `gbs` it doesn't change sec/it, yet it *divides the total number of steps* (see the formula in §1). `gbs8_bs8_ng2` needs 7,487 steps for an epoch; `gbs8_bs2_ng2` needs 29,946 steps at the same ~30 s/it. Same per-step cost, 4× difference in epoch wall-clock. **Larger `bs` is essentially free throughput here.**

**Why epoch time scales as `ng × gbs / bs`:** total epoch time = total_steps × sec/it ∝ `(ng/bs) × gbs`. This is why the projected full-run times in the notebook range from ~62 h (`gbs8_bs8_ng2`) to ~1,140 h (`gbs32_bs2_ng2`-style combos).

**Projected full-epoch hours (from the last tqdm ETA) and peak memory, per config:**

| ng | bs | gbs | projected total hours | last peak reserved (GB) |
|---:|---:|----:|----------------------:|------------------------:|
| 2 | 2 | 4 | 135.20 | 20.95 |
| 2 | 4 | 4 | 66.22 | 20.95 |
| 4 | 4 | 4 | 98.12 | 20.92 |
| 2 | 8 | 4 | — (failed) | — |
| 4 | 8 | 4 | — (failed) | — |
| 8 | 8 | 4 | — (failed) | — |
| 2 | 2 | 8 | 247.29 | 26.90 |
| 2 | 4 | 8 | 130.31 | 26.90 |
| 4 | 4 | 8 | 248.35 | 26.90 |
| 2 | 8 | 8 | **62.46** | 26.40 |
| 4 | 8 | 8 | 120.30 | 26.90 |
| 8 | 8 | 8 | 182.03 | 26.88 |
| 2 | 2 | 16 | 579.52 | 38.58 |
| 2 | 4 | 16 | 272.24 | 38.58 |
| 4 | 4 | 16 | 501.55 | 37.60 |
| 2 | 8 | 16 | 131.10 | 38.58 |
| 4 | 8 | 16 | 248.44 | 38.59 |
| 8 | 8 | 16 | 468.60 | 38.59 |
| 2 | 2 | 32 | 1133.59 | 62.21 |
| 2 | 4 | 32 | 573.58 | 62.21 |
| 4 | 4 | 32 | 1139.75 | 62.21 |
| 2 | 8 | 32 | 287.07 | 62.21 |
| 4 | 8 | 32 | 571.15 | 62.21 |
| 8 | 8 | 32 | 1067.16 | 62.21 |

The table is a direct visualization of the `ng × gbs / bs` law: read down any fixed `(ng, bs)` and hours roughly double with each doubling of `gbs`; read across `bs` at fixed `(ng, gbs)` and hours roughly halve with each doubling of `bs`; memory depends on `gbs` alone. (Projections extrapolate the last smoothed tqdm rate from 20-minute runs, so treat them as ±10–20% estimates.)

**The interesting second-order effect — `ng` makes runs slightly *faster*:** at every fixed `gbs`, the `ng=8` run is the fastest (21.9 vs ~30 s/it at gbs=8; 56.3 vs ~63–70 at gbs=16; 128.3 vs ~137 at gbs=32). Explanation: a generation round contains `gbs/ng` *unique* prompts, each repeated `ng` times, and the whole batch is left-padded to the longest prompt in the round. With `ng=8` and `gbs=8` there is only **one** unique prompt per round, so there is zero padding waste; with `ng=2` there are 4 unique prompts and the expected max length (hence wasted padded tokens) is higher. Fewer unique prompts per round → shorter padded length on average → fewer FLOPs per forward. The effect size (~10–25%) is consistent with typical prompt-length spread in GSM8K/MATH.

---

## 5. Reward: mostly noise — and the artifacts prove it

The `best_train_reward` column should **not** be used to rank configurations from this sweep. Three observations in the data itself demonstrate why:

**(a) The runs saw almost no training.** At 20 minutes, `gbs=4` runs completed ~44–70 optimizer steps, `gbs=16` runs ~10–14, and `gbs=32` runs only **4–6 steps**. With warmup of 100 steps, no run even left LR warmup; `gbs=32` runs did essentially zero learning. Their "best reward" is the *initial* model's reward on the first few batches, not a property of the hyperparameters.

**(b) The smoking gun: three different configs report the identical reward 0.34375 (= 11/32).** `gbs32_bs2_ng2`, `gbs32_bs4_ng2`, and `gbs32_bs8_ng2` all report exactly 11/32. With `seed: 123` fixed, all three see the same shuffled prompt order and generate the same first round of 32 completions; `bs` only changes how those identical completions are sliced into optimizer steps. 11 of the 32 completions were correct — a statement about the *base model and those particular prompts*, not about the config. The same quantization is visible elsewhere (0.3125 = 5/16 across three `gbs=16, ng=2` runs).

**(c) The apparent "high ng is bad" trend is a variance artifact.** At `gbs=16/32`, reward falls monotonically with `ng` (e.g. gbs=32: 0.344 → 0.198 → 0.052). But a generation round at `ng=8, gbs=32` contains only **4 unique prompts** (vs 16 at `ng=2`). The reward estimate is therefore an average over very few problems from a mixture (GSM8K = easier, MATH = much harder): draw 4 hard MATH problems and the round scores near zero (0.052 ≈ 1.7/32). Early, with few rounds observed, "best reward" at high `ng` is just a max over a handful of high-variance, few-prompt estimates. This says nothing about `ng=8` being worse for learning — in fact GRPO's advantage baseline is computed within each group of `ng` completions, so `ng=2` gives the *noisiest* learning signal even though it produced prettier early reward numbers here.

**(d) Why `gbs=8` produced the "best" rewards (0.35, 0.338):** these runs hit a sweet spot of measurement, not of learning — enough steps (~24–34) to log several reward estimates (so the max-statistic has more draws than `gbs=16/32` runs), over enough unique prompts per round (so each estimate is less noisy than high-`ng` rounds). "Best-of-N checkpoints" is biased upward in N.

**Correlation summary for reward:** reward correlates with nothing causally in this sweep. Its apparent correlations (positive-ish with `gbs`, negative with `ng`) are mediated by (i) number of reward measurements taken before the time limit, (ii) number of unique prompts per measurement, and (iii) the shared data order from the fixed seed.

---

## 6. The correlation structure, in one picture

```
                    gbs
                  ↗  |  ↘
        (linear)↗    |    ↘(linear)
     peak memory     |     sec/it
                     |        |
   bs ───────────────┼────────┼──→ total steps (∝ ng/bs) ──→ epoch hours ∝ ng·gbs/bs
                     |        |
   ng ──→ unique prompts/round (gbs/ng) ──→ padding waste (↑ sec/it slightly)
                              |                    ↘
                              └──→ steps done in 20 min ──→ reward measurements (noise)
```

- **memory ↔ sec/it: r ≈ 1.0, but it's confounding, not causation** — both are driven by `gbs`. You cannot trade one against the other in this design; lowering `gbs` improves both simultaneously.
- **`bs` is a pure throughput knob**: invisible in memory and sec/it, but inversely scales epoch length.
- **`ng` is a pure statistics knob with a small speed bonus**: it multiplies epoch length (`∝ ng`), slightly reduces sec/it via padding, and controls GRPO group size (advantage-estimate quality).
- **reward correlates with everything and means nothing** at this horizon (see §5).

---

## 7. What this implies for the next experiment

1. **Don't use `gbs=32`**: 89% memory utilization (one long prompt from OOM), 4.7× the per-round latency of `gbs=8`, and its "good" reward numbers are a seed artifact (§5b).
2. **Maximize `bs` subject to the constraints** (`bs % ng == 0`, `gbs % bs == 0`): it's free wall-clock. The 7,487-step epoch of `gbs8_bs8_ng2` (~62 h projected) is the cheapest full epoch observed.
3. **Don't pick `ng=2` because of the reward column.** Group size 2 gives GRPO a one-bit advantage signal per pair. `ng=4` (e.g. `gbs8_bs8_ng4` or `gbs16_bs8_ng4`) is a more defensible default; its lower early reward here is measurement variance, not a learning deficit.
4. **Sensible starting point:** `gbs=8 or 16, bs=8, ng=4` — ~27–39 GB (40–55% of the slice, with headroom for longer prompts), ~29–60 s/it, and a balanced statistics/throughput trade.
5. **To actually compare quality**, rerun 2–3 shortlisted configs with (i) equal *optimizer-step* budgets (not equal wall-clock), (ii) ≥ several hundred steps so runs clear the 100-step warmup, (iii) different seeds per replicate, and (iv) a held-out eval — train reward on self-generated batches is not an accuracy measurement.

## 8. Caveats

- All quality statements are about a **proxy** (train reward) observed during **LR warmup** on **one seed**. Nothing here measures converged performance.
- The per-sequence memory (~1.5 GB) and time (~4.3 s) coefficients are specific to this MIG slice, `max_prompt_length=200`, `max_completion_length=256`, `block_length=32`, and float32 log-likelihoods; changing any of these changes the linear laws, not just the constants.
- The padding-based explanation for the `ng` speed effect (§4) fits the data well but wasn't isolated experimentally; profiling one round with sorted-vs-shuffled prompts would confirm it.
