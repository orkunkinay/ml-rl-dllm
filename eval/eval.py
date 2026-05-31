#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import argparse
import json
import math
import os
import random
import re
import signal
import sys
import threading
import time
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.memory import log_cuda_memory
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from common.parsing.parse_and_get_acc import check_gsm_correct
from common.parsing.parse_and_get_acc import check_math_correct
from common.parsing.parse_and_get_acc import extract_gsm_answer
from common.parsing.parse_and_get_acc import extract_math_answer
from common.run_state import append_jsonl
from common.run_state import completed_jsonl_sample_ids
from common.run_state import iter_jsonl
from common.run_state import write_progress
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.humaneval import HumanEvalDataset
from data.loaders.math500 import MATH500Dataset
from data.loaders.mbpp import MBPPDataset
from data.sanitize import sanitize_humaneval
from data.sanitize import sanitize_mbpp

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}

FEW_SHOT_DEFAULTS = {
    "gsm8k": 0,  # NOTE: Fast-dLLM uses 5
    "math": 0,  # NOTE: Fast-dLLM uses 4
    "humaneval": 0,
    "mbpp": 3,
}


def build_generation_result_path(
    args: argparse.Namespace, model_name: str, extension: str
) -> Path:
    filename_parts = [
        args.dataset,
        model_name,
        args.gen_length,
        args.diffusion_steps,
        args.block_length,
        args.remasking,
        0,  # for legacy reasons we include the rank of the process
        "generations",
    ]
    return Path(args.output_dir) / (
        "_".join(map(str, filename_parts)) + extension
    )


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_baseline_checkpoint(name):
    name = name.replace("checkpoint-", "")
    if not name.startswith("baseline-"):
        return None

    params = {"method": name.split("-")[1]}

    # Extract K<number> (tokens per step)
    if match := re.search(r"K(\d+)", name):
        params["diffusion_steps"] = int(match.group(1))

    # Extract t<number> (threshold)
    if match := re.search(r"t([\d.]+)", name):
        params["thres"] = float(match.group(1))

    return params


def evaluate(
    model,
    tokenizer,
    dataloader,
    dataset_name,
    accelerator=None,
    policy=None,
    gen_length=128,
    temperature=0.0,
    steps=64,
    block_length=32,
    remasking="low_confidence",
    thres=0.7,
    sampling_mode="bernoulli",
    dpls_stop_logit=0.0,
    temperature_policy=1.0,
    policy_full_context=True,
    confidences_top_p=1,
    mask_id=126336,
    model_type=None,
    output_jsonl=None,
    completed_sample_ids=None,
    disable_tqdm=False,
    progress_dir=None,
    alpha=None,
    seed=None,
    tqdm_position=0,
    log_memory=False,
    memory_log_interval=50,
    reset_memory_peak_each_log=False,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    is_code_dataset = dataset_name in ["humaneval", "mbpp"]
    completed_sample_ids = set(completed_sample_ids or set())
    total_seen = len(completed_sample_ids)
    running_correct = 0
    running_nfe = []
    memory_log_interval = max(1, int(memory_log_interval or 50))
    first_memory_log_done = False
    last_memory_log_count = len(completed_sample_ids)

    def make_sample_id(batch, batch_index, index):
        if dataset_name == "humaneval":
            return str(batch["task_ids"][index])
        if dataset_name == "mbpp":
            return str(batch["task_ids"][index])
        if "questions" in batch:
            return f"{dataset_name}:{batch['questions'][index]}"
        return f"{dataset_name}:batch_{batch_index}:item_{index}"

    def progress_total():
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "num_samples"):
            return sampler.num_samples
        if sampler is not None:
            try:
                return len(sampler)
            except TypeError:
                pass
        if hasattr(dataloader, "dataset"):
            return len(dataloader.dataset)
        return None

    with torch.no_grad():
        total_for_progress = progress_total()
        progress_initial = (
            0
            if accelerator is not None and accelerator.num_processes > 1
            else min(len(completed_sample_ids), total_for_progress or 0)
        )
        progress = tqdm(
            total=total_for_progress,
            disable=disable_tqdm
            or (not accelerator.is_main_process if accelerator else False),
            dynamic_ncols=True,
            position=tqdm_position,
            initial=progress_initial,
            desc=f"dataset={dataset_name} sampler={remasking} alpha={alpha}",
        )
        try:
            for batch_index, batch in enumerate(dataloader):
                batch_sample_ids = [
                    make_sample_id(batch, batch_index, j)
                    for j in range(len(batch["prompts"]))
                ]
                if batch_sample_ids and all(
                    sid in completed_sample_ids for sid in batch_sample_ids
                ):
                    continue
                start_time = time.time()
                input_ids = batch["input_ids"].to(device)
                attn_masks = batch["attention_mask"].bool().to(device)
                prompts = batch["prompts"]

                if is_code_dataset:
                    if dataset_name == "humaneval":
                        raw_prompts = batch["raw_prompts"]
                        task_ids = batch["task_ids"]
                        test_cases = batch["test_cases"]
                        entry_points = batch["entry_points"]
                    elif dataset_name == "mbpp":
                        raw_prompts = batch["texts"]
                        task_ids = batch["task_ids"]
                        test_cases = batch["test_cases"]
                        entry_points = [None] * len(task_ids)
                else:
                    gt_answers = batch["answers"]
                    questions = batch["questions"]

                gen_kwargs = {
                    "model": model,
                    "prompt": input_ids,
                    "remasking": remasking,
                    "gen_length": gen_length,
                    "block_length": block_length,
                    "temperature": temperature,
                    "mask_id": mask_id,
                    "model_type": model_type,
                    "attention_mask": attn_masks,
                }

                if remasking == "policy":
                    if policy is None:
                        raise ValueError(
                            "policy remasking requires a policy to be provided"
                        )
                    gen_kwargs.update(
                        {
                            "policy": policy,
                            "sampling_mode": sampling_mode,
                            "dpls_stop_logit": dpls_stop_logit,
                            "temperature_policy": temperature_policy,
                            "full_context": policy_full_context,
                            "confidences_top_p": confidences_top_p,
                        }
                    )
                elif remasking == "fastdllm":
                    gen_kwargs["thres"] = thres
                else:
                    gen_kwargs["steps"] = steps

                result = generate_unified(**gen_kwargs)
                out = result.sequences

                if remasking == "policy":
                    steps_taken = result.steps_taken.tolist()
                elif remasking == "fastdllm":
                    steps_taken = [result.steps_taken.item()]
                else:
                    steps_taken = [result.steps_taken.item()] * len(input_ids)

                generated_texts = tokenizer.batch_decode(
                    out[:, -gen_length:], skip_special_tokens=True
                )

                batch_wall_time = time.time() - start_time
                wall_time_per_sample = batch_wall_time / len(generated_texts)

                if is_code_dataset:
                    sanitized_completions = []
                    for j, gen_text in enumerate(generated_texts):
                        if dataset_name == "humaneval":
                            try:
                                full_completion = raw_prompts[j] + gen_text
                                sanitized = sanitize_humaneval(
                                    full_completion, entry_points[j]
                                )
                                sanitized_completions.append(sanitized)
                            except Exception as e:
                                print(
                                    f"Warning: Failed to sanitize HumanEval completion for {task_ids[j]}: {e}"
                                )
                                sanitized_completions.append(raw_prompts[j] + gen_text)
                        elif dataset_name == "mbpp":
                            try:
                                sanitized = sanitize_mbpp(gen_text)
                                sanitized_completions.append(sanitized)
                            except Exception as e:
                                print(
                                    f"Warning: Failed to sanitize MBPP completion for {task_ids[j]}: {e}"
                                )
                                sanitized_completions.append(gen_text)

                    example_result = [
                        {
                            "task_id": task_ids[j],
                            "prompt": raw_prompts[j],
                            "prompt_input": prompts[j],
                            "generation_raw": generated_texts[j],
                            "generation_sanitized": sanitized_completions[j],
                            "test_cases": test_cases[j],
                            "entry_point": entry_points[j],
                            "steps": steps_taken[j].item()
                            if hasattr(steps_taken[j], "item")
                            else steps_taken[j],
                            "wall_time": wall_time_per_sample,
                        }
                        for j in range(len(task_ids))
                    ]
                else:
                    example_result = [
                        {
                            "question": questions[j],
                            "prompt_input": prompts[j],
                            "generations": generated_texts[j],
                            "ground_truth": gt_answers[j].item()
                            if hasattr(gt_answers[j], "item")
                            else gt_answers[j],
                            "steps": steps_taken[j].item()
                            if hasattr(steps_taken[j], "item")
                            else steps_taken[j],
                            "wall_time": wall_time_per_sample,
                        }
                        for j in range(len(gt_answers))
                    ]

                all_generations.extend(example_result)
                total_processed += len(generated_texts)
                wall_times.append(batch_wall_time)
                rows_to_append = []
                for j, item in enumerate(example_result):
                    sample_id = batch_sample_ids[j]
                    if sample_id in completed_sample_ids:
                        continue
                    if is_code_dataset:
                        prediction = item["generation_sanitized"]
                        answer = item["test_cases"]
                        prompt = item["prompt"]
                        nfe = item["steps"]
                    else:
                        prediction = item["generations"]
                        answer = item["ground_truth"]
                        prompt = item["prompt_input"]
                        nfe = item["steps"]
                    correct = None
                    if dataset_name == "gsm8k":
                        correct = check_gsm_correct(
                            extract_gsm_answer(prediction), answer
                        )
                    elif dataset_name == "math":
                        correct = check_math_correct(
                            extract_math_answer(prediction), answer
                        )
                    if correct is True:
                        running_correct += 1
                    row = {
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "prediction": prediction,
                        "answer": answer,
                        "correct": correct,
                        "nfe": nfe,
                        "sampler": remasking,
                        "alpha": alpha,
                        "seed": seed,
                        "dataset": dataset_name,
                        "raw_result": item,
                    }
                    rows_to_append.append(row)
                    completed_sample_ids.add(sample_id)
                    running_nfe.append(float(nfe))
                if output_jsonl is not None and (
                    accelerator is None or accelerator.is_main_process
                ):
                    for row in rows_to_append:
                        append_jsonl(output_jsonl, row)
                progress.update(len(rows_to_append))
                total_seen = len(completed_sample_ids)
                if log_memory and (accelerator is None or accelerator.is_main_process):
                    should_log_memory = False
                    if rows_to_append and not first_memory_log_done:
                        should_log_memory = True
                        first_memory_log_done = True
                    elif total_seen - last_memory_log_count >= memory_log_interval:
                        should_log_memory = True
                    if should_log_memory:
                        log_cuda_memory(
                            prefix=(
                                f"eval dataset={dataset_name} sampler={remasking} "
                                f"sample={total_seen}"
                            ),
                            reset_peak=reset_memory_peak_each_log,
                        )
                        last_memory_log_count = total_seen
                mean_nfe = sum(running_nfe) / len(running_nfe) if running_nfe else None
                progress.set_postfix(sample=total_seen, mean_nfe=mean_nfe)
                if progress_dir and (accelerator is None or accelerator.is_main_process):
                    write_progress(
                        progress_dir,
                        "running",
                        alpha=alpha,
                        seed=seed,
                        global_step=total_seen,
                        total_steps=total_for_progress,
                        completed_fraction=(
                            total_seen / total_for_progress
                            if total_for_progress
                            else None
                        ),
                        latest_metrics={
                            "mean_nfe": mean_nfe,
                            "correct": running_correct,
                        },
                        extra={"dataset": dataset_name, "sampler": remasking},
                    )

                if accelerator and accelerator.is_main_process:
                    idx = random.randint(0, len(prompts) - 1)
                    if is_code_dataset:
                        if dataset_name == "humaneval":
                            print(f"Task ID: {task_ids[idx]}")
                            print("-" * 50)
                            print("Generation (sanitized):")
                            print(sanitized_completions[idx])
                            print("-" * 50)
                        elif dataset_name == "mbpp":
                            print(f"Task: {raw_prompts[idx]}")
                            print("-" * 50)
                            print("Generation (sanitized):")
                            print(sanitized_completions[idx])
                            print("-" * 50)
                    else:
                        print(f"Question: {questions[idx]}")
                        print("-" * 50)
                        print("Generation:")
                        print(generated_texts[idx])
                        print("-" * 50)
                        print(f"Ground truth: {gt_answers[idx]}")
        finally:
            progress.close()
    if log_memory and (accelerator is None or accelerator.is_main_process):
        log_cuda_memory(
            prefix=f"eval end dataset={dataset_name} sampler={remasking}",
            reset_peak=reset_memory_peak_each_log,
        )
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def evaluate_code(generations, dataset_name):
    try:
        print(f"\n=== Running code evaluation for {dataset_name} ===")
        code_eval = hf_evaluate.load("code_eval")

        predictions = [[gen["generation_sanitized"]] for gen in generations]
        references = [gen["test_cases"] for gen in generations]

        print(f"Evaluating {len(predictions)} code samples...")
        pass_at_k, results = code_eval.compute(
            references=references, predictions=predictions, k=[1]
        )
        pass_at_1 = pass_at_k["pass@1"]

        print("Code evaluation results:")
        print(f"  pass@1: {pass_at_1:.4f}")

        for task_id, task_results in results.items():
            if len(task_results) > 0:
                _, result_dict = task_results[0]
                generations[task_id]["pass@1"] = 1.0 if result_dict["passed"] else 0.0
            else:
                generations[task_id]["pass@1"] = 0.0

        return {"pass@1": pass_at_1}

    except Exception as e:
        print(f"Error during code evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_local_path_and_save_results(
    results: dict,
    args: argparse.Namespace,
    model_name: str,
) -> Path | None:
    file_path = None
    if not args.dont_save:
        file_path = build_generation_result_path(args, model_name, ".json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=False)
        print(f"Saved results locally to {file_path}")
    return file_path


def get_incremental_jsonl_path(args: argparse.Namespace, model_name: str) -> Path:
    return build_generation_result_path(args, model_name, ".jsonl")


def generations_from_jsonl(path: str | Path) -> list[dict]:
    generations = []
    for row in iter_jsonl(path) or []:
        raw = row.get("raw_result")
        if raw is not None:
            generations.append(raw)
    return generations


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument("--model_path", type=str, required=False, default=None)
    parser.add_argument(
        "--few_shot",
        type=int,
        default=-1,
        help="Number of few-shot examples (default: -1 -> dataset-specific defaults)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "humaneval", "mbpp"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=0)
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--remasking", type=str, default="policy")
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--thres", type=float, default=0.7)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature_policy", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume evaluation from existing JSONL results. Use 'auto'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--tqdm_position", type=int, default=0)
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--memory_log_interval", type=int, default=50)
    parser.add_argument("--reset_memory_peak_each_log", action="store_true")
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=None,
        help="Sampling mode override (optional, uses config value if not specified)",
    )
    args = parser.parse_args()

    init_seed(args.seed)

    baseline_mode = False
    baseline_params = None
    if args.policy_path:
        checkpoint_name = Path(args.policy_path).parent.name
        baseline_params = parse_baseline_checkpoint(checkpoint_name)
        if baseline_params:
            baseline_mode = True
            print(f"Auto-detected baseline: {baseline_params}")

    # Load args from teh config (unless overriden)
    trl_parser = TrlParser((Config,))
    (grpo_config,) = trl_parser.parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    args.grpo_config = grpo_config
    if args.sampling_mode is None:
        args.sampling_mode = grpo_config.sampling_mode
    if args.block_length is None:
        args.block_length = grpo_config.block_length
    if args.gen_length is None:
        args.gen_length = grpo_config.max_completion_length
    # Override model_path from config if not explicitly provided
    if args.model_path is None:
        args.model_path = grpo_config.model_path
    args.dpls_stop_logit = grpo_config.dpls_stop_logit

    if args.remasking == "fastdllm":
        assert args.thres is not None, "thres must be provided for fastdllm"

    # NOTE: setting up the accelerator must be done after parsing config
    accelerator = Accelerator()

    # Check if we are running a baseline, if so get the args from the name
    args.baseline_mode = baseline_mode
    if baseline_mode:
        assert baseline_params is not None
        args.remasking = baseline_params["method"]
        if "thres" in baseline_params:
            args.thres = baseline_params["thres"]
        if "diffusion_steps" in baseline_params:
            args.diffusion_steps = baseline_params["diffusion_steps"]

        args.sampling_mode = None
        if args.remasking in {"random", "low_confidence"}:
            assert args.diffusion_steps > 0

    # Set few_shot to dataset-specific default if -1 is specified
    if args.few_shot == -1:
        args.few_shot = FEW_SHOT_DEFAULTS[args.dataset]
        if accelerator.is_main_process:
            print(
                f"Using dataset-specific few-shot setting for {args.dataset}: {args.few_shot}"
            )

    # Compute model name for output path
    model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    output_jsonl = get_incremental_jsonl_path(args, model_name)
    completed_sample_ids = set()
    if not args.dont_save:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.resume == "auto":
            completed_sample_ids = completed_jsonl_sample_ids(output_jsonl)
            if accelerator.is_main_process:
                print(
                    f"Resume auto: found {len(completed_sample_ids)} completed samples in {output_jsonl}"
                )
        elif output_jsonl.exists() and not args.overwrite:
            raise FileExistsError(
                f"Incremental results already exist: {output_jsonl}. "
                "Use --resume auto to continue or --overwrite to append a fresh run intentionally."
            )
        elif output_jsonl.exists() and args.overwrite and accelerator.is_main_process:
            output_jsonl.unlink()

    def handle_signal(signum, frame):
        if accelerator.is_main_process:
            write_progress(
                args.output_dir,
                "interrupted",
                alpha=args.grpo_config.alpha_compute_reward,
                seed=args.seed,
                global_step=len(completed_jsonl_sample_ids(output_jsonl)),
                last_checkpoint=str(output_jsonl),
                extra={
                    "signal": signum,
                    "dataset": args.dataset,
                    "sampler": args.remasking,
                },
            )
        sys.exit(128 + signum)

    previous_signal_handlers = {}
    if (
        accelerator.is_main_process
        and threading.current_thread() is threading.main_thread()
    ):
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)

    # Load the base model and tokenizer
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.log_memory:
        log_cuda_memory(
            prefix="eval after model loading",
            reset_peak=args.reset_memory_peak_each_log,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if "LLaDA" in args.model_path:
        mask_id = MASK_TOKENS_MAP["LLaDA"]
        _model_type = "LLaDA"
    elif "Dream" in args.model_path:
        mask_id = MASK_TOKENS_MAP["Dream"]
        _model_type = "Dream"
    else:
        raise ValueError(f"Model path {args.model_path} not supported")

    # Load the policy
    policy = None
    if args.remasking == "policy" and not args.baseline_mode:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = args.grpo_config
        if config.policy_type == "dit_hidden":
            assert _model_type == "LLaDA", (
                "dit_hidden policy is only supported with LLaDA models, not Dream"
            )
            policy_core = DiTHiddenStatePolicy(
                dllm=model,
                time_embed_dim=config.policy_time_embed_dim,
                num_blocks=config.policy_num_blocks,
                smart_init=config.policy_smart_init,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_confidence":
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTConfidencePolicy(
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                confidences_top_p=config.confidences_top_p,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        else:
            raise ValueError(
                f"Policy type {config.policy_type} not supported. "
                "Choose from ['dit_hidden', 'dit_confidence']"
            )
        policy = PolicyHFWrapper(policy_core, config.policy_type)

        if args.policy_path is not None:
            if accelerator.is_main_process:
                print(f"Loading policy from {args.policy_path}")
            state = load_file(args.policy_path)
            policy.load_state_dict(state)
        if args.log_memory and accelerator.is_main_process:
            log_cuda_memory(
                prefix="eval after policy loading",
                reset_peak=args.reset_memory_peak_each_log,
            )
    elif args.log_memory and accelerator.is_main_process:
        log_cuda_memory(
            prefix="eval after sampler setup",
            reset_peak=args.reset_memory_peak_each_log,
        )

    # Create the dataset
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": -1,
        "num_examples": args.few_shot,
    }
    if args.dataset in ["gsm8k", "math"]:
        dataset_kwargs["add_reasoning"] = True
    dataset = DATASET_MAP[args.dataset](**dataset_kwargs)

    # take only first args.n_test examples
    collate_fn = dataset.collate_fn
    if args.n_test is not None and len(dataset) > args.n_test:
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=collate_fn,
    )

    # Use accelerator to prepare model and policy, but NOT the dataloader
    # We manage distribution manually with CustomDistributedSampler to avoid padding
    if policy is not None:
        model, policy = accelerator.prepare(model, policy)
    else:
        model = accelerator.prepare(model)

    # Run evaluation
    results = evaluate(
        model,
        tokenizer,
        dataloader,
        dataset_name=args.dataset,
        accelerator=accelerator,
        policy=policy,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        remasking=args.remasking,
        thres=args.thres,
        sampling_mode=args.sampling_mode,
        dpls_stop_logit=args.dpls_stop_logit,
        temperature_policy=args.temperature_policy,
        mask_id=mask_id,
        model_type=_model_type,
        policy_full_context=args.grpo_config.policy_full_context
        if args.remasking == "policy"
        else False,
        confidences_top_p=args.grpo_config.confidences_top_p
        if args.remasking == "policy"
        else 1,
        output_jsonl=None if args.dont_save else output_jsonl,
        completed_sample_ids=completed_sample_ids,
        disable_tqdm=args.disable_tqdm,
        progress_dir=args.output_dir,
        alpha=args.grpo_config.alpha_compute_reward,
        seed=args.seed,
        tqdm_position=args.tqdm_position,
        log_memory=args.log_memory,
        memory_log_interval=args.memory_log_interval,
        reset_memory_peak_each_log=args.reset_memory_peak_each_log,
    )

    if accelerator.num_processes > 1:
        all_gpu_generations = gather_object(results["generations"])
        if accelerator.is_main_process:
            results["generations"] = all_gpu_generations

    if accelerator.is_main_process:
        if not args.dont_save and output_jsonl.exists():
            results["generations"] = generations_from_jsonl(output_jsonl)
            results["total_processed"] = len(results["generations"])
        if args.dataset in {"humaneval", "mbpp"}:
            results["code_eval_results"] = evaluate_code(
                results["generations"], args.dataset
            )
        results["metrics"] = {
            k: results.pop(k) for k in ("wall_time", "total_processed")
        }
        results.update(
            {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
                "remasking": args.remasking,
                "policy_path": args.policy_path,
                "thres": args.thres,
                "n_test": args.n_test,
            }
        )
        get_local_path_and_save_results(results, args, model_name)
        if not args.dont_save:
            write_progress(
                args.output_dir,
                "completed",
                alpha=args.grpo_config.alpha_compute_reward,
                seed=args.seed,
                global_step=len(results["generations"]),
                total_steps=len(dataset) if hasattr(dataset, "__len__") else None,
                completed_fraction=1.0,
                last_checkpoint=str(output_jsonl),
                latest_metrics=results.get("metrics", {}),
                extra={"dataset": args.dataset, "sampler": args.remasking},
            )

        # Before exiting, print some basic metrics about the test set to make sure we processed
        # as many samples as we expected
        actual_samples_processed = len(results["generations"])
        expected_dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        if hasattr(dataset, "dataset"):  # Handle Subset wrapper
            expected_dataset_size = (
                len(dataset.dataset) if args.n_test is None else args.n_test
            )
        elif args.n_test is not None:
            expected_dataset_size = args.n_test

        print("\n=== Test Set Verification ===")
        print(f"Dataset: {args.dataset}")
        print(f"Samples processed: {actual_samples_processed}")
        print(f"Expected dataset size: {expected_dataset_size}")
        if expected_dataset_size:
            print(
                f"Coverage: {actual_samples_processed}/{expected_dataset_size} ({100 * actual_samples_processed / expected_dataset_size:.1f}%)"
            )
        print(f"Batch size: {args.batch_size}")
        print(f"Multi-GPU processes: {accelerator.num_processes}")
        print("=============================\n")

    for sig, handler in previous_signal_handlers.items():
        signal.signal(sig, handler)

    accelerator.end_training()
    accelerator.free_memory()
