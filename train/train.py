#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import signal
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import accelerate
import torch
import transformers
import trl
import wandb
from dotenv import load_dotenv
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import ModelConfig
from trl import TrlParser

import train.reward_func as reward_func
from common.config import Config
from common.memory import log_cuda_memory
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from common.run_state import ClusterStateCallback
from common.run_state import atomic_write_json
from common.run_state import config_to_dict
from common.run_state import prepare_local_run_dir
from common.run_state import resolve_resume_checkpoint
from common.run_state import restore_rng_state
from common.s3 import S3UploadCallback
from common.s3 import download_s3_checkpoint
from common.s3 import get_latest_s3_checkpoint
from data.data_utils import get_gsm8k_and_math_and_kodcode_questions
from data.data_utils import get_gsm8k_and_math_questions
from data.data_utils import get_gsm8k_questions
from data.data_utils import get_kodcode_questions
from data.data_utils import get_math_questions
from data.data_utils import set_random_seed
from train.trainer import Trainer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

torch.set_float32_matmul_precision("high")

print("=== Library Versions ===")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"trl: {trl.__version__}")
try:
    import flash_attn

    print(f"flash_attn: {flash_attn.__version__}")
except ImportError:
    print("flash_attn: not installed")
print("========================")


def get_reward_functions(config: Config):
    """Get reward functions based on config."""
    if config.reward_functions is not None:
        reward_functions = []
        for func_name in config.reward_functions:
            func = getattr(reward_func, func_name, None)
            if func is None:
                raise ValueError(
                    f"Unknown reward function: {func_name}. Function not found in reward_func module."
                )
            reward_functions.append(func)
        return reward_functions
    else:
        raise ValueError("Reward functions must be manually specified.")


load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "Hugging Face token not found in environment variables. Please set HF_TOKEN."
    )


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    # During training, remasking must always be "policy"
    assert grpo_config.remasking == "policy", (
        f"Training only supports remasking='policy', got '{grpo_config.remasking}'"
    )

    assert grpo_config.per_device_train_batch_size % grpo_config.num_generations == 0, (
        f"per_device_train_batch_size ({grpo_config.per_device_train_batch_size}) must be "
        f"divisible by num_generations ({grpo_config.num_generations}) to ensure complete groups per GPU."
    )

    # ES (Expert Steering) currently only supports 1 group per GPU (generates samples for first prompt only)
    if grpo_config.es_thresholds:
        assert grpo_config.num_generations == grpo_config.per_device_train_batch_size, (
            "ES requires exactly 1 group per GPU (num_generations == per_device_train_batch_size)"
        )
        assert grpo_config.block_length == 256
        assert grpo_config.policy_full_context

    if grpo_config.dataset in {"mbpp", "humaneval"}:
        raise ValueError(
            f"Training not supported for {grpo_config.dataset}. "
            "This dataset is evaluation-only."
        )
    elif grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
    elif grpo_config.dataset == "gsm8k_and_math":
        dataset = get_gsm8k_and_math_questions("train", seed=grpo_config.seed)
        assert (
            "mixed_correctness_mult_reward_func" in grpo_config.reward_functions
            or "mixed_correctness_add_reward_func" in grpo_config.reward_functions
        )
    elif grpo_config.dataset == "gsm8k_and_math_and_kodcode":
        dataset = get_gsm8k_and_math_and_kodcode_questions(
            "train", seed=grpo_config.seed
        )
        assert (
            "mixed_correctness_mult_reward_func" in grpo_config.reward_functions
            or "mixed_correctness_add_reward_func" in grpo_config.reward_functions
        )
    elif grpo_config.dataset == "kodcode":
        dataset = get_kodcode_questions()
    else:
        raise ValueError(f"Dataset {grpo_config.dataset} not supported")

    reward_functions = get_reward_functions(grpo_config)
    dataset = dataset.shuffle(seed=grpo_config.seed)
    train_set = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration (only if enabled in ModelConfig)
    # For the paper, we left this turned off.
    bnb_config = None
    if model_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model and tokenizer
    if "LLaDA" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["LLaDA"]
        grpo_config.model_type = "LLaDA"
    elif "Dream" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["Dream"]
        grpo_config.model_type = "Dream"
    else:
        raise ValueError(f"Model path {grpo_config.model_path} not supported")

    if grpo_config.log_memory:
        log_cuda_memory(
            prefix="train after model loading",
            reset_peak=grpo_config.reset_memory_peak_each_log,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Create policy based on type
    if grpo_config.policy_type == "dit_hidden":
        assert grpo_config.model_type == "LLaDA", (
            "dit_hidden policy is only supported with LLaDA models, not Dream"
        )
        policy_core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            num_blocks=grpo_config.policy_num_blocks,
            smart_init=grpo_config.policy_smart_init,
            time_period=grpo_config.policy_time_period,
        ).to(device)

    elif grpo_config.policy_type == "dit_confidence":
        hidden_dim = grpo_config.policy_hidden_dim or 128
        feedforward_dim = grpo_config.policy_feedforward_dim or (4 * hidden_dim)

        policy_core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=grpo_config.policy_num_heads,
            dropout=grpo_config.policy_dropout,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            smart_init=grpo_config.policy_smart_init,
            confidences_top_p=grpo_config.confidences_top_p,
            num_blocks=grpo_config.policy_num_blocks,
            time_period=grpo_config.policy_time_period,
        ).to(device)
    else:
        raise ValueError(
            f"Policy type {grpo_config.policy_type} not supported. "
            "Choose from ['dit_hidden', 'dit_confidence']"
        )

    policy = PolicyHFWrapper(policy_core, grpo_config.policy_type)

    if grpo_config.log_memory:
        log_cuda_memory(
            prefix="train after policy loading",
            reset_peak=grpo_config.reset_memory_peak_each_log,
        )

    # Log policy parameter count
    total_params = sum(p.numel() for p in policy_core.parameters())
    trainable_params = sum(
        p.numel() for p in policy_core.parameters() if p.requires_grad
    )

    print(f"Policy type: {grpo_config.policy_type}")
    print(f"Total policy parameters: {total_params:,}")
    print(f"Trainable policy parameters: {trainable_params:,}")

    if wandb.run is not None:
        wandb.log(
            {
                "policy/total_parameters": total_params,
                "policy/trainable_parameters": trainable_params,
                "policy/policy_type": grpo_config.policy_type,
            },
            step=0,
        )

    output_dir = grpo_config.output_dir
    s3_output = "s3" in str(output_dir)
    run_dir = None
    cluster_callback = None
    if s3_output:
        # For remote paths we save checkpoints in a temp dir locally and then
        # use a callback to push them to aws
        context_manager = tempfile.TemporaryDirectory()
        callbacks = [S3UploadCallback(output_dir)]
    else:
        run_dir = prepare_local_run_dir(
            grpo_config,
            resume=grpo_config.resume,
            overwrite=grpo_config.overwrite,
            run_root=grpo_config.run_root,
            run_name=grpo_config.run_name,
        )
        output_dir = str(run_dir)
        grpo_config.output_dir = output_dir
        grpo_config.disable_tqdm = bool(grpo_config.disable_tqdm)
        config_snapshot = {
            "grpo_config": config_to_dict(grpo_config),
            "model_config": config_to_dict(model_config),
        }
        atomic_write_json(run_dir / "config.yaml", config_snapshot)
        context_manager = nullcontext(output_dir)
        callbacks = []
        cluster_callback = ClusterStateCallback(run_dir, grpo_config)
        callbacks.append(cluster_callback)
        os.makedirs(output_dir, exist_ok=True)

    with context_manager as local_output_dir:
        grpo_config.output_dir = local_output_dir

        # Check for existing checkpoint to resume from
        resume_from = None
        if s3_output:
            latest = get_latest_s3_checkpoint(output_dir)
            if latest and grpo_config.resume == "auto":
                resume_from = download_s3_checkpoint(
                    output_dir, latest, local_output_dir
                )
                print(
                    f"=== Auto-resume: found {latest}, downloaded to {resume_from} ==="
                )
        else:
            resume_from = resolve_resume_checkpoint(
                grpo_config.resume, local_output_dir
            )
            if grpo_config.resume == "auto" and resume_from is None:
                print("=== Auto-resume: no checkpoint found; starting fresh ===")
            elif resume_from is not None:
                print(f"=== Resuming from checkpoint: {resume_from} ===")
                sidecar = Path(local_output_dir) / "checkpoints" / "checkpoint_latest.pt"
                if sidecar.exists():
                    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
                    restore_rng_state(payload.get("rng_state"))
                    print(
                        "=== Resume state: "
                        f"step={payload.get('global_step')} "
                        f"epoch={payload.get('epoch')} "
                        f"alpha={payload.get('alpha')} "
                        f"seed={payload.get('seed')} ==="
                    )

        trainer = Trainer(
            args=grpo_config,
            model=policy,
            dllm=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        def handle_signal(signum, frame):
            if cluster_callback is not None and trainer.accelerator.is_main_process:
                print(f"\n=== Caught signal {signum}; saving emergency checkpoint ===")
                cluster_callback.save_emergency_checkpoint(trainer, signum=signum)
            sys.exit(128 + signum)

        previous_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)

        try:
            trainer.train(
                resume_from_checkpoint=str(resume_from) if resume_from else None
            )
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

        if resume_from:
            print(f"=== Resumed training from step {trainer.state.global_step} ===")
        else:
            print(
                f"=== Started fresh training, now at step {trainer.state.global_step} ==="
            )


if __name__ == "__main__":
    parser = TrlParser((Config, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )
    main(grpo_config=grpo_config, model_config=model_config)
