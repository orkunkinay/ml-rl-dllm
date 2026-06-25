#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cluster-safe run state, checkpoint, and progress helpers."""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import time
import tempfile
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments

from common.memory import get_cuda_memory_stats
from common.memory import log_cuda_memory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    )


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def iter_jsonl(path: str | Path):
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_host_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


def sanitize_run_component(value: Any) -> str:
    text = str(value).replace("/", "_").replace(" ", "_")
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep).strip("_")


def deterministic_run_name(config: Any) -> str:
    model_path = getattr(config, "model_path", "model")
    if "LLaDA" in model_path:
        model = "llada"
    elif "Dream" in model_path:
        model = "dream"
    else:
        model = sanitize_run_component(Path(str(model_path)).name or "model")
    bl = getattr(config, "block_length", "na")
    alpha = getattr(config, "alpha_compute_reward", "na")
    seed = getattr(config, "seed", "na")
    return f"paper_{model}_bl{bl}_alpha_{alpha}_seed_{seed}"


def prepare_local_run_dir(
    config: Any,
    resume: str | None,
    overwrite: bool = False,
    run_root: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    run_name = run_name or deterministic_run_name(config)
    run_dir = Path(run_root) / sanitize_run_component(run_name)
    exists_with_contents = run_dir.exists() and any(run_dir.iterdir())
    if exists_with_contents and overwrite:
        shutil.rmtree(run_dir)
        exists_with_contents = False
    if exists_with_contents and not overwrite and resume != "auto":
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --resume auto to continue "
            "or --overwrite to replace intentionally."
        )
    for child in ("checkpoints", "logs", "outputs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_hf_checkpoint(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    candidates = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir() or path.name == "checkpoint-best":
            continue
        try:
            step = int(path.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def _checkpoint_weight_files(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            payload = json.load(f)
        shard_names = sorted(set(payload.get("weight_map", {}).values()))
        return [checkpoint_dir / name for name in shard_names]

    single_path = checkpoint_dir / "model.safetensors"
    if single_path.exists():
        return [single_path]

    return sorted(checkpoint_dir.glob("*.safetensors"))


def _is_readable_safetensors_file(path: Path) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            list(handle.keys())
        return True
    except Exception:
        return False


def _is_valid_hf_checkpoint(checkpoint_dir: Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return False
    if not (checkpoint_dir / "trainer_state.json").exists():
        return False

    weight_files = _checkpoint_weight_files(checkpoint_dir)
    if not weight_files:
        return False

    return all(path.exists() and _is_readable_safetensors_file(path) for path in weight_files)


def resolve_resume_checkpoint(resume: str | None, run_dir: str | Path) -> Path | None:
    if not resume or str(resume).lower() in {"false", "none", "no"}:
        return None
    if resume == "auto":
        run_dir = Path(run_dir)
        candidates = []
        for path in run_dir.glob("checkpoint-*"):
            if not path.is_dir() or path.name == "checkpoint-best":
                continue
            try:
                step = int(path.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, path))

        for _, checkpoint_dir in sorted(candidates, reverse=True):
            if _is_valid_hf_checkpoint(checkpoint_dir):
                return checkpoint_dir
            warnings.warn(
                f"Skipping unreadable checkpoint at {checkpoint_dir}; it appears to be incomplete or corrupted."
            )
        return None

    path = Path(resume)
    if path.name.endswith(".pt") and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hf_path = payload.get("hf_checkpoint_path")
        if hf_path:
            path = Path(hf_path)

    if path.is_dir() and not _is_valid_hf_checkpoint(path):
        raise ValueError(
            f"Resolved resume checkpoint {path} is not a complete Hugging Face checkpoint."
        )

    return path


def write_progress(
    run_dir: str | Path,
    status: str,
    *,
    alpha: float | None = None,
    seed: int | None = None,
    epoch: float | int | None = None,
    global_step: int | None = None,
    total_steps: int | None = None,
    completed_fraction: float | None = None,
    last_checkpoint: str | None = None,
    latest_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    progress = {
        "status": status,
        "alpha": alpha,
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "completed_fraction": completed_fraction,
        "last_checkpoint": last_checkpoint,
        "latest_metrics": latest_metrics or {},
        "updated_at": utc_now(),
    }
    if extra:
        progress.update(extra)
    try:
        atomic_write_json(Path(run_dir) / "progress.json", progress)
    except Exception as exc:
        warnings.warn(f"Failed to write progress.json: {exc}")


def checkpoint_metadata(
    *,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    config: Any,
    state: Any,
    latest_metrics: dict[str, Any] | None,
    hf_checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    unwrapped = model.module if hasattr(model, "module") else model
    payload: dict[str, Any] = {
        "policy_model_state_dict": (
            unwrapped.state_dict() if unwrapped is not None else None
        ),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None
        ),
        "gradient_scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None and hasattr(scaler, "state_dict")
            else None
        ),
        "epoch": getattr(state, "epoch", None),
        "global_step": getattr(state, "global_step", None),
        "alpha": getattr(config, "alpha_compute_reward", None),
        "seed": getattr(config, "seed", None),
        "config": config_to_dict(config),
        "rng_state": capture_rng_state(),
        "latest_metrics": latest_metrics or {},
        "wall_clock_timestamp": time.time(),
        "time": utc_now(),
        "git_commit": get_git_commit(),
        "host": get_host_info(),
        "hf_checkpoint_path": str(hf_checkpoint_path) if hf_checkpoint_path else None,
    }
    return payload


def completed_jsonl_sample_ids(path: str | Path) -> set[str]:
    completed: set[str] = set()
    for row in iter_jsonl(path) or []:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            completed.add(str(sample_id))
    return completed


def jsonl_is_nonempty_parseable(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return any(True for _ in iter_jsonl(path))
    except Exception:
        return False


class ClusterStateCallback(TrainerCallback):
    """Write sidecar checkpoints, metrics.jsonl, and progress.json for clusters."""

    def __init__(self, run_dir: str | Path, config: Any):
        self.run_dir = Path(run_dir)
        self.config = config
        self.latest_metrics: dict[str, Any] = {}
        self.last_checkpoint: str | None = None

    def _total_steps(self, state: TrainerState) -> int | None:
        return state.max_steps if state.max_steps and state.max_steps > 0 else None

    def _completed_fraction(self, state: TrainerState) -> float | None:
        total = self._total_steps(state)
        if not total:
            return None
        return min(1.0, state.global_step / total)

    def _memory_enabled(self) -> bool:
        return bool(getattr(self.config, "log_memory", False))

    def _memory_interval(self) -> int:
        return max(1, int(getattr(self.config, "memory_log_interval", 50) or 50))

    def _reset_peak(self) -> bool:
        return bool(getattr(self.config, "reset_memory_peak_each_log", False))

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        write_progress(
            self.run_dir,
            "running",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=state.global_step,
            total_steps=self._total_steps(state),
            completed_fraction=self._completed_fraction(state),
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not state.is_world_process_zero or not logs:
            return
        self.latest_metrics.update(
            {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in logs.items()
                if isinstance(v, (int, float, str)) or hasattr(v, "item")
            }
        )
        if self._memory_enabled():
            self.latest_metrics["memory"] = get_cuda_memory_stats(reset_peak=False)
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "alpha": getattr(self.config, "alpha_compute_reward", None),
            "seed": getattr(self.config, "seed", None),
            "lr": logs.get("learning_rate"),
            "time": utc_now(),
            **self.latest_metrics,
        }
        append_jsonl(self.run_dir / "metrics.jsonl", row)
        write_progress(
            self.run_dir,
            "running",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=state.global_step,
            total_steps=self._total_steps(state),
            completed_fraction=self._completed_fraction(state),
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero or not self._memory_enabled():
            return
        step = state.global_step
        if step != 1 and step % self._memory_interval() != 0:
            return
        stats = log_cuda_memory(
            prefix=f"train step={step}",
            reset_peak=self._reset_peak(),
        )
        self.latest_metrics["memory"] = stats
        write_progress(
            self.run_dir,
            "running",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=step,
            total_steps=self._total_steps(state),
            completed_fraction=self._completed_fraction(state),
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
        )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        if self._memory_enabled():
            self.latest_metrics["memory"] = log_cuda_memory(
                prefix=f"train before checkpoint step={state.global_step}",
                reset_peak=self._reset_peak(),
            )
        hf_checkpoint = kwargs.get("hf_checkpoint_path") or (
            Path(args.output_dir) / f"checkpoint-{state.global_step}"
        )
        update_latest = kwargs.get("update_latest", True)
        alias_name = kwargs.get("alias_name")
        payload = checkpoint_metadata(
            model=kwargs.get("model"),
            optimizer=kwargs.get("optimizer"),
            scheduler=kwargs.get("lr_scheduler"),
            scaler=kwargs.get("scaler"),
            config=self.config,
            state=state,
            latest_metrics=self.latest_metrics,
            hf_checkpoint_path=hf_checkpoint,
        )
        ckpt_dir = self.run_dir / "checkpoints"
        step_path = ckpt_dir / (
            alias_name or f"checkpoint_step_{state.global_step}.pt"
        )
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        atomic_torch_save(payload, step_path)
        if update_latest:
            atomic_torch_save(payload, latest_path)
            self.last_checkpoint = str(latest_path.relative_to(self.run_dir))
        else:
            self.last_checkpoint = str(step_path.relative_to(self.run_dir))
        write_progress(
            self.run_dir,
            "running",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=state.global_step,
            total_steps=self._total_steps(state),
            completed_fraction=self._completed_fraction(state),
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        if self._memory_enabled():
            self.latest_metrics["memory"] = log_cuda_memory(
                prefix=f"train end step={state.global_step}",
                reset_peak=self._reset_peak(),
            )
        write_progress(
            self.run_dir,
            "completed",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=state.global_step,
            total_steps=self._total_steps(state),
            completed_fraction=1.0,
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
        )

    def save_emergency_checkpoint(self, trainer: Any, signum: int | None = None) -> Path:
        state = trainer.state
        step = getattr(state, "global_step", 0)
        checkpoint_dir = self.run_dir / f"checkpoint-emergency-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(checkpoint_dir))
        state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        payload = checkpoint_metadata(
            model=getattr(trainer, "model", None),
            optimizer=getattr(trainer, "optimizer", None),
            scheduler=getattr(trainer, "lr_scheduler", None),
            scaler=getattr(trainer, "scaler", None),
            config=self.config,
            state=state,
            latest_metrics=self.latest_metrics,
            hf_checkpoint_path=checkpoint_dir,
        )
        latest_path = self.run_dir / "checkpoints" / "checkpoint_latest.pt"
        emergency_path = (
            self.run_dir / "checkpoints" / f"checkpoint_emergency_{step}.pt"
        )
        atomic_torch_save(payload, emergency_path)
        atomic_torch_save(payload, latest_path)
        self.last_checkpoint = str(latest_path.relative_to(self.run_dir))
        write_progress(
            self.run_dir,
            "interrupted",
            alpha=getattr(self.config, "alpha_compute_reward", None),
            seed=getattr(self.config, "seed", None),
            epoch=state.epoch,
            global_step=step,
            total_steps=self._total_steps(state),
            completed_fraction=self._completed_fraction(state),
            last_checkpoint=self.last_checkpoint,
            latest_metrics=self.latest_metrics,
            extra={"signal": signum},
        )
        return emergency_path
