from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path.cwd().resolve()
if not (ROOT / "logs/last_timing_sweep_3520013_3520066/").exists() and (ROOT.parent / "logs/last_timing_sweep_3520013_3520066/").exists():
    ROOT = ROOT.parent
LOG_DIR = ROOT / "logs/last_timing_sweep_3520013_3520066/"
print(f"Reading logs from: {LOG_DIR}")

CONFIG_RE = {
    "num_generations": re.compile(r"num_generations:\s*(\d+)"),
    "per_device_train_batch_size": re.compile(r"per_device_train_batch_size:\s*(\d+)"),
    "generation_batch_size": re.compile(r"generation_batch_size:\s*(\d+)"),
    "run_name": re.compile(r"run_name:\s*(\S+)"),
    "config_path": re.compile(r"config_path:\s*(\S+)"),
    "gpu_name": re.compile(r"GPU:\s*(.+)"),
    "gpu_memory_gb": re.compile(r"GPU memory GB:\s*([0-9.]+)"),
}

CUDA_RE = re.compile(
    r"^(?P<label>.*?) CUDA memory \| .*?allocated=(?P<allocated_gb>[0-9.]+) GB "
    r"\| reserved=(?P<reserved_gb>[0-9.]+) GB "
    r"\| peak_allocated=(?P<peak_allocated_gb>[0-9.]+) GB "
    r"\| peak_reserved=(?P<peak_reserved_gb>[0-9.]+) GB"
)

TQDM_RE = re.compile(
    r"(?P<step>\d+)\/(?P<total>\d+) "
    r"\[(?P<elapsed>\d+:\d+(?::\d+)?)<(?P<remaining>[^,]+),\s*"
    r"(?P<sec_per_it>[0-9.]+)s\/it\]"
)

BEST_RE = re.compile(r"Saved checkpoint-best at step (\d+) with train reward ([0-9.eE+-]+)")
CANCEL_RE = re.compile(r"JOB\s+(?P<job_id>\d+).*?CANCELLED.*?DUE TO TIME LIMIT")
VALUE_ERROR_RE = re.compile(r"ValueError:\s*(.*)")


def parse_duration_to_seconds(value: str | None) -> float:
    if not value or not isinstance(value, str):
        return np.nan
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return np.nan
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60 + seconds
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return hours * 3600 + minutes * 60 + seconds
    return np.nan


def extract_config(out_text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, pattern in CONFIG_RE.items():
        match = pattern.search(out_text)
        if not match:
            config[key] = np.nan
            continue
        value = match.group(1).strip()
        if key in {"num_generations", "per_device_train_batch_size", "generation_batch_size"}:
            config[key] = int(value)
        elif key == "gpu_memory_gb":
            config[key] = float(value)
        else:
            config[key] = value
    return config


def parse_log_pair(out_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    err_path = out_path.with_suffix(".err")
    out_text = out_path.read_text(errors="replace")
    err_text = err_path.read_text(errors="replace") if err_path.exists() else ""
    stem = out_path.stem

    config = extract_config(out_text)
    cuda_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    tqdm_rows: list[dict[str, Any]] = []

    for line_idx, line in enumerate(out_text.splitlines(), start=1):
        match = CUDA_RE.search(line.strip())
        if match:
            row = {"run_id": stem, "line": line_idx, **match.groupdict()}
            for col in ["allocated_gb", "reserved_gb", "peak_allocated_gb", "peak_reserved_gb"]:
                row[col] = float(row[col])
            step_match = re.search(r"step=(\d+)", row["label"])
            row["step"] = int(step_match.group(1)) if step_match else np.nan
            cuda_rows.append(row)

        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                metrics = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                continue
            if isinstance(metrics, dict):
                metrics = {str(k): v for k, v in metrics.items()}
                metric_rows.append({"run_id": stem, "line": line_idx, "metric_row": len(metric_rows) + 1, **metrics})

    clean_err = err_text.replace("\r", "\n")
    for match_idx, match in enumerate(TQDM_RE.finditer(clean_err), start=1):
        row = {"run_id": stem, "tqdm_row": match_idx, **match.groupdict()}
        row["step"] = int(row["step"])
        row["total"] = int(row["total"])
        row["sec_per_it"] = float(row["sec_per_it"])
        row["elapsed_sec"] = parse_duration_to_seconds(row["elapsed"])
        row["remaining_sec"] = parse_duration_to_seconds(row["remaining"])
        tqdm_rows.append(row)

    best_checkpoints = [(int(s), float(r)) for s, r in BEST_RE.findall(out_text)]
    last_best_step, last_best_reward = best_checkpoints[-1] if best_checkpoints else (np.nan, np.nan)

    status = "completed_or_unknown"
    if "Traceback" in err_text:
        status = "failed"
    if "DUE TO TIME LIMIT" in err_text:
        status = "time_limit"

    value_error = VALUE_ERROR_RE.findall(err_text)
    failure_reason = value_error[-1] if value_error else ""
    if not failure_reason and status == "time_limit":
        failure_reason = "Cancelled due to time limit"

    last_cuda = cuda_rows[-1] if cuda_rows else {}
    last_tqdm = tqdm_rows[-1] if tqdm_rows else {}

    run = {
        "run_id": stem,
        "out_file": str(out_path.relative_to(ROOT)),
        "err_file": str(err_path.relative_to(ROOT)) if err_path.exists() else "",
        **config,
        "status": status,
        "failure_reason": failure_reason,
        "n_cuda_lines": len(cuda_rows),
        "n_metric_rows": len(metric_rows),
        "n_tqdm_rows": len(tqdm_rows),
        "last_cuda_label": last_cuda.get("label", ""),
        "last_allocated_gb": last_cuda.get("allocated_gb", np.nan),
        "last_reserved_gb": last_cuda.get("reserved_gb", np.nan),
        "last_peak_allocated_gb": last_cuda.get("peak_allocated_gb", np.nan),
        "last_peak_reserved_gb": last_cuda.get("peak_reserved_gb", np.nan),
        "last_tqdm_step": last_tqdm.get("step", np.nan),
        "last_tqdm_total": last_tqdm.get("total", np.nan),
        "last_tqdm_elapsed": last_tqdm.get("elapsed", ""),
        "last_tqdm_remaining": last_tqdm.get("remaining", ""),
        "last_tqdm_sec_per_it": last_tqdm.get("sec_per_it", np.nan),
        "last_tqdm_elapsed_sec": last_tqdm.get("elapsed_sec", np.nan),
        "last_tqdm_remaining_sec": last_tqdm.get("remaining_sec", np.nan),
        "best_checkpoint_step": last_best_step,
        "best_train_reward": last_best_reward,
        "n_best_checkpoints": len(best_checkpoints),
    }
    return run, cuda_rows, metric_rows, tqdm_rows


run_rows, cuda_rows, metric_rows, tqdm_rows = [], [], [], []
for out_path in sorted(LOG_DIR.glob("*.out")):
    run, cuda_part, metric_part, tqdm_part = parse_log_pair(out_path)
    run_rows.append(run)
    cuda_rows.extend(cuda_part)
    metric_rows.extend(metric_part)
    tqdm_rows.extend(tqdm_part)

runs = pd.DataFrame(run_rows)
cuda = pd.DataFrame(cuda_rows)
metrics = pd.DataFrame(metric_rows)
tqdm = pd.DataFrame(tqdm_rows)

# Derived run-level columns.
runs["global_batch_size_observed"] = runs["per_device_train_batch_size"]  # Logs are single-MIG/single-process runs.
runs["gbs_divisible_by_train_bs"] = (
    runs["generation_batch_size"] % runs["global_batch_size_observed"] == 0
)
runs["generation_batches_per_update"] = runs["generation_batch_size"] / runs["global_batch_size_observed"]
runs["memory_utilization_last_peak_reserved"] = runs["last_peak_reserved_gb"] / runs["gpu_memory_gb"]
runs["projected_total_sec_from_tqdm"] = runs["last_tqdm_elapsed_sec"] + runs["last_tqdm_remaining_sec"]
runs["projected_total_hours_from_tqdm"] = runs["projected_total_sec_from_tqdm"] / 3600
runs["projected_total_hours_from_sec_per_it"] = runs["last_tqdm_sec_per_it"] * runs["last_tqdm_total"] / 3600
runs["observed_progress_pct"] = 100 * runs["last_tqdm_step"] / runs["last_tqdm_total"]
runs["has_runtime"] = runs["last_tqdm_sec_per_it"].notna()
runs["has_memory"] = runs["last_peak_reserved_gb"].notna()
runs["has_quality_proxy"] = runs["best_train_reward"].notna()

# Make concise aliases for plotting.
runs = runs.rename(columns={
    "num_generations": "ng",
    "per_device_train_batch_size": "bs",
    "generation_batch_size": "gbs",
})

for frame in [cuda, metrics, tqdm]:
    if not frame.empty:
        frame[["ng", "bs", "gbs"]] = frame["run_id"].map(runs.set_index("run_id")[["ng", "bs", "gbs"]].to_dict("index")).apply(pd.Series)

print(f"Parsed {len(runs)} runs, {len(cuda)} CUDA memory rows, {len(metrics)} metric rows, {len(tqdm)} tqdm rows.")


cols = ["ng", "bs", "gbs", "projected_total_hours_from_tqdm", "last_peak_reserved_gb"]

summary = runs[cols].copy()
summary["projected_total_hours_from_tqdm"] = summary["projected_total_hours_from_tqdm"].round(2)

print(
    summary
    .sort_values(["gbs", "bs", "ng"])
    .to_string(index=False)
)
