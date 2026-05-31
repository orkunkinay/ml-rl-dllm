#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Small CUDA memory logging helpers for cluster runs."""

from __future__ import annotations

from typing import Any

import torch


def get_cuda_memory_stats(reset_peak: bool = False) -> dict[str, Any]:
    """Return CUDA memory statistics in GB, safely when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": None,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }

    device = torch.cuda.current_device()
    stats = {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    return stats


def format_cuda_memory(prefix: str = "", reset_peak: bool = False) -> str:
    stats = get_cuda_memory_stats(reset_peak=reset_peak)

    if not stats["cuda_available"]:
        return f"{prefix} CUDA unavailable"

    return (
        f"{prefix} CUDA memory | "
        f"device={stats['device_name']} | "
        f"allocated={stats['allocated_gb']:.2f} GB | "
        f"reserved={stats['reserved_gb']:.2f} GB | "
        f"peak_allocated={stats['max_allocated_gb']:.2f} GB | "
        f"peak_reserved={stats['max_reserved_gb']:.2f} GB"
    )


def log_cuda_memory(
    prefix: str = "",
    reset_peak: bool = False,
    logger=None,
) -> dict[str, Any]:
    """Print or logger.info CUDA memory stats and return the raw stats dict."""
    msg = format_cuda_memory(prefix=prefix, reset_peak=reset_peak)

    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)

    return get_cuda_memory_stats(reset_peak=False)
