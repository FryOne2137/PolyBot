"""Device selection, including Apple Metal Performance Shaders (MPS)."""

from __future__ import annotations

import os

# Must be configured before importing torch; it has no effect on non-MPS runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Return the requested device, preferring Apple Metal for ``auto``.

    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` allows PyTorch to execute a rare
    unsupported MPS operator on CPU instead of aborting a long training run.
    """

    requested = requested.lower()
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")

    mps_ready = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    cuda_ready = torch.cuda.is_available()
    if requested == "mps" and not mps_ready:
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
    if requested == "cuda" and not cuda_ready:
        raise RuntimeError("CUDA was requested but is unavailable")

    if requested == "mps" or (requested == "auto" and mps_ready):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and cuda_ready):
        return torch.device("cuda")
    return torch.device("cpu")
