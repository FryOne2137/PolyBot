"""Reproducible process-level random-number setup for training and evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and PyTorch from one non-negative 31-bit value.

    Native PolyEnv instances still receive their own explicit seeds.  This
    helper controls model initialisation, dropout and belief-particle sampling,
    which previously remained random even when the CLI ``--seed`` was fixed.
    """
    resolved = int(seed) & 0x7FFF_FFFF
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    return resolved
