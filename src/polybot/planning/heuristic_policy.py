"""Network-free legal-action teacher used for bootstrap and anchor games."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..data.polyenv_features import strategic_action_bias


def teacher_policy(packet: Dict[str, object], temperature: float = 0.35) -> np.ndarray:
    """Return a soft strategic policy over the packet's legal action order."""
    bias = strategic_action_bias(packet)
    scaled = bias / max(temperature, 1.0e-4)
    weights = np.exp(np.clip(scaled - scaled.max(), -30.0, 0.0))
    return (weights / weights.sum()).astype(np.float32)


def choose_teacher_action(packet: Dict[str, object], rng: np.random.Generator) -> int:
    """Sample one legal action from the teacher policy for diverse openings."""
    probabilities = teacher_policy(packet)
    return int(rng.choice(len(probabilities), p=probabilities))
