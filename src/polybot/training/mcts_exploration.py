"""Exploration primitives used only while collecting self-play data.

AlphaZero adds stochasticity to the *root prior* before search, then trains on
the visit distribution produced by that search.  Keeping the operations here
separate from generic MCTS makes it hard to accidentally enable them in an
evaluation match.
"""

from __future__ import annotations

import numpy as np

from ..config import MCTSConfig


def self_play_temperature(config: MCTSConfig, move: int) -> float:
    """Return the action-sampling temperature for one self-play decision."""
    if move < 0:
        raise ValueError("move must not be negative")
    if config.temperature_drop_move <= 0 or move < config.temperature_drop_move:
        return float(config.temperature)
    return float(config.late_temperature)


def mix_dirichlet_root_prior(
    prior: np.ndarray,
    rng: np.random.Generator,
    *,
    epsilon: float,
    concentration: float,
) -> np.ndarray:
    """Mix a legal root prior with Dirichlet noise.

    ``concentration`` is the total concentration, rather than AlphaZero's
    per-action alpha.  Keeping it fixed as the legal action count changes is
    important in PolyBot, where a position can have tens or thousands of
    actions.  It gives comparable total exploration mass at every root.
    """
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 1:
        raise ValueError("root prior must be rank one")
    if prior.size == 0:
        return prior.copy()
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Dirichlet epsilon must be between zero and one")
    cleaned = np.where(np.isfinite(prior) & (prior > 0.0), prior, 0.0)
    total = float(cleaned.sum())
    if total <= 0.0:
        cleaned.fill(1.0 / cleaned.size)
    else:
        cleaned /= total
    if epsilon == 0.0 or cleaned.size == 1:
        return cleaned
    if concentration <= 0.0:
        raise ValueError("Dirichlet concentration must be positive when noise is enabled")
    alpha = np.full(cleaned.size, float(concentration) / cleaned.size, dtype=np.float64)
    noise = rng.dirichlet(alpha)
    return (1.0 - epsilon) * cleaned + epsilon * noise


def add_dirichlet_noise_to_root_logits(
    logits: np.ndarray,
    action_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    epsilon: float,
    concentration: float,
) -> int:
    """In-place root-prior perturbation for the first native MCTS wave.

    Only legal action logits are replaced with log-probabilities.  The native
    engine still receives its ordinary action mask, so padding and illegal
    actions are untouched.
    """
    values = np.asarray(logits)
    mask = np.asarray(action_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("root logits and action mask must share shape [batch, actions]")
    if epsilon == 0.0:
        return 0
    applied = 0
    for row in range(values.shape[0]):
        legal = np.flatnonzero(mask[row])
        if legal.size == 0:
            continue
        legal_logits = np.asarray(values[row, legal], dtype=np.float64)
        finite = np.where(np.isfinite(legal_logits), legal_logits, -1.0e9)
        shifted = finite - float(finite.max())
        prior = np.exp(np.clip(shifted, -80.0, 0.0))
        mixed = mix_dirichlet_root_prior(
            prior, rng, epsilon=epsilon, concentration=concentration
        )
        values[row, legal] = np.log(np.maximum(mixed, 1.0e-30)).astype(values.dtype)
        applied += 1
    return applied
