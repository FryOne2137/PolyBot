"""In-memory self-play examples and bounded replay storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np


@dataclass
class TrainingExample:
    """One decision, with a full-map label kept separate from model inputs."""

    map_tokens: np.ndarray
    global_features: np.ndarray
    event_features: np.ndarray
    action_categorical: np.ndarray
    action_continuous: np.ndarray
    policy_target: np.ndarray
    full_map_target: np.ndarray
    hidden_mask: np.ndarray
    player_id: int
    shaped_reward: float = 0.0
    value_target: float = 0.0


class ReplayBuffer:
    """Fixed-size uniformly sampled training history."""

    def __init__(self, capacity: int) -> None:
        self._examples: Deque[TrainingExample] = deque(maxlen=capacity)

    def extend(self, examples: List[TrainingExample]) -> None:
        self._examples.extend(examples)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        priority_fraction: float = 0.0,
    ) -> List[TrainingExample]:
        if len(self._examples) < batch_size:
            raise ValueError("not enough replay examples for this batch")
        rows = list(self._examples)
        priority_count = int(round(batch_size * float(np.clip(priority_fraction, 0.0, 1.0))))
        priority_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.shaped_reward >= 0.5 or abs(row.value_target) >= 0.5
            ],
            dtype=np.int64,
        )
        chosen: List[int] = []
        if priority_indices.size:
            count = min(priority_count, priority_indices.size)
            chosen.extend(int(index) for index in rng.choice(priority_indices, size=count, replace=False))
        remaining = batch_size - len(chosen)
        available = np.setdiff1d(np.arange(len(rows), dtype=np.int64), np.asarray(chosen, dtype=np.int64))
        chosen.extend(int(index) for index in rng.choice(available, size=remaining, replace=False))
        indices = np.asarray(chosen, dtype=np.int64)
        return [self._examples[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._examples)
