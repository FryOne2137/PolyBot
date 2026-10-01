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
    action_history: np.ndarray | None = None  # [history, 12], public action memory
    action_history_mask: np.ndarray | None = None
    shaped_reward: float = 0.0
    value_target: float = 0.0
    # Direct-policy positions still teach the belief and value heads, but
    # must not distil a policy into itself.  Search positions keep this at 1.
    policy_loss_weight: float = 1.0
    # Trainer-imposed horizons are not game draws. Their value targets are
    # masked until a genuine terminal result (or an explicit bootstrap) is
    # available, while their policy and belief targets remain useful.
    value_loss_weight: float = 1.0
    # Mean root value returned by fog-safe particle MCTS.  It is a training
    # target bootstrap, never an additional observation exposed to the model.
    bootstrap_value: float | None = None
    # Reserve replay capacity in optimisation batches for positions where the
    # bridge/terrain relation can be learned.  This is metadata only and is
    # never exposed as a model input.
    bridge_priority: bool = False
    # PolyEnv found at least one missing or extra public action for a sampled
    # belief particle. ``engine_action_constraint_tiles`` is a fog-safe mask
    # of the public action endpoints implicated by those differences.
    engine_action_constraint_priority: bool = False
    engine_action_constraint_tiles: np.ndarray | None = None
    engine_action_constraint_score: float = 0.0
    # A high-confidence MCTS move disagreed with the raw policy at this public
    # position.  This is replay metadata, not a model input: selective MCTS
    # distillation can reserve part of a batch for the decisions most likely
    # to teach a strategically new move.
    critical_priority: bool = False
    critical_score: float = 0.0
    # A retained critical position whose MCTS target was recomputed in a later
    # league generation.  Sampling it explicitly lets expensive reanalysis
    # influence more than one optimisation batch without dominating fresh data.
    reanalysis_priority: bool = False
    # Frozen-champion value distillation is kept separate from the game/MCTS
    # target.  This lets the optimiser improve value estimates while limiting
    # catastrophic drift of the shared representation during a league rung.
    anchor_value_target: float = 0.0
    anchor_value_loss_weight: float = 0.0


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
        bridge_fraction: float = 0.0,
        critical_fraction: float = 0.0,
        reanalysis_fraction: float = 0.0,
        engine_action_constraint_fraction: float = 0.0,
    ) -> List[TrainingExample]:
        if len(self._examples) < batch_size:
            raise ValueError("not enough replay examples for this batch")
        rows = list(self._examples)
        priority_count = int(round(batch_size * float(np.clip(priority_fraction, 0.0, 1.0))))
        bridge_count = int(round(batch_size * float(np.clip(bridge_fraction, 0.0, 1.0))))
        critical_count = int(round(batch_size * float(np.clip(critical_fraction, 0.0, 1.0))))
        reanalysis_count = int(round(batch_size * float(np.clip(reanalysis_fraction, 0.0, 1.0))))
        constraint_count = int(round(batch_size * float(np.clip(engine_action_constraint_fraction, 0.0, 1.0))))
        priority_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.shaped_reward >= 0.5 or abs(row.value_target) >= 0.5
            ],
            dtype=np.int64,
        )
        bridge_indices = np.asarray(
            [index for index, row in enumerate(rows) if row.bridge_priority],
            dtype=np.int64,
        )
        critical_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.critical_priority
            ],
            dtype=np.int64,
        )
        reanalysis_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.reanalysis_priority
            ],
            dtype=np.int64,
        )
        constraint_indices = np.asarray(
            [index for index, row in enumerate(rows) if row.engine_action_constraint_priority],
            dtype=np.int64,
        )
        chosen: List[int] = []
        if reanalysis_indices.size:
            count = min(reanalysis_count, reanalysis_indices.size, batch_size)
            chosen.extend(
                int(index)
                for index in rng.choice(reanalysis_indices, size=count, replace=False)
            )
        if critical_indices.size:
            available_critical = np.setdiff1d(
                critical_indices, np.asarray(chosen, dtype=np.int64)
            )
            count = min(critical_count, available_critical.size, batch_size - len(chosen))
            chosen.extend(
                int(index)
                for index in rng.choice(available_critical, size=count, replace=False)
            )
        if bridge_indices.size:
            available_bridge = np.setdiff1d(
                bridge_indices, np.asarray(chosen, dtype=np.int64)
            )
            count = min(bridge_count, available_bridge.size, batch_size - len(chosen))
            chosen.extend(
                int(index)
                for index in rng.choice(available_bridge, size=count, replace=False)
            )
        if constraint_indices.size:
            available_constraint = np.setdiff1d(
                constraint_indices, np.asarray(chosen, dtype=np.int64)
            )
            count = min(constraint_count, available_constraint.size, batch_size - len(chosen))
            chosen.extend(
                int(index)
                for index in rng.choice(available_constraint, size=count, replace=False)
            )
        if priority_indices.size:
            available_priority = np.setdiff1d(
                priority_indices, np.asarray(chosen, dtype=np.int64)
            )
            count = min(priority_count, available_priority.size, batch_size - len(chosen))
            chosen.extend(
                int(index)
                for index in rng.choice(available_priority, size=count, replace=False)
            )
        remaining = batch_size - len(chosen)
        available = np.setdiff1d(np.arange(len(rows), dtype=np.int64), np.asarray(chosen, dtype=np.int64))
        chosen.extend(int(index) for index in rng.choice(available, size=remaining, replace=False))
        indices = np.asarray(chosen, dtype=np.int64)
        return [self._examples[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._examples)
