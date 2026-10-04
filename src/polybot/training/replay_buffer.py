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
    """Fixed-size training history with optional hard-example preservation."""

    def __init__(
        self,
        capacity: int,
        *,
        preserve_constraint_fraction: float = 0.0,
        preserve_bridge_fraction: float = 0.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        if not 0.0 <= preserve_constraint_fraction <= 1.0:
            raise ValueError("preserve_constraint_fraction must be in [0, 1]")
        if not 0.0 <= preserve_bridge_fraction <= 1.0:
            raise ValueError("preserve_bridge_fraction must be in [0, 1]")
        self._examples: Deque[TrainingExample] = deque(maxlen=capacity)
        self._capacity = int(capacity)
        self._preserve_constraint_fraction = float(preserve_constraint_fraction)
        self._preserve_bridge_fraction = float(preserve_bridge_fraction)
        self._rows_cache: list[TrainingExample] | None = None
        self._index_cache: dict[str, np.ndarray] | None = None

    def _invalidate_cache(self) -> None:
        self._rows_cache = None
        self._index_cache = None

    def _cached_rows_and_indices(
        self,
    ) -> tuple[list[TrainingExample], dict[str, np.ndarray]]:
        if self._rows_cache is not None and self._index_cache is not None:
            return self._rows_cache, self._index_cache
        rows = list(self._examples)
        indices = {
            "priority": np.asarray([
                index for index, row in enumerate(rows)
                if row.shaped_reward >= 0.5 or abs(row.value_target) >= 0.5
            ], dtype=np.int64),
            "bridge": np.asarray([
                index for index, row in enumerate(rows) if row.bridge_priority
            ], dtype=np.int64),
            "critical": np.asarray([
                index for index, row in enumerate(rows) if row.critical_priority
            ], dtype=np.int64),
            "reanalysis": np.asarray([
                index for index, row in enumerate(rows) if row.reanalysis_priority
            ], dtype=np.int64),
            "constraint": np.asarray([
                index for index, row in enumerate(rows)
                if row.engine_action_constraint_priority
            ], dtype=np.int64),
        }
        self._rows_cache = rows
        self._index_cache = indices
        return rows, indices

    def extend(self, examples: List[TrainingExample]) -> None:
        if not examples:
            return
        combined = list(self._examples) + list(examples)
        if len(combined) <= self._capacity:
            self._examples.extend(examples)
            self._invalidate_cache()
            return
        # FIFO silently forgets rare contradictions between a sampled world
        # and public actions. Keep recent hard examples and bridge geometry,
        # then fill the remaining slots with ordinary recent positions.
        selected: set[int] = set()
        constraint_quota = int(round(
            self._capacity * self._preserve_constraint_fraction
        ))
        for index in range(len(combined) - 1, -1, -1):
            if len(selected) >= constraint_quota:
                break
            if combined[index].engine_action_constraint_priority:
                selected.add(index)
        bridge_quota = int(round(self._capacity * self._preserve_bridge_fraction))
        bridge_selected = 0
        for index in range(len(combined) - 1, -1, -1):
            if bridge_selected >= bridge_quota:
                break
            if index not in selected and combined[index].bridge_priority:
                selected.add(index)
                bridge_selected += 1
        for index in range(len(combined) - 1, -1, -1):
            if len(selected) >= self._capacity:
                break
            selected.add(index)
        self._examples = deque(
            (combined[index] for index in sorted(selected)), maxlen=self._capacity
        )
        self._invalidate_cache()

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
        rows, cached_indices = self._cached_rows_and_indices()
        priority_count = int(round(batch_size * float(np.clip(priority_fraction, 0.0, 1.0))))
        bridge_count = int(round(batch_size * float(np.clip(bridge_fraction, 0.0, 1.0))))
        critical_count = int(round(batch_size * float(np.clip(critical_fraction, 0.0, 1.0))))
        reanalysis_count = int(round(batch_size * float(np.clip(reanalysis_fraction, 0.0, 1.0))))
        constraint_count = int(round(batch_size * float(np.clip(engine_action_constraint_fraction, 0.0, 1.0))))
        priority_indices = cached_indices["priority"]
        bridge_indices = cached_indices["bridge"]
        critical_indices = cached_indices["critical"]
        reanalysis_indices = cached_indices["reanalysis"]
        constraint_indices = cached_indices["constraint"]
        # ``np.setdiff1d`` sorts and duplicates its inputs. On a 100k belief
        # replay that happens several times per batch and can allocate enough
        # temporary memory to fail before an optimiser step. A boolean mask is
        # linear, preserves the same without-replacement semantics, and keeps
        # the hard-example fractions independent.
        available = np.ones(len(rows), dtype=bool)
        chosen: List[int] = []

        def reserve(indices: np.ndarray, requested: int) -> None:
            if requested <= 0 or len(chosen) >= batch_size or not indices.size:
                return
            candidates = indices[available[indices]]
            count = min(int(requested), int(candidates.size), batch_size - len(chosen))
            if count <= 0:
                return
            selected = np.asarray(
                rng.choice(candidates, size=count, replace=False), dtype=np.int64
            )
            available[selected] = False
            chosen.extend(int(index) for index in selected)

        reserve(reanalysis_indices, reanalysis_count)
        reserve(critical_indices, critical_count)
        reserve(bridge_indices, bridge_count)
        reserve(constraint_indices, constraint_count)
        reserve(priority_indices, priority_count)
        remaining = batch_size - len(chosen)
        if remaining:
            reserve(np.flatnonzero(available), remaining)
        indices = np.asarray(chosen, dtype=np.int64)
        return [rows[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._examples)

    def snapshot(self) -> List[TrainingExample]:
        """Return the bounded replay window in chronological order.

        Callers use this only to persist a calibration reservoir between
        short league invocations.  The examples themselves are immutable from
        the buffer's point of view, so copying the deque is sufficient.
        """
        return list(self._examples)
