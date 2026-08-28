"""Durable critical-position storage for MCTS reanalysis.

The training loop keeps normal replay in RAM.  This module deliberately keeps
only the rare, high-confidence MCTS disagreements on disk together with the
publicly reproducible game prefix that created each position.  A later league
round can recreate the position from its map seed and action ids, run search
again with the current candidate, and replace the stale policy target.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Iterable, Sequence

import numpy as np

from .replay_buffer import TrainingExample


@dataclass
class ReanalysePosition:
    """One reproducible candidate decision retained across league rounds."""

    example: TrainingExample
    map_seed: int
    action_prefix: tuple[int, ...]
    player_id: int
    refreshes: int = 0


class PersistentReanalyseBuffer:
    """Bounded, atomic on-disk store of MCTS-critical positions.

    The file is local training data, not an interchange format.  It is loaded
    only from a path selected by the person running the league, therefore
    pickle is appropriate here and preserves variable-width action tensors
    without losing their native order.
    """

    FORMAT_VERSION = 1

    def __init__(self, path: Path, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("reanalysis replay capacity must be positive")
        self.path = Path(path)
        self.capacity = int(capacity)
        self.positions: list[ReanalysePosition] = []

    def load(self) -> int:
        """Load retained positions, ignoring a buffer from an older format."""
        if not self.path.is_file():
            return 0
        try:
            with self.path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
            # A partially written or stale optional replay must never prevent
            # a league from starting.  The next successful save replaces it.
            return 0
        if not isinstance(payload, dict) or payload.get("version") != self.FORMAT_VERSION:
            return 0
        rows = payload.get("positions")
        if not isinstance(rows, list):
            return 0
        self.positions = [row for row in rows if isinstance(row, ReanalysePosition)]
        if len(self.positions) > self.capacity:
            self.positions = self.positions[-self.capacity :]
        return len(self.positions)

    def extend(self, positions: Iterable[ReanalysePosition]) -> int:
        """Append positions and retain the newest bounded history."""
        added = list(positions)
        if not added:
            return 0
        self.positions.extend(added)
        if len(self.positions) > self.capacity:
            self.positions = self.positions[-self.capacity :]
        return len(added)

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> Sequence[ReanalysePosition]:
        if count <= 0 or not self.positions:
            return ()
        selected = rng.choice(
            len(self.positions), size=min(int(count), len(self.positions)), replace=False
        )
        return tuple(self.positions[int(index)] for index in selected)

    def save(self) -> None:
        """Persist atomically so an interrupted training run keeps old data."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": self.FORMAT_VERSION, "positions": self.positions}
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
