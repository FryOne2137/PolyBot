"""Durable, memory-bounded storage for critical MCTS reanalysis positions."""

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
    map_size: int = 11
    map_type: str = "lakes"
    visible_action_history: int = 0
    refreshes: int = 0


def _array_bytes(value: object) -> int:
    return int(value.nbytes) if isinstance(value, np.ndarray) else 0


def reanalyse_position_bytes(position: ReanalysePosition) -> int:
    """Conservative footprint, including variable actions and action history."""
    example = position.example
    fields = (
        "map_tokens", "global_features", "event_features",
        "action_categorical", "action_continuous", "policy_target",
        "full_map_target", "hidden_mask", "action_history",
        "action_history_mask", "engine_action_constraint_tiles",
    )
    arrays = sum(_array_bytes(getattr(example, name, None)) for name in fields)
    # Python integers in an action prefix need more than eight bytes.  The
    # constant covers both dataclasses and scalar metadata.
    return int(arrays + len(position.action_prefix) * 36 + 1024)


def freeze_training_example(example: TrainingExample) -> TrainingExample:
    """Freeze replay arrays before safely sharing one example with reanalysis."""
    for name in (
        "map_tokens", "global_features", "event_features",
        "action_categorical", "action_continuous", "policy_target",
        "full_map_target", "hidden_mask", "action_history",
        "action_history_mask", "engine_action_constraint_tiles",
    ):
        value = getattr(example, name, None)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return example


class PersistentReanalyseBuffer:
    """Bounded atomic replay with independent item and byte limits.

    Version 3 writes each position as an independent pickle record. This
    avoids constructing a whole replay-sized Python list during load. Old
    monolithic files are deliberately ignored: reading them could recreate the
    memory spike this storage is intended to prevent.
    """

    FORMAT_VERSION = 3

    def __init__(
        self, path: Path, capacity: int, max_bytes: int = 64 * 1024**2
    ) -> None:
        if capacity < 1:
            raise ValueError("reanalysis replay capacity must be positive")
        if max_bytes < 1:
            raise ValueError("reanalysis replay max_bytes must be positive")
        self.path = Path(path)
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self.positions: list[ReanalysePosition] = []
        self.bytes_used = 0
        self.rejected_oversize = 0
        self.rejected_invalid = 0
        self.replaced_duplicates = 0
        self.evicted = 0

    @staticmethod
    def _key(position: ReanalysePosition) -> tuple[object, ...]:
        return (
            int(position.map_seed),
            tuple(int(action) for action in position.action_prefix),
            int(position.player_id),
            int(position.map_size),
            str(position.map_type),
            int(position.visible_action_history),
        )

    @staticmethod
    def _valid(position: ReanalysePosition) -> bool:
        example = position.example
        return bool(
            isinstance(example, TrainingExample)
            and int(position.map_seed) > 0
            and int(position.map_size) > 0
            and isinstance(position.action_prefix, tuple)
            and example.map_tokens.ndim == 2
            and example.full_map_target.ndim == 2
            and example.map_tokens.shape == example.full_map_target.shape
        )

    def _trim(self) -> None:
        while self.positions and (
            len(self.positions) > self.capacity or self.bytes_used > self.max_bytes
        ):
            removed = self.positions.pop(0)
            self.bytes_used -= reanalyse_position_bytes(removed)
            self.evicted += 1

    def add(self, position: ReanalysePosition) -> bool:
        """Admit one row immediately, evicting oldest rows if needed."""
        if not self._valid(position):
            self.rejected_invalid += 1
            return False
        size = reanalyse_position_bytes(position)
        if size > self.max_bytes:
            self.rejected_oversize += 1
            return False
        key = self._key(position)
        for index, previous in enumerate(self.positions):
            if self._key(previous) == key:
                self.bytes_used -= reanalyse_position_bytes(previous)
                del self.positions[index]
                self.replaced_duplicates += 1
                break
        position.example = freeze_training_example(position.example)
        self.positions.append(position)
        self.bytes_used += size
        self._trim()
        return True

    def load(self) -> int:
        """Load V3 incrementally, rejecting old monolithic files safely."""
        if not self.path.is_file():
            return 0
        try:
            with self.path.open("rb") as handle:
                header = pickle.load(handle)
                if (
                    not isinstance(header, dict)
                    or header.get("version") != self.FORMAT_VERSION
                ):
                    # Do not overwrite a potentially valuable V2 replay
                    # after refusing its monolithic load. Start a bounded V3
                    # sibling instead; the old artifact remains recoverable.
                    upgraded = self.path.with_name(
                        self.path.stem + "-v3" + self.path.suffix
                    )
                    if upgraded.is_file():
                        self.path = upgraded
                        return self.load()
                    self.path = upgraded
                    return 0
                while True:
                    try:
                        row = pickle.load(handle)
                    except EOFError:
                        break
                    if isinstance(row, ReanalysePosition):
                        self.add(row)
        except (MemoryError, OSError, pickle.UnpicklingError, AttributeError):
            return 0
        return len(self.positions)

    def extend(self, positions: Iterable[ReanalysePosition]) -> int:
        """Add rows one-by-one so no unbounded temporary list is created."""
        accepted = 0
        for position in positions:
            accepted += int(self.add(position))
        return accepted

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
        """Persist record-by-record through an atomic replacement."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {"version": self.FORMAT_VERSION},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for position in self.positions:
                pickle.dump(position, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
