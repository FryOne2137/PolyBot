"""Terminal-outcome critic fitting with a frozen, held-out reference arena."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .flat_experiments import FlatImprovements, isolated_rng


@dataclass(frozen=True)
class ValuePosition:
    observation: Mapping[str, np.ndarray]
    return_target: float
    has_terminal_value: bool = True


@dataclass
class ValueValidationCollector:
    games: int
    capacity_per_game: int
    seed: int
    positions: dict[int, list[Mapping[str, np.ndarray]]] = field(default_factory=dict)
    seen: dict[int, int] = field(default_factory=dict)
    examples: list[ValuePosition] = field(default_factory=list)
    completed_games: int = 0
    truncated_games: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed ^ 0x56414C)

    def add(self, game: int, batch: Mapping[str, np.ndarray], row: int) -> None:
        if game >= self.games:
            return
        from .vector_population import _ppo_snapshot
        self.seen[game] = self.seen.get(game, 0) + 1
        reservoir = self.positions.setdefault(game, [])
        index = len(reservoir) if len(reservoir) < self.capacity_per_game else int(
            self.rng.integers(self.seen[game])
        )
        if index >= self.capacity_per_game:
            return
        legal = np.flatnonzero(batch["action_mask"][row])
        observation, _ = _ppo_snapshot(batch, row, int(legal[0]))
        if index == len(reservoir):
            reservoir.append(observation)
        else:
            reservoir[index] = observation

    def finish(self, game: Mapping[str, Any]) -> None:
        key = int(game["game_index"])
        if key >= self.games:
            return
        positions = self.positions.pop(key, [])
        if game.get("truncated", 0):
            self.truncated_games += 1
            return
        self.completed_games += 1
        outcome = 2.0 * float(game["score_a"]) - 1.0
        self.examples.extend(ValuePosition(position, outcome) for position in positions)


def _states(model, examples, training):
    from .vector_population import _collate_public_observations
    from .vector_teacher import _vector_tensors
    batch = _collate_public_observations(examples)
    tensors = _vector_tensors(batch, next(model.parameters()).device, training)
    with torch.no_grad():
        state, _ = model.encode_map(
            tensors["map_tokens"], tensors["global_features"],
            tensors["event_features"], tensors["action_history"],
            tensors["action_history_mask"],
        )
    return state.detach()


def value_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if not len(targets):
        return {"count": 0, "valid": False}
    error = predictions - targets
    return {
        "valid": bool(np.isfinite(predictions).all()), "count": len(targets),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "zero_rmse": float(np.sqrt(np.square(targets).mean())),
        "zero_mae": float(np.abs(targets).mean()),
    }


def fit_value_head(
    model, training, replay: Sequence, validation: Sequence,
    settings: FlatImprovements, *, seed: int,
) -> dict[str, Any]:
    """Only the critic head changes; validation never supplies a gradient.

    Validation outcomes belong to a frozen reference policy, not an oracle
    for the evolving policy. They are a stable diagnostic and a fit guard;
    independent game-strength gates remain authoritative.
    """
    result = {"updates": 0, "kept": False, "train_positions": len(replay),
              "validation_positions": len(validation)}
    if not settings.value_updates or len(replay) < settings.value_batch_size or len(validation) < 8:
        return {**result, "reason": "insufficient_terminal_data"}
    model.eval()
    device = next(model.parameters()).device
    targets = np.asarray([e.return_target for e in validation], dtype=np.float32)
    # This evaluator and optimizer own no actor graph or policy targets.
    with isolated_rng():
        features = torch.cat([
            _states(model, validation[start:start + settings.value_batch_size], training)
            for start in range(0, len(validation), settings.value_batch_size)
        ])
        def measure():
            with torch.no_grad():
                predictions = model.value_head(features).squeeze(-1).float().cpu().numpy()
            metrics = value_metrics(predictions, targets)
            for phase, lo, hi in (("opening", 0, 20), ("mid", 20, 60), ("late", 60, float("inf"))):
                mask = np.asarray([
                    lo <= int(np.asarray(e.observation["state"]).reshape(-1)[0]) < hi
                    for e in validation
                ])
                metrics[phase] = value_metrics(predictions[mask], targets[mask])
            return metrics
        before = measure()
        start_state = deepcopy(model.value_head.state_dict())
        optimizer = torch.optim.AdamW(
            model.value_head.parameters(), lr=settings.value_learning_rate,
            weight_decay=0.0,
        )
        rng = np.random.default_rng(seed ^ 0x43524954)
        rows = list(replay)
        # Sample phases evenly when present. The validation retains its
        # original distribution and reports each phase separately.
        groups = {}
        for index, example in enumerate(rows):
            turn = int(np.asarray(example.observation["state"]).reshape(-1)[0])
            groups.setdefault(0 if turn < 20 else 1 if turn < 60 else 2, []).append(index)
        buckets = list(groups.values())
        for _ in range(settings.value_updates):
            indices = [
                int(rng.choice(buckets[int(rng.integers(len(buckets)))]))
                for _ in range(settings.value_batch_size)
            ]
            subset = [rows[index] for index in indices]
            state = _states(model, subset, training)
            target = torch.as_tensor(
                [e.return_target for e in subset], device=device, dtype=torch.float32
            )
            loss = F.mse_loss(model.value_head(state).squeeze(-1).float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
            if not torch.isfinite(norm):
                model.value_head.load_state_dict(start_state)
                return {**result, "reason": "nonfinite_gradient", "before": before}
            optimizer.step()
            result["updates"] += 1
        after = measure()
        kept = after["valid"] and after["rmse"] <= before["rmse"] + 1e-8
        if not kept:
            model.value_head.load_state_dict(start_state)
    return {**result, "before": before, "after": after, "kept": kept,
            "reason": "validation_improved" if kept else "value_regression"}
