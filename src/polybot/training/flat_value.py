"""Head-only critic fitting guarded by outcomes of the evaluated policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .flat_experiments import FlatImprovements, isolated_rng


VALUE_PHASES = ("opening", "mid", "late")


@dataclass(frozen=True)
class ValuePosition:
    observation: Mapping[str, np.ndarray]
    return_target: float
    has_terminal_value: bool = True
    game_key: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class EncodedValuePosition:
    features: np.ndarray
    turn: int
    return_target: float
    game_key: tuple[Any, ...]
    phase: str | None = None


def _turn(example: ValuePosition | EncodedValuePosition) -> int:
    if isinstance(example, EncodedValuePosition):
        return example.turn
    return int(np.asarray(example.observation["state"]).reshape(-1)[0])


def _phase(example: ValuePosition | EncodedValuePosition) -> str:
    if isinstance(example, EncodedValuePosition) and example.phase is not None:
        return example.phase
    turn = _turn(example)
    return VALUE_PHASES[0 if turn < 20 else 1 if turn < 60 else 2]


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
        game_key = (game.get("map_size"), game.get("map_type"), tuple(game.get("player_tribes", ())),
                    int(game.get("map_seed", self.seed + key // 2)),
                    int(game.get("model_a_player", key % 2)))
        self.examples.extend(ValuePosition(position, outcome, game_key=game_key)
                             for position in positions)


def _states(model, examples, training):
    if isinstance(examples[0], EncodedValuePosition):
        return torch.as_tensor(
            np.stack([example.features for example in examples]),
            device=next(model.parameters()).device,
        )
    from .vector_population import _collate_public_observations
    from .vector_teacher import _vector_tensors
    from dataclasses import replace
    import math
    groups = {}
    for i, example in enumerate(examples):
        tiles = len(example.observation["map_tokens"])
        groups.setdefault(tiles, []).append(i)
    states, indexes = [], []
    for tiles, rows in groups.items():
        batch = _collate_public_observations([examples[i] for i in rows])
        local = replace(training, map_size=math.isqrt(tiles))
        tensors = _vector_tensors(batch, next(model.parameters()).device, local)
        with torch.no_grad():
            state, encoded_tiles = model.encode_map(
                tensors["map_tokens"], tensors["global_features"],
                tensors["event_features"], tensors["action_history"], tensors["action_history_mask"],
            )
            _, state = model.action_encoder(state, encoded_tiles, tensors["map_tokens"],
                tensors["action_categorical"], tensors["action_continuous"], tensors["action_mask"])
        states.append(state.detach())
        indexes.extend(rows)
    stacked = torch.cat(states)
    return stacked[torch.as_tensor(np.argsort(indexes), device=stacked.device)]


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
        "prediction_mean": float(predictions.mean()),
        "outcome_mean": float(targets.mean()),
    }


def evaluate_value_head(model, training, examples: Sequence, *, batch_size: int = 128) -> dict[str, Any]:
    """Evaluate one frozen checkpoint; do not reuse collection-time predictions."""
    if not examples:
        return {"valid": False, "count": 0, "games": 0}
    model.eval()
    with isolated_rng(), torch.no_grad():
        predictions = torch.cat([
            model.value_head(_states(model, examples[start:start + batch_size], training)).squeeze(-1).float()
            for start in range(0, len(examples), batch_size)
        ]).cpu().numpy()
    targets = np.asarray([e.return_target for e in examples], dtype=np.float32)
    return _metrics_by_phase(predictions, targets, examples)


def _metrics_by_phase(predictions, targets, examples):
    metrics = value_metrics(predictions, targets)
    keys = {getattr(e, "game_key", None) for e in examples} - {None}
    metrics["games"] = len(keys)
    phases = np.asarray([_phase(example) for example in examples])
    for phase in VALUE_PHASES:
        mask = phases == phase
        metrics[phase] = value_metrics(predictions[mask], targets[mask])
    return metrics


def fit_value_head(
    model, training, replay: Sequence, validation: Sequence,
    settings: FlatImprovements, *, seed: int, validation_source: str = "explicit",
    keep_best_update: bool = False,
) -> dict[str, Any]:
    """Only the critic head changes; validation never supplies a gradient.

    The caller supplies held-out games of the same policy that is being fitted.
    Only the head changes, so those action trajectories remain applicable.
    A rejected actor must receive its own saved validation, never the rejected
    policy's labels. Validation positions never become training examples.
    When a later game benchmark will compare the trained head, callers can
    select the best finite update even if its offline RMSE is worse than before.
    """
    result = {"updates": 0, "kept": False, "train_positions": len(replay),
              "validation_positions": len(validation), "validation_source": validation_source}
    if not settings.value_updates or len(replay) < settings.value_batch_size or len(validation) < 8:
        return {**result, "reason": "insufficient_terminal_data"}
    if validation_source != "explicit" and len({getattr(e, "game_key", None) for e in validation} - {None}) < 2:
        return {**result, "reason": "insufficient_terminal_games"}
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
            return _metrics_by_phase(predictions, targets, validation)
        before = measure()
        if not before["valid"]:
            return {**result, "reason": "nonfinite_reference", "before": before}
        start_state = deepcopy(model.value_head.state_dict())
        rng = np.random.default_rng(seed ^ 0x43524954)
        rows = list(replay)
        # Sample phases evenly when present. The validation retains its
        # original distribution and reports each phase separately.
        groups = {}
        for index, example in enumerate(rows):
            groups.setdefault(_phase(example), []).append(index)
        buckets = list(groups.values())
        # The encoder is frozen. Reuse identical sampled features when trying
        # a smaller head learning rate rather than replaying the expensive net.
        batches = []
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
            batches.append((state, target))
        best_state, after = start_state, before
        best_step = 0
        attempted_updates = 0
        selected_lr = settings.value_learning_rate
        attempts = 0
        for attempt in range(settings.value_fit_backoffs + 1):
            attempts += 1
            model.value_head.load_state_dict(start_state)
            lr = settings.value_learning_rate * 0.5 ** attempt
            optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=lr, weight_decay=0.0)
            for step, (state, target) in enumerate(batches, 1):
                loss = F.mse_loss(model.value_head(state).squeeze(-1).float(), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
                if not torch.isfinite(norm):
                    break
                optimizer.step()
                attempted_updates += 1
                measured = measure()
                if measured["valid"] and (
                    (keep_best_update and best_step == 0) or measured["rmse"] < after["rmse"] - 1e-8
                ):
                    after, best_state = measured, deepcopy(model.value_head.state_dict())
                    best_step, selected_lr = step, lr
            if best_step and after["rmse"] < before["rmse"] - 1e-8:
                break
        kept = best_step > 0
        model.value_head.load_state_dict(best_state)
        result.update(updates=attempted_updates, selected_updates=best_step,
                      attempts=attempts, learning_rate=selected_lr)
        selection_improved = after["rmse"] < before["rmse"] - 1e-8
    reason = "validation_improved" if selection_improved else "best_updated_head" if kept else "value_regression"
    return {**result, "before": before, "after": after, "kept": kept,
            "selection_improved": selection_improved, "reason": reason}
