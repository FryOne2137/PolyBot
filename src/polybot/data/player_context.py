"""Observer-owned state and public score features, with explicit unknown masks.

Keep the first nine global fields and 23 categorical map fields stable for old
replay. Population is an auxiliary continuous map field, not a new belief
classification target. No opponent technologies or tribes enter this schema.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

TECH_COUNT = 25
PLAYER_SLOTS = 16
TRIBE_COUNT = 12
LEGACY_GLOBAL_SIZE = 9
CONTEXT_SIZE = 126
GLOBAL_SIZE = LEGACY_GLOBAL_SIZE + CONTEXT_SIZE
POPULATION_COLUMN = 23
UNKNOWN_POPULATION = -32768  # distinguish legacy unknown from real negative population
MAP_TYPE_IDS = {"lakes": 0, "drylands": 1, "water_world": 2}
NATIVE_CONTEXT_FIELDS = (
    "own_techs", "own_tribe", "initial_player_count", "player_scores",
    "player_scores_mask", "player_status", "map_type_id",
)


def canonical_map_type(value: Any) -> str:
    name = str(value).strip().lower().replace(" ", "_")
    if name == "waterworld":
        name = "water_world"
    if name not in MAP_TYPE_IDS:
        raise ValueError(f"unsupported map type: {value!r}")
    return name


def polyenv_map_type(value: str, enum_type: Any) -> Any:
    """Use the same three-way map identity for native pool construction."""
    names = {"lakes": "Lakes", "drylands": "Drylands", "water_world": "WaterWorld"}
    return getattr(enum_type, names[canonical_map_type(value)])


def pad_global_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == GLOBAL_SIZE:
        return values
    if values.shape[-1] != LEGACY_GLOBAL_SIZE:
        raise ValueError(f"global features must have 9 legacy or {GLOBAL_SIZE} fields")
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, CONTEXT_SIZE)])


def pad_map_tokens(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] == 24:
        return values
    if values.shape[-1] != 23:
        raise ValueError("map tokens must have 23 legacy or 24 population fields")
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)],
                  constant_values=UNKNOWN_POPULATION)


def native_player_context(batch: Mapping[str, Any], map_type: str | None = None) -> np.ndarray:
    state = np.asarray(batch["state"])
    leading = state.shape[:-1]
    if "player_context" in batch:
        result = np.asarray(batch["player_context"], dtype=np.float32).copy()
        if result.shape != (*leading, CONTEXT_SIZE):
            raise ValueError("player_context does not match state batch dimensions")
    else:
        result = np.zeros((*leading, CONTEXT_SIZE), dtype=np.float32)
        if "own_techs" in batch:
            techs = np.asarray(batch["own_techs"])
            if techs.shape != (*leading, TECH_COUNT) or np.any(~np.isin(techs, (-1, 0, 1))):
                raise ValueError("own_techs must have shape [...,25] with -1/0/1 values")
            known = (techs >= 0).all(axis=-1)
            result[..., :25] = (techs > 0) * known[..., None]
            result[..., 25] = known
        if "own_tribe" in batch:
            tribes = np.asarray(batch["own_tribe"])
            if tribes.shape != leading or np.any((tribes < -1) | (tribes > TRIBE_COUNT)):
                raise ValueError("own_tribe has an invalid shape or id")
            result[..., 26:38] = tribes[..., None] == np.arange(1, 13)
            result[..., 38] = tribes >= 1
        if "initial_player_count" in batch:
            counts = np.asarray(batch["initial_player_count"])
            if counts.shape != leading or np.any((counts != -1) & ((counts < 2) | (counts > PLAYER_SLOTS))):
                raise ValueError("initial_player_count must be 2..16 or -1")
            result[..., 39] = np.maximum(counts, 0) / PLAYER_SLOTS
            result[..., 40] = counts >= 2
        players = state[..., 1]
        valid = (players >= 0) & (players < PLAYER_SLOTS)
        result[..., 41:57] = (players[..., None] == np.arange(PLAYER_SLOTS)) & valid[..., None]
        result[..., 57] = valid
        # The active observer is known to be in play. The score mask denotes
        # original participants, never opponent alive/eliminated status.
        result[..., 58:74] = result[..., 41:57] * (state[..., 2] == 0)[..., None]
        result[..., 74:90] = result[..., 58:74]
        if "player_status" in batch:
            status = np.asarray(batch["player_status"])
            if status.shape != (*leading, PLAYER_SLOTS) or np.any(~np.isin(status, (-1, 0, 1))):
                raise ValueError("player_status must have shape [...,16] with -1/0/1 values")
            known = status >= 0
            result[..., 58:74] = np.where(known, status == 1, result[..., 58:74])
            result[..., 74:90] = np.maximum(known, result[..., 74:90])
        if "player_scores" in batch:
            scores = np.asarray(batch["player_scores"], dtype=np.float32)
            mask = np.asarray(batch.get("player_scores_mask", np.zeros_like(scores)), dtype=bool)
            if scores.shape != (*leading, PLAYER_SLOTS) or mask.shape != scores.shape:
                raise ValueError("public score arrays must have shape [...,16]")
            if not np.isfinite(scores[mask]).all():
                raise ValueError("public scores must be finite")
            result[..., 90:106] = np.arcsinh(np.where(mask, scores, 0) / 1000.0)
            result[..., 106:122] = mask
    if "map_type_id" in batch:
        types = np.asarray(batch["map_type_id"])
        if types.shape != leading or np.any(~np.isin(types, (-1, 0, 1, 2))):
            raise ValueError("map_type_id must be -1, 0, 1, or 2")
        result[..., 122:125] = types[..., None] == np.arange(3)
        result[..., 125] = types >= 0
    if map_type is not None:
        missing = result[..., 125] == 0
        result[..., 122 + MAP_TYPE_IDS[canonical_map_type(map_type)]] = np.where(
            missing, 1, result[..., 122 + MAP_TYPE_IDS[canonical_map_type(map_type)]]
        )
        result[..., 125] = 1
    return result


def context_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    state = np.zeros((1, 11), dtype=np.int32)
    state[0, 1] = int(obs.get("perspective_player", obs.get("current_player", obs.get("player", -1))))
    state[0, 2] = bool(obs.get("game_over", False))
    batch: dict[str, Any] = {"state": state}
    for name in NATIVE_CONTEXT_FIELDS:
        if name in obs:
            batch[name] = np.asarray(obs[name])[None]
    if "own_techs" not in batch and "techs" in obs:
        ids = np.asarray(obs["techs"], dtype=np.int64).reshape(-1)
        if np.any((ids < 0) | (ids >= TECH_COUNT)):
            raise ValueError("owned technology ids must be in 0..24")
        bits = np.zeros((1, TECH_COUNT), dtype=np.int32)
        bits[0, ids] = 1
        batch["own_techs"] = bits
    return native_player_context(batch, obs.get("map_type"))[0]


def flatten_player_context(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Flatten per-observer SelfPlayPool packets without mixing perspectives."""
    states = np.asarray(batch["state"])
    return {"state": states.reshape(-1, states.shape[-1]),
            "player_context": native_player_context(batch).reshape(-1, CONTEXT_SIZE)}
