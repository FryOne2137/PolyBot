"""Conversion between PolyEnv packets and neural-network tensors.

The model only receives ``model_request_numpy()`` packets.  Full maps are
handled separately as supervised labels in the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from .full_actions import require_complete_actions

from .player_context import (
    canonical_map_type, context_from_obs, native_player_context,
    pad_global_features, pad_map_tokens,
)

from .token_schema import (
    ACTION_CATEGORICAL_FIELDS,
    ACTION_CONTINUOUS_FIELDS,
    EVENT_SUMMARY_SIZE,
    MAP_FEATURE_MAX,
    VISIBLE_ACTION_HISTORY_FEATURES,
)

# Dense action layout used by PolyEnv's VectorGameEnv and MctsPool.  Keeping
# this conversion here lets both native APIs feed the same policy network
# without constructing a Python ``dict`` per environment or MCTS leaf.
NATIVE_ACTION_CATEGORICAL_COLUMNS = np.asarray(
    (0, 1, 2, 10, 3, 4, 5, 6, 7, 8, 9, 14), dtype=np.int64
)
NATIVE_ACTION_CONTINUOUS_COLUMNS = np.asarray((12, 13, 15, 16), dtype=np.int64)

# ``VectorGameEnv`` exposes the fog-safe journal as 26 event columns, while
# ``SelfPlayPool`` uses a compact 12-column action-history token.  These first
# ten compact fields are direct, visibility-filtered projections of the native
# event packet.  Keeping the mapping here gives flat PPO and MCTS one shared
# history representation.
_VECTOR_EVENT_TO_ACTION_HISTORY_COLUMNS = np.asarray(
    (2, 9, 10, 3, 4, 5, 14, 15, 16, 17), dtype=np.int64
)


@dataclass
class PacketTensors:
    map_tokens: torch.Tensor
    global_features: torch.Tensor
    event_features: torch.Tensor
    action_history: torch.Tensor
    action_history_mask: torch.Tensor
    action_categorical: torch.Tensor
    action_continuous: torch.Tensor
    action_mask: torch.Tensor


def vector_event_action_history(
    batch: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a native vector event window to compact fog-safe action tokens.

    PolyEnv already removed events and fields which the current observer could
    not see.  This function only changes the representation; it never consults
    the authoritative map or reconstructs hidden source/target information.
    Both input and output remain chronological and right-aligned.
    """
    events = np.asarray(batch["visible_event_features"], dtype=np.int32)
    mask = np.asarray(batch["visible_event_mask"], dtype=bool)
    if events.ndim != 3 or events.shape[2] != 26:
        raise ValueError(
            "visible_event_features must have shape [batch, history, 26]"
        )
    if mask.shape != events.shape[:2]:
        raise ValueError(
            "visible_event_mask must match visible_event_features[:2]"
        )
    history = np.zeros(
        (*events.shape[:2], VISIBLE_ACTION_HISTORY_FEATURES), dtype=np.int32
    )
    if events.shape[1] == 0:
        return history, mask
    history[..., :10] = events[..., _VECTOR_EVENT_TO_ACTION_HISTORY_COLUMNS]

    # Match PolyEnv SelfPlayPool's detail_kind/detail_value precedence:
    # tile action, building, spawned unit, then upgrade kind.
    detail_kind = np.zeros(events.shape[:2], dtype=np.int32)
    detail_value = np.full(events.shape[:2], -1, dtype=np.int32)
    for kind, column in reversed(((1, 11), (2, 12), (3, 13), (4, 22))):
        present = events[..., column] >= 0
        detail_kind = np.where(present, kind, detail_kind)
        detail_value = np.where(present, events[..., column], detail_value)
    history[..., 10] = detail_kind
    history[..., 11] = detail_value
    history[~mask] = 0
    return history, mask


def vector_event_summary(batch: Dict[str, np.ndarray]) -> np.ndarray:
    """Compress a native visible-event window to the model's event features."""
    events = np.asarray(batch["visible_event_features"], dtype=np.int32)
    mask = np.asarray(batch["visible_event_mask"], dtype=bool)
    if events.ndim != 3 or events.shape[2] != 26:
        raise ValueError(
            "visible_event_features must have shape [batch, history, 26]"
        )
    if mask.shape != events.shape[:2]:
        raise ValueError(
            "visible_event_mask must match visible_event_features[:2]"
        )
    result = np.zeros((events.shape[0], EVENT_SUMMARY_SIZE), dtype=np.float32)
    if events.shape[1] == 0:
        return result
    result[:, 0] = np.minimum(mask.sum(axis=1), 128) / 128.0
    types = events[:, :, 2]
    for event_type in range(8):
        result[:, 1 + event_type] = (
            ((types == event_type) & mask).sum(axis=1) / 32.0
        )
    result[:, 9] = (
        np.clip(
            np.where(mask, np.maximum(events[:, :, 6], 0), 0).sum(axis=1),
            0,
            200,
        )
        / 200.0
    )
    result[:, 10] = (
        ((events[:, :, 24] > 0) & mask).sum(axis=1) / 16.0
    )
    result[:, 11] = (
        ((events[:, :, 25] > 0) & mask).sum(axis=1) / 16.0
    )
    return result


def event_action_history(
    events: Dict[str, Any] | None,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one scalar GameEnv journal into a fixed compact history window."""
    if not 0 <= max_history <= 1024:
        raise ValueError("max_history must be between 0 and 1024")
    history = np.zeros(
        (max_history, VISIBLE_ACTION_HISTORY_FEATURES), dtype=np.int32
    )
    mask = np.zeros(max_history, dtype=bool)
    if max_history == 0 or not events or "type_id" not in events:
        return history, mask
    count = int(np.asarray(events["type_id"]).reshape(-1).size)
    if count == 0:
        return history, mask

    def column(name: str, default: int = -1) -> np.ndarray:
        value = events.get(name)
        if value is None:
            return np.full(count, default, dtype=np.int32)
        result = np.asarray(value, dtype=np.int32).reshape(-1)
        if result.size != count:
            raise ValueError(
                f"visible event field {name!r} has {result.size} rows, "
                f"expected {count}"
            )
        return result

    compact = np.stack(
        [
            column("type_id", 0),
            column("actor_player"),
            column("actor_tribe"),
            column("flags", 0),
            column("source_index"),
            column("target_index"),
            column("source_unit_type"),
            column("target_unit_type"),
            column("source_observed_unit_id"),
            column("target_observed_unit_id"),
            np.zeros(count, dtype=np.int32),
            np.full(count, -1, dtype=np.int32),
        ],
        axis=-1,
    )
    for kind, name in reversed(
        ((1, "tile_action_kind"), (2, "building_type"),
         (3, "spawn_type"), (4, "unit_upgrade_kind"))
    ):
        values = column(name)
        present = values >= 0
        compact[present, 10] = kind
        compact[present, 11] = values[present]
    kept = min(max_history, count)
    history[-kept:] = compact[-kept:]
    mask[-kept:] = True
    return history, mask


def native_batch_to_tensors(
    batch: Dict[str, np.ndarray],
    device: torch.device,
    map_size: int,
    map_type: str,
) -> PacketTensors:
    """Adapt one dense VectorGameEnv/MctsPool batch to model inputs.

    The native APIs already return padded arrays.  This conversion deliberately
    has no per-leaf Python packet construction, which is essential for MCTS
    throughput.
    """
    require_complete_actions(batch)
    state = np.asarray(batch["state"], dtype=np.int32)
    # PolyEnv already emits compact int32 action features.  Do not widen this
    # entire [leaves, actions, 17] tensor on CPU: with 512 MCTS leaves and
    # 1024 actions that transient copy alone costs about 34 MiB (or 68 MiB as
    # int64) and can exhaust pinned/pageable host memory on Windows.  Only the
    # selected categorical columns are widened later, directly on the device.
    features = np.asarray(batch["action_features"], dtype=np.int32)
    count = state.shape[0]
    global_features = native_global_features(batch, map_size, map_type)
    # Keep native packets as their compact integer dtype until they reach the
    # accelerator.  SelfPlayPool can provide pinned NumPy views, so this makes
    # the large map/action copies asynchronous on CUDA instead of first making
    # an unpinned CPU ``int64`` temporary for every MCTS batch.
    non_blocking = device.type == "cuda"
    map_tokens = torch.as_tensor(batch["map_tokens"]).to(device=device, non_blocking=non_blocking).long()
    raw_features = torch.as_tensor(features).to(device=device, non_blocking=non_blocking)
    categorical_columns = torch.as_tensor(NATIVE_ACTION_CATEGORICAL_COLUMNS, dtype=torch.long, device=device)
    continuous_columns = torch.as_tensor(NATIVE_ACTION_CONTINUOUS_COLUMNS, dtype=torch.long, device=device)
    if "visible_event_features" in batch:
        history, history_mask = vector_event_action_history(batch)
        summary = vector_event_summary(batch)
    else:
        history = np.asarray(batch.get(
            "visible_action_history",
            np.empty(
                (count, 0, VISIBLE_ACTION_HISTORY_FEATURES), dtype=np.int32
            ),
        ))
        history_mask = np.asarray(batch.get(
            "visible_action_history_mask",
            np.empty((count, 0), dtype=np.uint8),
        ), dtype=bool)
        summary = np.zeros((count, EVENT_SUMMARY_SIZE), dtype=np.float32)
    return PacketTensors(
        map_tokens=map_tokens,
        global_features=torch.as_tensor(global_features, dtype=torch.float32).to(
            device=device, non_blocking=non_blocking
        ),
        # MctsPool intentionally omits journals from its hot path, in which
        # case ``summary`` is the neutral all-zero representation. VectorGameEnv
        # supplies its bounded fog-safe event window here.
        event_features=torch.as_tensor(summary).to(
            device=device, non_blocking=non_blocking
        ).float(),
        action_history=torch.as_tensor(history).to(device=device, non_blocking=non_blocking).float(),
        action_history_mask=torch.as_tensor(history_mask).to(device=device, non_blocking=non_blocking).bool(),
        action_categorical=raw_features.index_select(-1, categorical_columns).long(),
        action_continuous=raw_features.index_select(-1, continuous_columns).float(),
        action_mask=torch.as_tensor(batch["action_mask"]).to(
            device=device, non_blocking=non_blocking
        ).bool(),
    )


def native_global_features(batch: Dict[str, np.ndarray] | np.ndarray, map_size: int, map_type: str) -> np.ndarray:
    """Shared scalar layout for vector collection, replay and native leaves."""
    if not isinstance(batch, dict) and not hasattr(batch, "keys"):
        batch = {"state": batch}
    state = np.asarray(batch["state"], dtype=np.int32)
    if state.ndim != 2:
        raise ValueError("native globals expect a flattened [batch,state] packet")
    base = np.empty((len(state), 9), dtype=np.float32)
    base[:, 0] = state[:, 0] / 100.0
    base[:, 1] = state[:, 1] / 16.0
    base[:, 2] = state[:, 4] / 50.0
    base[:, 3] = state[:, 7] / 20.0
    base[:, 4] = state[:, 5] / 32.0
    base[:, 5] = state[:, 6] / 16.0
    base[:, 6] = np.maximum(state[:, 8], 0) / 1024.0
    base[:, 7] = float(map_size) / 32.0
    base[:, 8] = float(canonical_map_type(map_type) == "drylands")
    if "domain_meta" in batch:
        from .player_context import MAP_TYPE_IDS
        meta = np.asarray(batch["domain_meta"])
        base[:, 7] = meta[:, 0] / 32.0
        base[:, 8] = meta[:, 1] == MAP_TYPE_IDS["drylands"]
        context = native_player_context(batch)
        known = meta[:, 1] >= 0
        context[known, 122:125] = meta[known, 1, None] == np.arange(3)
        context[known, 125] = 1
        return np.concatenate([base, context], axis=-1)
    return np.concatenate([base, native_player_context(batch, map_type)], axis=-1)


def _scalar(value: Any, default: float = 0.0) -> float:
    """Convert PolyEnv scalar/list values to a safe numeric feature."""
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    if array.ndim == 0:
        return float(array.item())
    return float(array.size)


def global_features_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Legacy global fields plus masked observer/public context."""
    map_type = canonical_map_type(obs.get("map_type", "lakes"))
    # Scalar ``GameEnv`` exposes the city which still needs a reward choice as
    # ``pending_city_id``.  The native path stores the same id in state[8].
    # Older PolyEnv packet names remain as a compatibility fallback, but must
    # not shadow the current field.
    pending = obs.get(
        "pending_city_id",
        obs.get("pending_city_reward", obs.get("pending_city_rewards", -1)),
    )
    values = [
        _scalar(obs.get("turn")) / 100.0,
        _scalar(obs.get("current_player", obs.get("player"))) / 16.0,
        _scalar(obs.get("player_stars", obs.get("stars"))) / 50.0,
        _scalar(obs.get("next_turn_star_income", obs.get("income"))) / 20.0,
        _scalar(obs.get("owns_units", obs.get("own_units", obs.get("units")))) / 32.0,
        _scalar(obs.get("own_cities", obs.get("cities"))) / 16.0,
        # Match ``native_batch_to_tensors``: this is an identifier/presence
        # feature, not an unbounded reward magnitude.
        max(_scalar(pending), 0.0) / 1024.0,
        _scalar(obs.get("map_size")) / 32.0,
        1.0 if map_type == "drylands" else 0.0,
    ]
    return np.concatenate([np.asarray(values, dtype=np.float32), context_from_obs(obs)])


def event_summary(
    events: Dict[str, Any] | None,
    max_events: int | None = None,
) -> np.ndarray:
    """Compress the *visible* journal into fixed-size belief-model context.

    The summary intentionally has no engine unit IDs.  Passing a raw history to
    a Transformer is a useful later upgrade; this compact form keeps the first
    implementation fast enough for MCTS while retaining movement/combat signal.
    """
    result = np.zeros(EVENT_SUMMARY_SIZE, dtype=np.float32)
    if not events or "type_id" not in events:
        return result
    types = np.asarray(events["type_id"], dtype=np.int64).reshape(-1)
    if max_events is not None:
        if max_events < 0:
            raise ValueError("max_events must not be negative")
        types = types[-max_events:] if max_events else types[:0]
    result[0] = min(types.size, 128) / 128.0
    for event_type in range(8):
        result[1 + event_type] = float(np.count_nonzero(types == event_type)) / 32.0
    if "damage" in events:
        damage = np.asarray(events["damage"]).reshape(-1)
        if max_events is not None:
            damage = damage[-max_events:] if max_events else damage[:0]
        result[9] = float(np.clip(damage.clip(min=0).sum(), 0, 200)) / 200.0
    if "unit_destroyed" in events:
        destroyed = np.asarray(events["unit_destroyed"]).reshape(-1)
        if max_events is not None:
            destroyed = destroyed[-max_events:] if max_events else destroyed[:0]
        result[10] = float(np.count_nonzero(destroyed > 0)) / 16.0
    if "source_unit_destroyed" in events:
        source_destroyed = np.asarray(events["source_unit_destroyed"]).reshape(-1)
        if max_events is not None:
            source_destroyed = (
                source_destroyed[-max_events:]
                if max_events else source_destroyed[:0]
            )
        result[11] = float(np.count_nonzero(source_destroyed > 0)) / 16.0
    return result


def visible_events_or_empty(env: Any, since: int = 0) -> Dict[str, Any]:
    """Read the optional visible-event API without requiring a new PolyEnv.

    Older PolyEnv installations do not expose ``visible_events_numpy`` yet.
    Empty event features are safe: they reduce belief quality but never reveal
    hidden information or prevent policy/value training.
    """
    read_events = getattr(env, "visible_events_numpy", None)
    if read_events is None:
        return {}
    return read_events(since)


def _field(actions: Dict[str, Any], name: str, count: int) -> np.ndarray:
    value = actions.get(name)
    if value is None:
        return np.full(count, -1, dtype=np.int64)
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size != count:
        raise ValueError("action field %r has %d rows, expected %d" % (name, result.size, count))
    return result


def action_features(actions: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return one categorical and one continuous feature row per legal action."""
    if "action_id" not in actions:
        raise ValueError("PolyEnv action packet is missing action_id")
    count = int(np.asarray(actions["action_id"]).reshape(-1).size)
    categorical = np.stack([_field(actions, name, count) for name in ACTION_CATEGORICAL_FIELDS], axis=-1)
    continuous = np.stack([_field(actions, name, count) for name in ACTION_CONTINUOUS_FIELDS], axis=-1)
    return categorical, continuous.astype(np.float32)


def capture_flags(packet: Dict[str, Any], action_row: int) -> tuple[bool, bool]:
    """Return whether a legal action captures a visible city/capital.

    The action list and its visible target are part of the normal player
    observation, so this helper is safe for both MCTS and evaluation.
    """
    actions = packet["actions"]
    full_names = actions.get("type_fullname", [])
    if action_row >= len(full_names) or str(full_names[action_row]) != "tile_action:capture_city":
        return False, False
    city_ids = np.asarray(actions.get("city", []), dtype=np.int64).reshape(-1)
    if action_row >= city_ids.size or city_ids[action_row] < 0:
        return True, False
    tokens = np.asarray(packet["map_tokens"], dtype=np.int64)
    city_id = city_ids[action_row]
    capital = bool(np.any((tokens[:, 15] == city_id) & (tokens[:, 9] == 1)))
    return True, capital


def native_capture_flags(
    batch: Dict[str, np.ndarray],
    selected_rows: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return capture-city/capital flags for selected native action rows.

    A native action with tile action 11 is *found city*, not capture city.  A
    tile action 14 is a capture; its city id can then be matched against the
    already-visible map to distinguish a capital.  Keeping this in the shared
    feature adapter prevents league evaluation and direct evaluation from
    drifting into different, misleading diagnostics.
    """
    active = np.asarray(active, dtype=bool)
    rows = np.flatnonzero(active)
    city = np.zeros(active.shape[0], dtype=bool)
    capital = np.zeros(active.shape[0], dtype=bool)
    if not rows.size:
        return city, capital
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    features = np.asarray(batch["action_features"], dtype=np.int32)[
        rows, selected_rows[rows]
    ]
    city[rows] = (features[:, 0] == 8) & (features[:, 8] == 14)
    for offset in np.flatnonzero(city[rows]):
        row = int(rows[offset])
        city_id = int(features[offset, 3])
        if city_id < 0:
            continue
        tokens = np.asarray(batch["map_tokens"], dtype=np.int32)[row]
        capital[row] = bool(
            np.any((tokens[:, 15] == city_id) & (tokens[:, 9] == 1))
        )
    return city, capital


def strategic_action_bias(
    packet: Dict[str, Any],
    *,
    capture_city_bonus: float = 2.0,
    capture_capital_bonus: float = 4.0,
    found_city_bonus: float = 3.0,
    attack_bonus: float = 0.8,
    end_turn_penalty: float = 2.0,
) -> np.ndarray:
    """Small safe prior biases for productive legal actions.

    Every value comes from the visible packet and legal-action list.  This is
    a curriculum aid for early self-play, not a substitute for learned policy.
    """
    actions = packet["actions"]
    action_ids = np.asarray(actions["action_id"], dtype=np.int64).reshape(-1)
    count = action_ids.size
    bias = np.zeros(count, dtype=np.float32)
    full_names = actions.get("type_fullname", [])
    damage = np.asarray(actions.get("damage_dealt", np.zeros(count)), dtype=np.float32).reshape(-1)
    targets = np.asarray(actions.get("target_index", np.full(count, -1)), dtype=np.int64).reshape(-1)
    sources = np.asarray(actions.get("source_index", np.full(count, -1)), dtype=np.int64).reshape(-1)
    tokens = np.asarray(packet["map_tokens"], dtype=np.int64)
    player = int(packet["obs"].get("current_player", 0))
    width = int(packet.get("spec", {}).get("map_width", max(1, round(len(tokens) ** 0.5))))
    visible = tokens[:, 0] == 1
    enemy_city = visible & (tokens[:, 14] == 2) & (tokens[:, 16] >= 0) & (tokens[:, 16] != player)
    village = visible & (tokens[:, 14] == 1)
    enemy_unit = visible & (tokens[:, 4] >= 0) & (tokens[:, 3] >= 0) & (tokens[:, 3] != player)
    goal_indices = np.flatnonzero(enemy_city | village | enemy_unit)

    def distance(index: int, goals: np.ndarray) -> int:
        x, y = index % width, index // width
        goal_x, goal_y = goals % width, goals // width
        return int(np.min(np.abs(goal_x - x) + np.abs(goal_y - y)))

    for row in range(count):
        name = str(full_names[row]) if row < len(full_names) else ""
        captures_city, captures_capital = capture_flags(packet, row)
        if captures_capital:
            bias[row] += capture_capital_bonus
        elif captures_city:
            bias[row] += capture_city_bonus
        elif name == "tile_action:found_city":
            bias[row] += found_city_bonus
        elif name == "attack":
            dealt = damage[row] if row < damage.size else 0.0
            bias[row] += attack_bonus + min(max(float(dealt), 0.0), 10.0) * 0.12
        elif name in {"tile_action:ruin", "tile_action:explorer"}:
            bias[row] += 0.8
        elif name.startswith("build:") or name.startswith("upgrade_city:"):
            bias[row] += 0.45
        elif name.startswith("spawn_unit:") or name.startswith("buy_tech:"):
            bias[row] += 0.25
        elif name == "move":
            target = targets[row] if row < targets.size else -1
            if 0 <= target < len(tokens) and tokens[target, 0] == 0:
                bias[row] += 0.5
            source = sources[row] if row < sources.size else -1
            if 0 <= source < len(tokens) and 0 <= target < len(tokens) and goal_indices.size:
                improvement = distance(source, goal_indices) - distance(target, goal_indices)
                if improvement > 0:
                    # Prefer moving toward visible objectives, but preserve
                    # learned policy between equally useful paths.
                    bias[row] += min(1.2, 0.4 * improvement)

    if count > 1:
        for row in range(count):
            name = str(full_names[row]) if row < len(full_names) else ""
            if name == "end_turn":
                bias[row] -= end_turn_penalty
    return bias


def packet_to_tensors(
    packet: Dict[str, Any],
    device: torch.device,
    events: Dict[str, Any] | None = None,
    *,
    visible_action_history: int = 0,
    include_event_summary: bool = True,
) -> PacketTensors:
    """Make a batch of one, retaining the current legal-action order."""
    map_tokens = np.asarray(packet["map_tokens"], dtype=np.int64)
    if map_tokens.ndim != 2 or map_tokens.shape[1] not in (23, 24):
        raise ValueError("expected map_tokens with shape [tiles, 23 or 24]")
    categorical, continuous = action_features(packet["actions"])
    history, history_mask = event_action_history(
        events, visible_action_history
    )
    return PacketTensors(
        map_tokens=torch.as_tensor(map_tokens[None], dtype=torch.long, device=device),
        global_features=torch.as_tensor(global_features_from_obs(packet["obs"])[None], device=device),
        event_features=torch.as_tensor(
            (
                event_summary(
                    events,
                    max_events=(
                        visible_action_history
                        if visible_action_history > 0 else None
                    ),
                )
                if include_event_summary
                else np.zeros(EVENT_SUMMARY_SIZE, dtype=np.float32)
            )[None],
            dtype=torch.float32,
            device=device,
        ),
        action_history=torch.as_tensor(
            history[None], dtype=torch.float32, device=device
        ),
        action_history_mask=torch.as_tensor(
            history_mask[None], dtype=torch.bool, device=device
        ),
        action_categorical=torch.as_tensor(categorical[None], dtype=torch.long, device=device),
        action_continuous=torch.as_tensor(continuous[None], dtype=torch.float32, device=device),
        action_mask=torch.ones((1, categorical.shape[0]), dtype=torch.bool, device=device),
    )


def packets_to_tensors(
    packets: Iterable[Dict[str, Any]],
    device: torch.device,
    events: Iterable[Dict[str, Any] | None] | None = None,
    *,
    include_event_summary: bool = True,
) -> PacketTensors:
    """Pad several player-visible packets into one inference batch.

    MCTS produces positions with differing legal-action counts.  Padding them
    here lets the GPU evaluate many tree leaves in one forward pass; masks make
    the padding invisible to the policy head.  This function deliberately uses
    exactly the same player-view fields as :func:`packet_to_tensors`.
    """
    rows = list(packets)
    if not rows:
        raise ValueError("cannot batch zero model packets")
    event_rows = list(events) if events is not None else [None] * len(rows)
    if len(event_rows) != len(rows):
        raise ValueError("events must have one row per packet")
    maps = [pad_map_tokens(np.asarray(packet["map_tokens"], dtype=np.int64)) for packet in rows]
    if any(row.ndim != 2 or row.shape[1] != 24 for row in maps):
        raise ValueError("expected every map_tokens row to have shape [tiles, 23 or 24]")
    if any(row.shape != maps[0].shape for row in maps[1:]):
        raise ValueError("MCTS inference batch mixes incompatible board shapes")
    action_rows = [action_features(packet["actions"]) for packet in rows]
    max_actions = max(categorical.shape[0] for categorical, _ in action_rows)
    if max_actions == 0:
        raise ValueError("cannot evaluate a packet with no legal actions")
    categorical = np.full(
        (len(rows), max_actions, len(ACTION_CATEGORICAL_FIELDS)), -1, dtype=np.int64
    )
    continuous = np.zeros(
        (len(rows), max_actions, len(ACTION_CONTINUOUS_FIELDS)), dtype=np.float32
    )
    mask = np.zeros((len(rows), max_actions), dtype=bool)
    for index, (cats, conts) in enumerate(action_rows):
        count = cats.shape[0]
        categorical[index, :count] = cats
        continuous[index, :count] = conts
        mask[index, :count] = True
    return PacketTensors(
        map_tokens=torch.as_tensor(np.stack(maps), dtype=torch.long, device=device),
        global_features=torch.as_tensor(
            np.stack([global_features_from_obs(packet["obs"]) for packet in rows]),
            dtype=torch.float32,
            device=device,
        ),
        event_features=torch.as_tensor(
            (
                np.stack([event_summary(row) for row in event_rows])
                if include_event_summary
                else np.zeros(
                    (len(rows), EVENT_SUMMARY_SIZE), dtype=np.float32
                )
            ),
            dtype=torch.float32,
            device=device,
        ),
        action_history=torch.empty((len(rows), 0, VISIBLE_ACTION_HISTORY_FEATURES), dtype=torch.float32, device=device),
        action_history_mask=torch.empty((len(rows), 0), dtype=torch.bool, device=device),
        action_categorical=torch.as_tensor(categorical, dtype=torch.long, device=device),
        action_continuous=torch.as_tensor(continuous, dtype=torch.float32, device=device),
        action_mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def stack_examples(examples: Iterable[Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Pad variable-length legal-action lists for one optimisation batch."""
    rows: List[Any] = list(examples)
    if not rows:
        raise ValueError("cannot stack an empty batch")
    max_actions = max(int(row.action_categorical.shape[0]) for row in rows)
    batch_size = len(rows)
    tiles, features = rows[0].map_tokens.shape
    cat_dim = rows[0].action_categorical.shape[1]
    cont_dim = rows[0].action_continuous.shape[1]

    maps = np.stack([pad_map_tokens(row.map_tokens) for row in rows]).astype(np.int64)
    globals_ = np.stack([pad_global_features(row.global_features) for row in rows]).astype(np.float32)
    events_ = np.stack([row.event_features for row in rows]).astype(np.float32)
    full_maps = np.stack([pad_map_tokens(row.full_map_target) for row in rows]).astype(np.int64)
    hidden = np.stack([row.hidden_mask for row in rows]).astype(bool)
    constraint_masks = np.stack([
        np.zeros_like(row.hidden_mask, dtype=bool)
        if getattr(row, "engine_action_constraint_tiles", None) is None
        else np.asarray(row.engine_action_constraint_tiles, dtype=bool)
        for row in rows
    ])
    constraint_scores = np.asarray([
        getattr(row, "engine_action_constraint_score", 0.0) for row in rows
    ], dtype=np.float32)
    values = np.asarray([row.value_target for row in rows], dtype=np.float32)
    policy_weights = np.asarray(
        [getattr(row, "policy_loss_weight", 1.0) for row in rows],
        dtype=np.float32,
    )
    value_weights = np.asarray([row.value_loss_weight for row in rows], dtype=np.float32)
    anchor_values = np.asarray(
        [getattr(row, "anchor_value_target", 0.0) for row in rows],
        dtype=np.float32,
    )
    anchor_value_weights = np.asarray(
        [getattr(row, "anchor_value_loss_weight", 0.0) for row in rows],
        dtype=np.float32,
    )
    cats = np.full((batch_size, max_actions, cat_dim), -1, dtype=np.int64)
    conts = np.zeros((batch_size, max_actions, cont_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_actions), dtype=bool)
    policies = np.zeros((batch_size, max_actions), dtype=np.float32)
    max_history = max(0 if row.action_history is None else int(row.action_history.shape[0]) for row in rows)
    histories = np.zeros((batch_size, max_history, VISIBLE_ACTION_HISTORY_FEATURES), dtype=np.float32)
    history_masks = np.zeros((batch_size, max_history), dtype=bool)
    for index, row in enumerate(rows):
        length = row.action_categorical.shape[0]
        cats[index, :length] = row.action_categorical
        conts[index, :length] = row.action_continuous
        masks[index, :length] = True
        policies[index, :length] = row.policy_target
        if row.action_history is not None:
            history_length = min(max_history, row.action_history.shape[0])
            # Python interprets ``-0:`` as ``0:``, i.e. the full dimension.
            # Old replay can contain a real, shaped [0, features] history;
            # it must remain an all-padding row when mixed with a newer
            # non-empty history instead of attempting a full-row broadcast.
            if history_length == 0:
                continue
            # Histories are chronological and right-aligned in live packets.
            # Keep that invariant when old 16-event replay is mixed with a
            # newer 32-event example: the newest event must use the same
            # positional embedding in every batch.
            histories[index, -history_length:] = row.action_history[-history_length:]
            if row.action_history_mask is None:
                history_masks[index, -history_length:] = True
            else:
                history_masks[index, -history_length:] = row.action_history_mask[-history_length:]

    return {
        "map_tokens": torch.as_tensor(maps, dtype=torch.long, device=device),
        "global_features": torch.as_tensor(globals_, dtype=torch.float32, device=device),
        "event_features": torch.as_tensor(events_, dtype=torch.float32, device=device),
        "action_history": torch.as_tensor(histories, dtype=torch.float32, device=device),
        "action_history_mask": torch.as_tensor(history_masks, dtype=torch.bool, device=device),
        "action_categorical": torch.as_tensor(cats, dtype=torch.long, device=device),
        "action_continuous": torch.as_tensor(conts, dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(masks, dtype=torch.bool, device=device),
        "policy_target": torch.as_tensor(policies, dtype=torch.float32, device=device),
        "policy_loss_weight": torch.as_tensor(
            policy_weights, dtype=torch.float32, device=device
        ),
        "value_target": torch.as_tensor(values, dtype=torch.float32, device=device),
        "value_loss_weight": torch.as_tensor(value_weights, dtype=torch.float32, device=device),
        "anchor_value_target": torch.as_tensor(
            anchor_values, dtype=torch.float32, device=device
        ),
        "anchor_value_loss_weight": torch.as_tensor(
            anchor_value_weights, dtype=torch.float32, device=device
        ),
        "full_map_target": torch.as_tensor(full_maps, dtype=torch.long, device=device),
        "hidden_mask": torch.as_tensor(hidden, dtype=torch.bool, device=device),
        "engine_action_constraint_mask": torch.as_tensor(
            constraint_masks, dtype=torch.bool, device=device
        ),
        "engine_action_constraint_score": torch.as_tensor(
            constraint_scores, dtype=torch.float32, device=device
        ),
    }
