"""Stable schema for PolyEnv model packets.

Keeping categorical limits and field order here makes the neural model and data
conversion agree without duplicating magic indexes.
"""

from __future__ import annotations

from typing import Sequence


# Largest valid value in the first 23 categorical map-token columns.
# PolyEnv's 24th field, own_city_population, is a signed continuous auxiliary
# input (not a belief head). The model accepts 23-field legacy observations.
MAP_FEATURE_MAX: Sequence[int] = (
    1, 1, 100, 16, 127, 100, 1023, 3, 16, 1, 1, 16, 32, 32,
    4, 1023, 16, 100, 6, 4, 16, 100, 1023,
)

ACTION_CATEGORICAL_FIELDS: Sequence[str] = (
    "type_id",
    "source_index",
    "target_index",
    "unit_id",
    "city",
    "tech",
    "building",
    "spawn_type",
    "upgrade",
    "tile_action",
    "unit_upgrade",
    "affordable",
)

ACTION_CONTINUOUS_FIELDS: Sequence[str] = (
    "cost_stars",
    "stars_before",
    "damage_dealt",
    "damage_received",
)

EVENT_SUMMARY_SIZE = 12
VISIBLE_ACTION_HISTORY_FEATURES = 12
