"""Training-only horizon value from authoritative economy advantage."""

from __future__ import annotations

import numpy as np


def economy_value(env: object, player_id: int) -> float:
    """Return tanh of player-vs-opponent economy advantage.

    This is a target after a trainer-imposed horizon only. It is never exposed
    to the policy, belief model or fog-of-war MCTS input.
    """
    tokens = np.asarray(env.full_map_numpy(), dtype=np.int64)
    city = tokens[:, 14] == 2
    owner = tokens[:, 16]
    level = np.maximum(tokens[:, 13], 0)
    scores: dict[int, float] = {}
    for player in (0, 1):
        observation = env.observation(player_id=player, visible_only=False)
        stars = float(observation.get("player_stars", observation.get("stars", 0.0)))
        income = float(observation.get("next_turn_star_income", observation.get("income", 0.0)))
        owned = city & (owner == player)
        scores[player] = (
            0.60 * float(owned.sum())
            + 0.15 * float(level[owned].sum())
            + 0.03 * income
            + 0.01 * stars
        )
    opponent = 1 - int(player_id)
    return float(np.tanh((scores[int(player_id)] - scores[opponent]) / 4.0))
