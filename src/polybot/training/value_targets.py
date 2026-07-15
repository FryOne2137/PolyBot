"""Value-target construction shared by MCTS self-play trainers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..config import TrainingConfig
from .replay_buffer import TrainingExample


Transition = Tuple[int, float, Optional[TrainingExample]]


def _terminal_value(winner: object, player_id: int) -> float:
    try:
        winner_id = int(winner)
    except (TypeError, ValueError):
        return 0.0
    if winner_id < 0:
        return 0.0
    return 1.0 if winner_id == player_id else -1.0


def assign_td_lambda_targets(
    examples: Sequence[TrainingExample],
    transitions: Sequence[Transition],
    winner: object,
    training: TrainingConfig,
) -> None:
    """Assign signed TD(lambda) targets from an entire game kept in RAM.

    A self-play example is recorded immediately before an action.  The next
    recorded example is therefore a valid bootstrap state after one or more
    game transitions.  Its MCTS root value is converted to the perspective of
    the earlier player and blended with its later return.  If a data source
    has no MCTS bootstrap (teacher / legacy fallback), this naturally becomes
    a Monte-Carlo return, preserving its old behaviour.
    """
    positions = [
        (index, example)
        for index, (_, _, example) in enumerate(transitions)
        if example is not None
    ]
    if not positions:
        return

    targets: dict[int, float] = {}
    lambda_ = float(np.clip(training.td_lambda, 0.0, 1.0))
    gamma = float(np.clip(training.reward_discount, 0.0, 1.0))
    reward_weight = float(training.shaping_target_weight)

    for position_index in range(len(positions) - 1, -1, -1):
        transition_index, example = positions[position_index]
        next_position = positions[position_index + 1] if position_index + 1 < len(positions) else None
        end = next_position[0] if next_position is not None else len(transitions)

        reward_sum = 0.0
        discount = 1.0
        for actor, shaped_reward, _ in transitions[transition_index:end]:
            sign = 1.0 if int(actor) == example.player_id else -1.0
            reward_sum += discount * reward_weight * sign * float(shaped_reward)
            discount *= gamma

        if next_position is None:
            continuation = _terminal_value(winner, example.player_id)
        else:
            _, next_example = next_position
            sign = 1.0 if next_example.player_id == example.player_id else -1.0
            later_return = sign * targets[id(next_example)]
            bootstrap = next_example.bootstrap_value
            if bootstrap is None or not np.isfinite(bootstrap):
                continuation = later_return
            else:
                continuation = (1.0 - lambda_) * sign * float(bootstrap) + lambda_ * later_return

        example.value_target = float(np.clip(reward_sum + discount * continuation, -1.0, 1.0))
