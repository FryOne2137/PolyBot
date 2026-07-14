"""Run reproducible matches without writing games, replay buffers or logs."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..data.polyenv_features import capture_flags
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.belief_mcts import BeliefMCTS


def _new_env(seed: int, training: TrainingConfig) -> object:
    try:
        from PolyEnv import Bardur, GameEnv, Imperius
    except ImportError as error:
        raise RuntimeError("PolyEnv is required for evaluation") from error
    return GameEnv(
        seed=seed,
        map_size=training.map_size,
        players=(Bardur, Imperius),
        map_type=training.map_type,
    )


def evaluate_matches(
    model_a: PolicyValueBeliefNet,
    device: torch.device,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    model_b: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Evaluate A against random play or B on fixed seeds with greedy MCTS.

    Model A alternates between player slots 0 and 1 to remove first-player
    bias.  The function keeps all game state in memory and only returns a
    compact terminal summary.
    """
    model_a.eval()
    if model_b is not None:
        model_b.eval()
    planner_a = BeliefMCTS(model_a, device, mcts)
    planner_b = BeliefMCTS(model_b, device, mcts) if model_b is not None else None
    rng = np.random.default_rng(training.seed)
    totals: Dict[str, float] = {
        "games": float(games),
        "model_a_wins": 0.0,
        "model_a_losses": 0.0,
        "draws": 0.0,
        "model_a_city_captures": 0.0,
        "model_a_capital_captures": 0.0,
        "opponent_city_captures": 0.0,
        "opponent_capital_captures": 0.0,
        "model_a_particles": 0.0,
        "model_a_decisions": 0.0,
        "moves": 0.0,
    }
    opponent_name = "model_b" if model_b is not None else "random"

    with tqdm(total=games, desc=f"evaluate A vs {opponent_name}", unit="game", dynamic_ncols=True) as progress:
        for game_index in range(games):
            env = _new_env(training.seed + game_index, training)
            model_a_player = game_index % 2
            moves = 0
            winner = -1
            while not env.is_done() and moves < training.max_moves:
                packet = env.model_request_numpy()
                player = int(packet["obs"].get("current_player", env.current_player()))
                is_a_turn = player == model_a_player
                if is_a_turn:
                    decision = planner_a.decide(env)
                    action_id = decision.action_id
                    action_row = decision.action_row
                    totals["model_a_particles"] += decision.valid_particles
                    totals["model_a_decisions"] += 1.0
                elif planner_b is not None:
                    decision = planner_b.decide(env)
                    action_id = decision.action_id
                    action_row = decision.action_row
                else:
                    action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
                    action_row = int(rng.integers(len(action_ids)))
                    action_id = int(action_ids[action_row])
                city, capital = capture_flags(packet, action_row)
                ok, _, _, winner, _ = env.step_fast(action_id)
                if not ok:
                    raise RuntimeError("evaluation selected an action rejected by PolyEnv")
                prefix = "model_a" if is_a_turn else "opponent"
                if city:
                    totals[prefix + "_city_captures"] += 1.0
                if capital:
                    totals[prefix + "_capital_captures"] += 1.0
                moves += 1

            totals["moves"] += moves
            if winner == model_a_player:
                totals["model_a_wins"] += 1.0
            elif winner >= 0:
                totals["model_a_losses"] += 1.0
            else:
                totals["draws"] += 1.0
            progress.update(1)
            progress.set_postfix(
                wins=int(totals["model_a_wins"]),
                captures=int(totals["model_a_city_captures"]),
            )

    totals["model_a_win_rate"] = totals["model_a_wins"] / max(1, games)
    totals["mean_moves"] = totals["moves"] / max(1, games)
    totals["mean_model_a_particles"] = (
        totals["model_a_particles"] / max(1, totals["model_a_decisions"])
    )
    return totals
