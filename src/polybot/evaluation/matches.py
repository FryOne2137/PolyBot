"""Run reproducible matches without writing games, replay buffers or logs."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..data.polyenv_features import capture_flags, packet_to_tensors
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.belief_mcts import BeliefMCTS


def _paired_seed_and_slot(base_seed: int, game_index: int) -> tuple[int, int]:
    """Use the same map for consecutive games and swap the candidate slot."""
    return int(base_seed) + int(game_index) // 2, int(game_index) % 2


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
    game_callback: Optional[Callable[[Dict[str, float | int]], None]] = None,
) -> Dict[str, float]:
    """Evaluate A against random play or B on fixed seeds with greedy MCTS.

    Model A alternates between player slots 0 and 1 to remove first-player
    bias.  The function keeps all game state in memory and only returns a
    compact terminal summary.
    """
    seed_everything(training.seed)
    model_a.eval()
    if model_b is not None:
        model_b.eval()
    planner_a = BeliefMCTS(model_a, device, mcts, map_size=training.map_size, map_type=training.map_type)
    planner_b = BeliefMCTS(model_b, device, mcts, map_size=training.map_size, map_type=training.map_type) if model_b is not None else None
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

    game_scores = []
    with tqdm(total=games, desc=f"evaluate A vs {opponent_name}", unit="game", dynamic_ncols=True) as progress:
        for game_index in range(games):
            # Evaluate every generated map twice with player slots swapped.
            # This controls map-specific first-player bias instead of merely
            # alternating slots across two unrelated maps.
            map_seed, model_a_player = _paired_seed_and_slot(training.seed, game_index)
            env = _new_env(map_seed, training)
            before = totals.copy()
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
                game_scores.append(1.0)
            elif winner >= 0:
                totals["model_a_losses"] += 1.0
                game_scores.append(0.0)
            else:
                totals["draws"] += 1.0
                game_scores.append(0.5)
            if game_callback is not None:
                game_callback({
                    "game_index": game_index,
                    "map_seed": map_seed,
                    "model_a_player": model_a_player,
                    "winner_player": int(winner),
                    "score_a": game_scores[-1],
                    "moves": moves,
                    "model_a_city_captures": int(
                        totals["model_a_city_captures"] - before["model_a_city_captures"]
                    ),
                    "model_a_capital_captures": int(
                        totals["model_a_capital_captures"] - before["model_a_capital_captures"]
                    ),
                    "opponent_city_captures": int(
                        totals["opponent_city_captures"] - before["opponent_city_captures"]
                    ),
                    "opponent_capital_captures": int(
                        totals["opponent_capital_captures"] - before["opponent_capital_captures"]
                    ),
                    "model_a_decisions": int(
                        totals["model_a_decisions"] - before["model_a_decisions"]
                    ),
                    "model_a_particles": float(
                        totals["model_a_particles"] - before["model_a_particles"]
                    ),
                    "truncated": int(winner < 0 and not env.is_done()),
                })
            progress.update(1)
            progress.set_postfix(
                wins=int(totals["model_a_wins"]),
                captures=int(totals["model_a_city_captures"]),
            )

    totals["model_a_win_rate"] = totals["model_a_wins"] / max(1, games)
    totals["evaluation_seed"] = float(training.seed)
    totals["model_a_score"] = float(np.mean(game_scores)) if game_scores else 0.0
    # The two slot-swapped games share a map and are therefore not independent.
    # Estimate uncertainty across map pairs rather than individual games.
    map_scores = [
        float(np.mean(game_scores[index:index + 2]))
        for index in range(0, len(game_scores), 2)
    ]
    totals["evaluation_maps"] = float(len(map_scores))
    totals["score_standard_error"] = (
        float(np.std(map_scores, ddof=1) / np.sqrt(len(map_scores)))
        if len(map_scores) > 1 else 0.0
    )
    totals["score_lower_95"] = max(
        0.0, totals["model_a_score"] - 1.96 * totals["score_standard_error"]
    )
    totals["mean_moves"] = totals["moves"] / max(1, games)
    totals["mean_model_a_particles"] = (
        totals["model_a_particles"] / max(1, totals["model_a_decisions"])
    )
    return totals


@torch.inference_mode()
def _direct_policy_action(
    model: PolicyValueBeliefNet,
    packet: Dict[str, object],
    device: torch.device,
) -> int:
    """Choose a greedy legal action without belief completion or MCTS."""
    tensors = packet_to_tensors(packet, device)
    output = model(
        tensors.map_tokens,
        tensors.global_features,
        tensors.action_categorical,
        tensors.action_continuous,
        tensors.action_mask,
        tensors.event_features,
        action_history=tensors.action_history,
        action_history_mask=tensors.action_history_mask,
        include_belief=False,
    )
    return int(output.policy_logits[0].argmax().item())


def evaluate_direct_matches(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    device: torch.device,
    training: TrainingConfig,
    games: int,
    game_callback: Optional[Callable[[Dict[str, float | int]], None]] = None,
) -> Dict[str, float]:
    """Paired greedy-policy arena used before MCTS is allowed to train."""
    if games < 2 or games % 2:
        raise ValueError("direct evaluation games must be positive and even")
    seed_everything(training.seed)
    model_a.eval()
    model_b.eval()
    totals: Dict[str, float] = {
        "games": float(games),
        "model_a_wins": 0.0,
        "model_a_losses": 0.0,
        "draws": 0.0,
        "model_a_city_captures": 0.0,
        "model_a_capital_captures": 0.0,
        "opponent_city_captures": 0.0,
        "opponent_capital_captures": 0.0,
        "model_a_decisions": 0.0,
        "moves": 0.0,
        "terminated_games": 0.0,
        "truncated_games": 0.0,
        "evaluation_seed": float(training.seed),
    }
    scores: list[float] = []
    with tqdm(
        total=games,
        desc="evaluate direct policy A vs model_b",
        unit="game",
        dynamic_ncols=True,
    ) as progress:
        for game_index in range(games):
            map_seed, model_a_player = _paired_seed_and_slot(
                training.seed, game_index
            )
            env = _new_env(map_seed, training)
            winner = -1
            moves = 0
            before = totals.copy()
            while not env.is_done() and moves < training.max_moves:
                packet = env.model_request_numpy()
                player = int(
                    packet["obs"].get("current_player", env.current_player())
                )
                is_a_turn = player == model_a_player
                action_row = _direct_policy_action(
                    model_a if is_a_turn else model_b, packet, device
                )
                action_ids = np.asarray(
                    packet["actions"]["action_id"], dtype=np.int64
                )
                city, capital = capture_flags(packet, action_row)
                ok, _, _, winner, _ = env.step_fast(int(action_ids[action_row]))
                if not ok:
                    raise RuntimeError(
                        "direct policy selected an action rejected by PolyEnv"
                    )
                prefix = "model_a" if is_a_turn else "opponent"
                totals[prefix + "_city_captures"] += float(city)
                totals[prefix + "_capital_captures"] += float(capital)
                totals["model_a_decisions"] += float(is_a_turn)
                moves += 1
            totals["moves"] += float(moves)
            truncated = bool(winner < 0 and not env.is_done())
            totals["truncated_games"] += float(truncated)
            totals["terminated_games"] += float(env.is_done())
            score = 0.5 if winner < 0 else float(winner == model_a_player)
            scores.append(score)
            if score == 1.0:
                totals["model_a_wins"] += 1.0
            elif score == 0.0:
                totals["model_a_losses"] += 1.0
            else:
                totals["draws"] += 1.0
            if game_callback is not None:
                game_callback({
                    "game_index": game_index,
                    "map_seed": map_seed,
                    "model_a_player": model_a_player,
                    "winner_player": int(winner),
                    "score_a": score,
                    "moves": moves,
                    "model_a_city_captures": int(
                        totals["model_a_city_captures"]
                        - before["model_a_city_captures"]
                    ),
                    "model_a_capital_captures": int(
                        totals["model_a_capital_captures"]
                        - before["model_a_capital_captures"]
                    ),
                    "opponent_city_captures": int(
                        totals["opponent_city_captures"]
                        - before["opponent_city_captures"]
                    ),
                    "opponent_capital_captures": int(
                        totals["opponent_capital_captures"]
                        - before["opponent_capital_captures"]
                    ),
                    "truncated": int(truncated),
                })
            progress.update(1)
            progress.set_postfix(
                wins=int(totals["model_a_wins"]),
                draws=int(totals["draws"]),
            )
    totals["model_a_win_rate"] = totals["model_a_wins"] / max(1.0, float(games))
    totals["model_a_score"] = float(np.mean(scores)) if scores else 0.0
    map_scores = [
        float(np.mean(scores[index:index + 2]))
        for index in range(0, len(scores), 2)
    ]
    totals["evaluation_maps"] = float(len(map_scores))
    totals["score_standard_error"] = (
        float(np.std(map_scores, ddof=1) / np.sqrt(len(map_scores)))
        if len(map_scores) > 1
        else 0.0
    )
    totals["score_lower_95"] = max(
        0.0,
        totals["model_a_score"] - 1.96 * totals["score_standard_error"],
    )
    totals["mean_moves"] = totals["moves"] / max(1.0, float(games))
    totals["completion_rate"] = totals["terminated_games"] / max(
        1.0, float(games)
    )
    totals["model_a_city_captures_per_game"] = totals[
        "model_a_city_captures"
    ] / max(1.0, float(games))
    return totals
