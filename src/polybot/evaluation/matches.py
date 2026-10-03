"""Run reproducible matches without writing games, replay buffers or logs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..data.polyenv_features import (
    capture_flags,
    native_capture_flags,
    native_batch_to_tensors,
    packet_to_tensors,
    visible_events_or_empty,
)
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.belief_mcts import BeliefMCTS


def _paired_seed_and_slot(base_seed: int, game_index: int) -> tuple[int, int]:
    """Use the same map for consecutive games and swap the candidate slot."""
    return int(base_seed) + int(game_index) // 2, int(game_index) % 2


def _new_env(
    seed: int,
    training: TrainingConfig,
    *,
    visible_action_history: int = 0,
) -> object:
    try:
        from PolyEnv import Bardur, GameEnv, Imperius
    except ImportError as error:
        raise RuntimeError("PolyEnv is required for evaluation") from error
    return GameEnv(
        seed=seed,
        map_size=training.map_size,
        players=(Bardur, Imperius),
        map_type=training.map_type,
        visible_event_history=int(visible_action_history),
    )


def evaluate_matches(
    model_a: PolicyValueBeliefNet,
    device: torch.device,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    model_b: Optional[PolicyValueBeliefNet] = None,
    game_callback: Optional[Callable[[Dict[str, float | int]], None]] = None,
    *,
    visible_action_history: int = 0,
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
    if not 0 <= int(visible_action_history) <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    planner_a = BeliefMCTS(
        model_a, device, mcts, map_size=training.map_size,
        map_type=training.map_type,
        visible_action_history=visible_action_history,
    )
    planner_b = BeliefMCTS(
        model_b, device, mcts, map_size=training.map_size,
        map_type=training.map_type,
        visible_action_history=visible_action_history,
    ) if model_b is not None else None
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
            env = _new_env(
                map_seed, training,
                visible_action_history=visible_action_history,
            )
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
    totals["visible_action_history"] = float(visible_action_history)
    return totals


@torch.inference_mode()
def _direct_policy_action(
    model: PolicyValueBeliefNet,
    packet: Dict[str, object],
    device: torch.device,
    events: Dict[str, object] | None = None,
    visible_action_history: int = 0,
) -> int:
    """Choose a greedy legal action without belief completion or MCTS."""
    tensors = packet_to_tensors(
        packet,
        device,
        events,
        visible_action_history=visible_action_history,
    )
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


def _new_paired_vector_envs(
    training: TrainingConfig,
    maps: int,
    *,
    num_threads: int | None,
    max_actions: int,
    visible_action_history: int,
) -> tuple[object, object]:
    """Create two identical seed batches for the slot-swapped arena."""
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError(
            "PolyEnv VectorGameEnv is required for direct evaluation"
        ) from error
    arguments = {
        "num_envs": maps,
        "seed": training.seed,
        "map_size": training.map_size,
        "players": (Bardur, Imperius),
        "map_type": training.map_type,
        "num_threads": num_threads,
        "max_actions": max_actions,
        "auto_reset": False,
        # Scalar GameEnv packets used by the playable bot include exact attack
        # previews. Keep the promotion arena semantically identical.
        "include_combat_preview": True,
        "visible_event_history": visible_action_history,
    }
    return VectorGameEnv(**arguments), VectorGameEnv(**arguments)


def _allocate_vector_buffers(
    env: object,
    device: torch.device,
) -> tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    """Allocate reusable native output, pinning the large model inputs on CUDA."""
    buffers: Dict[str, np.ndarray] = {}
    owners: Dict[str, torch.Tensor] = {}
    model_fields = {
        "map_tokens",
        "state",
        "action_features",
        "action_mask",
        "visible_event_features",
        "visible_event_mask",
    }
    torch_dtypes = {
        np.dtype(np.int32): torch.int32,
        np.dtype(np.uint8): torch.uint8,
    }
    for name, field in env.batch_spec().items():
        shape = tuple(int(value) for value in field["shape"])
        dtype = np.dtype(field["dtype"])
        if (
            device.type == "cuda"
            and name in model_fields
            and dtype in torch_dtypes
        ):
            owner = torch.empty(
                shape,
                dtype=torch_dtypes[dtype],
                device="cpu",
                pin_memory=True,
            )
            owners[name] = owner
            buffers[name] = owner.numpy()
        else:
            buffers[name] = np.empty(shape, dtype=dtype)
    return buffers, owners


def _packet_rows(
    tensors: Any,
    rows: np.ndarray,
) -> Dict[str, torch.Tensor]:
    index = torch.as_tensor(
        rows, dtype=torch.long, device=tensors.map_tokens.device
    )
    return {
        name: getattr(tensors, name).index_select(0, index)
        for name in (
            "map_tokens",
            "global_features",
            "action_categorical",
            "action_continuous",
            "action_mask",
            "event_features",
            "action_history",
            "action_history_mask",
        )
    }


def _join_tensor_rows(
    groups: Sequence[Dict[str, torch.Tensor]],
    name: str,
) -> torch.Tensor:
    values = [group[name] for group in groups]
    return values[0] if len(values) == 1 else torch.cat(values, dim=0)


@torch.inference_mode()
def _batched_direct_policy_actions(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    batches: Sequence[Dict[str, np.ndarray]],
    candidate_players: Sequence[int],
    active: Sequence[np.ndarray],
    device: torch.device,
    training: TrainingConfig,
) -> list[np.ndarray]:
    """Route all live rows through at most one forward per frozen model."""
    tensor_batches = [
        native_batch_to_tensors(
            batch, device, training.map_size, training.map_type
        )
        for batch in batches
    ]
    selected = [
        np.full(np.asarray(batch["state"]).shape[0], -1, dtype=np.int64)
        for batch in batches
    ]
    for model, candidate_turn in ((model_a, True), (model_b, False)):
        groups: list[Dict[str, torch.Tensor]] = []
        placements: list[tuple[int, np.ndarray]] = []
        for batch_index, (batch, candidate_player, live) in enumerate(
            zip(batches, candidate_players, active)
        ):
            players = np.asarray(batch["state"], dtype=np.int32)[:, 1]
            is_candidate = players == int(candidate_player)
            rows = np.flatnonzero(
                np.asarray(live, dtype=bool)
                & (is_candidate if candidate_turn else ~is_candidate)
            )
            if rows.size:
                groups.append(_packet_rows(tensor_batches[batch_index], rows))
                placements.append((batch_index, rows))
        if not groups:
            continue
        output = model(
            _join_tensor_rows(groups, "map_tokens"),
            _join_tensor_rows(groups, "global_features"),
            _join_tensor_rows(groups, "action_categorical"),
            _join_tensor_rows(groups, "action_continuous"),
            _join_tensor_rows(groups, "action_mask"),
            _join_tensor_rows(groups, "event_features"),
            action_history=_join_tensor_rows(groups, "action_history"),
            action_history_mask=_join_tensor_rows(
                groups, "action_history_mask"
            ),
            include_belief=False,
        )
        choices = output.policy_logits.argmax(dim=-1).cpu().numpy()
        offset = 0
        for batch_index, rows in placements:
            count = int(rows.size)
            selected[batch_index][rows] = choices[offset:offset + count]
            offset += count
    return selected


def _vector_capture_flags(
    batch: Dict[str, np.ndarray],
    selected_rows: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the shared native capture classifier."""
    return native_capture_flags(batch, selected_rows, active)


def evaluate_direct_matches(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    device: torch.device,
    training: TrainingConfig,
    games: int,
    game_callback: Optional[Callable[[Dict[str, float | int]], None]] = None,
    *,
    model_a_label: str = "policy A",
    model_b_label: str = "model B",
    visible_action_history: int = 0,
    num_threads: int | None = None,
    max_actions: int = 1024,
) -> Dict[str, float]:
    """Evaluate paired greedy policies in two native vector batches.

    The two batches use identical seed sequences. Candidate slot zero lives in
    the first batch and candidate slot one in the second, preserving the old
    paired-map schedule while reducing model inference from one forward per
    game action to at most one batched forward per model and vector step.
    """
    if games < 2 or games % 2:
        raise ValueError("direct evaluation games must be positive and even")
    if not 0 <= visible_action_history <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if training.max_moves < 1:
        raise ValueError("direct evaluation max_moves must be positive")
    seed_everything(training.seed)
    model_a.eval()
    model_b.eval()
    maps = games // 2
    envs = _new_paired_vector_envs(
        training,
        maps,
        num_threads=num_threads,
        max_actions=max_actions,
        visible_action_history=visible_action_history,
    )
    allocated = [_allocate_vector_buffers(env, device) for env in envs]
    batches = [item[0] for item in allocated]
    # ``allocated`` deliberately stays alive: its tensor owners back the NumPy
    # views used by PolyEnv for pinned CUDA transfers.
    for env, batch in zip(envs, batches):
        env.reset_into(batch, seed=training.seed)

    candidate_players = (0, 1)
    done = [np.zeros(maps, dtype=bool), np.zeros(maps, dtype=bool)]
    terminated = [np.zeros(maps, dtype=bool), np.zeros(maps, dtype=bool)]
    winners = [
        np.full(maps, -1, dtype=np.int32),
        np.full(maps, -1, dtype=np.int32),
    ]
    moves = [np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)]
    candidate_decisions = [
        np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)
    ]
    candidate_city = [
        np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)
    ]
    candidate_capital = [
        np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)
    ]
    opponent_city = [
        np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)
    ]
    opponent_capital = [
        np.zeros(maps, dtype=np.int32), np.zeros(maps, dtype=np.int32)
    ]
    vector_steps = 0

    with tqdm(
        total=games,
        desc=f"evaluate direct {model_a_label} vs {model_b_label}",
        unit="game",
        dynamic_ncols=True,
    ) as progress:
        while not all(bool(np.all(item)) for item in done):
            live = [~item for item in done]
            for batch, active in zip(batches, live):
                truncated_actions = np.asarray(
                    batch["action_truncated"], dtype=bool
                )
                if np.any(truncated_actions & active):
                    required = int(np.max(np.asarray(
                        batch["total_legal_action_count"], dtype=np.int32
                    )[truncated_actions & active]))
                    raise RuntimeError(
                        "direct evaluation action capacity is too small: "
                        f"max_actions={max_actions}, required at least {required}"
                    )
            players_before = [
                np.asarray(batch["state"], dtype=np.int32)[:, 1].copy()
                for batch in batches
            ]
            selected_rows = _batched_direct_policy_actions(
                model_a,
                model_b,
                batches,
                candidate_players,
                live,
                device,
                training,
            )
            previous_completed = sum(int(item.sum()) for item in done)
            for batch_index, (
                env,
                batch,
                active,
                selected,
                candidate_player,
            ) in enumerate(zip(
                envs,
                batches,
                live,
                selected_rows,
                candidate_players,
            )):
                live_rows = np.flatnonzero(active)
                legal = np.asarray(batch["action_mask"], dtype=bool)
                if np.any(~legal[live_rows, selected[live_rows]]):
                    raise RuntimeError(
                        "direct policy selected a padded VectorGameEnv action"
                    )
                city, capital = _vector_capture_flags(
                    batch, selected, active
                )
                candidate_turn = players_before[batch_index] == candidate_player
                candidate_city[batch_index] += (
                    city & candidate_turn & active
                ).astype(np.int32)
                candidate_capital[batch_index] += (
                    capital & candidate_turn & active
                ).astype(np.int32)
                opponent_city[batch_index] += (
                    city & ~candidate_turn & active
                ).astype(np.int32)
                opponent_capital[batch_index] += (
                    capital & ~candidate_turn & active
                ).astype(np.int32)
                candidate_decisions[batch_index] += (
                    candidate_turn & active
                ).astype(np.int32)
                action_ids = np.full(maps, -1, dtype=np.int32)
                action_ids[live_rows] = np.asarray(
                    batch["action_id"], dtype=np.int32
                )[live_rows, selected[live_rows]]
                env.step_into(action_ids, batch)
                valid = np.asarray(batch["action_valid"], dtype=bool)
                if np.any(~valid[live_rows]):
                    raise RuntimeError(
                        "direct policy selected an action rejected by PolyEnv"
                    )
                moves[batch_index][live_rows] += 1
                terminal_now = (
                    np.asarray(batch["terminated"], dtype=bool) & active
                )
                limit_now = (moves[batch_index] >= training.max_moves) & active
                newly_done = terminal_now | limit_now
                terminated[batch_index] |= terminal_now
                state_winners = np.asarray(
                    batch["state"], dtype=np.int32
                )[:, 3]
                winners[batch_index][terminal_now] = state_winners[terminal_now]
                done[batch_index] |= newly_done

            vector_steps += 1
            completed = sum(int(item.sum()) for item in done)
            if completed > previous_completed:
                progress.update(completed - previous_completed)
            if completed > previous_completed or vector_steps % 10 == 0:
                completed_scores = []
                for map_index in range(maps):
                    for player in candidate_players:
                        if not done[player][map_index]:
                            continue
                        winner = int(winners[player][map_index])
                        completed_scores.append(
                            0.5 if winner < 0 else float(winner == player)
                        )
                progress.set_postfix(
                    wins=sum(score == 1.0 for score in completed_scores),
                    draws=sum(score == 0.5 for score in completed_scores),
                    losses=sum(score == 0.0 for score in completed_scores),
                    live=games - completed,
                    moves=max(int(item.max()) for item in moves),
                )

    game_rows: list[Dict[str, float | int]] = []
    scores: list[float] = []
    for map_index in range(maps):
        for model_a_player in candidate_players:
            batch_index = model_a_player
            game_index = 2 * map_index + model_a_player
            winner = int(winners[batch_index][map_index])
            score = 0.5 if winner < 0 else float(winner == model_a_player)
            scores.append(score)
            row: Dict[str, float | int] = {
                "game_index": game_index,
                "map_seed": int(training.seed) + map_index,
                "model_a_player": model_a_player,
                "winner_player": winner,
                "score_a": score,
                "moves": int(moves[batch_index][map_index]),
                "model_a_city_captures": int(
                    candidate_city[batch_index][map_index]
                ),
                "model_a_capital_captures": int(
                    candidate_capital[batch_index][map_index]
                ),
                "opponent_city_captures": int(
                    opponent_city[batch_index][map_index]
                ),
                "opponent_capital_captures": int(
                    opponent_capital[batch_index][map_index]
                ),
                "truncated": int(not terminated[batch_index][map_index]),
            }
            game_rows.append(row)
            if game_callback is not None:
                game_callback(row)

    totals: Dict[str, float] = {
        "games": float(games),
        "model_a_wins": float(sum(score == 1.0 for score in scores)),
        "model_a_losses": float(sum(score == 0.0 for score in scores)),
        "draws": float(sum(score == 0.5 for score in scores)),
        "model_a_city_captures": float(sum(
            int(row["model_a_city_captures"]) for row in game_rows
        )),
        "model_a_capital_captures": float(sum(
            int(row["model_a_capital_captures"]) for row in game_rows
        )),
        "opponent_city_captures": float(sum(
            int(row["opponent_city_captures"]) for row in game_rows
        )),
        "opponent_capital_captures": float(sum(
            int(row["opponent_capital_captures"]) for row in game_rows
        )),
        "model_a_decisions": float(sum(
            int(item.sum()) for item in candidate_decisions
        )),
        "moves": float(sum(int(item.sum()) for item in moves)),
        "terminated_games": float(sum(
            int(item.sum()) for item in terminated
        )),
        "truncated_games": float(sum(
            int((~item).sum()) for item in terminated
        )),
        "evaluation_seed": float(training.seed),
    }
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
    totals["visible_action_history"] = float(visible_action_history)
    return totals
