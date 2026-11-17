"""Compare turn-reply search with native MCTS on a fixed paired arena."""

from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from ..infrastructure.device import select_device
from ..infrastructure.randomness import seed_everything
from ..planning.turn_reply import TurnReplySearch, _outcome, _step_world
from ..training.routed_self_play_pool import evaluate_matches_pool
from .matches import _new_env


def _play_turn_reply_games(search, training, seeds, *, max_envs):
    records = {}
    stats = dict(turns=0, scored_turns=0, simulated_actions=0, unfinished_branches=0,
                 accepted_particles=0, comparable_particles=0)
    with tqdm(total=len(seeds), desc="evaluate turn-reply search vs raw opponent", unit="game") as progress:
        for start in range(0, len(seeds), max_envs):
            batch = seeds[start:start + max_envs]
            games = [
                _new_env(seed, training, visible_action_history=search.visible_action_history)
                for seed in batch
            ]
            sides = (start + np.arange(len(batch))) % 2
            previous_players = np.full(len(batch), -1)
            active = set(range(len(batch)))
            for _ in range(training.max_moves):
                if not active:
                    break
                actions = {}
                policy_rows = ([], [])
                for row in sorted(active):
                    player = games[row].current_player()
                    if player != sides[row]:
                        policy_rows[1].append(row)
                    elif player == previous_players[row]:
                        policy_rows[0].append(row)
                    else:
                        decision = search.decide(games[row])
                        actions[row] = decision.action_id
                        stats["turns"] += 1
                        stats["scored_turns"] += int(decision.comparable_particles > 0)
                        for name in ("simulated_actions", "unfinished_branches", "accepted_particles", "comparable_particles"):
                            stats[name] += getattr(decision, name)
                    previous_players[row] = player
                for opponent, rows in enumerate(policy_rows):
                    if rows:
                        chosen = search.policy_actions([games[row] for row in rows], opponent=bool(opponent))
                        actions.update(zip(rows, chosen, strict=True))
                for row, action in actions.items():
                    winner = _step_world(games[row], action)
                    if winner is not None:
                        records[batch[row], int(sides[row])] = (1.0 + _outcome(winner, int(sides[row]))) / 2
                        active.remove(row)
                        progress.update(1)
            progress.update(len(active))
    stats["policy_only_turns"] = stats["turns"] - stats["scored_turns"]
    return records, stats


def evaluate_turn_reply_benchmark(
    model, opponent, training, mcts, games, *,
    num_envs, num_threads, visible_action_history, max_pending_leaves,
    candidates, max_rollout_actions, belief_retries,
):
    """Keep map seeds, seat swaps, opponent and real-game limits fixed."""
    seeds = [training.seed + index // 2 for index in range(games)]
    baseline = {}
    invalid_maps = []
    batch_size = num_envs - num_envs % 2

    def run_mcts(schedule):
        def record(game):
            if not game["truncated"]:
                baseline[game["map_seed"], game["model_a_player"]] = game["score_a"]

        result = evaluate_matches_pool(
            model, opponent, training, mcts, len(schedule), model_b_uses_mcts=False,
            num_threads=num_threads, visible_action_history=visible_action_history,
            max_pending_leaves=max_pending_leaves, max_envs=num_envs, belief_retries=belief_retries,
            belief_action_mode="root_shield", leaf_value_scale=0.0, opponent_leaf_value_scale=1.0,
            map_seeds=schedule, replace_failed_pairs=False, model_a_label="baseline (0,1)",
            model_b_label="raw opponent", game_callback=record,
        )
        return bool(result["valid"])

    start_time = perf_counter()
    for start in range(0, games, batch_size):
        schedule = seeds[start:start + batch_size]
        if run_mcts(schedule):
            continue
        if len(schedule) == 2:
            invalid_maps.append(schedule[0])
        else:
            for index in range(0, len(schedule), 2):
                if not run_mcts(schedule[index:index + 2]):
                    invalid_maps.append(schedule[index])
    baseline_seconds = perf_counter() - start_time

    seed_everything(training.seed)
    model.eval()
    opponent.eval()
    search = TurnReplySearch(
        model, opponent, select_device(training.device),
        candidates=candidates, particles=mcts.particles, visible_action_history=visible_action_history,
        max_rollout_actions=max_rollout_actions, belief_retries=belief_retries, belief_temperature=0.0,
    )
    start_time = perf_counter()
    turn_reply, stats = _play_turn_reply_games(search, training, seeds, max_envs=batch_size)
    turn_reply_seconds = perf_counter() - start_time
    common_maps = [
        seed for seed in seeds[::2]
        if all((seed, side) in baseline and (seed, side) in turn_reply for side in (0, 1))
    ]
    arms = {}
    for name, records, seconds in (
        ("mcts_0_1", baseline, baseline_seconds), ("turn_reply", turn_reply, turn_reply_seconds),
    ):
        scores = [records[seed, side] for seed in common_maps for side in (0, 1)]
        arms[name] = {
            "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
            "paired_score": float(np.mean(scores)) if scores else None,
            "completed_games": len(records), "elapsed_seconds": seconds,
        }
    arms["turn_reply"].update(stats)
    arms["turn_reply"]["truncated_games"] = games - len(turn_reply)
    arms["mcts_0_1"]["invalid_map_seeds"] = invalid_maps
    arms["mcts_0_1"]["truncated_games"] = games - 2 * len(invalid_maps) - len(baseline)
    differences = [
        sum(turn_reply[seed, side] - baseline[seed, side] for side in (0, 1)) / 2
        for seed in common_maps
    ]
    return {
        "valid": bool(common_maps), "baseline_simulations": mcts.simulations,
        "turn_candidates": candidates, "turn_max_rollout_actions": max_rollout_actions,
        "planned_games_per_arm": games, "scheduled_map_seeds": seeds[::2],
        "compared_games_per_arm": 2 * len(common_maps), "compared_map_seeds": common_maps,
        "arms": arms, "paired_score_delta": float(np.mean(differences)) if differences else None,
    }
