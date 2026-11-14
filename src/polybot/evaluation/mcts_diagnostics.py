"""Diagnose MCTS choices and replay alternatives on paired maps."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..training.belief_recovery import BeliefRecoveryRequired
from ..training.flat_value import value_metrics
from ..training.routed_self_play_pool import (
    RoutedPoolSearch,
    _evaluation_belief_retries,
    _evaluation_search_with_recovery,
    evaluate_matches_pool,
)


def _position_fingerprint(request: Mapping[str, np.ndarray], row: int) -> str:
    """Check that replay restored the public inputs used in the original search."""
    digest = sha256()
    for key in (
        "map_tokens", "state", "own_techs", "own_tribe", "player_scores",
        "player_scores_mask", "initial_player_count", "map_seed",
        "to_play", "agent_id", "visible_action_history",
        "visible_action_history_mask",
    ):
        digest.update(np.asarray(request[key][row]).tobytes())
    legal = np.asarray(request["action_mask"][row], dtype=bool)
    for key in ("action_id", "action_features"):
        digest.update(request[key][row, legal].tobytes())
    return digest.hexdigest()


@dataclass
class _ReplayPosition:
    game: int
    seed: int
    player: int
    prefix_length: int
    fingerprint: str


@dataclass
class _RawPosition(_ReplayPosition):
    raw_action: int
    raw_value: float


@dataclass
class _Disagreement(_RawPosition):
    belief_action: int
    belief_value: float
    belief_visits: int


def _true_map_worlds(
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
) -> np.ndarray:
    # Preserve each observer's visible tokens and fog mask. Replace only
    # hidden-map hypotheses, keeping the same ISMCTS routing and leaf model.
    state_ids = request["state_id"]
    public = search.pool.all_player_belief_requests()["map_tokens"]
    truth = search.pool.belief_targets_numpy(state_ids)
    visible = public[..., :1] == 1
    completed = np.where(visible, public, truth[:, None])
    completed[..., 0] = public[..., 0]
    worlds = np.repeat(completed[:, :, None], max(1, search.mcts.particles), axis=2)
    return np.ascontiguousarray(worlds, dtype=np.int32)


def _search_submitted_maps(
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
    worlds: np.ndarray,
) -> Mapping[str, np.ndarray]:
    search.pool.submit_beliefs(request["state_id"], worlds)
    active = np.ones(search.num_envs, dtype=bool)
    return search._search_submitted_roots(request, active, 0.0, 0.0, ~active, None, None)


def _root_decisions(root: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    decisions = []
    for row, mask in enumerate(root["action_mask"]):
        legal = np.flatnonzero(mask)
        chosen = legal[root["policy"][row, legal].argmax()]
        visits = root["visit_count"][row, legal]
        decisions.append({
            "action_id": int(root["action_id"][row, chosen]),
            "root_value": float(root["root_value"][row]),
            "root_visits": int(root["root_visit_count"][row]),
            "chosen_action_visits": int(root["visit_count"][row, chosen]),
            "action_visits": dict(zip(root["action_id"][row, legal].tolist(), visits.tolist())),
        })
    return decisions


def _compare_leaf_scales(
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    worlds: list[np.ndarray] = []
    active = np.ones(search.num_envs, dtype=bool)
    hold = np.full(search.num_envs, -1, dtype=np.int32)
    state_ids = request["state_id"].copy()
    fingerprints = [_position_fingerprint(request, row) for row in range(search.num_envs)]
    generator = search._action_generator
    generator_state = generator.get_state() if generator is not None else None

    def clear_forest() -> None:
        # A checked step with every slot held clears the forest without
        # advancing any game. The next search starts with zero visits.
        search.pool.step_into_checked(hold, state_ids, request)
        if not np.array_equal(request["state_id"], state_ids) or any(
            _position_fingerprint(request, row) != fingerprint
            for row, fingerprint in enumerate(fingerprints)
        ):
            raise RuntimeError("clearing the MCTS forest changed a diagnostic position")
        if generator_state is not None:
            generator.set_state(generator_state)

    search.leaf_value_scale = 1.0
    root, _ = _evaluation_search_with_recovery(
        search, request, active, on_submitted_beliefs=worlds.append,
    )
    decisions = {"belief_value_1": _root_decisions(root)}
    clear_forest()
    search.leaf_value_scale = 0.0
    root = _search_submitted_maps(search, request, worlds[0])
    decisions["belief_value_0"] = _root_decisions(root)

    clear_forest()
    truth = _true_map_worlds(search, request)
    search.leaf_value_scale = 1.0
    root = _search_submitted_maps(search, request, truth)
    decisions["true_map_value_1"] = _root_decisions(root)
    clear_forest()
    search.leaf_value_scale = 0.0
    root = _search_submitted_maps(search, request, truth)
    decisions["true_map_value_0"] = _root_decisions(root)
    return decisions


def _search_true_maps(
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return _search_submitted_maps(search, request, _true_map_worlds(search, request))


def _replay_positions(
    search: RoutedPoolSearch,
    positions: Sequence[_ReplayPosition],
    prefixes: Mapping[int, Sequence[int]],
) -> Mapping[str, np.ndarray]:
    request = search.reset(
        np.array([position.seed for position in positions], dtype=np.uint32),
        np.array([(0, 1) if position.player == 0 else (1, 0) for position in positions], dtype=np.int32),
    )
    for move in range(max(position.prefix_length for position in positions)):
        actions = np.array([
            prefixes[position.game][move] if move < position.prefix_length else -1
            for position in positions
        ], dtype=np.int32)
        search.step_public_policy(actions, request["state_id"], request)
        if not np.all(request["action_valid"][actions >= 0]):
            raise RuntimeError("a saved game action was rejected during diagnostic replay")
    for row, position in enumerate(positions):
        if _position_fingerprint(request, row) != position.fingerprint:
            raise RuntimeError(
                f"diagnostic replay differs from seed {position.seed}, "
                f"move {position.prefix_length + 1}, player {position.player}"
            )
    return request


def _diagnostic_search(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    positions: Sequence[_ReplayPosition],
    *,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    model_b_search_top_actions: int | None,
    model_b_search_random_actions: int | None,
) -> RoutedPoolSearch:
    return RoutedPoolSearch(
        training, mcts, (model_a, model_b),
        num_envs=len(positions), num_threads=num_threads, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history, belief_retries=belief_retries,
        seed=positions[0].seed, belief_action_mode="root_shield",
        search_top_actions_by_agent=(
            mcts.search_top_actions,
            mcts.search_top_actions if model_b_search_top_actions is None else model_b_search_top_actions,
        ),
        search_random_actions_by_agent=(
            mcts.search_random_actions,
            mcts.search_random_actions if model_b_search_random_actions is None else model_b_search_random_actions,
        ),
    )


def _play_continuations(
    search: RoutedPoolSearch,
    positions: Sequence[_ReplayPosition],
    first_actions: Sequence[int],
    prefixes: Mapping[int, Sequence[int]],
    max_moves: int,
    *,
    mcts_value_scale: float | None = None,
) -> list[dict[str, Any]]:
    """Force the first move, then use raw policies or MCTS on model A's turns."""
    request = _replay_positions(search, positions, prefixes)
    moves = np.array([position.prefix_length for position in positions], dtype=np.int32)
    terminated = np.zeros(len(positions), dtype=bool)
    winners = np.full(len(positions), -1, dtype=np.int32)

    def step(actions: np.ndarray, *, searched: bool = False) -> None:
        if searched:
            search.pool.step_into_checked(actions, request["state_id"], request)
        else:
            search.step_public_policy(actions, request["state_id"], request)
        played = actions >= 0
        if not np.all(request["action_valid"][played]):
            raise RuntimeError("a continuation action was rejected on the original map")
        moves[played] += 1
        finished = played & request["terminated"].astype(bool)
        # Subsequent held steps in this slot may clear the winner field.
        winners[finished] = request["winner"][finished]
        terminated[:] |= finished

    if mcts_value_scale is not None:
        search.leaf_value_scale = mcts_value_scale
    step(np.asarray(first_actions, dtype=np.int32))
    while True:
        active = ~terminated & (moves < max_moves)
        rows = np.flatnonzero(active)
        if not rows.size:
            break
        mcts_mask = active & (request["agent_id"] == 0)
        searched = mcts_value_scale is not None and bool(mcts_mask.any())
        if searched:
            root, _ = _evaluation_search_with_recovery(search, request, active, mcts_mask=mcts_mask)
            policy, action_ids = root["policy"], root["action_id"]
        else:
            policy, _ = search.direct_policy_value(request, rows)
            action_ids = request["action_id"]
        actions = np.full(len(positions), -1, dtype=np.int32)
        actions[rows] = action_ids[rows, policy[rows].argmax(axis=1)]
        step(actions, searched=searched)

    results = []
    for row, position in enumerate(positions):
        winner = int(winners[row]) if terminated[row] else None
        results.append({
            "score_a": None if winner is None else (0.5 if winner < 0 else float(winner == position.player)),
            "winner_player": winner,
            "truncated": not bool(terminated[row]),
            "moves_from_position": int(moves[row]) - position.prefix_length,
        })
    return results


def _continuation_pair_summary(
    scores: Sequence[tuple[float | None, float | None]],
) -> dict[str, Any]:
    deltas = [one - zero for zero, one in scores if zero is not None and one is not None]
    return {
        "pairs": len(scores),
        "both_terminal": len(deltas),
        "incomplete": len(scores) - len(deltas),
        "value_1_better": sum(delta > 0.0 for delta in deltas),
        "value_0_better": sum(delta < 0.0 for delta in deltas),
        "same_outcome": sum(delta == 0.0 for delta in deltas),
        "mean_score_delta_value_1_minus_value_0": sum(deltas) / len(deltas) if deltas else None,
    }


def _play_disagreement_pairs(
    disagreements: Sequence[tuple[_ReplayPosition, dict[str, Any]]],
    prefixes: Mapping[int, Sequence[int]],
    new_search: Callable[[Sequence[_ReplayPosition]], RoutedPoolSearch],
    max_envs: int,
    max_moves: int,
) -> dict[str, Any]:
    pairs: dict[str, list[tuple[dict[int, dict[str, Any]], int, int]]] = {"belief": [], "true_map": []}
    branches = []
    for position, result in disagreements:
        continuations: dict[int, dict[str, Any]] = {}
        result["continuations"] = continuations
        actions: set[int] = set()
        for world, world_pairs in pairs.items():
            action_0 = result[f"{world}_value_0"]["action_id"]
            action_1 = result[f"{world}_value_1"]["action_id"]
            if action_0 != action_1:
                world_pairs.append((continuations, action_0, action_1))
                actions.update((action_0, action_1))
        # The same real action has the same raw continuation regardless of
        # which search produced it. Play it once and reuse the outcome.
        branches.extend((position, action, continuations) for action in sorted(actions))

    terminal_branches = 0
    for start in tqdm(range(0, len(branches), max_envs), desc="play raw continuations", unit="batch"):
        chunk = branches[start:start + max_envs]
        positions = [position for position, _, _ in chunk]
        first_actions = [action for _, action, _ in chunk]
        search = new_search(positions)
        outcomes = _play_continuations(search, positions, first_actions, prefixes, max_moves)
        for (_, action, continuations), outcome in zip(chunk, outcomes):
            continuations[action] = outcome
            terminal_branches += not outcome["truncated"]
        del search

    summary: dict[str, Any] = {
        "policy": "raw_argmax",
        "played_branches": len(branches),
        "terminal_branches": terminal_branches,
        "truncated_branches": len(branches) - terminal_branches,
    }
    for world, world_pairs in pairs.items():
        summary[world] = _continuation_pair_summary([
            (continuations[action_0]["score_a"], continuations[action_1]["score_a"])
            for continuations, action_0, action_1 in world_pairs
        ])
    return summary


def evaluate_mcts_diagnostics(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    positions_per_game: int,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    max_envs: int,
    model_a_label: str,
    model_b_label: str,
    model_b_search_top_actions: int | None = None,
    model_b_search_random_actions: int | None = None,
) -> dict[str, Any]:
    """Measure decision changes and value error on model A's belief-MCTS games."""
    if positions_per_game < 1:
        raise ValueError("positions_per_game must be positive")
    if not (0 <= training.seed <= 2**31 - 1):
        raise ValueError("diagnostic seed must be in [0, 2**31 - 1]")
    rng = np.random.default_rng(training.seed ^ 0xD1A6)
    prefixes: dict[int, list[int]] = defaultdict(list)
    samples: dict[int, list[_Disagreement]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    outcomes: dict[int, Mapping[str, Any]] = {}

    def record_decisions(indices, raw, root, actions) -> None:
        for row in np.flatnonzero(actions >= 0):
            game = int(indices[row])
            prefix = prefixes[game]
            if int(raw["agent_id"][row]) == 0:
                legal = raw["action_mask"][row].astype(bool)
                raw_action = int(raw["action_id"][row, legal][raw["policy"][row, legal].argmax()])
                belief_action = int(actions[row])
                if raw_action != belief_action:
                    seen[game] += 1
                    reservoir = samples[game]
                    index = len(reservoir) if len(reservoir) < positions_per_game else int(
                        rng.integers(seen[game])
                    )
                    if index < positions_per_game:
                        sample = _Disagreement(
                            game, int(raw["map_seed"][row]), int(raw["to_play"][row]), len(prefix),
                            _position_fingerprint(raw, row), raw_action, float(raw["root_value"][row]),
                            belief_action, float(root["root_value"][row]),
                            int(root["root_visit_count"][row]),
                        )
                        if index == len(reservoir):
                            reservoir.append(sample)
                        else:
                            reservoir[index] = sample
            prefix.append(int(actions[row]))

    def record_outcome(game) -> None:
        outcomes[int(game["game_index"])] = game

    evaluation = evaluate_matches_pool(
        model_a, model_b, training, mcts, games,
        num_threads=num_threads, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history, belief_retries=belief_retries,
        max_envs=max_envs, model_b_uses_mcts=False, replace_failed_pairs=False,
        belief_action_mode="root_shield", model_a_label=model_a_label, model_b_label=model_b_label,
        model_b_search_top_actions=model_b_search_top_actions,
        model_b_search_random_actions=model_b_search_random_actions,
        decision_callback=record_decisions, game_callback=record_outcome,
    )
    report: dict[str, Any] = {
        "belief_mcts_evaluation": evaluation,
        "disagreements_seen": sum(seen.values()),
        "positions_per_game": positions_per_game,
        "diagnostics_valid": False,
    }
    if not evaluation["valid"]:
        return report
    positions = sorted(
        (sample for game in samples.values() for sample in game),
        key=lambda sample: (sample.game, sample.prefix_length),
    )
    results = []
    seed_everything(training.seed)
    for start in tqdm(range(0, len(positions), max_envs), desc="replay true-map MCTS", unit="batch"):
        chunk = positions[start:start + max_envs]
        search = _diagnostic_search(
            model_a, model_b, training, mcts, chunk,
            num_threads=num_threads, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_retries=belief_retries,
            model_b_search_top_actions=model_b_search_top_actions,
            model_b_search_random_actions=model_b_search_random_actions,
        )
        request = _replay_positions(search, chunk, prefixes)
        root = _search_true_maps(search, request)
        for row, sample in enumerate(chunk):
            legal = root["action_mask"][row].astype(bool)
            true_action = int(root["action_id"][row, legal][root["policy"][row, legal].argmax()])
            game = outcomes[sample.game]
            results.append({
                "game": sample.game, "seed": sample.seed, "move": sample.prefix_length + 1,
                "player": sample.player, "raw_action": sample.raw_action,
                "belief_action": sample.belief_action, "true_map_action": true_action,
                "raw_value": sample.raw_value, "belief_root_value": sample.belief_value,
                "true_map_root_value": float(root["root_value"][row]),
                "belief_visits": sample.belief_visits,
                "true_map_visits": int(root["root_visit_count"][row]),
                "observed_belief_game_outcome": None if game["truncated"] else 2 * game["score_a"] - 1,
            })
        del root, request, search

    terminal = [result for result in results if result["observed_belief_game_outcome"] is not None]
    target = np.asarray([result["observed_belief_game_outcome"] for result in terminal])
    report.update(
        diagnostics_valid=bool(results),
        compared_positions=len(results),
        terminal_positions=len(terminal),
        terminal_games=len({result["game"] for result in terminal}),
        true_map_agrees_with_raw=sum(row["true_map_action"] == row["raw_action"] for row in results),
        true_map_agrees_with_belief=sum(row["true_map_action"] == row["belief_action"] for row in results),
        true_map_prefers_third_action=sum(
            row["true_map_action"] not in (row["raw_action"], row["belief_action"]) for row in results
        ),
        value_vs_observed_belief_game={
            key: value_metrics(np.asarray([result[key] for result in terminal]), target)
            for key in ("raw_value", "belief_root_value", "true_map_root_value")
        },
        positions=results,
    )
    return report


def evaluate_mcts_value_diagnostics(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    positions_per_game: int,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    max_envs: int,
    model_a_label: str,
    model_b_label: str,
    model_b_search_top_actions: int | None = None,
    model_b_search_random_actions: int | None = None,
    play_disagreement_games: bool = False,
) -> dict[str, Any]:
    """Compare leaf value scales on sampled apprentice turns from paired raw games."""
    if positions_per_game < 1:
        raise ValueError("positions_per_game must be positive")
    if not (0 <= training.seed <= 2**31 - 1):
        raise ValueError("diagnostic seed must be in [0, 2**31 - 1]")
    rng = np.random.default_rng(training.seed ^ 0xD1A6)
    seeds = [training.seed + index // 2 for index in range(games)]
    prefixes: dict[int, list[int]] = defaultdict(list)
    samples: dict[int, list[_RawPosition]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    outcomes: dict[int, Mapping[str, Any]] = {}

    def record_decisions(indices, raw, _root, actions) -> None:
        for row in np.flatnonzero(actions >= 0):
            game = int(indices[row])
            prefix = prefixes[game]
            if int(raw["agent_id"][row]) == 0:
                seen[game] += 1
                reservoir = samples[game]
                index = len(reservoir) if len(reservoir) < positions_per_game else int(
                    rng.integers(seen[game])
                )
                if index < positions_per_game:
                    position = _RawPosition(
                        game, int(raw["map_seed"][row]), int(raw["to_play"][row]), len(prefix),
                        _position_fingerprint(raw, row), int(actions[row]), float(raw["root_value"][row]),
                    )
                    if index == len(reservoir):
                        reservoir.append(position)
                    else:
                        reservoir[index] = position
            prefix.append(int(actions[row]))

    def record_outcome(game) -> None:
        outcomes[int(game["game_index"])] = game

    raw = evaluate_matches_pool(
        model_a, model_b, training, mcts, games,
        num_threads=num_threads, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history, belief_retries=belief_retries, max_envs=max_envs,
        model_a_uses_mcts=False, model_b_uses_mcts=False, map_seeds=seeds, replace_failed_pairs=False,
        model_a_label=model_a_label, model_b_label=model_b_label,
        decision_callback=record_decisions, game_callback=record_outcome,
    )
    positions = sorted(
        (position for game in samples.values() for position in game),
        key=lambda position: (position.game, position.prefix_length),
    )
    failures: list[dict[str, Any]] = []
    report = {
        "raw": raw,
        "simulations": mcts.simulations,
        "positions_per_game": positions_per_game,
        "candidate_turns": sum(seen.values()),
        "sampled_positions": len(positions),
        "raw_games": list(outcomes.values()),
        "diagnostics_valid": False,
        "failed_positions": failures,
        "positions": [],
    }
    if not raw["valid"] or raw["evaluated_map_seeds"] != seeds:
        return report

    compared = 0
    changes = {"belief_only": 0, "true_map_only": 0, "both": 0, "neither": 0}
    agreement = {"value_1": 0, "value_0": 0}
    disagreements: list[tuple[_ReplayPosition, dict[str, Any]]] = []
    seed_everything(training.seed)
    new_search = partial(
        _diagnostic_search, model_a, model_b, training, mcts,
        num_threads=num_threads, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history,
        belief_retries=_evaluation_belief_retries(belief_retries),
        model_b_search_top_actions=model_b_search_top_actions,
        model_b_search_random_actions=model_b_search_random_actions,
    )
    batches = deque(positions[start:start + max_envs] for start in range(0, len(positions), max_envs))
    with tqdm(total=len(positions), desc="compare MCTS leaf values", unit="position") as progress:
        while batches:
            chunk = batches.popleft()
            search = new_search(chunk)
            request = _replay_positions(search, chunk, prefixes)
            try:
                decisions = _compare_leaf_scales(search, request)
            except BeliefRecoveryRequired as error:
                # A failed particle in one slot can block the whole native
                # submission. Retry the other positions without that slot.
                failed = {int(row) for row in error.env_ids if 0 <= row < len(chunk)}
                if not failed:
                    failed = set(range(len(chunk)))
                for row in sorted(failed):
                    position = chunk[row]
                    failures.append({
                        "game": position.game, "seed": position.seed, "move": position.prefix_length + 1,
                        "player": position.player, "reason": str(error),
                    })
                remaining = [position for row, position in enumerate(chunk) if row not in failed]
                if remaining:
                    batches.appendleft(remaining)
                progress.update(len(failed))
                continue
            finally:
                del request, search

            for row, position in enumerate(chunk):
                choices = {name: batch[row] for name, batch in decisions.items()}
                actions = {name: choice["action_id"] for name, choice in choices.items()}
                belief_changed = actions["belief_value_1"] != actions["belief_value_0"]
                truth_changed = actions["true_map_value_1"] != actions["true_map_value_0"]
                if belief_changed:
                    change = "both" if truth_changed else "belief_only"
                else:
                    change = "true_map_only" if truth_changed else "neither"
                changes[change] += 1
                for scale in agreement:
                    agreement[scale] += actions[f"belief_{scale}"] == actions[f"true_map_{scale}"]
                if change != "neither":
                    game = outcomes[position.game]
                    disagreements.append((position, {
                        "game": position.game, "seed": position.seed, "move": position.prefix_length + 1,
                        "player": position.player, "raw_action": position.raw_action, "raw_value": position.raw_value,
                        "observed_raw_game_outcome": None if game["truncated"] else 2 * game["score_a"] - 1,
                        "change": change, **choices,
                    }))
            compared += len(chunk)
            progress.update(len(chunk))

    report.update(
        diagnostics_valid=compared > 0 and not failures,
        compared_positions=compared,
        changes=changes,
        belief_true_map_action_agreement=agreement,
        positions=[result for _, result in disagreements],
    )
    if play_disagreement_games:
        report["continuations"] = _play_disagreement_pairs(
            disagreements, prefixes, new_search, max_envs, training.max_moves,
        )
    return report


def evaluate_mcts_first_divergence(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    max_envs: int,
    model_a_label: str,
    model_b_label: str,
    model_b_search_top_actions: int | None = None,
    model_b_search_random_actions: int | None = None,
    continuation_value_scale: float = 1.0,
) -> dict[str, Any]:
    """Replay the first different action in paired MCTS games with opposite winners."""
    if continuation_value_scale not in (0.0, 1.0):
        raise ValueError("continuation value scale must be 0 or 1")
    seeds = [training.seed + index // 2 for index in range(games)]
    if not (0 <= training.seed <= 2**31 - 1) or any(seed >= 2**31 for seed in seeds):
        raise ValueError("diagnostic map seeds must be in [0, 2**31 - 1]")
    schedule = [(seed, index % 2) for index, seed in enumerate(seeds)]
    baselines: dict[str, Any] = {}
    report: dict[str, Any] = {
        "baseline": baselines,
        "diagnostics_valid": False,
        "positions": [],
        "simulations": mcts.simulations,
    }
    traces: dict[str, dict[int, list[tuple[int, int, str]]]] = {}
    outcomes: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = {}
    for label, scale in (("value_1", 1.0), ("value_0", 0.0)):
        trace: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
        games_by_side: dict[tuple[int, int], Mapping[str, Any]] = {}

        def record_actions(indices, request, actions) -> None:
            for row in np.flatnonzero(actions >= 0):
                trace[int(indices[row])].append((
                    int(actions[row]), int(request["to_play"][row]), _position_fingerprint(request, row),
                ))

        def record_game(game) -> None:
            games_by_side[game["map_seed"], game["model_a_player"]] = game

        baseline = evaluate_matches_pool(
            model_a, model_b, training, mcts, games,
            num_threads=num_threads, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_retries=belief_retries,
            max_envs=max_envs, model_b_uses_mcts=False, leaf_value_scale=scale,
            map_seeds=seeds, replace_failed_pairs=False, belief_action_mode="root_shield",
            model_b_search_top_actions=model_b_search_top_actions,
            model_b_search_random_actions=model_b_search_random_actions,
            model_a_label=f"{model_a_label} ({label})", model_b_label=model_b_label,
            action_callback=record_actions, game_callback=record_game,
        )
        baselines[label] = baseline
        if not baseline["valid"]:
            report["invalid_reason"] = f"{label} did not finish the fixed evaluation schedule"
            return report
        if baseline["evaluated_map_seeds"] != seeds or set(games_by_side) != set(schedule):
            raise RuntimeError(f"{label} evaluation changed the paired map and side schedule")
        traces[label] = trace
        outcomes[label] = games_by_side

    positions: list[tuple[_ReplayPosition, dict[str, Any]]] = []
    prefixes: dict[int, list[int]] = {}
    paired_games = []
    excluded_truncated = excluded_draws = 0
    for index, (seed, player) in enumerate(schedule):
        zero = outcomes["value_0"][seed, player]
        one = outcomes["value_1"][seed, player]
        paired = {
            "seed": seed, "model_a_player": player,
            "winner_value_0": None if zero["truncated"] else zero["winner_player"],
            "winner_value_1": None if one["truncated"] else one["winner_player"],
        }
        paired_games.append(paired)
        if zero["truncated"] or one["truncated"]:
            excluded_truncated += 1
            continue
        if min(zero["winner_player"], one["winner_player"]) < 0:
            excluded_draws += 1
            continue
        if zero["winner_player"] == one["winner_player"]:
            continue

        zero_steps = traces["value_0"][zero["game_index"]]
        one_steps = traces["value_1"][one["game_index"]]
        prefix = []
        for step_0, step_1 in zip(zero_steps, one_steps):
            action_0, actor, fingerprint = step_0
            action_1, actor_1, fingerprint_1 = step_1
            if (actor, fingerprint) != (actor_1, fingerprint_1):
                raise RuntimeError(f"paired observations differ before the first different action on seed {seed}")
            if action_0 != action_1:
                position = _ReplayPosition(index, seed, player, len(prefix), fingerprint)
                positions.append((position, {
                    **paired, "move": len(prefix) + 1, "to_play": actor,
                    "actor": "apprentice" if actor == player else "opponent",
                    "action_value_0": action_0, "action_value_1": action_1,
                    "continuations": {},
                }))
                break
            prefix.append(action_0)
        else:
            raise RuntimeError(f"the winner changed without a different action on seed {seed}, side {player}")
        prefixes[index] = prefix

    new_search = partial(
        _diagnostic_search, model_a, model_b, training, mcts,
        num_threads=num_threads, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history,
        belief_retries=_evaluation_belief_retries(belief_retries),
        model_b_search_top_actions=model_b_search_top_actions,
        model_b_search_random_actions=model_b_search_random_actions,
    )
    summaries = {}
    batch_pairs = max_envs // 2
    for policy, scale in (("raw", None), ("mcts", continuation_value_scale)):
        seed_everything(training.seed)
        batches = deque(positions[start:start + batch_pairs] for start in range(0, len(positions), batch_pairs))
        failed_pairs = 0
        with tqdm(total=len(positions), desc=f"play first divergence ({policy})", unit="pair") as progress:
            while batches:
                batch = batches.popleft()
                branch_positions = [position for position, _ in batch for _ in range(2)]
                first_actions = [row[key] for _, row in batch for key in ("action_value_0", "action_value_1")]
                search = new_search(branch_positions)
                try:
                    branches = _play_continuations(
                        search, branch_positions, first_actions, prefixes, training.max_moves,
                        mcts_value_scale=scale,
                    )
                except BeliefRecoveryRequired as error:
                    if len(batch) > 1:
                        middle = len(batch) // 2
                        batches.appendleft(batch[middle:])
                        batches.appendleft(batch[:middle])
                        continue
                    failed_pairs += 1
                    branches = [{"score_a": None, "failure_reason": str(error)}] * 2
                finally:
                    del search
                for row_index, (_, row) in enumerate(batch):
                    row["continuations"][policy] = dict(zip(
                        ("from_value_0_action", "from_value_1_action"), branches[2 * row_index:2 * row_index + 2],
                    ))
                progress.update(len(batch))

        scores = [
            (row["continuations"][policy]["from_value_0_action"]["score_a"],
             row["continuations"][policy]["from_value_1_action"]["score_a"])
            for _, row in positions
        ]
        summaries[policy] = {
            **_continuation_pair_summary(scores), "failed_pairs": failed_pairs,
            "leaf_value_scale": scale,
        }

    report.update(
        diagnostics_valid=not any(summary["failed_pairs"] for summary in summaries.values()),
        paired_games=paired_games,
        excluded_truncated_pairs=excluded_truncated,
        excluded_draw_pairs=excluded_draws,
        changed_winner_pairs=len(positions),
        opponent_first_divergences=sum(row["actor"] == "opponent" for _, row in positions),
        continuation_policies={"raw": "both_raw", "mcts": "apprentice_mcts_opponent_raw"},
        continuations=summaries,
        positions=[row for _, row in positions],
    )
    return report
