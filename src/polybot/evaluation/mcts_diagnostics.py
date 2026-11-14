"""Compare belief search and true-map search on replayed evaluation positions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import numpy as np
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..training.flat_value import value_metrics
from ..training.routed_self_play_pool import RoutedPoolSearch, evaluate_matches_pool


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
class _Disagreement:
    game: int
    seed: int
    player: int
    prefix_length: int
    fingerprint: str
    raw_action: int
    belief_action: int
    raw_value: float
    belief_value: float
    belief_visits: int


def _search_true_maps(
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    # This separate, offline pool never plays a move chosen by this search.
    # Preserve each observer's visible tokens and fog mask. Replace only
    # hidden-map hypotheses, keeping the same ISMCTS routing and leaf model.
    state_ids = request["state_id"]
    public = search.pool.all_player_belief_requests()["map_tokens"]
    truth = search.pool.belief_targets_numpy(state_ids)
    visible = public[..., :1] == 1
    completed = np.where(visible, public, truth[:, None])
    completed[..., 0] = public[..., 0]
    worlds = np.repeat(completed[:, :, None], max(1, search.mcts.particles), axis=2)
    search.pool.submit_beliefs(state_ids, np.ascontiguousarray(worlds, dtype=np.int32))
    active = np.ones(search.num_envs, dtype=bool)
    return search._search_submitted_roots(request, active, 0.0, 0.0, ~active, None, None)


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
                            _position_fingerprint(raw, row), raw_action, belief_action,
                            float(raw["root_value"][row]), float(root["root_value"][row]),
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
        search = RoutedPoolSearch(
            training, mcts, (model_a, model_b),
            num_envs=len(chunk), num_threads=num_threads, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_retries=belief_retries,
            seed=chunk[0].seed, belief_action_mode="root_shield",
            search_top_actions_by_agent=(
                mcts.search_top_actions,
                mcts.search_top_actions if model_b_search_top_actions is None else model_b_search_top_actions,
            ),
            search_random_actions_by_agent=(
                mcts.search_random_actions,
                mcts.search_random_actions if model_b_search_random_actions is None else model_b_search_random_actions,
            ),
        )
        request = search.reset(
            np.array([sample.seed for sample in chunk], dtype=np.uint32),
            np.array([(0, 1) if sample.player == 0 else (1, 0) for sample in chunk], dtype=np.int32),
        )
        for move in range(max(sample.prefix_length for sample in chunk)):
            actions = np.array([
                prefixes[sample.game][move] if move < sample.prefix_length else -1
                for sample in chunk
            ], dtype=np.int32)
            search.step_public_policy(actions, request["state_id"], request)
            if not np.all(request["action_valid"][actions >= 0]):
                raise RuntimeError("a saved game action was rejected during diagnostic replay")
        for row, sample in enumerate(chunk):
            if _position_fingerprint(request, row) != sample.fingerprint:
                raise RuntimeError(
                    f"diagnostic replay differs from seed {sample.seed}, "
                    f"move {sample.prefix_length + 1}, player {sample.player}"
                )
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
