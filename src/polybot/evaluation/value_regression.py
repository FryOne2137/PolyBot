"""Compare terminal accuracy and first move changes from a fitted value head."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..training.belief_recovery import BeliefRecoveryRequired
from ..training.flat_value import evaluate_value_head
from ..training.mcts_value import collect_mcts_terminal_value_games
from ..training.routed_self_play_pool import RoutedPoolSearch, _evaluation_belief_retries
from .mcts_diagnostics import (
    _ReplayPosition,
    _compare_leaf_scales,
    _paired_rollout_rngs,
    _play_continuations,
    _position_fingerprint,
    _step_replayed_action,
)


def _require_only_value_differs(old: PolicyValueBeliefNet, new: PolicyValueBeliefNet) -> None:
    old_state, new_state = old.state_dict(), new.state_dict()
    if old.config != new.config or old_state.keys() != new_state.keys() or any(
        not torch.equal(value, new_state[name])
        for name, value in old_state.items()
        if not name.startswith("value_head.")
    ):
        raise ValueError("old and fitted checkpoints must share their encoder, policy and belief weights")


def _first_value_divergence(search, variants, seed: int, side: int, max_moves: int):
    request = search.reset(
        np.array([seed], dtype=np.uint32),
        np.array([[0, 1] if side == 0 else [1, 0]], dtype=np.int32),
    )
    prefix: list[int] = []
    while len(prefix) < max_moves and not request["terminated"][0]:
        if request["agent_id"][0] == 0:
            roots = _compare_leaf_scales(search, request, variants=variants, include_true_map=False)
            choices = {name: roots[f"belief_{name}"][0] for name, _, _ in variants}
            actions = list(dict.fromkeys(choice["action_id"] for choice in choices.values()))
            if len(actions) > 1:
                position = _ReplayPosition(0, seed, side, len(prefix), _position_fingerprint(request, 0))
                for choice in choices.values():
                    visits = choice.pop("action_visits")
                    choice["compared_action_visits"] = {action: visits.get(action, 0) for action in actions}
                return position, prefix, actions, choices
            action = np.asarray(actions, dtype=np.int32)
            search.pool.step_into_checked(action, request["state_id"], request)
        else:
            policy, _ = search.direct_policy_value(request, np.array([0], dtype=np.int32))
            action = np.array([request["action_id"][0, policy[0].argmax()]], dtype=np.int32)
            _step_replayed_action(search, request, action)
        if not request["action_valid"][0]:
            raise RuntimeError(f"first-divergence search attempted an illegal action on seed {seed}")
        prefix.append(int(action[0]))
    return None


def _roll_out_actions(search, position, prefix, actions, repeats, temperature):
    positions = [position] * len(actions)
    common_scores = []
    rows = []
    for repeat in range(repeats):
        try:
            outcomes = _play_continuations(
                search, positions, actions, {position.game: prefix}, search.training.max_moves,
                mcts_value_scale=0.0, raw_temperature=temperature,
                opponent_rngs=_paired_rollout_rngs(search.training.seed, positions, repeat, 1),
            )
        except BeliefRecoveryRequired as error:
            rows.append({"repeat": repeat + 1, "belief_error": str(error)})
            continue
        scores = [outcome["score_a"] for outcome in outcomes]
        if all(score is not None for score in scores):
            common_scores.append(scores)
        rows.append({"repeat": repeat + 1, "score_a": dict(zip(actions, scores, strict=True))})
    return {
        "completed_paired_repeats": len(common_scores),
        "mean_score_a": dict(zip(actions, np.mean(common_scores, axis=0).tolist(), strict=True))
        if common_scores else {},
        "repeats": rows,
    }


def evaluate_mcts_value_regression(
    new: PolicyValueBeliefNet,
    old: PolicyValueBeliefNet,
    opponent: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    positions_per_game: int,
    divergence_games: int,
    branch_rollouts: int,
    rollout_temperature: float,
    num_threads: int | None,
    max_envs: int,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
) -> dict[str, Any]:
    """Score identical held-out positions, then replay the first divergent action."""
    if games < 2 or games % 2 or positions_per_game < 3 or max_envs < 3:
        raise ValueError("require paired games, three phase samples and at least three environments")
    if not 1 <= training.seed <= 2**31 - games // 2:
        raise ValueError("diagnostic map seeds must be in [1, 2**31 - 1]")
    if divergence_games < 1 or branch_rollouts < 1 or not np.isfinite(rollout_temperature) or rollout_temperature <= 0:
        raise ValueError("require positive diagnostic game and rollout counts and a finite positive temperature")
    _require_only_value_differs(old, new)
    old.eval()
    new.eval()
    opponent.eval()

    datasets = {}
    cohorts = {}
    for label, scale in (("zero", 0.0), ("new_quarter", 0.25)):
        datasets[label], cohorts[label] = collect_mcts_terminal_value_games(
            new, opponent, training, mcts, games,
            positions_per_game=positions_per_game, num_envs=max_envs, num_threads=num_threads,
            visible_action_history=visible_action_history, max_pending_leaves=max_pending_leaves,
            opponent_value_scale=1.0, excluded_map_seeds=(), label=f"diagnostic {label}",
            leaf_value_scale=scale, replace_incomplete=False, belief_retries=belief_retries,
        )
    common_maps = sorted(set(cohorts["zero"]["map_seeds"]) & set(cohorts["new_quarter"]["map_seeds"]))
    common_map_set = set(common_maps)
    metrics = {}
    for label, dataset in datasets.items():
        common = [sample for sample in dataset if sample.game_key[0] in common_map_set]
        metrics[label] = {
            name: evaluate_value_head(model, training, common, value_scale=scale)
            for name, model, scale in (
                ("zero", new, 0.0), ("old", old, 1.0), ("new", new, 1.0),
                ("old_quarter", old, 0.25), ("new_quarter", new, 0.25),
            )
        }

    def make_search(seed: int, num_envs: int) -> RoutedPoolSearch:
        return RoutedPoolSearch(
            training, mcts, (new, opponent), num_envs=num_envs, num_threads=num_threads,
            seed=seed, visible_action_history=visible_action_history,
            max_pending_leaves=max_pending_leaves, belief_retries=_evaluation_belief_retries(belief_retries),
            belief_action_mode="root_shield", leaf_value_scale=0.0, opponent_leaf_value_scale=1.0,
        )

    variants = (("zero", 0.0, new), ("old_quarter", 0.25, old), ("new_quarter", 0.25, new))
    seed_everything(training.seed)
    schedule = [
        (seed, (index + offset) % 2)
        for offset in (0, 1) for index, seed in enumerate(common_maps)
    ][:divergence_games]
    divergences = []
    for seed, side in tqdm(schedule, desc="first value divergences", unit="game"):
        search = make_search(seed, 1)
        row = {"map_seed": seed, "model_a_player": side}
        try:
            result = _first_value_divergence(search, variants, seed, side, training.max_moves)
        except BeliefRecoveryRequired as error:
            row.update(status="belief_error", belief_error=str(error))
            divergences.append(row)
            continue
        finally:
            del search
        if result is None:
            row.update(status="no_divergence")
        else:
            position, prefix, actions, choices = result
            row.update(
                status="diverged", move=len(prefix) + 1, prefix_actions=prefix,
                first_actions=choices, root_fingerprint=position.fingerprint,
            )
            row["continuations"] = _roll_out_actions(
                make_search(seed, len(actions)), position, prefix, actions, branch_rollouts, rollout_temperature,
            )
        divergences.append(row)

    return {
        "valid": bool(common_maps),
        "simulations": mcts.simulations, "particles": mcts.particles,
        "opponent_leaf_value_scale": 1.0,
        "cohorts": cohorts, "compared_map_seeds": common_maps, "rmse": metrics,
        "continuation_candidate_scale": 0.0, "continuation_opponent_temperature": rollout_temperature,
        "rollouts_per_action": branch_rollouts,
        "games_scanned": len(schedule),
        "first_divergences": divergences,
    }
