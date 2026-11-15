"""Collect terminal targets after alternative actions at MCTS decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..data.polyenv_features import native_batch_to_tensors
from ..evaluation.mcts_diagnostics import (
    _play_replayed_continuations,
    _position_fingerprint,
    _replay_positions,
    _ReplayPosition,
)
from ..models.policy_value_belief import PolicyValueBeliefNet
from .flat_value import EncodedValuePosition, VALUE_PHASES
from .routed_self_play_pool import RoutedPoolSearch, _slice_rows, evaluate_matches_pool


@torch.inference_mode()
def _encode_successors(
    model: PolicyValueBeliefNet,
    search: RoutedPoolSearch,
    request: Mapping[str, np.ndarray],
) -> dict[int, tuple[np.ndarray, int, int]]:
    rows = np.flatnonzero(~request["terminated"].astype(bool))
    if not rows.size:
        return {}
    batch = _slice_rows(request, rows)
    tensors = native_batch_to_tensors(batch, search.device, search.training.map_size, search.training.map_type)
    output = model(
        tensors.map_tokens, tensors.global_features, tensors.action_categorical,
        tensors.action_continuous, tensors.action_mask, tensors.event_features,
        action_history=tensors.action_history, action_history_mask=tensors.action_history_mask,
        include_belief=False,
    )
    return dict(zip(rows.tolist(), zip(
        output.state.float().cpu().numpy(), batch["state"][:, 0].tolist(), batch["to_play"].tolist(),
    )))


def _branch_examples(
    position: _ReplayPosition,
    outcomes: Sequence[Mapping[str, Any]],
    successors: Sequence[tuple[np.ndarray, int, int]],
    *,
    phase: str | None = None,
) -> list[EncodedValuePosition]:
    """Use both observed terminal outcomes, expressed in each successor's perspective."""
    if len(successors) != 2 or any(outcome["score_a"] is None for outcome in outcomes):
        return []
    key = (position.seed, position.player, position.prefix_length)
    return [
        EncodedValuePosition(
            features, turn, (2 * outcome["score_a"] - 1) * (1 if actor == position.player else -1), key, phase,
        )
        for (features, turn, actor), outcome in zip(successors, outcomes)
    ]


def collect_mcts_value_branches(
    model: PolicyValueBeliefNet,
    opponent: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    positions_per_game: int,
    num_envs: int,
    num_threads: int,
    visible_action_history: int,
    max_pending_leaves: int,
    label: str,
) -> tuple[list[EncodedValuePosition], dict[str, Any]]:
    seeds = [training.seed + index // 2 for index in range(games)]
    rng = np.random.default_rng(training.seed ^ 0x56414C)
    prefixes: dict[int, list[int]] = defaultdict(list)
    eligible: dict[int, list[tuple[_ReplayPosition, int, bool]]] = defaultdict(list)
    terminal_source_games: set[int] = set()

    def record(indices, request, actions) -> None:
        for row in np.flatnonzero(actions >= 0):
            game = int(indices[row])
            prefix = prefixes[game]
            action = int(actions[row])
            if request["agent_id"][row] == 0 and request["action_mask"][row].sum() > 1:
                early_end = action == 0 and request["state"][row, 0] < 20
                position = _ReplayPosition(
                    game, int(request["map_seed"][row]), int(request["to_play"][row]),
                    len(prefix), _position_fingerprint(request, row),
                )
                eligible[game].append((position, action, bool(early_end)))
            prefix.append(action)

    def record_game(game) -> None:
        if not game["truncated"]:
            terminal_source_games.add(int(game["game_index"]))

    evaluation = evaluate_matches_pool(
        model, opponent, training, mcts, games, model_b_uses_mcts=False, map_seeds=seeds,
        num_threads=num_threads, max_envs=num_envs, max_pending_leaves=max_pending_leaves,
        visible_action_history=visible_action_history, belief_action_mode="root_shield",
        replace_failed_pairs=False, leaf_value_scale=0.0, opponent_leaf_value_scale=0.0,
        model_a_label=label, model_b_label="raw opponent", action_callback=record, game_callback=record_game,
    )
    report: dict[str, Any] = {
        "valid": bool(evaluation["valid"] and evaluation.get("evaluated_map_seeds") == seeds),
        "map_seeds": sorted(set(evaluation.get("evaluated_map_seeds", ()))),
        "evaluation": evaluation,
        "labelled_maps": 0,
        "positions": 0,
        "model_a_leaf_value_scale": 0.0,
        "opponent_leaf_value_scale": 0.0,
        "phase_definition": "thirds_of_completed_game_by_action_index",
    }
    if not report["valid"]:
        return [], report

    positions: list[tuple[_ReplayPosition, int, str]] = []
    early_end_samples = 0
    phase_counts = {phase: {"available_roots": 0, "sampled_roots": 0, "labelled_pairs": 0,
                            "pairs_with_different_outcomes": 0} for phase in VALUE_PHASES}
    quota, remainder = divmod(positions_per_game, len(VALUE_PHASES))
    for game in sorted(terminal_source_games):
        buckets: dict[str, list[tuple[_ReplayPosition, int, bool]]] = defaultdict(list)
        for sample in eligible[game]:
            phase = VALUE_PHASES[len(VALUE_PHASES) * sample[0].prefix_length // len(prefixes[game])]
            buckets[phase].append(sample)
        for index, phase in enumerate(VALUE_PHASES):
            bucket = buckets[phase]
            limit = quota + int((index - game) % len(VALUE_PHASES) < remainder)
            count = min(limit, len(bucket))
            phase_counts[phase]["available_roots"] += len(bucket)
            phase_counts[phase]["sampled_roots"] += count
            for sample_index in rng.choice(len(bucket), size=count, replace=False):
                position, action, early_end = bucket[sample_index]
                positions.append((position, action, phase))
                early_end_samples += early_end
    positions.sort(key=lambda sample: (sample[0].game, sample[0].prefix_length))

    examples = []
    terminal_branches = informative_pairs = 0
    batch_pairs = num_envs // 2
    for start in tqdm(range(0, len(positions), batch_pairs), desc=f"{label} raw branches", unit="batch"):
        chunk = positions[start:start + batch_pairs]
        branch_positions = [sample[0] for sample in chunk for _ in range(2)]
        search = RoutedPoolSearch(
            training, mcts, (model, opponent), num_envs=len(branch_positions),
            num_threads=num_threads, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_retries=0, seed=chunk[0][0].seed,
        )
        request = _replay_positions(search, branch_positions, prefixes)
        policy, _ = search.direct_policy_value(request, np.arange(0, len(branch_positions), 2))
        first_actions = []
        for row, (_, chosen, _) in zip(range(0, len(branch_positions), 2), chunk):
            legal = request["action_mask"][row].astype(bool) & (request["action_id"][row] != chosen)
            alternate = np.flatnonzero(legal)
            best_alternate = alternate[policy[row, alternate].argmax()]
            first_actions.extend((chosen, int(request["action_id"][row, best_alternate])))

        successors = {}
        outcomes = _play_replayed_continuations(
            search, request, branch_positions, first_actions, training.max_moves,
            after_first_action=lambda packet: successors.update(_encode_successors(model, search, packet)),
        )
        for row, (position, _, phase) in zip(range(0, len(branch_positions), 2), chunk):
            pair = outcomes[row:row + 2]
            terminal_branches += sum(outcome["score_a"] is not None for outcome in pair)
            labelled = _branch_examples(
                position, pair, [successors[i] for i in (row, row + 1) if i in successors], phase=phase,
            )
            if labelled:
                informative = pair[0]["score_a"] != pair[1]["score_a"]
                informative_pairs += informative
                phase_counts[phase]["labelled_pairs"] += 1
                phase_counts[phase]["pairs_with_different_outcomes"] += informative
                examples.extend(labelled)
        del request, search

    report.update(
        sampled_roots=len(positions), early_end_turn_roots=early_end_samples,
        phases=phase_counts, terminal_source_games=len(terminal_source_games),
        truncated_source_games=games - len(terminal_source_games),
        terminal_branches=terminal_branches, truncated_branches=2 * len(positions) - terminal_branches,
        labelled_pairs=len(examples) // 2, pairs_with_different_outcomes=informative_pairs,
        labelled_maps=len({example.game_key[0] for example in examples}), positions=len(examples),
    )
    return examples, report


def evaluate_mcts_value_heads(
    model: PolicyValueBeliefNet,
    opponent: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    num_envs: int,
    num_threads: int,
    visible_action_history: int,
    max_pending_leaves: int,
    opponent_value_scale: float,
) -> dict[str, Any]:
    seeds = [training.seed + index // 2 for index in range(games)]
    arms = {}
    outcomes: dict[str, dict[tuple[int, int], float]] = {}
    for name, scale in (("fitted", 1.0), ("zero", 0.0)):
        terminal: dict[tuple[int, int], float] = {}

        def record(game) -> None:
            if not game["truncated"]:
                terminal[game["map_seed"], game["model_a_player"]] = game["score_a"]

        arms[name] = evaluate_matches_pool(
            model, opponent, training, mcts, games, model_b_uses_mcts=False, map_seeds=seeds,
            num_threads=num_threads, max_envs=num_envs, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_action_mode="root_shield",
            replace_failed_pairs=False, leaf_value_scale=scale,
            opponent_leaf_value_scale=opponent_value_scale, model_a_label=f"value {name}",
            model_b_label="raw opponent", game_callback=record,
        )
        outcomes[name] = terminal

    valid = all(arm["valid"] and arm.get("evaluated_map_seeds") == seeds for arm in arms.values())
    complete = [seed for seed in seeds[::2] if all(
        (seed, side) in scores for scores in outcomes.values() for side in (0, 1)
    )] if valid else []
    scores = {
        name: float(np.mean([terminal[seed, side] for seed in complete for side in (0, 1)])) if complete else None
        for name, terminal in outcomes.items()
    }
    return {
        "arms": arms, "valid": bool(complete), "opponent_value_scale": opponent_value_scale,
        "compared_map_seeds": complete, "compared_games_per_arm": 2 * len(complete),
        "paired_scores": scores,
        "beats_zero": bool(complete and scores["fitted"] > scores["zero"]),
    }
