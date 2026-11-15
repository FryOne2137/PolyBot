"""Collect terminal targets after alternative actions at MCTS decisions."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, TrainingConfig
from ..data.polyenv_features import native_batch_to_tensors
from ..evaluation.mcts_diagnostics import (
    _compare_leaf_scales,
    _play_replayed_continuations,
    _position_fingerprint,
    _replay_positions,
    _ReplayPosition,
)
from ..models.policy_value_belief import PolicyValueBeliefNet
from .belief_recovery import BeliefRecoveryRequired
from .flat_value import EncodedValuePosition, VALUE_PHASES
from .routed_self_play_pool import RoutedPoolSearch, _evaluation_belief_retries, _slice_rows, evaluate_matches_pool


@dataclass(frozen=True)
class _BranchProbe:
    position: _ReplayPosition
    phase: str
    turn: int


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
    """Convert both mean terminal scores to their successors' perspectives."""
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
    opponent_value_scale: float,
    disagreement_probes_per_position: int,
    branch_rollouts: int,
    rollout_temperature: float,
    label: str,
) -> tuple[list[EncodedValuePosition], dict[str, Any]]:
    seeds = [training.seed + index // 2 for index in range(games)]
    rng = np.random.default_rng(training.seed ^ 0x56414C)
    prefixes: dict[int, list[int]] = defaultdict(list)
    eligible: dict[int, list[tuple[_ReplayPosition, int]]] = defaultdict(list)
    terminal_source_games: set[int] = set()

    def record(indices, request, actions) -> None:
        for row in np.flatnonzero(actions >= 0):
            game = int(indices[row])
            prefix = prefixes[game]
            action = int(actions[row])
            if request["agent_id"][row] == 0 and request["action_mask"][row].sum() > 1:
                position = _ReplayPosition(
                    game, int(request["map_seed"][row]), int(request["to_play"][row]),
                    len(prefix), _position_fingerprint(request, row),
                )
                eligible[game].append((position, int(request["state"][row, 0])))
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
        "disagreement_opponent_leaf_value_scale": opponent_value_scale,
        "branch_rollouts_per_move": branch_rollouts,
        "rollout_temperature": rollout_temperature,
        "phase_definition": "thirds_of_completed_game_by_action_index",
    }
    if not report["valid"]:
        return [], report

    phase_counts = {phase: dict.fromkeys((
        "available_roots", "probed_roots", "failed_probes", "disagreement_roots", "sampled_roots",
        "labelled_pairs", "pairs_with_different_mean_scores",
    ), 0) for phase in VALUE_PHASES}
    probes: list[_BranchProbe] = []
    limits: dict[tuple[int, str], int] = {}
    quota, remainder = divmod(positions_per_game, len(VALUE_PHASES))
    for game in sorted(terminal_source_games):
        buckets: dict[str, list[_BranchProbe]] = defaultdict(list)
        for position, turn in eligible[game]:
            phase = VALUE_PHASES[len(VALUE_PHASES) * position.prefix_length // len(prefixes[game])]
            buckets[phase].append(_BranchProbe(position, phase, turn))
        for index, phase in enumerate(VALUE_PHASES):
            bucket = buckets[phase]
            limit = quota + int((index - game) % len(VALUE_PHASES) < remainder)
            limits[game, phase] = limit
            count = min(limit * disagreement_probes_per_position, len(bucket))
            phase_counts[phase]["available_roots"] += len(bucket)
            probes.extend(bucket[row] for row in rng.choice(len(bucket), size=count, replace=False))

    def new_search(positions: Sequence[_ReplayPosition], belief_retries: int = 0) -> RoutedPoolSearch:
        return RoutedPoolSearch(
            training, mcts, (model, opponent), num_envs=len(positions),
            num_threads=num_threads, max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history, belief_retries=belief_retries,
            seed=positions[0].seed, belief_action_mode="root_shield", opponent_leaf_value_scale=opponent_value_scale,
        )

    probes.sort(key=lambda probe: (probe.position.game, probe.position.prefix_length))
    candidates: dict[tuple[int, str], list[tuple[_BranchProbe, tuple[int, int]]]] = defaultdict(list)
    batches = deque(probes[start:start + num_envs] for start in range(0, len(probes), num_envs))
    with tqdm(total=len(probes), desc=f"{label} value disagreements", unit="position") as progress:
        while batches:
            batch = batches.popleft()
            positions = [probe.position for probe in batch]
            search = new_search(positions, belief_retries=_evaluation_belief_retries(4))
            request = _replay_positions(search, positions, prefixes)
            try:
                comparison = _compare_leaf_scales(search, request, include_true_map=False)
            except BeliefRecoveryRequired as error:
                failed = {int(row) for row in error.env_ids if 0 <= row < len(batch)} or set(range(len(batch)))
                remaining = [probe for row, probe in enumerate(batch) if row not in failed]
                if remaining:
                    batches.appendleft(remaining)
                for row in failed:
                    phase_counts[batch[row].phase]["failed_probes"] += 1
                progress.update(len(failed))
                continue
            finally:
                del request, search
            for row, probe in enumerate(batch):
                action_0 = comparison["belief_value_0"][row]["action_id"]
                action_1 = comparison["belief_value_1"][row]["action_id"]
                phase_counts[probe.phase]["probed_roots"] += 1
                if action_0 != action_1:
                    phase_counts[probe.phase]["disagreement_roots"] += 1
                    candidates[probe.position.game, probe.phase].append((probe, (action_0, action_1)))
            progress.update(len(batch))

    selected = []
    early_end_samples = 0
    for key, bucket in candidates.items():
        count = min(limits[key], len(bucket))
        phase_counts[key[1]]["sampled_roots"] += count
        for row in rng.choice(len(bucket), size=count, replace=False):
            probe, actions = bucket[row]
            selected.append((probe, actions))
            early_end_samples += probe.turn < 20 and 0 in actions
    selected.sort(key=lambda item: (item[0].position.game, item[0].position.prefix_length))

    examples = []
    terminal_branches = informative_pairs = completed_rollout_pairs = partial_rollout_roots = 0
    batch_pairs = num_envs // 2
    for start in tqdm(range(0, len(selected), batch_pairs), desc=f"{label} raw branches", unit="batch"):
        chunk = selected[start:start + batch_pairs]
        branch_positions = [probe.position for probe, _ in chunk for _ in range(2)]
        search = new_search(branch_positions)
        first_actions = [action for _, actions in chunk for action in actions]
        successors = {}
        pair_scores: list[list[tuple[float, float]]] = [[] for _ in chunk]

        def record_successors(packet) -> None:
            successors.update(_encode_successors(model, search, packet))

        for repeat in range(branch_rollouts):
            request = _replay_positions(search, branch_positions, prefixes)
            # Identical per-ply random streams for the two actions at one root.
            raw_rngs = [np.random.default_rng((training.seed, position.seed, position.player, position.prefix_length, repeat))
                        for position in branch_positions]
            outcomes = _play_replayed_continuations(
                search, request, branch_positions, first_actions, training.max_moves,
                after_first_action=record_successors if repeat == 0 else None,
                raw_rngs=raw_rngs, raw_temperature=rollout_temperature,
            )
            for index, scores in enumerate(pair_scores):
                score_0, score_1 = (outcome["score_a"] for outcome in outcomes[2 * index:2 * index + 2])
                terminal_branches += int(score_0 is not None) + int(score_1 is not None)
                if score_0 is not None and score_1 is not None:
                    scores.append((score_0, score_1))

        for index, ((probe, _), scores) in enumerate(zip(chunk, pair_scores)):
            completed_rollout_pairs += len(scores)
            partial_rollout_roots += len(scores) < branch_rollouts
            if not scores:
                continue
            score_0, score_1 = np.mean(scores, axis=0).tolist()
            labelled = _branch_examples(
                probe.position, [{"score_a": score_0}, {"score_a": score_1}],
                [successors[row] for row in (2 * index, 2 * index + 1) if row in successors], phase=probe.phase,
            )
            if labelled:
                informative = score_0 != score_1
                informative_pairs += informative
                phase_counts[probe.phase]["labelled_pairs"] += 1
                phase_counts[probe.phase]["pairs_with_different_mean_scores"] += informative
                examples.extend(labelled)
        del request, search

    report.update(
        sampled_roots=len(selected), early_end_turn_roots=early_end_samples, requested_probes=len(probes),
        phases=phase_counts, terminal_source_games=len(terminal_source_games),
        truncated_source_games=games - len(terminal_source_games),
        terminal_branches=terminal_branches, truncated_branches=2 * len(selected) * branch_rollouts - terminal_branches,
        completed_rollout_pairs=completed_rollout_pairs, roots_with_incomplete_rollouts=partial_rollout_roots,
        labelled_pairs=len(examples) // 2, pairs_with_different_mean_scores=informative_pairs,
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
