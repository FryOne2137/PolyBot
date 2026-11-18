"""Fit only a checkpoint's value head on game outcomes or saved branch pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig, TrainingConfig
from polybot.evaluation.matches import evaluate_direct_matches
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model, save_model
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.cli import (
    add_environment_options,
    add_run_options,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.flat_experiments import FlatImprovements
from polybot.training.flat_value import EncodedValuePosition, evaluate_value_head, fit_paired_value_heads, fit_value_head


def collect_value_games(
    model: PolicyValueBeliefNet,
    opponent: PolicyValueBeliefNet,
    training: TrainingConfig,
    games: int,
    *,
    positions_per_game: int,
    num_envs: int,
    num_threads: int,
    visible_action_history: int,
    label: str,
) -> tuple[list[EncodedValuePosition], dict[str, Any]]:
    rng = np.random.default_rng(training.seed ^ 0x56414C)
    examples: list[EncodedValuePosition] = []
    map_seeds: set[int] = set()
    terminal_games = truncated_games = 0
    device = next(model.parameters()).device
    batch_games = num_envs - num_envs % 2
    num_batches = (games + batch_games - 1) // batch_games

    for start in range(0, games, batch_games):
        count = min(batch_games, games - start)
        seen = [0] * count
        positions: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]

        def record_position(game: int, turn: int, features: np.ndarray) -> None:
            seen[game] += 1
            sampled = positions[game]
            index = len(sampled) if len(sampled) < positions_per_game else int(
                rng.integers(seen[game])
            )
            if index == len(sampled):
                sampled.append((turn, features.copy()))
            elif index < positions_per_game:
                sampled[index] = (turn, features.copy())

        def record_outcome(game: Mapping[str, float | int]) -> None:
            key = (int(game["map_seed"]), int(game["model_a_player"]))
            map_seeds.add(key[0])
            if not game["truncated"]:
                target = 2.0 * float(game["score_a"]) - 1.0
                examples.extend(
                    EncodedValuePosition(features, turn, target, key)
                    for turn, features in positions[int(game["game_index"])]
                )

        evaluation = evaluate_direct_matches(
            model, opponent, device, replace(training, seed=training.seed + start // 2),
            count, num_threads=num_threads, visible_action_history=visible_action_history,
            game_callback=record_outcome, raw_value_callback=record_position,
            # Match routed MCTS leaves, which carry compact action history
            # and use a neutral event summary.
            include_event_summary=False,
            model_a_label=f"{label} {start // batch_games + 1}/{num_batches}",
        )
        terminal_games += int(evaluation["terminated_games"])
        truncated_games += int(evaluation["truncated_games"])

    labelled_maps = {example.game_key[0] for example in examples}
    return examples, {
        "map_seeds": sorted(map_seeds),
        "labelled_maps": len(labelled_maps),
        "terminal_games": terminal_games,
        "truncated_games": truncated_games,
        "positions": len(examples),
    }


def fit_mcts_terminal_value(args, model, opponent, config, training, schedules, run) -> dict[str, Any]:
    from polybot.training.mcts_value import (
        MCTS_VALUE_SCALES,
        collect_mcts_terminal_value_games,
        evaluate_mcts_value_heads,
    )

    mcts = MCTSConfig(simulations=args.simulations, particles=args.particles)
    turn_starts_only = args.position_source == "mcts-turn-start"
    benchmark_config = (
        {"turn_candidates": args.turn_candidates, "turn_max_rollout_actions": args.turn_max_rollout_actions}
        if turn_starts_only else {"benchmark_leaf_value_scales": MCTS_VALUE_SCALES}
    )
    pool = dict(
        num_envs=args.num_envs, num_threads=args.num_threads,
        visible_action_history=args.visible_action_history, max_pending_leaves=args.max_pending_leaves,
        opponent_value_scale=args.opponent_value_scale,
    )
    run.write_config(
        training=training, model=config, mcts=mcts, schedules=schedules, loss="mse",
        collection_leaf_value_scales=(0.0, args.opponent_value_scale), turn_starts_only=turn_starts_only,
        **benchmark_config,
    )
    datasets = {}
    collection = {}
    report: dict[str, Any] = {"collection": collection, "saved_checkpoint": None, "mcts_improved": False}
    reserved = {label: set(range(seed, seed + games // 2)) for label, seed, games in schedules}
    attempted_seeds: set[int] = set()
    for label, seed, games in schedules[:2]:
        excluded = attempted_seeds.union(*(maps for split, maps in reserved.items() if split != label))
        datasets[label], summary = collect_mcts_terminal_value_games(
            model, opponent, replace(training, seed=seed), mcts, games,
            positions_per_game=args.positions_per_game, excluded_map_seeds=excluded,
            label=f"value {label}", turn_starts_only=turn_starts_only, **pool,
        )
        collection[label] = summary
        attempted_seeds.update(summary["attempted_map_seeds"])
        run.event("terminal_collection", split=label, **summary)
        if not summary["valid"] or summary["labelled_maps"] < 2 or not all(summary["phases"].values()):
            return {**report, "reason": f"{label}_insufficient_terminal_data"}

    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size,
        value_learning_rate=args.learning_rate, value_validation_games=args.selection_games,
        value_capacity=max(len(datasets["train"]), args.batch_size), value_fit_backoffs=0,
    )
    fit = fit_value_head(
        model, training, datasets["train"], datasets["selection"], settings, seed=training.seed,
        validation_source="mcts_turn_start_selection" if turn_starts_only else "mcts_terminal_selection",
        keep_best_update=True,
    )
    report["fit"] = fit
    run.log(
        updates=fit["updates"], selected_updates=fit.get("selected_updates"),
        selection_original_rmse=(fit.get("before") or {}).get("rmse"),
        selection_fitted_rmse=(fit.get("after") or {}).get("rmse"),
        selection_zero_rmse=(fit.get("after") or {}).get("zero_rmse"),
    )
    if not fit["kept"]:
        return {**report, "reason": fit["reason"]}
    save_model(args.model_out, model, config)
    report["saved_checkpoint"] = str(args.model_out)
    run.event("experimental_checkpoint_saved", loss="mse", checkpoint=args.model_out)

    _, seed, games = schedules[2]
    if turn_starts_only:
        from polybot.evaluation.turn_reply import evaluate_turn_reply_benchmark

        benchmark = evaluate_turn_reply_benchmark(
            model, opponent, replace(training, seed=seed), mcts, games,
            num_envs=args.num_envs, num_threads=args.num_threads,
            visible_action_history=args.visible_action_history, max_pending_leaves=args.max_pending_leaves,
            candidates=args.turn_candidates, max_rollout_actions=args.turn_max_rollout_actions, belief_retries=4,
        )
        report["turn_reply_benchmark"] = benchmark
        run.log(compared_mcts_games=benchmark["compared_games_per_arm"], turn_reply_score_delta=benchmark["paired_score_delta"])
        if benchmark["valid"]:
            report["mcts_improved"] = benchmark["paired_score_delta"] > 0.0
            report["reason"] = "mcts_improved" if report["mcts_improved"] else "mcts_no_demonstrated_improvement"
        else:
            report["reason"] = "mcts_benchmark_invalid"
        return report

    benchmark = evaluate_mcts_value_heads(
        model, opponent, replace(training, seed=seed), mcts, games, value_scales=MCTS_VALUE_SCALES, **pool,
    )
    report["mcts_benchmark"] = benchmark
    report["mcts_improved"] = benchmark["beats_zero"]
    if benchmark["valid"]:
        scales = benchmark["leaf_value_scales"]
        scores = benchmark["paired_scores"]
        best = max(scales, key=scores.__getitem__)
        report["best_benchmark_value_scale"] = scales[best]
        report["reason"] = "mcts_improved" if benchmark["beats_zero"] else "mcts_no_demonstrated_improvement"
    else:
        report["reason"] = "mcts_benchmark_invalid"
    run.log(compared_mcts_games=benchmark["compared_games_per_arm"], **{
        f"mcts_scale_{scale}_score": score for scale, score in benchmark["paired_scores"].items()
    })
    return report


def fit_mcts_branch_value(args, model, opponent, config, training, schedules, run) -> dict[str, Any]:
    from polybot.training.mcts_value import collect_mcts_value_branches, evaluate_mcts_value_heads

    mcts = MCTSConfig(simulations=args.simulations, particles=args.particles)
    pool = dict(
        num_envs=args.num_envs, num_threads=args.num_threads,
        visible_action_history=args.visible_action_history, max_pending_leaves=args.max_pending_leaves,
    )
    run.write_config(
        training=training, model=config, mcts=mcts, schedules=schedules,
        collection_leaf_value_scales=(0.0, 0.0), benchmark_opponent_value_scale=args.opponent_value_scale,
        disagreement_opponent_value_scale=args.opponent_value_scale,
        disagreement_probes_per_position=args.disagreement_probes_per_position,
        branch_rollouts=args.branch_rollouts, rollout_temperature=args.rollout_temperature,
        difference_loss_weight=args.difference_loss_weight,
    )
    datasets = {}
    collection = {}
    report: dict[str, Any] = {
        "collection": collection, "pair_datasets": {}, "saved_checkpoints": {}, "mcts_improved": False,
    }
    for label, seed, games in schedules[:2]:
        path = run.directory / f"{label}_pairs.jsonl"
        report["pair_datasets"][label] = str(path)
        with path.open("w", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps({
                "schema_version": 1, "split": label, "seed": seed,
                "branch_order": ["value_0", "value_1"], "config": "config.json",
            }) + "\n")

            def save_pair(pair) -> None:
                stream.write(json.dumps(pair, allow_nan=False) + "\n")

            examples, summary = collect_mcts_value_branches(
                model, opponent, replace(training, seed=seed), mcts, games,
                positions_per_game=args.positions_per_game, label=f"value {label}",
                opponent_value_scale=args.opponent_value_scale,
                disagreement_probes_per_position=args.disagreement_probes_per_position,
                branch_rollouts=args.branch_rollouts, rollout_temperature=args.rollout_temperature,
                pair_callback=save_pair, **pool,
            )
        collection[label] = summary
        run.event("branch_collection", split=label, **{key: value for key, value in summary.items() if key != "evaluation"})
        if not summary["valid"]:
            return {**report, "reason": f"{label}_collection_invalid"}
        if summary["map_seeds"] != list(range(seed, seed + games // 2)):
            raise RuntimeError(f"unexpected map schedule in {label} games")
        if summary["labelled_maps"] < 2:
            return {**report, "reason": f"{label}_insufficient_terminal_maps"}
        datasets[label] = examples

    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size,
        value_learning_rate=args.learning_rate, value_validation_games=args.selection_games,
        value_capacity=max(len(datasets["train"]), args.batch_size),
    )
    heads, fit = fit_paired_value_heads(
        model, datasets["train"], datasets["selection"], settings,
        seed=training.seed, difference_weight=args.difference_loss_weight,
    )
    report["fit"] = fit
    for name, result in fit.items():
        selected = result["after"] or {}
        run.log(**{f"{name}_updates": result["updates"], f"{name}_selection_rmse": selected.get("rmse"),
                    f"{name}_selection_gap_rmse": selected.get("gap_rmse"),
                    f"{name}_selection_preference_accuracy": selected.get("preference_accuracy")})
    if len(heads) != 2:
        return {**report, "reason": "missing_finite_loss_variant"}

    for name, path in (("mse", args.mse_model_out), ("mse_delta", args.model_out)):
        model.value_head.load_state_dict(heads[name])
        save_model(path, model, config)
        report["saved_checkpoints"][name] = str(path)
        run.event("experimental_checkpoint_saved", loss=name, checkpoint=path)
    _, seed, games = schedules[2]
    benchmark = evaluate_mcts_value_heads(
        model, opponent, replace(training, seed=seed), mcts, games,
        opponent_value_scale=args.opponent_value_scale, fitted_heads=heads, **pool,
    )
    improved = benchmark["beats_zero"]
    reason = "mcts_improved" if improved else "mcts_no_demonstrated_improvement"
    report.update(mcts_benchmark=benchmark, mcts_improved=improved, reason=reason)
    run.log(compared_mcts_games=benchmark["compared_games_per_arm"], **{
        f"mcts_{arm}_score": score for arm, score in benchmark["paired_scores"].items()
    })
    return report


def fit_mcts_soft_z_value(args, model, opponent, config, training, schedules, run) -> dict[str, Any]:
    from polybot.planning.turn_reply import TurnReplySearch
    from polybot.training.mcts_value import evaluate_mcts_value_heads
    from polybot.training.turn_value import (
        collect_value_roots, evaluate_move_preferences, fit_matched_value_heads, label_soft_z,
    )

    mcts = MCTSConfig(simulations=args.simulations, particles=args.particles)
    pool = dict(
        num_envs=args.num_envs, num_threads=args.num_threads,
        visible_action_history=args.visible_action_history, max_pending_leaves=args.max_pending_leaves,
        opponent_value_scale=args.opponent_value_scale,
    )
    search = TurnReplySearch(
        model, opponent, next(model.parameters()).device, candidates=args.turn_candidates,
        particles=args.particles, visible_action_history=args.visible_action_history,
        max_rollout_actions=args.turn_max_rollout_actions, belief_retries=4,
        belief_temperature=mcts.belief_particle_temperature,
    )
    run.write_config(
        training=training, model=config, mcts=mcts, schedules=schedules, targets=["terminal", "soft_z"],
        collection_leaf_value_scales=(0.0, args.opponent_value_scale), benchmark_opponent="raw",
        teacher_simulations_per_post_reply_root=args.teacher_simulations,
        turn_candidates=args.turn_candidates, turn_max_rollout_actions=args.turn_max_rollout_actions,
        ranking_games=args.ranking_games, ranking_positions=args.ranking_positions,
        ranking_rollouts=args.branch_rollouts, rollout_temperature=args.rollout_temperature,
    )
    report: dict[str, Any] = {"collection": {}, "saved_checkpoints": {}, "mcts_improved": False}
    reserved = {name: set(range(seed, seed + games // 2)) for name, seed, games in schedules}
    attempted: set[int] = set()
    datasets = {}
    for name, seed, games in schedules[:2]:
        split = replace(training, seed=seed)
        roots, prefixes, info = collect_value_roots(
            model, opponent, split, mcts, games, label=f"value {name}", positions_per_game=args.positions_per_game,
            excluded_map_seeds=attempted.union(*(maps for key, maps in reserved.items() if key != name)),
            **pool,
        )
        attempted.update(info["attempted_map_seeds"])
        report["collection"][name] = info
        run.event("turn_value_collection", split=name, **info)
        if not info["valid"]:
            return {**report, "reason": f"{name}_incomplete_collection"}
        path = run.directory / f"{name}_matched_targets.jsonl"
        with path.open("w", encoding="utf-8", buffering=1) as stream:
            datasets[name], labels = label_soft_z(
                search, roots, prefixes, split, mcts, simulations=args.teacher_simulations,
                num_threads=args.num_threads, opponent_value_scale=args.opponent_value_scale,
                record=lambda row: stream.write(json.dumps(row, allow_nan=False) + "\n"), label=f"value {name}",
            )
        info.update(labels=labels, label_file=str(path))
        run.event("turn_value_matched_labels", split=name, **labels)
        if labels["maps"] < 2 or not all(labels["phases"].values()):
            return {**report, "reason": f"{name}_insufficient_matched_labels"}

    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size, value_learning_rate=args.learning_rate,
        value_capacity=max(len(datasets["train"]["terminal"]), args.batch_size), value_fit_backoffs=0,
    )
    heads, report["fit"] = fit_matched_value_heads(model, training, datasets, settings, seed=training.seed)
    if len(heads) != 2:
        return {**report, "reason": "missing_fitted_value_head"}
    original = deepcopy(model.value_head.state_dict())
    try:
        for name, path in (("terminal", args.terminal_model_out), ("soft_z", args.model_out)):
            model.value_head.load_state_dict(heads[name])
            save_model(path, model, config)
            report["saved_checkpoints"][name] = str(path)
            run.event("experimental_checkpoint_saved", target=name, checkpoint=path)
    finally:
        model.value_head.load_state_dict(original)

    _, seed, games = schedules[2]
    validation = replace(training, seed=seed)
    benchmark = evaluate_mcts_value_heads(
        model, opponent, validation, mcts, games, fitted_heads=heads, isolate_failed_pairs=True, **pool,
    )
    scores = benchmark["paired_scores"]
    benchmark["soft_z_minus_terminal"] = scores["soft_z"] - scores["terminal"] if benchmark["valid"] else None
    report.update(mcts_benchmark=benchmark, mcts_improved=benchmark["beats_zero"])
    if not benchmark["valid"]:
        report["reason"] = "mcts_benchmark_invalid"
    else:
        report["reason"] = "mcts_improved" if benchmark["beats_zero"] else "mcts_no_demonstrated_improvement"
    run.log(compared_mcts_games=benchmark["compared_games_per_arm"], **{
        f"mcts_{arm}_score": score for arm, score in benchmark["paired_scores"].items()
    })

    roots, prefixes, info = collect_value_roots(
        model, opponent, validation, mcts, args.ranking_games, label="ranking reference", raw=True,
        excluded_map_seeds=attempted, positions_per_game=args.positions_per_game, **pool,
    )
    path = run.directory / "ranking_pairs.jsonl"
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        rankings = evaluate_move_preferences(
            search, heads, validation, roots, prefixes, num_envs=args.num_envs, positions=args.ranking_positions,
            branch_rollouts=args.branch_rollouts, rollout_temperature=args.rollout_temperature,
            record=lambda row: stream.write(json.dumps(row, allow_nan=False) + "\n"),
        )
    report["move_preferences"] = {**rankings, "collection": info, "pair_file": str(path)}
    run.log(**{
        f"{arm}_{metric}": measurements[metric]
        for arm, measurements in rankings["arms"].items()
        for metric in ("preference_accuracy", "gap_rmse")
    })
    return report


def fit_saved_mcts_pairs(args, model, opponent, config, training, schedules, run) -> dict[str, Any]:
    from polybot.training.mcts_value import evaluate_mcts_value_heads
    from polybot.training.saved_value_pairs import load_saved_value_pairs, split_value_pairs

    examples, source = load_saved_value_pairs(
        model, training, args.saved_pairs, history=args.visible_action_history,
    )
    _, validation_seed, games = schedules[2]
    validation_seeds = set(range(validation_seed, validation_seed + games // 2))
    if validation_seeds.intersection(source["reserved_map_seeds"]):
        raise ValueError("--validation-seed overlaps the saved dataset's source maps")
    datasets, split = split_value_pairs(
        examples, seed=args.selection_seed, selection_fraction=args.pair_selection_fraction,
    )
    mcts = MCTSConfig(simulations=args.simulations, particles=args.particles)
    run.write_config(
        training=training, model=config, mcts=mcts, source=source, split=split,
        loss="MSE(value, terminal_mean) + weight * MSE(root_value_gap / 2, root_target_gap / 2)",
        difference_loss_weight=args.difference_loss_weight, validation_map_seeds=sorted(validation_seeds),
        frozen_parameters="all except value_head", opponent_value_scale=args.opponent_value_scale,
    )
    print(
        f"saved value pairs | loaded={source['labelled_pairs']} | "
        f"train={split['train']['pairs']} pairs/{len(split['train']['map_seeds'])} maps | "
        f"selection={split['selection']['pairs']} pairs/{len(split['selection']['map_seeds'])} maps",
        flush=True,
    )
    report: dict[str, Any] = {
        "source": source, "split": split, "saved_checkpoint": None, "head_updated": False,
        "mcts_improved": False,
    }
    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size, value_learning_rate=args.learning_rate,
        value_capacity=max(len(datasets["train"]), args.batch_size), value_fit_backoffs=0,
    )
    heads, fits = fit_paired_value_heads(
        model, datasets["train"], datasets["selection"], settings,
        seed=training.seed, difference_weight=args.difference_loss_weight,
        loss_variants=("mse_delta",), preserve_reference=True,
    )
    report["fit"] = fit = fits.get("mse_delta", {})
    if "mse_delta" not in heads:
        return {**report, "reason": "no_finite_value_head"}
    for stage in ("before", "after"):
        fit[stage].update(
            games=len({row.game_key[:2] for row in datasets["selection"]}),
            maps=len(split["selection"]["map_seeds"]), pairs=split["selection"]["pairs"],
        )
    report["head_updated"] = fit["kept_update"]
    run.log(updates=fit["updates"], selected_updates=fit["selected_updates"], **{
        f"{stage}_{metric}": fit[stage][metric]
        for stage in ("before", "after")
        for metric in ("rmse", "gap_rmse", "preference_accuracy", "selection_loss")
    })
    print(
        f"saved value fit | selected_updates={fit['selected_updates']}/{fit['updates']} | "
        f"selection_loss={fit['before']['selection_loss']:.6f}->{fit['after']['selection_loss']:.6f} | "
        f"gap_rmse={fit['before']['gap_rmse']:.4f}->{fit['after']['gap_rmse']:.4f}",
        flush=True,
    )
    original = deepcopy(model.value_head.state_dict())
    try:
        model.value_head.load_state_dict(heads["mse_delta"])
        save_model(args.model_out, model, config)
    finally:
        model.value_head.load_state_dict(original)
    report["saved_checkpoint"] = str(args.model_out)
    report["reason"] = "mcts_benchmark_pending"
    run.event("saved_pair_value_checkpoint", checkpoint=args.model_out, head_updated=report["head_updated"])
    run.finish(report, plots=False)

    benchmark = evaluate_mcts_value_heads(
        model, opponent, replace(training, seed=validation_seed), mcts, games,
        fitted_heads={"mse_delta": heads["mse_delta"]}, isolate_failed_pairs=True,
        num_envs=args.num_envs, num_threads=args.num_threads, visible_action_history=args.visible_action_history,
        max_pending_leaves=args.max_pending_leaves, opponent_value_scale=args.opponent_value_scale,
    )
    report["mcts_benchmark"] = benchmark
    report["mcts_improved"] = benchmark["beats_zero"]
    report["reason"] = (
        "mcts_benchmark_invalid" if not benchmark["valid"]
        else "mcts_improved" if benchmark["beats_zero"] else "mcts_no_demonstrated_improvement"
    )
    run.log(compared_mcts_games=benchmark["compared_games_per_arm"], **{
        f"mcts_{arm}_score": score for arm, score in benchmark["paired_scores"].items()
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--position-source", choices=(
        "raw", "mcts-branches", "mcts-terminal", "mcts-turn-start", "mcts-soft-z", "saved-mcts-pairs",
    ), default="raw")
    parser.add_argument("--saved-pairs", type=Path, help="mcts-value-disagreements output directory")
    parser.add_argument("--pair-selection-fraction", type=float, default=0.25,
                        help="Fraction of whole saved maps reserved for head selection")
    parser.add_argument("--train-games", type=int, default=128)
    parser.add_argument("--selection-games", type=int, default=32)
    parser.add_argument("--validation-games", type=int, default=48)
    parser.add_argument("--selection-seed", type=int, default=447618319)
    parser.add_argument("--validation-seed", type=int, default=682954103)
    parser.add_argument("--positions-per-game", type=int, default=64,
                        help="Total position budget per game, divided across thirds in MCTS modes")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--visible-action-history", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=32, help="Search simulations per decision in MCTS modes")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--max-pending-leaves", type=int, default=4)
    parser.add_argument("--opponent-value-scale", type=float, default=1.0,
                        help="Fixed opponent leaf scale for terminal collection, disagreement probes and benchmarks")
    parser.add_argument("--turn-candidates", type=int, default=3,
                        help="Number of legal first actions compared by turn-reply search")
    parser.add_argument("--turn-max-rollout-actions", type=int, default=96,
                        help="Limit per simulated branch in turn-reply search; incomplete worlds are excluded")
    parser.add_argument("--teacher-simulations", type=int, default=128,
                        help="Deeper simulations per post-reply root in mcts-soft-z")
    parser.add_argument("--ranking-games", type=int, default=8,
                        help="Number of validation games replayed with raw policies to sample ranking roots")
    parser.add_argument("--ranking-positions", type=int, default=12,
                        help="Maximum number of held-out two-action positions for the ranking test")
    parser.add_argument("--disagreement-probes-per-position", type=int, default=8,
                        help="Maximum probes per desired sample within each phase in mcts-branches mode")
    parser.add_argument("--branch-rollouts", type=int, default=3,
                        help="Number of paired sampled raw rollouts per action for branch targets or ranking")
    parser.add_argument("--rollout-temperature", type=float, default=0.5,
                        help="Common raw-policy sampling temperature for both actions in branch rollouts")
    parser.add_argument("--difference-loss-weight", type=float, default=1.0,
                        help="Weight of the branch difference term in the second mcts-branches variant")
    parser.set_defaults(learning_rate=0.0001, max_moves=1600)
    args = parser.parse_args()
    training = training_config_from_args(args)
    saved_pairs_mode = args.position_source == "saved-mcts-pairs"

    if args.model_out.resolve() in {args.model_in.resolve(), args.opponent.resolve()}:
        parser.error("--model-out must be separate from the input checkpoints")
    if args.updates < 1 or args.batch_size < 1 or args.positions_per_game < 1:
        parser.error("--updates, --batch-size and --positions-per-game must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.num_envs < 2 or args.num_threads < 1 or args.max_moves < 1:
        parser.error("require at least two environments, one thread and one move")
    if not 0 <= args.visible_action_history <= 1024:
        parser.error("--visible-action-history must be in [0, 1024]")
    if args.position_source != "raw" and (
        args.simulations < 1 or args.particles < 1 or not 1 <= args.max_pending_leaves <= 8
        or not 0.0 <= args.opponent_value_scale <= 1.0
    ):
        parser.error("require positive simulations and particles, 1–8 pending leaves and opponent value scale in [0, 1]")
    if args.position_source in {"mcts-terminal", "mcts-turn-start", "mcts-soft-z"} and args.positions_per_game < 3:
        parser.error("terminal MCTS collection requires at least three positions per game for phase sampling")
    if args.position_source in {"mcts-turn-start", "mcts-soft-z"} and (
        args.turn_candidates < 2 or args.turn_max_rollout_actions < 1
    ):
        parser.error("turn-reply search requires at least two candidates and a positive rollout limit")
    if args.position_source == "mcts-branches" and (
        args.disagreement_probes_per_position < 1 or args.branch_rollouts < 1
        or not np.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0.0
    ):
        parser.error("require positive probe and rollout counts and a finite positive rollout temperature")
    if args.position_source == "mcts-branches":
        if args.batch_size < 2 or args.batch_size % 2:
            parser.error("mcts-branches requires an even batch size of at least two to keep branch pairs together")
        if not np.isfinite(args.difference_loss_weight) or args.difference_loss_weight <= 0.0:
            parser.error("--difference-loss-weight must be finite and positive")
        args.mse_model_out = args.model_out.with_name(f"{args.model_out.stem}-mse{args.model_out.suffix}")
        if args.mse_model_out.resolve() in {args.model_in.resolve(), args.opponent.resolve(), args.model_out.resolve()}:
            parser.error("derived MSE output must be separate from the input checkpoints and --model-out")
    if saved_pairs_mode:
        if args.saved_pairs is None or not args.saved_pairs.is_dir():
            parser.error("saved-mcts-pairs requires --saved-pairs pointing to the diagnostic directory")
        if args.batch_size < 2 or args.batch_size % 2:
            parser.error("saved pair training requires an even batch size of at least two")
        if not 0 < args.pair_selection_fraction < 1 or args.selection_seed < 0:
            parser.error("require --pair-selection-fraction in (0, 1) and a non-negative split seed")
        if not np.isfinite(args.difference_loss_weight) or args.difference_loss_weight <= 0.0:
            parser.error("--difference-loss-weight must be finite and positive")
    elif args.saved_pairs is not None:
        parser.error("--saved-pairs requires --position-source saved-mcts-pairs")
    if args.position_source == "mcts-soft-z":
        if args.teacher_simulations <= args.simulations or args.teacher_simulations < args.turn_candidates:
            parser.error("--teacher-simulations must exceed --simulations and cover all turn candidates")
        if not 4 <= args.ranking_games <= args.validation_games or args.ranking_games % 2:
            parser.error("--ranking-games must be even and in [4, --validation-games]")
        if (args.ranking_positions < 1 or args.branch_rollouts < 1
                or not np.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0.0):
            parser.error("ranking requires positive position and rollout counts and a finite positive temperature")
        args.terminal_model_out = args.model_out.with_name(f"{args.model_out.stem}-terminal{args.model_out.suffix}")
        if args.terminal_model_out.resolve() in {args.model_in.resolve(), args.opponent.resolve(), args.model_out.resolve()}:
            parser.error("derived terminal output must be separate from the input checkpoints and --model-out")

    schedules = (
        ("train", training.seed, args.train_games),
        ("selection", args.selection_seed, args.selection_games),
        ("validation", args.validation_seed, args.validation_games),
    )
    used_seeds: set[int] = set()
    for label, seed, games in schedules[2:] if saved_pairs_mode else schedules:
        if games < 4 or games % 2:
            parser.error(f"{label} requires at least four games and an even game count")
        if seed < 1 or seed + games // 2 - 1 > 2**31 - 1:
            parser.error(f"{label} map seeds must be in [1, 2**31 - 1]")
        map_seeds = set(range(seed, seed + games // 2))
        if used_seeds & map_seeds:
            parser.error("training, selection and validation map seeds must be disjoint")
        used_seeds.update(map_seeds)

    device = select_device(training.device)
    model, config = load_model(args.model_in, device)
    opponent, _ = load_model(args.opponent, device)
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    opponent.requires_grad_(False)
    run = training_run_from_args(args, "value")
    if args.position_source != "raw":
        fit_mcts = {
            "mcts-branches": fit_mcts_branch_value, "mcts-soft-z": fit_mcts_soft_z_value,
            "saved-mcts-pairs": fit_saved_mcts_pairs,
        }.get(
            args.position_source, fit_mcts_terminal_value,
        )
        report = fit_mcts(args, model, opponent, config, training, schedules, run)
        report["run_dir"] = str(run.finish(report, plots=False))
        print_summary(report)
        return
    run.write_config(training=training, model=config, schedules=schedules)
    datasets = {}
    collection = {}
    collected_seeds: set[int] = set()
    for label, seed, games in schedules:
        datasets[label], collection[label] = collect_value_games(
            model, opponent, replace(training, seed=seed), games,
            positions_per_game=args.positions_per_game, num_envs=args.num_envs,
            num_threads=args.num_threads, visible_action_history=args.visible_action_history,
            label=f"{args.model_in.stem} value {label}",
        )
        map_seeds = set(collection[label]["map_seeds"])
        if map_seeds != set(range(seed, seed + games // 2)) or map_seeds & collected_seeds:
            raise RuntimeError(f"unexpected map schedule in {label} games")
        collected_seeds.update(map_seeds)

    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size,
        value_learning_rate=args.learning_rate, value_validation_games=args.selection_games,
        value_capacity=max(len(datasets["train"]), args.batch_size),
    )
    report: dict[str, Any] = {"collection": collection, "saved_checkpoint": None}
    if any(row["labelled_maps"] < 2 for row in collection.values()):
        report["reason"] = "insufficient_terminal_maps"
    else:
        validation = datasets["validation"]
        before = evaluate_value_head(model, training, validation, batch_size=args.batch_size)
        fit = fit_value_head(
            model, training, datasets["train"], datasets["selection"], settings,
            seed=training.seed, validation_source="raw_selection",
        )
        after = evaluate_value_head(model, training, validation, batch_size=args.batch_size)
        beats_zero = bool(after["valid"] and after["rmse"] < after["zero_rmse"])
        beats_original = bool(before["valid"] and after["valid"] and after["rmse"] < before["rmse"])
        report.update(
            fit=fit, validation_before=before, validation_after=after,
            beats_zero=beats_zero, beats_original=beats_original,
        )
        run.log(
            original_rmse=before.get("rmse"),
            fitted_rmse=after.get("rmse"),
            zero_rmse=after.get("zero_rmse"),
            updates=fit["updates"],
        )
        if fit["kept"] and beats_zero and beats_original:
            save_model(args.model_out, model, config)
            report.update(saved_checkpoint=str(args.model_out), reason="validation_improved")
        else:
            report["reason"] = "validation_gate_failed" if fit["kept"] else fit["reason"]
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary(report)


if __name__ == "__main__":
    main()
