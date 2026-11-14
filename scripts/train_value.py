"""Fit only a checkpoint's value head on terminal raw-policy games."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import TrainingConfig
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
from polybot.training.flat_value import EncodedValuePosition, evaluate_value_head, fit_value_head


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--train-games", type=int, default=128)
    parser.add_argument("--selection-games", type=int, default=32)
    parser.add_argument("--validation-games", type=int, default=48)
    parser.add_argument("--selection-seed", type=int, default=447618319)
    parser.add_argument("--validation-seed", type=int, default=682954103)
    parser.add_argument("--positions-per-game", type=int, default=64)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--visible-action-history", type=int, default=32)
    parser.set_defaults(learning_rate=0.0001, max_moves=1600)
    args = parser.parse_args()
    training = training_config_from_args(args)

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

    schedules = (
        ("train", training.seed, args.train_games),
        ("selection", args.selection_seed, args.selection_games),
        ("validation", args.validation_seed, args.validation_games),
    )
    used_seeds: set[int] = set()
    for label, seed, games in schedules:
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
