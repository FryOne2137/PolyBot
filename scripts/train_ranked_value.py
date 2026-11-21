"""Fit informative branch rankings with terminal calibration and a gated MCTS arena."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model, save_model
from polybot.training.calibrated_value import (
    cached_terminal_positions, fit_calibrated_heads, informative_pairs, load_ranking_sources,
)
from polybot.training.cli import (
    add_environment_options, add_run_options, print_summary, training_config_from_args, training_run_from_args,
)
from polybot.training.flat_experiments import FlatImprovements, model_fingerprint
from polybot.training.decisive_value_pairs import DecisivePairCollection, collect_decisive_value_pairs
from polybot.training.mcts_value import evaluate_mcts_value_heads
from polybot.training.saved_value_pairs import split_value_pairs
from polybot.training.value_routing_audit import load_routing_inputs, routing_counts
from polybot.training.flat_value import _paired_value_tensors
from polybot.training.value_routing import paired_value_admission


def run_experiment(args, model, opponent, config, training, run):
    routed = getattr(args, "routed_value", False)
    if routed and (args.calibration_policy != "mcts-zero" or args.pair_rollout_temperature != 0
                   or getattr(args, "pair_source_policy", "raw") != "mcts-zero"):
        raise ValueError("routed value requires fixed MCTS (0,1) sources and greedy raw opponents for all labels")
    collect_target = getattr(args, "collect_decisive_pairs", 0)
    examples, source = load_ranking_sources(model, training, args.saved_pairs, history=args.visible_action_history)
    weighted_pairs = (getattr(args, "pair_collection_mode", "confirmed") == "pooled" or any(
        item.get("source_config", {}).get("label_mode") == "weighted_soft" for item in source["sources"]
    ))
    pair_value_weight = getattr(args, "pair_value_loss_weight", None)
    if pair_value_weight is None:
        pair_value_weight = 1.0 if weighted_pairs else 0.0
    if collect_target:
        ranking, split = {"train": [], "selection": []}, {"pending_new_collection": True}
    else:
        ranking, split = split_value_pairs(examples, seed=args.pair_split_seed, selection_fraction=args.pair_selection_fraction)
        if not weighted_pairs:
            ranking = {name: informative_pairs(rows) for name, rows in ranking.items()}
    report = {
        "ranking_sources": source, "ranking_split": split, "saved_checkpoint": None, "head_updated": False,
        "benchmark_performed": False, "beats_zero": None, "calibration": {},
        "informative_pairs": {name: len(informative_pairs(rows)) // 2 for name, rows in ranking.items()},
    }
    schedules = {
        "train": (training.seed, args.calibration_games),
        "selection": (args.calibration_selection_seed, args.calibration_selection_games),
        "arena": (args.validation_seed, args.validation_games),
    }
    planned = {name: set(range(seed, seed + games // 2)) for name, (seed, games) in schedules.items()}
    extra_reserved = set(getattr(args, "extra_reserved_seeds", ()))
    reserved = set(source["reserved_map_seeds"]) | extra_reserved
    for name, maps in planned.items():
        if maps & reserved:
            raise ValueError(f"{name} seeds overlap ranking or other calibration/evaluation maps")
        reserved.update(maps)
    mcts = getattr(args, "mcts_config", None) or MCTSConfig(simulations=args.simulations, particles=args.particles)
    pool = dict(num_envs=args.num_envs, num_threads=args.num_threads, visible_action_history=args.visible_action_history,
                max_pending_leaves=args.max_pending_leaves, opponent_value_scale=1.0)
    configuration = dict(training=training, model=config, mcts=mcts, ranking_sources=source, ranking_split=split,
                     checkpoint_fingerprints={"model": model_fingerprint(model), "opponent": model_fingerprint(opponent)},
                     schedules=schedules, calibration_weights=args.calibration_weights,
                     calibration_policy=args.calibration_policy, rmse_regression=args.calibration_rmse_regression,
                     bias_regression=args.calibration_bias_regression, frozen_parameters="all except value_head",
                     pair_value_loss_weight=pair_value_weight, balance_pair_maps=weighted_pairs)
    if routed:
        configuration.update(value_routing="native_agent_id", label_policy="fixed_mcts_0_1_vs_greedy_raw")
        report.update(value_routing="native_agent_id", ready_for_mcts=False)
    run.write_config(**configuration)
    if collect_target:
        print(f"value ranking collection | target={collect_target} | minimum={args.minimum_collected_pairs}", flush=True)
    else:
        counts = report["informative_pairs"]
        print(f"value ranking data | train_pairs={counts['train']} | selection_pairs={counts['selection']} | "
              "split=whole_map", flush=True)
    if not all(ranking.values()) and not collect_target:
        return {**report, "reason": "insufficient_informative_ranking_pairs"}
    calibration = {}
    attempted = set(source["reserved_map_seeds"]) | extra_reserved
    for name in ("train", "selection"):
        seed, games = schedules[name]
        excluded = attempted | set().union(*(maps for key, maps in planned.items() if key != name))
        calibration[name], summary = cached_terminal_positions(
            model, opponent, replace(training, seed=seed), mcts, games,
            cache_directory=args.calibration_cache, excluded_map_seeds=excluded,
            positions_per_game=args.positions_per_game, pool=pool, policy=args.calibration_policy,
            require_existing=getattr(args, "require_calibration_cache", False),
        )
        report["calibration"][name] = summary
        attempted.update(summary["attempted_map_seeds"])
        run.event("calibration_collection", split=name, **summary)
        print(f"value calibration data | split={name} | positions={len(calibration[name])} | "
              f"maps={summary['labelled_maps']} | cache_loaded={int(summary['cache_loaded'])}", flush=True)
        if not summary["valid"] or summary["labelled_maps"] < 2 or not calibration[name]:
            return {**report, "reason": f"{name}_insufficient_calibration_data"}
    if collect_target:
        collection = _collection_settings(args)
        summary = collect_decisive_value_pairs(
            model, opponent, training, mcts, output=args.collected_pairs_dir, settings=collection,
            excluded_map_seeds=attempted | planned["arena"], pool=pool,
            **(dict(resume_rollouts_only=True) if getattr(args, "resume_rollouts_only", False) else {}),
        )
        report["decisive_collection"] = summary
        run.event("decisive_pair_collection", **summary)
        collected_count = summary.get("usable_pairs", summary["confirmed_pairs"]) if weighted_pairs else summary["confirmed_pairs"]
        if collected_count < args.minimum_collected_pairs:
            return {**report, "reason": "insufficient_usable_new_pairs" if weighted_pairs else "insufficient_confirmed_new_pairs"}
        examples, new_source = load_ranking_sources(model, training, [args.collected_pairs_dir], history=args.visible_action_history)
        if len(examples) != 2 * collected_count:
            raise ValueError("confirmed pair count differs from the saved label count")
        if set(new_source["reserved_map_seeds"]) & (attempted | planned["arena"]):
            raise ValueError("new ranking source overlaps old source, calibration or arena maps")
        ranking, split = split_value_pairs(examples, seed=args.pair_split_seed, selection_fraction=args.pair_selection_fraction)
        if not weighted_pairs:
            ranking = {name: informative_pairs(rows) for name, rows in ranking.items()}
        report.update(previous_ranking_sources=source, ranking_sources=new_source, ranking_split=split,
                      informative_pairs={name: len(informative_pairs(rows)) // 2 for name, rows in ranking.items()},
                      training_pairs={name: len(rows) // 2 for name, rows in ranking.items()})
        configuration.update(previous_ranking_sources=source, ranking_sources=new_source, ranking_split=split,
                             decisive_collection=summary, collection_settings=asdict(collection))
        run.write_config(**configuration)
        counts = report["training_pairs"]
        print(f"{'weighted' if weighted_pairs else 'confirmed'} ranking data | train_pairs={counts['train']} | selection_pairs={counts['selection']} | "
              f"target_reached={int(summary['reached_target'])}", flush=True)
        if not all(ranking.values()):
            return {**report, "reason": "insufficient_informative_ranking_pairs"}
    routing_lookup = None
    if routed:
        ranking_source = report["ranking_sources"]
        for item in ranking_source["sources"]:
            saved = item["source_config"]
            identity = saved.get("collector_identity", {})
            if (saved.get("continuation") != "fixed_mcts_0_1_vs_greedy_raw"
                    or saved.get("source_policy") != "mcts-zero"
                    or identity.get("model") != model_fingerprint(model)
                    or identity.get("opponent") != model_fingerprint(opponent)):
                raise ValueError("routed training pairs must come from the same frozen MCTS (0,1) reference; collect a new dataset")
        counts = report["informative_pairs"]
        if (counts["train"] < args.minimum_informative_train
                or counts["selection"] < args.minimum_informative_selection):
            return {**report, "reason": "insufficient_ranking_evidence", "required_informative_pairs": {
                "train": args.minimum_informative_train, "selection": args.minimum_informative_selection}}
        routing_lookup = load_routing_inputs(model, opponent, training, ranking, ranking_source["sources"],
            history=args.visible_action_history, require_agent_id=True)
        report["routing_counts"] = {name: routing_counts(_paired_value_tensors(rows, "cpu")[-1], routing_lookup)
                                    for name, rows in ranking.items()}
        configuration.update(routing_counts=report["routing_counts"],
                             successor_files=[branch for pair in routing_lookup.values() for branch in pair])
        run.write_config(**configuration)
    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size, value_learning_rate=args.learning_rate,
        value_capacity=max(args.batch_size, len(calibration["train"])), value_fit_backoffs=0,
    )

    def record_routed_head(weight, step, metrics, state):
        if step % args.log_interval == 0 or step == args.updates:
            directory = run.directory / f"calibration-{weight!r}"
            directory.mkdir(exist_ok=True)
            torch.save({"value_head": state, "update": step, "metrics": metrics,
                        "input_checkpoint_fingerprint": configuration["checkpoint_fingerprints"]["model"],
                        "value_routing": "native_agent_id", "diagnostic_only": True}, directory / "last-head.pt")

    state, report["fit"] = fit_calibrated_heads(
        model, ranking, calibration, training, settings, weights=args.calibration_weights,
        rmse_regression=args.calibration_rmse_regression, bias_regression=args.calibration_bias_regression,
        run=run, log_interval=args.log_interval,
        pair_value_weight=pair_value_weight, balance_pair_maps=weighted_pairs,
        **(dict(routing_lookup=routing_lookup, update_callback=record_routed_head) if routed else {}),
    )
    if state is None:
        return {**report, "reason": "no_calibrated_ranking_improvement"}
    original = deepcopy(model.value_head.state_dict())
    try:
        model.value_head.load_state_dict(state)
        candidate_path = run.directory / "diagnostic-candidate.pt" if routed else args.model_out
        save_model(candidate_path, model, config)
    finally:
        model.value_head.load_state_dict(original)
    report.update(saved_checkpoint=None if routed else str(args.model_out),
                  diagnostic_checkpoint=str(candidate_path), head_updated=True, reason="arena_pending")
    run.finish(report, plots=False)
    benchmark = evaluate_mcts_value_heads(
        model, opponent, replace(training, seed=args.validation_seed), mcts, args.validation_games,
        fitted_heads={"calibrated": state, "original": original}, isolate_failed_pairs=True, **pool,
    )
    report.update(mcts_benchmark=benchmark, benchmark_performed=True,
                  beats_zero=benchmark["beats_zero"] if benchmark["valid"] else None,
                  reason="arena_complete" if benchmark["valid"] else "arena_invalid")
    if benchmark["valid"]:
        scores = benchmark["paired_scores"]
        report["beats_zero"] = scores["calibrated"] > scores["zero"]
        report["score_delta_vs_original"] = scores["calibrated"] - scores["original"]
        report["score_delta_vs_zero"] = scores["calibrated"] - scores["zero"]
    if routed:
        admission = paired_value_admission(benchmark, planned_games=args.validation_games,
                                           minimum_maps=args.minimum_arena_maps)
        report.update(gameplay_admission=admission, ready_for_mcts=admission["accepted"], reason=admission["reason"])
        if admission["accepted"]:
            try:
                model.value_head.load_state_dict(state)
                save_model(args.model_out, model, config)
                report["saved_checkpoint"] = str(args.model_out)
            finally:
                model.value_head.load_state_dict(original)
    return report


def _collection_settings(args):
    return DecisivePairCollection(
        target=args.collect_decisive_pairs, seed=args.pair_source_seed, source_games=args.pair_source_games,
        source_batch_games=args.pair_source_batch_games, probes_per_game=args.pair_probes_per_game,
        min_rollouts=args.pair_min_rollouts, max_rollouts=args.pair_max_rollouts, rollout_step=args.pair_rollout_step,
        confidence=args.pair_confidence, temperature=args.pair_rollout_temperature,
        max_pairs_per_map=args.pair_max_pairs_per_map, hours=args.pair_collection_hours,
        mode=getattr(args, "pair_collection_mode", "confirmed"),
        candidate_pool=getattr(args, "pair_candidate_pool", 100),
        screening_rollouts=getattr(args, "pair_screen_rollouts", 2),
        scout_fraction=getattr(args, "pair_scout_fraction", 0.2),
        source_policy=getattr(args, "pair_source_policy", "raw"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--saved-pairs", type=Path, action="append", required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--require-calibration-cache", action="store_true",
                        help="Fail before collection if the matching terminal cache is missing.")
    parser.add_argument("--collect-decisive-pairs", type=int, default=0)
    parser.add_argument("--collected-pairs-dir", type=Path)
    parser.add_argument("--minimum-collected-pairs", type=int,
                        help="Minimum labelled pairs to fit after the time limit; defaults to the target.")
    parser.add_argument("--pair-collection-mode", choices=("confirmed", "pooled"), default="confirmed")
    parser.add_argument("--pair-candidate-pool", type=int, default=100)
    parser.add_argument("--pair-screen-rollouts", type=int, default=2)
    parser.add_argument("--pair-scout-fraction", type=float, default=0.2)
    parser.add_argument("--pair-value-loss-weight", type=float,
                        help="MSE weight on saved branch means; pooled mode defaults to 1, legacy to 0.")
    parser.add_argument("--pair-source-seed", type=int, default=1838127001)
    parser.add_argument("--pair-source-games", type=int, default=256)
    parser.add_argument("--pair-source-batch-games", type=int, default=8)
    parser.add_argument("--pair-probes-per-game", type=int, default=12)
    parser.add_argument("--pair-min-rollouts", type=int, default=4)
    parser.add_argument("--pair-max-rollouts", type=int, default=16)
    parser.add_argument("--pair-rollout-step", type=int, default=4)
    parser.add_argument("--pair-confidence", type=float, default=0.95)
    parser.add_argument("--pair-rollout-temperature", type=float, default=0.5)
    parser.add_argument("--pair-max-pairs-per-map", type=int, default=2)
    parser.add_argument("--pair-collection-hours", type=float, default=8.0)
    parser.add_argument("--calibration-policy", choices=("mcts-zero", "raw"), default="mcts-zero")
    parser.add_argument("--calibration-games", type=int, default=24)
    parser.add_argument("--calibration-selection-games", type=int, default=8)
    parser.add_argument("--calibration-selection-seed", type=int, default=2038127001)
    parser.add_argument("--validation-games", type=int, default=16)
    parser.add_argument("--validation-seed", type=int, default=2108127001)
    parser.add_argument("--pair-split-seed", type=int, default=1768127001)
    parser.add_argument("--pair-selection-fraction", type=float, default=0.25)
    parser.add_argument("--positions-per-game", type=int, default=24)
    parser.add_argument("--calibration-weights", default="0.01,0.05,0.2,1.0")
    parser.add_argument("--calibration-rmse-regression", type=float, default=0.0)
    parser.add_argument("--calibration-bias-regression", type=float, default=0.02)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-pending-leaves", type=int, default=4)
    parser.add_argument("--visible-action-history", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.set_defaults(learning_rate=0.0001, seed=1968127001, max_moves=1600)
    args = parser.parse_args()
    if args.pair_value_loss_weight is not None and (not np.isfinite(args.pair_value_loss_weight) or args.pair_value_loss_weight < 0):
        parser.error("--pair-value-loss-weight must be finite and non-negative")
    if args.collect_decisive_pairs < 0:
        parser.error("--collect-decisive-pairs must be non-negative")
    if args.collect_decisive_pairs:
        if args.collected_pairs_dir is None or args.collected_pairs_dir.resolve() in {path.resolve() for path in args.saved_pairs}:
            parser.error("new collection needs --collected-pairs-dir distinct from every --saved-pairs directory")
        if args.minimum_collected_pairs is None:
            args.minimum_collected_pairs = args.collect_decisive_pairs
        if not 2 <= args.minimum_collected_pairs <= args.collect_decisive_pairs:
            parser.error("require 2 <= --minimum-collected-pairs <= --collect-decisive-pairs")
        try:
            _collection_settings(args)
        except ValueError as error:
            parser.error(str(error))
    try:
        args.calibration_weights = tuple(float(value) for value in args.calibration_weights.split(","))
    except ValueError:
        parser.error("--calibration-weights must contain comma-separated numbers")
    if (not args.calibration_weights or len(set(args.calibration_weights)) != len(args.calibration_weights)
            or any(not np.isfinite(weight) or weight <= 0 for weight in args.calibration_weights)):
        parser.error("calibration weights must be unique finite positive values")
    if not 0 < args.pair_selection_fraction < 1 or args.pair_split_seed < 0:
        parser.error("require a selection fraction in (0, 1) and non-negative split seed")
    if args.model_out.resolve() in {args.model_in.resolve(), args.opponent.resolve()}:
        parser.error("--model-out must differ from both inputs")
    if len({path.resolve() for path in args.saved_pairs}) != len(args.saved_pairs) or any(not path.is_dir() for path in args.saved_pairs):
        parser.error("--saved-pairs must identify distinct existing directories")
    if (args.updates < 1 or args.batch_size < 1 or args.positions_per_game < 3 or args.simulations < 1
            or args.particles < 1 or args.num_envs < 2 or args.num_threads < 1 or args.log_interval < 1
            or args.max_moves < 1 or not 1 <= args.max_pending_leaves <= 8
            or not 0 <= args.visible_action_history <= 1024):
        parser.error("invalid training, collection, search or history budget")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if any(not np.isfinite(value) or value < 0 for value in (
        args.calibration_rmse_regression, args.calibration_bias_regression,
    )):
        parser.error("calibration regression tolerances must be finite and non-negative")
    training = training_config_from_args(args)
    for seed, games in ((training.seed, args.calibration_games),
                        (args.calibration_selection_seed, args.calibration_selection_games),
                        (args.validation_seed, args.validation_games)):
        if games < 4 or games % 2 or seed < 1 or seed + games // 2 > 2**31:
            parser.error("each collection/arena requires even game counts >= 4 and valid 31-bit map seeds")
    device = select_device(training.device)
    model, config = load_model(args.model_in, device)
    opponent, _ = load_model(args.opponent, device)
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    opponent.requires_grad_(False)
    run = training_run_from_args(args, "value")
    report = run_experiment(args, model, opponent, config, training, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary(report)


if __name__ == "__main__":
    main()
