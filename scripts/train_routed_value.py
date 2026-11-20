"""Fit a frozen-encoder value head on fresh, policy-consistent MCTS successor pairs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import secrets
from types import SimpleNamespace

import numpy as np

from _bootstrap import add_local_source_root

add_local_source_root()

from reconstruct_ranked_value import load_source_data, read_source
from train_ranked_value import run_experiment
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import model_fingerprint


def training_options(args, source):
    previous, config = source["args"], source["config"]
    if config["calibration_policy"] != "mcts-zero":
        raise ValueError("the reused calibration cache must label fixed MCTS (0,1), not raw continuations")
    if config["mcts"]["simulations"] != 32:
        raise ValueError("this experiment compares MCTS@32 throughout collection, calibration and arena")
    source_paths = [Path(row["directory"]) for row in config["ranking_sources"]["sources"]]
    inputs = {Path(previous["model_in"]).resolve(), Path(previous["opponent"]).resolve()}
    if args.model_out.resolve() in inputs or args.model_out.exists():
        raise ValueError("--model-out must be a new path distinct from both input checkpoints")
    if args.pairs_dir.resolve() in {path.resolve() for path in source_paths}:
        raise ValueError("new routed pairs must not overwrite a historical source dataset")
    if args.collect_pairs == 0 and not (args.pairs_dir / "config.json").is_file():
        raise ValueError("--collect-pairs 0 needs a previously collected routed dataset")
    values = dict(
        model_in=Path(previous["model_in"]), opponent=Path(previous["opponent"]), model_out=args.model_out,
        saved_pairs=source_paths if args.collect_pairs else [args.pairs_dir],
        calibration_cache=args.calibration_cache, require_calibration_cache=True,
        routed_value=True, pair_source_policy="mcts-zero", pair_rollout_temperature=0.0,
        pair_collection_mode="pooled", collect_decisive_pairs=args.collect_pairs,
        minimum_collected_pairs=min(args.collect_pairs, args.minimum_pairs) if args.collect_pairs else 0,
        collected_pairs_dir=args.pairs_dir, pair_source_seed=args.pair_source_seed,
        pair_source_games=args.source_games, pair_source_batch_games=previous["num_envs"] - previous["num_envs"] % 2,
        pair_probes_per_game=args.probes_per_game, pair_min_rollouts=args.rollouts,
        pair_screen_rollouts=args.rollouts, pair_max_rollouts=args.max_rollouts, pair_rollout_step=4,
        pair_candidate_pool=max(args.collect_pairs, args.candidate_pool), pair_scout_fraction=0.35,
        pair_confidence=0.95, pair_max_pairs_per_map=2, pair_collection_hours=args.collection_hours,
        pair_split_seed=args.split_seed, pair_selection_fraction=0.25,
        pair_value_loss_weight=1.0, minimum_informative_train=args.minimum_informative_train,
        minimum_informative_selection=args.minimum_informative_selection, minimum_arena_maps=args.minimum_arena_maps,
        calibration_policy="mcts-zero", calibration_games=config["schedules"]["train"][1],
        calibration_selection_games=config["schedules"]["selection"][1],
        calibration_selection_seed=config["schedules"]["selection"][0], calibration_weights=args.calibration_weights,
        calibration_rmse_regression=0.0, calibration_bias_regression=0.02,
        validation_seed=args.validation_seed, validation_games=args.validation_games,
        positions_per_game=previous["positions_per_game"], updates=args.updates,
        batch_size=args.batch_size, learning_rate=args.learning_rate, log_interval=args.log_interval,
        simulations=config["mcts"]["simulations"], particles=config["mcts"]["particles"],
        **{key: previous[key] for key in ("num_envs", "num_threads", "max_pending_leaves", "visible_action_history")},
    )
    return SimpleNamespace(**values)


def run_training(args, model, opponent, source, run):
    options = training_options(args, source)
    model.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    # Verify original checkpoints, all saved pairs and both cache files before collecting anything.
    data = load_source_data(source, model, opponent, device=args.device, calibration_cache=args.calibration_cache)
    options.mcts_config = data["mcts"]
    arena_maps = set(range(args.validation_seed, args.validation_seed + args.validation_games // 2))
    if arena_maps & set(data["reserved_map_seeds"]):
        raise ValueError("the new arena overlaps a historical ranking, calibration or arena map")
    reused_cache_maps = {seed for summary in data["cache_reports"].values() for seed in summary["attempted_map_seeds"]}
    reused_cache_maps.update(seed for name in ("train", "selection") for seed in range(
        source["config"]["schedules"][name][0],
        source["config"]["schedules"][name][0] + source["config"]["schedules"][name][1] // 2))
    options.extra_reserved_seeds = sorted(set(data["reserved_map_seeds"]) - reused_cache_maps)
    training = replace(data["training"], learning_rate=args.learning_rate, batch_size=args.batch_size)
    fingerprints = model_fingerprint(model), model_fingerprint(opponent)
    model.value_head.requires_grad_(True)
    report = run_experiment(options, model, opponent, model.config, training, run)
    report["input_weights_unchanged"] = fingerprints == (model_fingerprint(model), model_fingerprint(opponent))
    report["source_run"] = str(source["directory"])
    report["pinned_cache"] = data["cache_reports"]
    report["old_cache_reused"] = True
    if not report["input_weights_unchanged"]:
        raise RuntimeError("value experiment changed an input network in memory")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps", "auto"), default="cuda")
    parser.add_argument("--collect-pairs", type=int, default=192,
                        help="Fresh pooled successor pairs; zero reuses --pairs-dir without games until the arena.")
    parser.add_argument("--minimum-pairs", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=256)
    parser.add_argument("--source-games", type=int, default=192)
    parser.add_argument("--probes-per-game", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-rollouts", type=int, default=16)
    parser.add_argument("--collection-hours", type=float, default=8)
    parser.add_argument("--pair-source-seed", type=int, default=178621901)
    parser.add_argument("--split-seed", type=int, default=218621901)
    parser.add_argument("--minimum-informative-train", type=int, default=24)
    parser.add_argument("--minimum-informative-selection", type=int, default=12)
    parser.add_argument("--validation-seed", type=int,
                        help="Fresh random arena seed when omitted; the resolved seed is saved for reproduction.")
    parser.add_argument("--validation-games", type=int, default=64)
    parser.add_argument("--minimum-arena-maps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--calibration-weights", default="0.01,0.05,0.2,1.0")
    parser.add_argument("--log-interval", type=int, default=64)
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-routed-value")
    args = parser.parse_args()
    if args.validation_seed is None:
        args.validation_seed = secrets.randbelow(max(1, 2**31 - args.validation_games // 2 - 1)) + 1
    try:
        args.calibration_weights = tuple(float(item) for item in args.calibration_weights.split(","))
        source = read_source(args.source_run)
        training_options(args, source)
        if (args.collect_pairs < 0 or args.minimum_pairs < 2 or args.candidate_pool < 2
                or (args.collect_pairs > 0 and args.collect_pairs < args.minimum_pairs) or args.split_seed < 0
                or args.source_games < 2 or args.source_games % 2 or args.probes_per_game < 1
                or args.rollouts < 4 or args.max_rollouts > 128 or args.max_rollouts < args.rollouts
                or (args.max_rollouts - args.rollouts) % 4 or args.updates < 1 or args.batch_size < 1
                or args.minimum_informative_train < 1 or args.minimum_informative_selection < 1
                or args.minimum_arena_maps < 2 or args.validation_games < 2 * args.minimum_arena_maps
                or args.validation_games % 2 or args.log_interval < 1 or not 0 < args.learning_rate < 1
                or not 0 < args.collection_hours < float("inf") or not args.calibration_weights
                or len(set(args.calibration_weights)) != len(args.calibration_weights)
                or any(not np.isfinite(weight) or weight <= 0 for weight in args.calibration_weights)):
            raise ValueError("invalid collection, evidence, optimizer or arena budget")
        for seed, games in ((args.pair_source_seed, args.source_games), (args.validation_seed, args.validation_games)):
            if not 1 <= seed or seed + games // 2 > 2**31:
                raise ValueError("new collection and arena require valid 31-bit map seeds")
        protected = {args.source_run.resolve(), args.pairs_dir.resolve(), args.calibration_cache.resolve()}
        if (args.project / "value" / args.name).resolve() in protected:
            raise ValueError("run artifacts must not overwrite input data")
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    device = select_device(args.device)
    model, _ = load_model(Path(source["args"]["model_in"]), device)
    opponent, _ = load_model(Path(source["args"]["opponent"]), device)
    run = training_run_from_args(args, "value")
    report = run_training(args, model, opponent, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary({key: report.get(key) for key in ("run_dir", "reason", "informative_pairs", "routing_counts",
        "ready_for_mcts", "saved_checkpoint", "diagnostic_checkpoint", "input_weights_unchanged",
        "gameplay_admission", "score_delta_vs_original", "score_delta_vs_zero")})


if __name__ == "__main__":
    main()
