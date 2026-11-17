"""Evaluate one checkpoint against random play or a second checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig
from polybot.evaluation.matches import evaluate_direct_matches, evaluate_matches
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_environment_options, print_summary, training_config_from_args
from polybot.training.routed_self_play_pool import evaluate_matches_pool


def _map_seed_group(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(seed) for seed in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integer map seeds") from error
    if any(not 1 <= seed <= 2**31 - 1 for seed in seeds):
        raise argparse.ArgumentTypeError("map seeds must be in [1, 2**31 - 1]")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint; write no games or metrics files")
    add_environment_options(parser)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, help="optional MCTS opponent; default is random")
    parser.add_argument(
        "--mode",
        choices=(
            "mcts", "flat", "mcts-vs-raw", "mcts-diagnostics",
            "mcts-value-ablation", "mcts-value-diagnostics", "mcts-first-divergence", "mcts-value-matrix",
            "mcts-branch-agreement", "mcts-value-regression",
        ),
        default="mcts",
        help="Evaluate search, raw policy, search gain, or belief/value diagnostics (leaf scales 1, 0.25, 0)",
    )
    parser.add_argument("--games", type=int, default=10, help="Game count outside mcts-value-matrix")
    parser.add_argument("--branch-pairs", type=Path, help="Saved train_value.py pair JSONL for mcts-branch-agreement")
    parser.add_argument("--branch-positions", type=int, default=6, help="Maximum saved pairs to replay or games to scan for a first value divergence")
    parser.add_argument("--old-value-model", type=Path, help="Checkpoint before fitting value, for mcts-value-regression")
    parser.add_argument(
        "--branch-rollouts", type=int,
        help="Re-run raw and MCTS this many times per action on saved positions that raw distinguished",
    )
    parser.add_argument(
        "--branch-rollout-temperature", type=float,
        help="Raw action temperature in paired branch rollouts (default: 0.5)",
    )
    parser.add_argument(
        "--map-seed-group", type=_map_seed_group, action="append", metavar="SEED[,SEED...]",
        help="In mcts-value-matrix, play both sides per map; repeat for separate evaluation seed resets",
    )
    parser.add_argument(
        "--positions-per-game", type=int, default=4,
        help="Maximum diagnostic positions per game (disagreements in mcts-diagnostics, raw turns in mcts-value-diagnostics)",
    )
    parser.add_argument(
        "--play-disagreement-games", action="store_true",
        help="In mcts-value-diagnostics, play both actions on the real map and finish with raw policies",
    )
    parser.add_argument(
        "--continuation-value-scale", type=float, choices=(0.0, 1.0),
        help="Common MCTS leaf value scale for both continuations in mcts-first-divergence (default: 1)",
    )
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--mcts-threads", type=int,
        help="Persistent native worker threads (SelfPlayPool for model-vs-model)",
    )
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument(
        "--mcts-top-actions", type=int, default=0,
        help="PolyBot-only candidate-prior experiment; applies to both neural players.",
    )
    parser.add_argument("--mcts-random-actions", type=int, default=0)
    parser.add_argument("--mcts-min-prior-mass", type=float, default=0.0)
    parser.add_argument(
        "--model-b-mcts-top-actions", type=int,
        help="Override model B only; use 0 to compare filtered A against full-search B.",
    )
    parser.add_argument("--model-b-mcts-random-actions", type=int)
    parser.add_argument("--num-envs", type=int, default=128, help="Maximum parallel model-vs-model games")
    parser.add_argument("--max-pending-leaves", type=int, default=4)
    parser.add_argument("--visible-action-history", type=int, default=32)
    parser.add_argument("--belief-retries", type=int, default=4)
    parser.add_argument(
        "--strategic-prior-weight", type=float, default=0.0,
        help="Keep at 0.0 for a pure checkpoint comparison; nonzero values evaluate a hand-written curriculum.",
    )
    parser.add_argument(
        "--teacher-prior-weight", type=float, default=0.0,
        help="Set to a nonzero value only when evaluating the anchored search policy.",
    )
    args = parser.parse_args()
    diagnostic_modes = {"mcts-diagnostics", "mcts-value-diagnostics"}
    if args.mode != "mcts" and args.model_b is None:
        parser.error(f"--mode {args.mode} requires --model-b")
    if args.mode in diagnostic_modes and args.positions_per_game < 1:
        parser.error("--positions-per-game must be positive")
    if args.play_disagreement_games and args.mode != "mcts-value-diagnostics":
        parser.error("--play-disagreement-games requires --mode mcts-value-diagnostics")
    if args.continuation_value_scale is not None and args.mode != "mcts-first-divergence":
        parser.error("--continuation-value-scale requires --mode mcts-first-divergence")
    if args.mode == "mcts-value-matrix":
        if not args.map_seed_group:
            parser.error("--mode mcts-value-matrix requires --map-seed-group")
        seeds = [seed for group in args.map_seed_group for seed in group]
        if len(set(seeds)) != len(seeds):
            parser.error("each map seed must appear once across --map-seed-group arguments")
    elif args.map_seed_group is not None:
        parser.error("--map-seed-group requires --mode mcts-value-matrix")
    value_regression = args.mode == "mcts-value-regression"
    if value_regression:
        if args.old_value_model is None:
            parser.error("mcts-value-regression requires --old-value-model")
        if args.games < 2 or args.games % 2 or args.positions_per_game < 3 or args.num_envs < 3:
            parser.error("mcts-value-regression requires paired games, at least three positions per game and three environments")
        if args.branch_pairs is not None:
            parser.error("--branch-pairs requires --mode mcts-branch-agreement")
    elif args.old_value_model is not None:
        parser.error("--old-value-model requires --mode mcts-value-regression")

    if args.mode == "mcts-branch-agreement":
        if args.branch_pairs is None or not args.branch_pairs.is_file():
            parser.error("--mode mcts-branch-agreement requires an existing --branch-pairs file")
        if args.num_envs < 2:
            parser.error("mcts-branch-agreement requires --num-envs >= 2")
    if args.mode == "mcts-branch-agreement" or value_regression:
        if args.branch_positions < 1:
            parser.error("--branch-positions must be positive")
        if args.branch_rollouts is not None and args.branch_rollouts < 1:
            parser.error("--branch-rollouts must be positive")
        if args.branch_rollout_temperature is not None and (
            (args.branch_rollouts is None and not value_regression)
            or not math.isfinite(args.branch_rollout_temperature)
            or args.branch_rollout_temperature <= 0.0
        ):
            parser.error("--branch-rollout-temperature must be finite and positive, and requires rollouts")
    elif any(value is not None for value in (args.branch_pairs, args.branch_rollouts, args.branch_rollout_temperature)):
        parser.error("branch rollout options require mcts-branch-agreement or mcts-value-regression")

    training = training_config_from_args(args)
    device = select_device(training.device)
    model_a, _ = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)[0] if args.model_b is not None else None
    if args.mode == "flat":
        summary = evaluate_direct_matches(
            model_a,
            model_b,
            device,
            training,
            args.games,
            model_a_label=args.model_a.stem,
            model_b_label=args.model_b.stem,
            visible_action_history=args.visible_action_history,
            num_threads=args.mcts_threads,
            max_actions=args.mcts_max_actions,
        )
        print_summary(summary)
        return
    # Greedy selection makes repeated evaluation on the same seeds comparable.
    mcts = MCTSConfig(
        simulations=args.simulations,
        particles=args.particles,
        temperature=0.0,
        belief_particle_temperature=0.0,
        inference_batch_size=args.inference_batch_size,
        native_threads=args.mcts_threads,
        native_max_actions=args.mcts_max_actions,
        search_top_actions=args.mcts_top_actions,
        search_random_actions=args.mcts_random_actions,
        search_min_prior_mass=args.mcts_min_prior_mass,
        strategic_prior_weight=args.strategic_prior_weight,
        teacher_prior_weight=args.teacher_prior_weight,
        capture_city_prior_bonus=args.capture_city_prior_bonus,
        capture_capital_prior_bonus=args.capture_capital_prior_bonus,
        found_city_prior_bonus=args.found_city_prior_bonus,
        attack_prior_bonus=args.attack_prior_bonus,
        end_turn_prior_penalty=args.end_turn_prior_penalty,
    )
    if model_b is None:
        # A random opponent is not a neural agent and retains the small scalar
        # diagnostic path. Checkpoint-vs-checkpoint evaluation uses the fast,
        # routed native pool below.
        summary = evaluate_matches(
            model_a,
            device,
            training,
            mcts,
            args.games,
            None,
            visible_action_history=args.visible_action_history,
        )
    else:
        pool_options = dict(
            num_threads=args.mcts_threads,
            max_pending_leaves=args.max_pending_leaves,
            visible_action_history=args.visible_action_history,
            belief_retries=args.belief_retries,
            max_envs=args.num_envs,
            model_b_search_top_actions=args.model_b_mcts_top_actions,
            model_b_search_random_actions=args.model_b_mcts_random_actions,
            model_a_label=args.model_a.stem,
            model_b_label=args.model_b.stem,
        )
        if args.mode in diagnostic_modes:
            from polybot.evaluation.mcts_diagnostics import (
                evaluate_mcts_diagnostics,
                evaluate_mcts_value_diagnostics,
            )

            diagnose = (
                evaluate_mcts_value_diagnostics if args.mode == "mcts-value-diagnostics"
                else evaluate_mcts_diagnostics
            )
            if args.mode == "mcts-value-diagnostics":
                pool_options["play_disagreement_games"] = args.play_disagreement_games
            summary = diagnose(
                model_a, model_b, training, mcts, args.games,
                positions_per_game=args.positions_per_game, **pool_options,
            )
        elif args.mode == "mcts-first-divergence":
            from polybot.evaluation.mcts_diagnostics import evaluate_mcts_first_divergence

            summary = evaluate_mcts_first_divergence(
                model_a, model_b, training, mcts, args.games,
                continuation_value_scale=(
                    1.0 if args.continuation_value_scale is None else args.continuation_value_scale
                ),
                **pool_options,
            )
        elif args.mode == "mcts-value-matrix":
            from polybot.evaluation.mcts_diagnostics import evaluate_mcts_value_matrix

            summary = evaluate_mcts_value_matrix(
                model_a, model_b, training, mcts, args.map_seed_group, **pool_options,
            )
        elif value_regression:
            from polybot.evaluation.value_regression import evaluate_mcts_value_regression

            old_model, _ = load_model(args.old_value_model, device)
            summary = evaluate_mcts_value_regression(
                model_a, old_model, model_b, training, mcts, args.games,
                positions_per_game=args.positions_per_game, divergence_games=args.branch_positions,
                branch_rollouts=3 if args.branch_rollouts is None else args.branch_rollouts,
                rollout_temperature=0.5 if args.branch_rollout_temperature is None else args.branch_rollout_temperature,
                num_threads=args.mcts_threads, max_envs=args.num_envs,
                visible_action_history=args.visible_action_history, max_pending_leaves=args.max_pending_leaves,
                belief_retries=args.belief_retries,
            )
        elif args.mode == "mcts-branch-agreement":
            from polybot.evaluation.mcts_diagnostics import evaluate_saved_branch_agreement

            summary = evaluate_saved_branch_agreement(
                model_a, model_b, training, mcts, args.branch_pairs,
                max_positions=args.branch_positions, max_envs=args.num_envs,
                num_threads=args.mcts_threads, max_pending_leaves=args.max_pending_leaves,
                visible_action_history=args.visible_action_history, belief_retries=args.belief_retries,
                branch_rollouts=args.branch_rollouts,
                rollout_temperature=0.5 if args.branch_rollout_temperature is None else args.branch_rollout_temperature,
            )
        elif args.mode == "mcts-value-ablation":
            seeds = [training.seed + index // 2 for index in range(args.games)]
            pool_options.update(
                model_b_uses_mcts=False,
                map_seeds=seeds,
                replace_failed_pairs=False,
                belief_action_mode="root_shield",
            )
            summary = {}
            for name, scale in (
                ("learned_value", 1.0),
                ("quarter_leaf_value", 0.25),
                ("zero_leaf_value", 0.0),
            ):
                result = evaluate_matches_pool(
                    model_a, model_b, training, mcts, args.games,
                    leaf_value_scale=scale,
                    **{**pool_options, "model_a_label": f"{args.model_a.stem} (leaf value x{scale})"},
                )
                summary[name] = {**result, "leaf_value_scale": scale}

            baseline = summary["learned_value"]
            for variant, metric in (
                ("quarter_leaf_value", "score_gain_quarter_value"),
                ("zero_leaf_value", "score_gain_zero_value"),
            ):
                result = summary[variant]
                valid = (
                    baseline["valid"] and result["valid"]
                    and baseline.get("evaluated_map_seeds") == result.get("evaluated_map_seeds") == seeds
                )
                summary[metric] = (
                    result["model_a_score"] - baseline["model_a_score"] if valid else None
                )
        elif args.mode == "mcts-vs-raw":
            seeds = [training.seed + index // 2 for index in range(args.games)]
            pool_options.update(
                model_b_uses_mcts=False,
                map_seeds=seeds,
                replace_failed_pairs=False,
                belief_action_mode="root_shield",
            )
            raw = evaluate_matches_pool(
                model_a, model_b, training, mcts, args.games,
                model_a_uses_mcts=False, **pool_options,
            )
            search = evaluate_matches_pool(
                model_a, model_b, training, mcts, args.games, **pool_options,
            )
            valid = (
                bool(raw["valid"] and search["valid"])
                and raw.get("evaluated_map_seeds") == search.get("evaluated_map_seeds") == seeds
            )
            summary = {
                "raw": raw,
                "mcts": search,
                "score_gain": search["model_a_score"] - raw["model_a_score"] if valid else None,
            }
        else:
            summary = evaluate_matches_pool(
                model_a, model_b, training, mcts, args.games, **pool_options,
            )
    print_summary(summary)


if __name__ == "__main__":
    main()
