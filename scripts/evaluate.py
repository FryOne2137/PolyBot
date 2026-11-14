"""Evaluate one checkpoint against random play or a second checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig
from polybot.evaluation.matches import evaluate_direct_matches, evaluate_matches
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_environment_options, print_summary, training_config_from_args
from polybot.training.routed_self_play_pool import evaluate_matches_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint; write no games or metrics files")
    add_environment_options(parser)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, help="optional MCTS opponent; default is random")
    parser.add_argument(
        "--mode",
        choices=("mcts", "flat", "mcts-vs-raw"),
        default="mcts",
        help="Evaluate search, raw policy, or the gain from adding search to model A only",
    )
    parser.add_argument("--games", type=int, default=10)
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
    if args.mode in {"flat", "mcts-vs-raw"} and args.model_b is None:
        parser.error(f"--mode {args.mode} requires --model-b")

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
        if args.mode == "mcts-vs-raw":
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
