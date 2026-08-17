"""PFSP population league with deterministic, regression-safe promotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.infrastructure.device import select_device
from polybot.training.cli import (
    add_environment_options,
    add_run_options,
    mcts_config_from_args,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.league import train_promotion_league


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train against a historical population and save only robust promotions"
    )
    add_environment_options(parser)
    # The league adds a separate geometry curriculum on top of policy/value
    # learning. A conservative default keeps that auxiliary head from
    # dominating the shared representation of an already competent champion.
    parser.set_defaults(bridge_constraint_weight=0.10)
    add_run_options(parser)
    parser.add_argument("--champion-in", type=Path, required=True, help="Initial frozen champion checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/league"))
    parser.add_argument("--checkpoint-prefix", default="champion")
    parser.add_argument(
        "--league-db", type=Path,
        help="SQLite history (default: <checkpoint-dir>/league.sqlite3)",
    )
    parser.add_argument(
        "--population-model", type=Path, action="append", default=[],
        help="Additional protected teacher/historical checkpoint; repeat as needed",
    )
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument(
        "--population-games", type=int, default=64,
        help="Candidate-vs-population training games per round",
    )
    parser.add_argument("--population-updates-per-game", type=int, default=2)
    parser.add_argument(
        "--population-workers", type=int, default=16,
        help="Persistent native SelfPlayPool worker threads for population games",
    )
    parser.add_argument("--pfsp-alpha", type=float, default=2.0)
    parser.add_argument("--champion-probability", type=float, default=0.40)
    parser.add_argument("--minimum-sampling-probability", type=float, default=0.05)
    parser.add_argument(
        "--champion-anchor-weight", type=float, default=0.30,
        help="Frozen champion fraction blended into policy targets in both training phases",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Candidate train-and-gate cycles")
    parser.add_argument("--promotion-games", type=int, default=100, help="Fixed-seed MCTS games per gate")
    parser.add_argument("--promotion-score", type=float, default=0.55, help="Win=1 / draw=0.5 score required")
    parser.add_argument("--promotion-lower-bound", type=float, default=0.50)
    parser.add_argument(
        "--historical-games", type=int, default=20,
        help="Paired held-out games against every retained historical model",
    )
    parser.add_argument("--population-score", type=float, default=0.52)
    parser.add_argument("--historical-score-floor", type=float, default=0.45)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--evaluation-seed", type=int, default=1_500_000_007, help="Held-out paired-map schedule")
    parser.add_argument("--evaluation-seed-stride", type=int, default=20_000_033)
    parser.add_argument("--round-seed-stride", type=int, default=10_000_019)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1_000, help="Vector decisions per candidate")
    parser.add_argument("--updates", type=int, default=500, help="Online optimiser updates per candidate")
    parser.add_argument("--num-threads", type=int, help="Persistent native SelfPlayPool worker count")
    parser.add_argument("--max-pending-leaves", type=int, default=4, help="Virtual-loss leaves per tree, 1-8")
    parser.add_argument("--visible-action-history", type=int, default=0, help="0 keeps train/evaluation inputs identical")
    parser.add_argument("--belief-retries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exploration-rate", type=float, default=0.10)
    parser.add_argument("--teacher-policy-target-weight", type=float, default=0.10)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument(
        "--mcts-top-actions", type=int, default=0,
        help="PolyBot-only leaf prior filter; 0 disables, 48 is a conservative experiment.",
    )
    parser.add_argument(
        "--mcts-random-actions", type=int, default=0,
        help="Extra random legal candidates per leaf in addition to top-K and mandatory tactics.",
    )
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    args = parser.parse_args()

    training = training_config_from_args(args)
    print(f"device: {select_device(training.device)} | league rounds: {args.rounds}", flush=True)
    print_summary(train_promotion_league(
        training,
        champion_in=args.champion_in,
        checkpoint_dir=args.checkpoint_dir,
        rounds=args.rounds,
        promotion_games=args.promotion_games,
        promotion_score=args.promotion_score,
        promotion_lower_bound=args.promotion_lower_bound,
        evaluation_seed=args.evaluation_seed,
        evaluation_seed_stride=args.evaluation_seed_stride,
        round_seed_stride=args.round_seed_stride,
        checkpoint_prefix=args.checkpoint_prefix,
        mcts=mcts_config_from_args(args),
        num_envs=args.num_envs,
        steps=args.steps,
        updates=args.updates,
        num_threads=args.num_threads,
        max_pending_leaves=args.max_pending_leaves,
        visible_action_history=args.visible_action_history,
        belief_retries=args.belief_retries,
        database_path=args.league_db,
        population_model_paths=args.population_model,
        population_size=args.population_size,
        population_games=args.population_games,
        population_updates_per_game=args.population_updates_per_game,
        population_workers=args.population_workers,
        pfsp_alpha=args.pfsp_alpha,
        champion_probability=args.champion_probability,
        minimum_sampling_probability=args.minimum_sampling_probability,
        champion_anchor_weight=args.champion_anchor_weight,
        historical_games=args.historical_games,
        population_score=args.population_score,
        historical_score_floor=args.historical_score_floor,
        elo_k=args.elo_k,
        run=training_run_from_args(args, "league"),
    ))


if __name__ == "__main__":
    main()
