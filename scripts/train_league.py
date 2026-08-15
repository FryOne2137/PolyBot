"""Repeated SelfPlayPool training with deterministic champion promotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.infrastructure.device import select_device
from polybot.training.cli import add_environment_options, mcts_config_from_args, print_summary, training_config_from_args
from polybot.training.league import train_promotion_league


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SelfPlayPool candidates and save only checkpointed promotions"
    )
    add_environment_options(parser)
    parser.add_argument("--champion-in", type=Path, required=True, help="Initial frozen champion checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/league"))
    parser.add_argument("--checkpoint-prefix", default="champion")
    parser.add_argument("--rounds", type=int, default=10, help="Candidate train-and-gate cycles")
    parser.add_argument("--promotion-games", type=int, default=100, help="Fixed-seed MCTS games per gate")
    parser.add_argument("--promotion-score", type=float, default=0.55, help="Win=1 / draw=0.5 score required")
    parser.add_argument("--evaluation-seed", type=int, default=91_337, help="Stable seed schedule for every gate")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1_000, help="Vector decisions per candidate")
    parser.add_argument("--updates", type=int, default=500, help="Online optimiser updates per candidate")
    parser.add_argument("--num-threads", type=int, help="Persistent native SelfPlayPool worker count")
    parser.add_argument("--max-pending-leaves", type=int, default=4, help="Virtual-loss leaves per tree, 1-8")
    parser.add_argument("--visible-action-history", type=int, default=0, help="0 keeps train/evaluation inputs identical")
    parser.add_argument("--belief-retries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=1.0)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
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
        evaluation_seed=args.evaluation_seed,
        checkpoint_prefix=args.checkpoint_prefix,
        mcts=mcts_config_from_args(args),
        num_envs=args.num_envs,
        steps=args.steps,
        updates=args.updates,
        num_threads=args.num_threads,
        max_pending_leaves=args.max_pending_leaves,
        visible_action_history=args.visible_action_history,
        belief_retries=args.belief_retries,
    ))


if __name__ == "__main__":
    main()
