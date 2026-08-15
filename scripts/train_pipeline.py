"""Full RAM-only curriculum; persists only one final model file."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import ModelConfig
from polybot.training.cli import (
    add_environment_options,
    mcts_config_from_args,
    print_summary,
    training_config_from_args,
)
from polybot.training.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PolyBot's full no-log curriculum and save final weights only")
    add_environment_options(parser)
    parser.add_argument("--belief-positions", type=int, default=20_000)
    parser.add_argument("--belief-batch-size", type=int, default=16)
    parser.add_argument("--teacher-games", type=int, default=100)
    parser.add_argument("--teacher-updates-per-game", type=int, default=16)
    parser.add_argument("--vector-teacher-envs", type=int, default=128)
    parser.add_argument("--vector-teacher-steps", type=int, default=10_000)
    parser.add_argument("--vector-teacher-updates-per-step", type=int, default=1)
    parser.add_argument("--flat-envs", type=int, default=64)
    parser.add_argument("--flat-steps", type=int, default=10_000)
    parser.add_argument(
        "--self-play-pool-envs", type=int, default=0,
        help="Live native games in the persistent SelfPlayPool stage; 0 disables that stage.",
    )
    parser.add_argument(
        "--self-play-pool-steps", type=int, default=0,
        help="Vector decisions for the persistent native SelfPlayPool MCTS stage.",
    )
    parser.add_argument(
        "--self-play-pool-updates", type=int, default=0,
        help="Total optimiser updates interleaved with the SelfPlayPool rollout.",
    )
    parser.add_argument("--self-play-pool-threads", type=int, help="Native worker threads owned by SelfPlayPool")
    parser.add_argument(
        "--self-play-pool-pending-leaves", type=int, default=4,
        help="Concurrent virtual-loss leaves per tree (1-8); 4 batches GPU inference efficiently.",
    )
    parser.add_argument("--self-play-games", type=int, default=200, help="anchored MCTS foundation games")
    parser.add_argument("--refinement-games", type=int, default=100, help="lower-noise MCTS games after the foundation")
    parser.add_argument("--updates-per-game", type=int, default=8)
    parser.add_argument("--refinement-updates-per-game", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exploration-rate", type=float, default=0.20)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-policy-target-weight", type=float, default=0.10)
    parser.add_argument("--teacher-replay-fraction", type=float, default=0.35)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--mcts-threads", type=int)
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=32, help="active games in the native global MctsPool")
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--model-out", type=Path, default=Path("models/polybot.pt"))
    parser.add_argument(
        "--champion-model",
        type=Path,
        action="append",
        default=[],
        help="Frozen opponent checkpoint; first one is the promotion champion, repeat to build a pool.",
    )
    parser.add_argument(
        "--promotion-games",
        type=int,
        default=30,
        help="Fixed-seed MCTS gate games against the first --champion-model; 0 disables the gate.",
    )
    parser.add_argument(
        "--promotion-score",
        type=float,
        default=0.55,
        help="Required win=1/draw=0.5 match score to replace the champion.",
    )
    args = parser.parse_args()
    summary = run_pipeline(
        training_config_from_args(args),
        ModelConfig(),
        mcts_config_from_args(args),
        belief_positions=args.belief_positions,
        belief_batch_size=args.belief_batch_size,
        self_play_games=args.self_play_games,
        updates_per_game=args.updates_per_game,
        flat_envs=args.flat_envs,
        flat_steps=args.flat_steps,
        self_play_pool_envs=args.self_play_pool_envs,
        self_play_pool_steps=args.self_play_pool_steps,
        self_play_pool_updates=args.self_play_pool_updates,
        model_out=args.model_out,
        model_in=args.model_in,
        teacher_games=args.teacher_games,
        teacher_updates_per_game=args.teacher_updates_per_game,
        vector_teacher_envs=args.vector_teacher_envs,
        vector_teacher_steps=args.vector_teacher_steps,
        vector_teacher_updates_per_step=args.vector_teacher_updates_per_step,
        self_play_pool_threads=args.self_play_pool_threads,
        self_play_pool_pending_leaves=args.self_play_pool_pending_leaves,
        mcts_workers=args.workers,
        teacher_replay_fraction=args.teacher_replay_fraction,
        refinement_games=args.refinement_games,
        refinement_updates_per_game=args.refinement_updates_per_game,
        champion_model_paths=args.champion_model,
        promotion_games=args.promotion_games,
        promotion_score=args.promotion_score,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
