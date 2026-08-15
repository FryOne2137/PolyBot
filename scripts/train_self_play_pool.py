"""Fast persistent-native MCTS stage backed by PolyEnv SelfPlayPool."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import (
    add_environment_options,
    mcts_config_from_args,
    print_summary,
    training_config_from_args,
)
from polybot.training.self_play_pool_train import train_self_play_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAM-only persistent SelfPlayPool belief-MCTS; writes only the requested model checkpoint"
    )
    add_environment_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/self_play_pool.pt"))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2_000, help="Vector decisions; each step advances every live game once")
    parser.add_argument("--updates", type=int, default=1_000, help="Total optimiser updates interleaved with rollout steps")
    parser.add_argument("--num-threads", type=int, help="Persistent native SelfPlayPool worker count")
    parser.add_argument("--max-pending-leaves", type=int, default=4, help="Virtual-loss leaves per tree, 1-8")
    parser.add_argument(
        "--visible-action-history", type=int, default=0,
        help="Keep at 0 until direct GameEnv evaluation exposes the same history tensor.",
    )
    parser.add_argument("--belief-retries", type=int, default=4, help="Fresh belief samples for each rejected environment")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=1.0, help="1.0 propagates the exact final result through the full game.")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2, help="Per-player ISMCTS belief particles (K)")
    parser.add_argument("--mcts-threads", type=int, help="Accepted for config compatibility; use --num-threads for this pool")
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    args = parser.parse_args()

    training = training_config_from_args(args)
    model, model_config = load_model(args.model_in, select_device(training.device))
    print_summary(train_self_play_pool(
        training,
        model_config,
        mcts_config_from_args(args),
        num_envs=args.num_envs,
        steps=args.steps,
        updates=args.updates,
        num_threads=args.num_threads,
        max_pending_leaves=args.max_pending_leaves,
        visible_action_history=args.visible_action_history,
        belief_retries=args.belief_retries,
        model_out=args.model_out,
        model=model,
    ))


if __name__ == "__main__":
    main()
