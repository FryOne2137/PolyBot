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
    add_run_options,
    mcts_config_from_args,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.self_play_pool_train import train_self_play_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAM-only persistent SelfPlayPool belief-MCTS with checkpoint and run plots"
    )
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/self_play_pool.pt"))
    parser.add_argument(
        "--anchor-model", type=Path,
        help="Frozen predecessor for conservative policy targets; defaults to --model-in.",
    )
    parser.add_argument("--anchor-policy-weight", type=float, default=0.20)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2_000, help="Vector decisions; each step advances every live game once")
    parser.add_argument("--updates", type=int, default=1_000, help="Total optimiser updates interleaved with rollout steps")
    parser.add_argument("--num-threads", type=int, help="Persistent native SelfPlayPool worker count")
    parser.add_argument("--max-pending-leaves", type=int, default=4, help="Virtual-loss leaves per tree, 1-8")
    parser.add_argument(
        "--visible-action-history", type=int, default=32,
        help="Fog-safe compact action history (0 disables, default: 32).",
    )
    parser.add_argument("--belief-retries", type=int, default=4, help="Fresh belief samples for each rejected environment")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=0.95, help="Blend terminal returns with MCTS value bootstraps.")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2, help="Per-player ISMCTS belief particles (K)")
    parser.add_argument("--mcts-threads", type=int, help="Accepted for config compatibility; use --num-threads for this pool")
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument(
        "--mcts-top-actions", type=int, default=0,
        help="PolyBot-only quality experiment: top-K policy priors plus mandatory tactics.",
    )
    parser.add_argument("--mcts-random-actions", type=int, default=0)
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    args = parser.parse_args()

    training = training_config_from_args(args)
    device = select_device(training.device)
    model, model_config = load_model(args.model_in, device)
    anchor_model = None
    if args.anchor_policy_weight > 0.0:
        anchor_model, anchor_config = load_model(args.anchor_model or args.model_in, device)
        if anchor_config != model_config:
            raise RuntimeError("anchor and candidate ModelConfig differ")
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
        anchor_model=anchor_model,
        anchor_policy_weight=args.anchor_policy_weight,
        run=training_run_from_args(args, "self_play_pool"),
    ))


if __name__ == "__main__":
    main()
