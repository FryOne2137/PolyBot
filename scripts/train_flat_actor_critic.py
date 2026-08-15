"""Stage 3: high-throughput gradient actor-critic curriculum without MCTS."""

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
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.vector_actor_critic import train_vector_actor_critic


def main() -> None:
    parser = argparse.ArgumentParser(description="RAM-only actor-critic with model weights and run plots")
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/flat_actor_critic.pt"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--max-actions", type=int, default=1024)
    parser.add_argument("--entropy-weight", type=float, default=0.008)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.995)
    args = parser.parse_args()
    training = training_config_from_args(args)
    model, model_config = load_model(args.model_in, select_device(training.device))
    print_summary(train_vector_actor_critic(
        training, model_config, args.num_envs, args.steps, args.num_threads,
        args.max_actions, args.entropy_weight, args.value_weight, args.teacher_weight, args.gamma,
        args.model_out, model, run=training_run_from_args(args, "actor_critic"),
    ))


if __name__ == "__main__":
    main()
