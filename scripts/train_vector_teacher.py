"""High-throughput VectorGameEnv policy curriculum with checkpoint and run metrics."""

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
from polybot.training.vector_teacher import train_vector_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a policy curriculum with native PolyEnv VectorGameEnv")
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/vector_teacher.pt"))
    parser.add_argument(
        "--anchor-model", type=Path,
        help="Frozen checkpoint whose policy/value keep a recovery refresh stable",
    )
    parser.add_argument("--policy-anchor-weight", type=float, default=0.30)
    parser.add_argument("--value-anchor-weight", type=float, default=0.20)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--num-threads", type=int, help="C++ simulation threads; default uses logical CPU count")
    parser.add_argument("--max-actions", type=int, default=1024)
    parser.add_argument("--visible-event-history", type=int, default=0)
    parser.add_argument("--updates-per-step", type=int, default=1)
    args = parser.parse_args()
    training = training_config_from_args(args)
    model, model_config = load_model(args.model_in, select_device(training.device))
    anchor_model = None
    if args.anchor_model is not None:
        anchor_model, anchor_config = load_model(
            args.anchor_model, select_device(training.device)
        )
        if anchor_config != model_config:
            parser.error("--anchor-model has a different ModelConfig")
    print_summary(train_vector_teacher(
        training, model_config, args.num_envs, args.steps,
        num_threads=args.num_threads,
        max_actions=args.max_actions,
        visible_event_history=args.visible_event_history,
        updates_per_step=args.updates_per_step,
        model_out=args.model_out,
        model=model,
        anchor_model=anchor_model,
        policy_anchor_weight=args.policy_anchor_weight,
        value_anchor_weight=args.value_anchor_weight,
        run=training_run_from_args(args, "vector_teacher"),
    ))


if __name__ == "__main__":
    main()
