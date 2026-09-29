"""Stage 1: train hidden-map belief from online PolyEnv observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import ModelConfig
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.belief_warmup import warmup_belief
from polybot.training.cli import (
    add_environment_options,
    add_run_options,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up belief and save model weights plus run plots")
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--positions", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--min-hidden-tiles", type=int, default=24,
        help="Keep only naturally foggy positions with at least this many hidden map tiles",
    )
    parser.add_argument(
        "--max-moves-per-world", type=int, default=96,
        help="Reset a random warm-up game before late, almost-revealed positions dominate",
    )
    parser.add_argument("--model-in", type=Path, help="Optional checkpoint whose policy/value weights are preserved")
    parser.add_argument(
        "--teacher-action-probability", type=float, default=0.0,
        help="Share of belief-world rollout actions sampled from the public heuristic teacher",
    )
    parser.add_argument(
        "--policy-action-probability", type=float, default=0.0,
        help="Share of belief-world rollout actions sampled from --model-in policy",
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/belief.pt"))
    args = parser.parse_args()
    training = training_config_from_args(args)
    model, model_config = (
        load_model(args.model_in, select_device(training.device))
        if args.model_in is not None else (None, ModelConfig())
    )
    summary = warmup_belief(
        training,
        model_config,
        positions=args.positions,
        batch_size=args.batch_size,
        min_hidden_tiles=args.min_hidden_tiles,
        max_moves_per_world=args.max_moves_per_world,
        model_out=args.model_out,
        model=model,
        preserve_policy_value=args.model_in is not None,
        teacher_action_probability=args.teacher_action_probability,
        policy_action_probability=args.policy_action_probability,
        run=training_run_from_args(args, "belief"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
