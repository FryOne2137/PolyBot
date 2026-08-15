"""Stage 1b: behavior-cloning bootstrap from the built-in heuristic teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import ModelConfig
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import (
    add_environment_options,
    add_run_options,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.teacher_bootstrap import train_teacher_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the teacher and save weights plus run plots")
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, help="optional belief checkpoint to continue from")
    parser.add_argument("--model-out", type=Path, default=Path("models/teacher_bootstrap.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--updates-per-game", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    args = parser.parse_args()
    training = training_config_from_args(args)
    model = None
    model_config = ModelConfig()
    if args.model_in is not None:
        model, model_config = load_model(args.model_in, select_device(training.device))
    print_summary(train_teacher_bootstrap(
        training, model_config, args.games, args.updates_per_game, args.model_out, model,
        run=training_run_from_args(args, "teacher"),
    ))


if __name__ == "__main__":
    main()
