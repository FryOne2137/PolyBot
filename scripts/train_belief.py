"""Stage 1: train hidden-map belief from online PolyEnv observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import ModelConfig
from polybot.training.belief_warmup import warmup_belief
from polybot.training.cli import add_environment_options, print_summary, training_config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up PolyBot's belief model; save only model weights")
    add_environment_options(parser)
    parser.add_argument("--positions", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-out", type=Path, default=Path("models/belief.pt"))
    args = parser.parse_args()
    summary = warmup_belief(
        training_config_from_args(args),
        ModelConfig(),
        positions=args.positions,
        batch_size=args.batch_size,
        model_out=args.model_out,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
