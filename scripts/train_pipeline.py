"""Stages 1 and 2 in sequence; persists only one final model file."""

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
    parser = argparse.ArgumentParser(description="Run PolyBot's complete no-log training pipeline")
    add_environment_options(parser)
    parser.add_argument("--belief-positions", type=int, default=20_000)
    parser.add_argument("--belief-batch-size", type=int, default=16)
    parser.add_argument("--self-play-games", type=int, default=200)
    parser.add_argument("--updates-per-game", type=int, default=4)
    parser.add_argument("--exploration-rate", type=float, default=0.20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--model-out", type=Path, default=Path("models/polybot.pt"))
    args = parser.parse_args()
    summary = run_pipeline(
        training_config_from_args(args),
        ModelConfig(),
        mcts_config_from_args(args),
        belief_positions=args.belief_positions,
        belief_batch_size=args.belief_batch_size,
        self_play_games=args.self_play_games,
        updates_per_game=args.updates_per_game,
        model_out=args.model_out,
        model_in=args.model_in,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
