"""Evaluate one checkpoint against random play or a second checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig
from polybot.evaluation.matches import evaluate_matches
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_environment_options, print_summary, training_config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint; write no games or metrics files")
    add_environment_options(parser)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, help="optional MCTS opponent; default is random")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    args = parser.parse_args()

    training = training_config_from_args(args)
    device = select_device(training.device)
    model_a, _ = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)[0] if args.model_b is not None else None
    # Greedy selection makes repeated evaluation on the same seeds comparable.
    mcts = MCTSConfig(
        simulations=args.simulations,
        particles=args.particles,
        temperature=0.0,
        inference_batch_size=args.inference_batch_size,
    )
    print_summary(evaluate_matches(model_a, device, training, mcts, args.games, model_b))


if __name__ == "__main__":
    main()
