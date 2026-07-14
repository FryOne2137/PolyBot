"""Shared command-line options for repository-local training scripts."""

from __future__ import annotations

import argparse
import json
from typing import Dict

from ..config import MCTSConfig, TrainingConfig


def add_environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=2137)
    parser.add_argument("--map-size", type=int, default=11)
    parser.add_argument("--map-type", default="lakes", choices=["lakes", "drylands"])
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        seed=args.seed,
        device=args.device,
        map_size=args.map_size,
        map_type=args.map_type,
        max_moves=args.max_moves,
        batch_size=getattr(args, "batch_size", 16),
        exploration_rate=getattr(args, "exploration_rate", 0.20),
        priority_replay_fraction=getattr(args, "priority_replay_fraction", 0.35),
        learning_rate=args.learning_rate,
    )


def mcts_config_from_args(args: argparse.Namespace) -> MCTSConfig:
    return MCTSConfig(simulations=args.simulations, particles=args.particles)


def print_summary(summary: Dict[str, object]) -> None:
    """Console output only; deliberately no metrics file is created."""
    print(json.dumps(summary, indent=2, sort_keys=True))
