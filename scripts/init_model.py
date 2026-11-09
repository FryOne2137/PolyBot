"""Create a reproducible random model and preserve its initialization record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import ModelConfig
from polybot.infrastructure.model_io import save_model
from polybot.infrastructure.randomness import seed_everything
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import model_fingerprint, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--architecture", choices=("relational", "factorized", "attention"), default="relational",
                        help="Map-encoding profile; all profiles share one action encoder and policy scorer")
    parser.add_argument("--seed", type=int, default=2137)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--action-interaction-dim", type=int, default=64)
    parser.add_argument("--action-encoder-dim", type=int, default=64)
    parser.add_argument("--action-encoder-heads", type=int, default=4)
    parser.add_argument("--action-latent-count", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--max-board-side", type=int, default=32)
    args = parser.parse_args()
    if args.model_out.exists():
        parser.error("model-out already exists; choose a new path")
    if args.d_model < 1 or args.heads < 1 or args.d_model % args.heads or args.layers < 1:
        parser.error("positive d-model divisible by heads and positive layers are required")
    if args.action_interaction_dim < 1 or args.max_board_side < 1 or not 0 <= args.dropout < 1:
        parser.error("invalid interaction dimension, board side or dropout")
    if min(args.action_encoder_dim, args.action_encoder_heads, args.action_latent_count) < 1 or args.action_encoder_dim % args.action_encoder_heads:
        parser.error("action encoder width must be positive and divisible by action heads; latent count must be positive")
    seed = seed_everything(args.seed)
    config = ModelConfig(
        d_model=args.d_model, transformer_layers=args.layers,
        attention_heads=args.heads, dropout=args.dropout,
        max_board_side=args.max_board_side, policy_architecture=args.architecture,
        action_interaction_dim=args.action_interaction_dim,
        action_encoder_dim=args.action_encoder_dim, action_encoder_heads=args.action_encoder_heads,
        action_latent_count=args.action_latent_count,
    )
    model = PolicyValueBeliefNet(config).eval()
    run = training_run_from_args(args, "initialization")
    run.write_config(model=config, seed=seed, initialized_from="random")
    write_manifest(
        run.directory, stage="initialization", model_config=asdict(config),
        initialization_seed=seed, fingerprint=model_fingerprint(model),
    )
    save_model(args.model_out, model, config)
    # Keep the original random weights with the source/environment snapshot.
    save_model(run.directory / "initial_model.pt", model, config)
    summary = {
        "initialized_from": "random", "initialization_seed": seed,
        "model_path": str(args.model_out.resolve()), "model_config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "fingerprint": model_fingerprint(model),
    }
    run.event("model_initialized", **summary)
    summary["run_dir"] = str(run.finish(summary))
    print_summary(summary)


if __name__ == "__main__":
    main()
