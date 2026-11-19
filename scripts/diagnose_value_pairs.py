"""Compare three value losses and inspect saved branch features offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import (
    add_environment_options, add_run_options, print_summary, training_config_from_args, training_run_from_args,
)
from polybot.training.flat_experiments import FlatImprovements
from polybot.training.value_pair_diagnostics import compare_saved_value_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_options(parser)
    add_run_options(parser)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--saved-pairs", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=1768127001)
    parser.add_argument("--pair-selection-fraction", type=float, default=0.25)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--difference-loss-weight", type=float, default=1.0)
    parser.add_argument("--visible-action-history", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=16)
    parser.add_argument("--memorization-test", action="store_true",
                        help="Fit delta only on all informative training pairs in one full batch")
    parser.set_defaults(learning_rate=0.00001, max_moves=1600)
    args = parser.parse_args()
    if not args.model_in.is_file() or not args.saved_pairs.is_dir():
        parser.error("require an existing --model-in and --saved-pairs directory")
    if args.updates < 1 or args.batch_size < 2 or args.batch_size % 2 or args.log_interval < 1:
        parser.error("require positive updates, an even batch size of at least two and a positive log interval")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not np.isfinite(args.difference_loss_weight) or args.difference_loss_weight <= 0:
        parser.error("--difference-loss-weight must be finite and positive")
    if not 0 < args.pair_selection_fraction < 1 or args.selection_seed < 0:
        parser.error("require a selection fraction in (0, 1) and a non-negative split seed")
    if not 0 <= args.visible_action_history <= 1024:
        parser.error("--visible-action-history must be in [0, 1024]")
    training = training_config_from_args(args)
    model, _ = load_model(args.model_in, select_device(training.device))
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=args.batch_size,
        value_learning_rate=args.learning_rate, value_fit_backoffs=0,
        value_capacity=max(args.batch_size, 128),
    )
    run = training_run_from_args(args, "value")
    report = compare_saved_value_losses(
        model, training, args.saved_pairs, settings, run, history=args.visible_action_history,
        split_seed=args.selection_seed, selection_fraction=args.pair_selection_fraction,
        difference_weight=args.difference_loss_weight, log_interval=args.log_interval,
        memorization_test=args.memorization_test,
    )
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary(report)


if __name__ == "__main__":
    main()
