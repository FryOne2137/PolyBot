"""Stage 1 entry point: train and validate only the belief decoders."""

from __future__ import annotations

from train_league import main


if __name__ == "__main__":
    main(fixed_training_stage="belief")
