"""Stage 2 entry point: train policy/value from MCTS after belief admission."""

from __future__ import annotations

from train_league import main


if __name__ == "__main__":
    main(fixed_training_stage="mcts")
