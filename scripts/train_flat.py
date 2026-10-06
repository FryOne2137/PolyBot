"""Stage 0 entry point: train the raw policy/value game without MCTS."""

from __future__ import annotations

from train_league import main


if __name__ == "__main__":
    main(fixed_training_stage="flat")
