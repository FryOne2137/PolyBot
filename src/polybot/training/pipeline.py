"""Full, disk-minimal training sequence: belief warm-up then MCTS self-play."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .belief_warmup import warmup_belief
from .self_play_train import train_self_play


def run_pipeline(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts_config: MCTSConfig,
    belief_positions: int,
    belief_batch_size: int,
    self_play_games: int,
    updates_per_game: int,
    model_out: Path,
    model_in: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run all stages and write exactly one final model checkpoint."""
    device = select_device(training.device)
    if model_in is None:
        model = PolicyValueBeliefNet(model_config).to(device)
    else:
        model, model_config = load_model(model_in, device)

    belief = warmup_belief(
        training,
        model_config,
        positions=belief_positions,
        batch_size=belief_batch_size,
        model=model,
    )
    self_play = train_self_play(
        training,
        model_config,
        mcts_config,
        games=self_play_games,
        updates_per_game=updates_per_game,
        model=model,
    )
    save_model(model_out, model, model_config)
    return {"belief_warmup": belief, "self_play": self_play}
