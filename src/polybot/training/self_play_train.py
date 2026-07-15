"""Simple in-memory AlphaZero-style self-play training after belief warm-up."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import stack_examples
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..infrastructure.progress import ConsoleProgress
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.belief_mcts import BeliefMCTS
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayRunner
from .teacher_bootstrap import _teacher_game


def train_self_play(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts_config: MCTSConfig,
    games: int,
    updates_per_game: int,
    opponent: str = "self_play",
    random_opponent_rate: float = 0.5,
    mixed_opponent: str = "heuristic",
    teacher_replay_fraction: float = 0.25,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Play one environment at a time and keep all training data in RAM.

    This deliberately prioritises low-overhead, observable training over Python
    process orchestration: each environment receives its own live MCTS progress
    bar and the model is updated after that environment finishes.
    """
    if opponent not in {"self_play", "random", "heuristic", "mixed"}:
        raise ValueError("opponent must be self_play, random, heuristic or mixed")
    if not 0.0 <= random_opponent_rate <= 1.0:
        raise ValueError("random_opponent_rate must be between 0 and 1")
    if mixed_opponent not in {"random", "heuristic"}:
        raise ValueError("mixed_opponent must be random or heuristic")
    if not 0.0 <= teacher_replay_fraction <= 1.0:
        raise ValueError("teacher_replay_fraction must be between 0 and 1")

    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    planner = BeliefMCTS(model, device, mcts_config)
    runner = SelfPlayRunner(planner, training, logger=None)
    replay = ReplayBuffer(training.replay_capacity)
    teacher_replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed)
    schedule_rng = np.random.default_rng(training.seed ^ 0xA17C0)
    progress = ConsoleProgress()
    progress.stage(
        f"device: {device} | games: {games} | MCTS simulations: "
        f"{mcts_config.simulations} | particles: {mcts_config.particles}"
    )

    losses = []
    rewards = []
    exploration_actions = 0.0
    opponent_games = 0.0
    reward_totals: Dict[str, float] = {}
    action_totals: Dict[str, float] = {}

    for game_index in range(games):
        model.eval()
        use_anchor_opponent = opponent in {"random", "heuristic"} or (
            opponent == "mixed" and schedule_rng.random() < random_opponent_rate
        )
        opponent_agent = (mixed_opponent if opponent == "mixed" else opponent) if use_anchor_opponent else None
        train_player = game_index % 2 if use_anchor_opponent else None

        def report_search(move: int, completed: int, total: int, particles: int) -> None:
            progress.mcts(game_index + 1, games, move, completed, total, particles)

        examples, game_metrics = runner.play_game(
            training.seed + game_index,
            iteration=0,
            game_index=game_index,
            train_player=train_player,
            opponent_agent=opponent_agent,
            on_search_progress=report_search,
        )
        progress.game_complete(game_index + 1, games, int(game_metrics["moves"]), len(examples))
        replay.extend(examples)
        if teacher_replay_fraction > 0.0:
            teacher_examples, _ = _teacher_game(training.seed + 1_000_000 + game_index, training)
            teacher_replay.extend(teacher_examples)

        opponent_games += float(use_anchor_opponent)
        exploration_actions += game_metrics.get("exploration_actions", 0.0)
        rewards.extend(value for name, value in game_metrics.items() if name.startswith("shaped_reward_player_"))
        for name, value in game_metrics.items():
            if name.startswith("reward_"):
                reward_totals[name] = reward_totals.get(name, 0.0) + value
            if name == "model_actions" or name.startswith("model_action_"):
                action_totals[name] = action_totals.get(name, 0.0) + value

        if len(replay) < training.batch_size:
            continue
        for _ in range(updates_per_game):
            teacher_count = min(int(round(training.batch_size * teacher_replay_fraction)), len(teacher_replay))
            self_play_count = training.batch_size - teacher_count
            if len(replay) < self_play_count:
                teacher_count = training.batch_size
                self_play_count = 0
            examples_batch = []
            if self_play_count:
                examples_batch.extend(replay.sample(self_play_count, rng, training.priority_replay_fraction))
            if teacher_count:
                examples_batch.extend(teacher_replay.sample(teacher_count, rng, training.priority_replay_fraction))
            losses.append(train_batch(model, optimiser, stack_examples(examples_batch, device), training))

    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float] = {
        "games": float(games),
        "replay_examples": float(len(replay)),
        "teacher_replay_examples": float(len(teacher_replay)),
        "teacher_replay_fraction": teacher_replay_fraction,
        "mean_shaped_reward": float(np.mean(rewards)) if rewards else 0.0,
        "actual_exploration_rate": exploration_actions / max(1, len(replay)),
        "opponent_games": opponent_games,
    }
    total_examples = max(1, len(replay))
    for name, value in reward_totals.items():
        summary["mean_" + name] = value / total_examples
    model_actions = max(1.0, action_totals.get("model_actions", 0.0))
    for name, value in action_totals.items():
        if name != "model_actions":
            summary[name + "_rate"] = value / model_actions
    if losses:
        for name in losses[0]:
            summary[name] = float(np.mean([loss[name] for loss in losses]))
    return summary
