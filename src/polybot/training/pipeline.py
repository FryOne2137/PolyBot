"""Full, disk-minimal curriculum from belief warm-up to anchored MCTS."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .belief_warmup import warmup_belief
from .self_play_train import train_self_play
from .teacher_bootstrap import train_teacher_bootstrap
from .vector_actor_critic import train_vector_actor_critic
from .vector_teacher import train_vector_teacher


def run_pipeline(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts_config: MCTSConfig,
    belief_positions: int,
    belief_batch_size: int,
    self_play_games: int,
    updates_per_game: int,
    flat_envs: int,
    flat_steps: int,
    model_out: Path,
    model_in: Path | None = None,
    teacher_games: int = 0,
    teacher_updates_per_game: int = 16,
    vector_teacher_envs: int = 0,
    vector_teacher_steps: int = 0,
    vector_teacher_updates_per_step: int = 1,
    mcts_workers: int = 1,
    teacher_replay_fraction: float = 0.35,
    refinement_games: int = 0,
    refinement_updates_per_game: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Run all stages in RAM and write exactly one final model checkpoint.

    The order intentionally gives the MCTS stage a competent prior:
    supervised belief -> legal-action teacher -> vector teacher -> flat
    actor-critic -> anchored MCTS -> lower-noise MCTS refinement. Games, replay
    buffers and optimiser states are never written; the one requested model
    path is overwritten at safe stage boundaries for crash recovery.
    """
    device = select_device(training.device)
    if model_in is None:
        model = PolicyValueBeliefNet(model_config).to(device)
    else:
        model, model_config = load_model(model_in, device)

    belief = {}
    if belief_positions:
        belief = warmup_belief(
            training,
            model_config,
            positions=belief_positions,
            batch_size=belief_batch_size,
            model=model,
        )
    teacher = {}
    if teacher_games:
        teacher = train_teacher_bootstrap(
            training, model_config, teacher_games, teacher_updates_per_game, model=model
        )
    vector_teacher = {}
    if vector_teacher_steps:
        vector_teacher = train_vector_teacher(
            training,
            model_config,
            num_envs=vector_teacher_envs,
            steps=vector_teacher_steps,
            max_actions=mcts_config.native_max_actions,
            updates_per_step=vector_teacher_updates_per_step,
            model=model,
        )
    flat = {}
    if flat_steps:
        flat = train_vector_actor_critic(
            training,
            model_config,
            num_envs=flat_envs,
            steps=flat_steps,
            max_actions=mcts_config.native_max_actions,
            model=model,
        )
    # Keep a single recoverable model artifact at ``model_out``. It is
    # overwritten by the final model, but protects the expensive warm-up if a
    # native extension terminates the process during MCTS.
    save_model(model_out, model, model_config)
    self_play = {}
    if self_play_games:
        self_play = train_self_play(
            training,
            model_config,
            mcts_config,
            games=self_play_games,
            updates_per_game=updates_per_game,
            opponent="mixed",
            random_opponent_rate=0.35,
            mixed_opponent="heuristic",
            teacher_replay_fraction=teacher_replay_fraction,
            workers=mcts_workers,
            model=model,
        )
        # If refinement fails, leave the completed foundation model at the
        # same permitted output path instead of reverting to pre-MCTS weights.
        save_model(model_out, model, model_config)
    refinement = {}
    if refinement_games:
        refinement_training = replace(
            training,
            learning_rate=training.learning_rate * 0.5,
            exploration_rate=min(training.exploration_rate, 0.03),
            teacher_policy_target_weight=min(training.teacher_policy_target_weight, 0.04),
        )
        refinement_mcts = replace(
            mcts_config,
            teacher_prior_weight=min(mcts_config.teacher_prior_weight, 0.08),
        )
        refinement = train_self_play(
            refinement_training,
            model_config,
            refinement_mcts,
            games=refinement_games,
            updates_per_game=refinement_updates_per_game,
            opponent="mixed",
            random_opponent_rate=0.20,
            mixed_opponent="heuristic",
            teacher_replay_fraction=min(teacher_replay_fraction, 0.20),
            workers=mcts_workers,
            model=model,
        )
    save_model(model_out, model, model_config)
    return {
        "belief_warmup": belief,
        "teacher_bootstrap": teacher,
        "vector_teacher": vector_teacher,
        "flat_actor_critic": flat,
        "mcts_foundation": self_play,
        "mcts_refinement": refinement,
    }
