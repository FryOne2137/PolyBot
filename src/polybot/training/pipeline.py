"""Full, disk-minimal curriculum from belief warm-up to anchored MCTS."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..evaluation.matches import evaluate_matches
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from .belief_warmup import warmup_belief
from .run_artifacts import TrainingRun
from .self_play_pool_train import train_self_play_pool
from .self_play_train import train_self_play
from .teacher_bootstrap import train_teacher_bootstrap
from .vector_actor_critic import train_vector_actor_critic
from .vector_teacher import train_vector_teacher


_MAX_SEED = 2**31 - 1


def _offset_seed(seed: int, offset: int) -> int:
    """Return a stable positive seed without relying on process RNG state."""
    return (int(seed) - 1 + int(offset)) % _MAX_SEED + 1


def _seed_schedule(base_seed: int, count: int) -> set[int]:
    """Materialise practical map schedules for overlap validation."""
    return {_offset_seed(base_seed, index) for index in range(max(0, int(count)))}


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
    self_play_pool_envs: int,
    self_play_pool_steps: int,
    self_play_pool_updates: int,
    model_out: Path,
    model_in: Path | None = None,
    teacher_games: int = 0,
    teacher_updates_per_game: int = 16,
    vector_teacher_envs: int = 0,
    vector_teacher_steps: int = 0,
    vector_teacher_updates_per_step: int = 1,
    self_play_pool_threads: int | None = None,
    self_play_pool_pending_leaves: int = 4,
    mcts_workers: int = 1,
    teacher_replay_fraction: float = 0.35,
    refinement_games: int = 0,
    refinement_updates_per_game: int = 4,
    champion_model_paths: Sequence[Path] = (),
    promotion_games: int = 30,
    promotion_score: float = 0.55,
    promotion_lower_bound: float = 0.50,
    self_play_pool_anchor_weight: float = 0.20,
    refinement_seed_offset: int = 10_000_019,
    evaluation_seed: int = 1_500_000_007,
    run: TrainingRun | None = None,
) -> Dict[str, object]:
    """Run all stages in RAM and write exactly one final model checkpoint.

    The order intentionally gives the MCTS stage a competent prior:
    supervised belief -> legal-action teacher -> vector teacher -> flat
    actor-critic -> persistent native SelfPlayPool MCTS -> anchored MCTS ->
    lower-noise MCTS refinement. Games, replay
    buffers and optimiser states are never written; the one requested model
    path is overwritten at safe stage boundaries for crash recovery.
    """
    if not 0.0 <= promotion_score <= 1.0:
        raise ValueError("promotion_score must be between 0 and 1")
    if not 0.0 <= promotion_lower_bound <= 1.0:
        raise ValueError("promotion_lower_bound must be between 0 and 1")
    if not 0.0 <= self_play_pool_anchor_weight <= 1.0:
        raise ValueError("self_play_pool_anchor_weight must be between 0 and 1")
    if promotion_games < 0:
        raise ValueError("promotion_games must not be negative")
    if promotion_games and promotion_games % 2:
        raise ValueError("promotion_games must be even so every map is evaluated from both player slots")
    if refinement_seed_offset == 0:
        raise ValueError("refinement_seed_offset must separate foundation and refinement maps")

    training = replace(training, seed=_offset_seed(training.seed, 0))
    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "pipeline")
    refinement_seed = _offset_seed(training.seed, refinement_seed_offset)
    resolved_evaluation_seed = _offset_seed(evaluation_seed, 0)
    foundation_maps = _seed_schedule(training.seed, self_play_games)
    refinement_maps = _seed_schedule(refinement_seed, refinement_games)
    evaluation_maps = _seed_schedule(resolved_evaluation_seed, (promotion_games + 1) // 2)
    if foundation_maps & refinement_maps:
        raise ValueError("foundation and refinement map seed schedules overlap")
    if evaluation_maps & (foundation_maps | refinement_maps):
        raise ValueError("promotion evaluation map seeds must be held out from training")

    def record_stage(stage_index: int, summary: Dict[str, object]) -> None:
        run.log(stage=stage_index, **summary)

    if model_in is None:
        model = PolicyValueBeliefNet(model_config).to(device)
    else:
        model, model_config = load_model(model_in, device)
    run.write_config(
        training=training,
        model=model_config,
        mcts=mcts_config,
        stage={
            "belief_positions": belief_positions,
            "belief_batch_size": belief_batch_size,
            "teacher_games": teacher_games,
            "vector_teacher_steps": vector_teacher_steps,
            "flat_steps": flat_steps,
            "self_play_pool_steps": self_play_pool_steps,
            "self_play_games": self_play_games,
            "refinement_games": refinement_games,
            "promotion_games": promotion_games,
            "promotion_score": promotion_score,
            "promotion_lower_bound": promotion_lower_bound,
            "self_play_pool_anchor_weight": self_play_pool_anchor_weight,
            "foundation_seed": training.seed,
            "refinement_seed": refinement_seed,
            "evaluation_seed": resolved_evaluation_seed,
            "model_in": model_in,
            "model_out": model_out,
        },
    )

    # A pool can contain the current champion plus older, still-useful styles.
    # They remain frozen for the whole candidate run and are never saved here.
    frozen_models = []
    frozen_configs = []
    for path in champion_model_paths:
        frozen_model, frozen_config = load_model(path, device)
        frozen_models.append(frozen_model.eval())
        frozen_configs.append(frozen_config)

    # Publish a known-good recovery point before entering native training.
    # Atomic replacement in save_model means an interrupted write cannot
    # corrupt the previous champion either.
    if frozen_models:
        save_model(model_out, frozen_models[0], frozen_configs[0])
    else:
        save_model(model_out, model, model_config)

    belief = {}
    if belief_positions:
        belief = warmup_belief(
            training,
            model_config,
            positions=belief_positions,
            batch_size=belief_batch_size,
            model=model,
            preserve_policy_value=model_in is not None,
            run=run.child("belief"),
        )
        record_stage(1, belief)
    teacher = {}
    if teacher_games:
        teacher = train_teacher_bootstrap(
            training, model_config, teacher_games, teacher_updates_per_game, model=model,
            run=run.child("teacher"),
        )
        record_stage(2, teacher)
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
            run=run.child("vector_teacher"),
        )
        record_stage(3, vector_teacher)
    flat = {}
    if flat_steps:
        flat = train_vector_actor_critic(
            training,
            model_config,
            num_envs=flat_envs,
            steps=flat_steps,
            max_actions=mcts_config.native_max_actions,
            model=model,
            run=run.child("actor_critic"),
        )
        record_stage(4, flat)
    pool_self_play = {}
    if self_play_pool_steps:
        if self_play_pool_envs < 1:
            raise ValueError("self_play_pool_envs must be positive when the SelfPlayPool stage is enabled")
        # Distil a frozen pre-stage snapshot into the noisy MCTS target.  This
        # makes the fast same-network pool conservative instead of allowing a
        # few bad online updates to erase the teacher/champion policy.
        pool_anchor = frozen_models[0] if frozen_models else copy.deepcopy(model).eval()
        pool_self_play = train_self_play_pool(
            training,
            model_config,
            mcts_config,
            num_envs=self_play_pool_envs,
            steps=self_play_pool_steps,
            updates=self_play_pool_updates,
            num_threads=self_play_pool_threads,
            max_pending_leaves=self_play_pool_pending_leaves,
            model=model,
            anchor_model=pool_anchor,
            anchor_policy_weight=self_play_pool_anchor_weight,
            run=run.child("self_play_pool"),
        )
        if not frozen_models:
            del pool_anchor
        record_stage(5, pool_self_play)
    # Keep one recoverable model artifact. With a promotion champion, that
    # file must remain the verified champion until the final gate succeeds;
    # otherwise it is safe to preserve the current candidate as before.
    if frozen_models:
        save_model(model_out, frozen_models[0], frozen_configs[0])
    else:
        save_model(model_out, model, model_config)
    self_play = {}
    if self_play_games:
        # With a frozen pool, the foundation is entirely candidate-vs-champion
        # play. Without one we retain the bootstrap curriculum for a first run.
        foundation_opponent = "model" if frozen_models else "mixed"
        foundation_mixed_opponent = "model" if frozen_models else "heuristic"
        foundation_anchor_rate = 1.0 if frozen_models else 0.35
        self_play = train_self_play(
            training,
            model_config,
            mcts_config,
            games=self_play_games,
            updates_per_game=updates_per_game,
            opponent=foundation_opponent,
            random_opponent_rate=foundation_anchor_rate,
            mixed_opponent=foundation_mixed_opponent,
            teacher_replay_fraction=teacher_replay_fraction,
            workers=mcts_workers,
            model=model,
            opponent_models=frozen_models,
            run=run.child("mcts_foundation"),
        )
        record_stage(6, self_play)
        # A failed refinement must never publish an untested candidate over a
        # verified champion. Without a champion keep the existing resume aid.
        if frozen_models:
            save_model(model_out, frozen_models[0], frozen_configs[0])
        else:
            save_model(model_out, model, model_config)
    refinement = {}
    if refinement_games:
        refinement_training = replace(
            training,
            seed=refinement_seed,
            learning_rate=training.learning_rate * 0.5,
            exploration_rate=min(training.exploration_rate, 0.03),
            teacher_policy_target_weight=min(training.teacher_policy_target_weight, 0.04),
        )
        refinement_mcts = replace(
            mcts_config,
            teacher_prior_weight=min(mcts_config.teacher_prior_weight, 0.08),
        )
        # Refinement sees both its own current play and the frozen pool. This
        # prevents the candidate from optimising solely against one opponent.
        refinement_opponent = "mixed" if frozen_models else "mixed"
        refinement_mixed_opponent = "model" if frozen_models else "heuristic"
        refinement_anchor_rate = 0.50 if frozen_models else 0.20
        refinement = train_self_play(
            refinement_training,
            model_config,
            refinement_mcts,
            games=refinement_games,
            updates_per_game=refinement_updates_per_game,
            opponent=refinement_opponent,
            random_opponent_rate=refinement_anchor_rate,
            mixed_opponent=refinement_mixed_opponent,
            teacher_replay_fraction=min(teacher_replay_fraction, 0.20),
            workers=mcts_workers,
            model=model,
            opponent_models=frozen_models,
            run=run.child("mcts_refinement"),
        )
        record_stage(7, refinement)
    promotion: Dict[str, float] = {}
    if frozen_models and promotion_games:
        # Evaluation has its own held-out seed schedule. Every generated map
        # is played twice with player slots swapped.
        evaluation_training = replace(training, seed=resolved_evaluation_seed)
        gate_mcts = replace(
            mcts_config,
            temperature=0.0,
            strategic_prior_weight=0.0,
            teacher_prior_weight=0.0,
        )
        match = evaluate_matches(
            model, device, evaluation_training, gate_mcts, promotion_games, model_b=frozen_models[0],
        )
        candidate_score = float(match["model_a_score"])
        score_lower_95 = float(match["score_lower_95"])
        promoted = (
            candidate_score >= promotion_score
            and score_lower_95 >= promotion_lower_bound
        )
        promotion = {
            "candidate_score": float(candidate_score),
            "promotion_threshold": float(promotion_score),
            "promotion_lower_bound": float(promotion_lower_bound),
            "promoted": float(promoted),
            **{"match_" + key: float(value) for key, value in match.items()},
        }
        # A rejected candidate must not replace the usable model artifact.
        # Save the first supplied checkpoint (the champion) back to model_out.
        if not promoted:
            model, model_config = frozen_models[0], frozen_configs[0]
        promotion_run = run.child("promotion")
        promotion_run.log(**promotion)
        promotion["run_dir"] = str(promotion_run.finish(promotion))
        record_stage(8, promotion)
    save_model(model_out, model, model_config)
    summary: Dict[str, object] = {
        "belief_warmup": belief,
        "teacher_bootstrap": teacher,
        "vector_teacher": vector_teacher,
        "flat_actor_critic": flat,
        "self_play_pool": pool_self_play,
        "mcts_foundation": self_play,
        "mcts_refinement": refinement,
        "promotion": promotion,
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
