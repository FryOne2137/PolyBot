"""Regression-safe policy curriculum before belief MCTS is enabled."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping

import torch

from ..config import ModelConfig, TrainingConfig
from ..evaluation.matches import evaluate_direct_matches
from ..models.policy_value_belief import PolicyValueBeliefNet
from .run_artifacts import TrainingRun
from .vector_actor_critic import train_vector_actor_critic
from .vector_teacher import train_vector_teacher


def model_state_on_cpu(model: PolicyValueBeliefNet) -> Dict[str, torch.Tensor]:
    """Take an in-memory rollback point without creating another checkpoint."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def restore_model_state(
    model: PolicyValueBeliefNet,
    state: Dict[str, torch.Tensor],
) -> None:
    """Restore a rollback point on the model's existing device."""
    model.load_state_dict(state, strict=True)


def direct_readiness(
    result: Mapping[str, Any],
    *,
    score: float,
    city_captures_per_game: float,
    completion_rate: float,
) -> bool:
    """Return whether a raw policy has learned enough to make MCTS useful."""
    return (
        float(result.get("model_a_score", 0.0)) >= float(score)
        and float(result.get("model_a_city_captures_per_game", 0.0))
        >= float(city_captures_per_game)
        and float(result.get("completion_rate", 0.0))
        >= float(completion_rate)
    )


def _objective_progress(
    result: Mapping[str, Any],
    city_captures_per_game: float,
    completion_rate: float,
) -> float:
    """Bounded progress signal used to select equal-score direct policies."""
    captures = float(result.get("model_a_city_captures_per_game", 0.0))
    completion = float(result.get("completion_rate", 0.0))
    capture_progress = (
        1.0
        if city_captures_per_game <= 0.0
        else min(1.0, captures / city_captures_per_game)
    )
    completion_progress = (
        1.0
        if completion_rate <= 0.0
        else min(1.0, completion / completion_rate)
    )
    return capture_progress + completion_progress


def train_direct_policy_curriculum(
    training: TrainingConfig,
    model_config: ModelConfig,
    candidate: PolicyValueBeliefNet,
    champion: PolicyValueBeliefNet,
    *,
    teacher_envs: int,
    teacher_steps: int,
    actor_envs: int,
    actor_steps: int,
    actor_block_steps: int,
    actor_max_steps: int | None = None,
    num_threads: int | None,
    max_actions: int,
    gate_games: int,
    gate_seed: int,
    gate_max_moves: int | None = None,
    gate_score: float = 0.50,
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    teacher_weight: float = 0.50,
    shaping_weight: float = 0.20,
    readiness_score: float = 0.50,
    readiness_city_captures_per_game: float = 0.05,
    readiness_completion_rate: float = 0.05,
    readiness_patience: int = 2,
    gate_game_callback: Callable[[str, int, Mapping[str, Any]], None] | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, Any]:
    """Learn visible objectives directly and roll back every regression.

    The fixed mini-arena is only a model-selection instrument inside one
    round. The league's later held-out MCTS gate remains the sole promotion
    authority, so no mini-gate result can replace the champion.
    """
    if teacher_steps < 0 or actor_steps < 0:
        raise ValueError("direct curriculum step counts must not be negative")
    if (
        actor_max_steps is not None
        and actor_steps > 0
        and int(actor_max_steps) < actor_steps
    ):
        raise ValueError("actor_max_steps must not be smaller than actor_steps")
    maximum_actor_steps = (
        actor_steps
        if actor_max_steps is None or actor_steps == 0
        else int(actor_max_steps)
    )
    if teacher_steps and teacher_envs < 1:
        raise ValueError("teacher_envs must be positive when teacher is enabled")
    if maximum_actor_steps and actor_envs < 1:
        raise ValueError("actor_envs must be positive when actor is enabled")
    if maximum_actor_steps and actor_block_steps < 1:
        raise ValueError("actor_block_steps must be positive when actor is enabled")
    if gate_games < 2 or gate_games % 2:
        raise ValueError("direct gate games must be positive and even")
    resolved_gate_max_moves = (
        training.max_moves if gate_max_moves is None else int(gate_max_moves)
    )
    if resolved_gate_max_moves < 1:
        raise ValueError("gate_max_moves must be positive")
    for name, value in {
        "gate_score": gate_score,
        "readiness_score": readiness_score,
        "readiness_completion_rate": readiness_completion_rate,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if readiness_city_captures_per_game < 0.0:
        raise ValueError(
            "readiness_city_captures_per_game must not be negative"
        )
    if readiness_patience < 1:
        raise ValueError("readiness_patience must be positive")
    run = run or TrainingRun.create("runs", "direct_curriculum")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "teacher_envs": teacher_envs,
            "teacher_steps": teacher_steps,
            "actor_envs": actor_envs,
            "actor_steps": actor_steps,
            "actor_max_steps": maximum_actor_steps,
            "actor_block_steps": actor_block_steps,
            "gate_games": gate_games,
            "gate_seed": gate_seed,
            "gate_max_moves": resolved_gate_max_moves,
            "gate_score": gate_score,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
            "readiness_score": readiness_score,
            "readiness_city_captures_per_game": readiness_city_captures_per_game,
            "readiness_completion_rate": readiness_completion_rate,
            "readiness_patience": readiness_patience,
        },
    )
    device = next(candidate.parameters()).device
    champion.to(device).eval()
    for parameter in champion.parameters():
        parameter.requires_grad_(False)
    best_state = model_state_on_cpu(candidate)
    best_score = 0.0
    best_progress = 0.0
    best_city_capture_rate = 0.0
    best_completion_rate = 0.0
    accepted_blocks = 0
    rollback_blocks = 0
    readiness_streak = 0
    ready = False
    gate_rows: list[Dict[str, float | str]] = []

    def gate(label: str, block: int, *, baseline: bool = False) -> bool:
        nonlocal best_state, best_score, best_progress
        nonlocal best_city_capture_rate, best_completion_rate
        nonlocal accepted_blocks, rollback_blocks, readiness_streak, ready
        gate_training = replace(
            training,
            seed=int(gate_seed),
            max_moves=resolved_gate_max_moves,
        )
        result = evaluate_direct_matches(
            candidate,
            champion,
            device,
            gate_training,
            gate_games,
            game_callback=(
                None
                if gate_game_callback is None
                else lambda game: gate_game_callback(label, block, game)
            ),
        )
        score = float(result["model_a_score"])
        progress = _objective_progress(
            result,
            readiness_city_captures_per_game,
            readiness_completion_rate,
        )
        accepted = baseline or (
            score >= float(gate_score)
            and (
                score > best_score + 1e-8
                or (
                    score >= best_score - 1e-8
                    and progress >= best_progress - 1e-8
                )
            )
        )
        if accepted:
            best_score = score
            best_progress = progress
            best_city_capture_rate = float(
                result.get("model_a_city_captures_per_game", 0.0)
            )
            best_completion_rate = float(result.get("completion_rate", 0.0))
            best_state = model_state_on_cpu(candidate)
            if not baseline:
                accepted_blocks += 1
            is_ready = direct_readiness(
                result,
                score=readiness_score,
                city_captures_per_game=readiness_city_captures_per_game,
                completion_rate=readiness_completion_rate,
            )
            readiness_streak = readiness_streak + 1 if is_ready else 0
            ready = readiness_streak >= readiness_patience
        else:
            restore_model_state(candidate, best_state)
            rollback_blocks += 1
            readiness_streak = 0
            ready = False
        row: Dict[str, float | str] = {
            "curriculum_stage": label,
            "curriculum_block": float(block),
            "direct_gate_score": score,
            "direct_gate_lower_95": float(result["score_lower_95"]),
            "direct_gate_accepted": float(accepted),
            "direct_best_score": float(best_score),
            "direct_gate_city_captures_per_game": float(
                result.get("model_a_city_captures_per_game", 0.0)
            ),
            "direct_gate_completion_rate": float(
                result.get("completion_rate", 0.0)
            ),
            "direct_objective_progress": float(progress),
            "direct_readiness_streak": float(readiness_streak),
            "direct_ready": float(ready),
            "direct_rollbacks": float(rollback_blocks),
        }
        gate_rows.append(row)
        run.log(**row)
        return accepted

    # Protect a retained challenger against its own previous performance,
    # rather than assuming every round starts from a 0.50 champion clone.
    gate("baseline", -1, baseline=True)

    teacher_summary: Dict[str, Any] = {}
    if teacher_steps:
        teacher_summary = train_vector_teacher(
            replace(training, seed=int(training.seed)),
            model_config,
            num_envs=teacher_envs,
            steps=teacher_steps,
            num_threads=num_threads,
            max_actions=max_actions,
            updates_per_step=1,
            model=candidate,
            anchor_model=champion,
            policy_anchor_weight=policy_anchor_weight,
            value_anchor_weight=value_anchor_weight,
            learner_action_probability=0.50,
            run=run.child("teacher"),
        )
        gate("teacher", 0)

    actor_summaries: list[Dict[str, Any]] = []
    completed = 0
    block = 0
    while completed < maximum_actor_steps and (
        completed < actor_steps or not ready
    ):
        block += 1
        block_steps = min(actor_block_steps, maximum_actor_steps - completed)
        block_training = replace(
            training,
            seed=(int(training.seed) - 1 + 1_000_003 * block) % (2**31 - 1) + 1,
        )
        summary = train_vector_actor_critic(
            block_training,
            model_config,
            num_envs=actor_envs,
            steps=block_steps,
            num_threads=num_threads,
            max_actions=max_actions,
            entropy_weight=0.004,
            value_weight=0.50,
            teacher_weight=teacher_weight,
            gamma=0.995,
            model=candidate,
            anchor_model=champion,
            policy_anchor_weight=policy_anchor_weight,
            value_anchor_weight=value_anchor_weight,
            shaping_weight=shaping_weight,
            run=run.child(f"actor_{block:03d}"),
        )
        actor_summaries.append(summary)
        completed += block_steps
        gate("actor", block)

    restore_model_state(candidate, best_state)
    summary: Dict[str, Any] = {
        "direct_teacher_steps": float(teacher_steps),
        "direct_actor_min_steps": float(actor_steps),
        "direct_actor_max_steps": float(maximum_actor_steps),
        "direct_actor_steps": float(completed),
        "direct_actor_blocks": float(block),
        "direct_gate_games": float(gate_games),
        "direct_best_score": float(best_score),
        "direct_best_city_captures_per_game": float(best_city_capture_rate),
        "direct_best_completion_rate": float(best_completion_rate),
        "direct_ready": float(ready),
        "direct_readiness_streak": float(readiness_streak),
        "direct_readiness_patience": float(readiness_patience),
        "direct_accepted_blocks": float(accepted_blocks),
        "direct_rollback_blocks": float(rollback_blocks),
        "direct_policy_anchor_weight": float(policy_anchor_weight),
        "direct_value_anchor_weight": float(value_anchor_weight),
        "direct_teacher_summary": copy.deepcopy(teacher_summary),
        "direct_actor_summaries": copy.deepcopy(actor_summaries),
        "direct_gate_rows": copy.deepcopy(gate_rows),
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
