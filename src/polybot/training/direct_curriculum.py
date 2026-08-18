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
    num_threads: int | None,
    max_actions: int,
    gate_games: int,
    gate_seed: int,
    gate_score: float = 0.50,
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    teacher_weight: float = 0.50,
    shaping_weight: float = 0.20,
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
    if teacher_steps and teacher_envs < 1:
        raise ValueError("teacher_envs must be positive when teacher is enabled")
    if actor_steps and actor_envs < 1:
        raise ValueError("actor_envs must be positive when actor is enabled")
    if actor_steps and actor_block_steps < 1:
        raise ValueError("actor_block_steps must be positive when actor is enabled")
    if gate_games < 2 or gate_games % 2:
        raise ValueError("direct gate games must be positive and even")
    if not 0.0 <= gate_score <= 1.0:
        raise ValueError("direct gate score must be between zero and one")
    run = run or TrainingRun.create("runs", "direct_curriculum")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "teacher_envs": teacher_envs,
            "teacher_steps": teacher_steps,
            "actor_envs": actor_envs,
            "actor_steps": actor_steps,
            "actor_block_steps": actor_block_steps,
            "gate_games": gate_games,
            "gate_seed": gate_seed,
            "gate_score": gate_score,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
        },
    )
    device = next(candidate.parameters()).device
    champion.to(device).eval()
    for parameter in champion.parameters():
        parameter.requires_grad_(False)
    best_state = model_state_on_cpu(candidate)
    best_score = 0.50
    accepted_blocks = 0
    rollback_blocks = 0
    gate_rows: list[Dict[str, float | str]] = []

    def gate(label: str, block: int) -> bool:
        nonlocal best_state, best_score, accepted_blocks, rollback_blocks
        gate_training = replace(training, seed=int(gate_seed))
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
        accepted = score >= max(float(gate_score), float(best_score))
        if accepted:
            best_score = score
            best_state = model_state_on_cpu(candidate)
            accepted_blocks += 1
        else:
            restore_model_state(candidate, best_state)
            rollback_blocks += 1
        row: Dict[str, float | str] = {
            "curriculum_stage": label,
            "curriculum_block": float(block),
            "direct_gate_score": score,
            "direct_gate_lower_95": float(result["score_lower_95"]),
            "direct_gate_accepted": float(accepted),
            "direct_best_score": float(best_score),
            "direct_rollbacks": float(rollback_blocks),
        }
        gate_rows.append(row)
        run.log(**row)
        return accepted

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
    while completed < actor_steps:
        block += 1
        block_steps = min(actor_block_steps, actor_steps - completed)
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
        "direct_actor_steps": float(actor_steps),
        "direct_actor_blocks": float(block),
        "direct_gate_games": float(gate_games),
        "direct_best_score": float(best_score),
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
