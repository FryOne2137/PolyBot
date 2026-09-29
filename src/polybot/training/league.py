"""Population-based self-play league with regression-safe promotion gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from ..config import MCTSConfig, TrainingConfig
from ..evaluation.matches import evaluate_direct_matches
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .league_db import LeagueDatabase
from .belief_recovery import BeliefQualityError, BeliefRecoveryRequired
from .direct_curriculum import (
    model_state_on_cpu,
    restore_model_state,
    train_direct_policy_curriculum,
)
from .routed_self_play_pool import (
    evaluate_matches_pool,
    require_routed_self_play_pool,
    train_population_pool,
)
from .run_artifacts import TrainingRun
from .self_play_pool_train import train_self_play_pool
from .vector_population import (
    evaluate_direct_population_matches,
    train_direct_population_curriculum,
)


_MAX_SEED = 2**31 - 1


def _offset_seed(seed: int, offset: int) -> int:
    return (int(seed) - 1 + int(offset)) % _MAX_SEED + 1


def _next_checkpoint_path(directory: Path, prefix: str) -> Path:
    """Return an unused, monotonically numbered champion checkpoint path."""
    index = 1
    while True:
        path = directory / f"{prefix}-{index:03d}.pt"
        if not path.exists():
            return path
        index += 1


def _promotion_gate_decision(
    evaluation_results: Sequence[Mapping[str, Any]],
    champion_id: int,
    *,
    promotion_score: float,
    promotion_lower_bound: float,
    population_score: float,
    historical_score_floor: float,
) -> Dict[str, float | bool]:
    """Apply one promotion rule to either raw-policy or MCTS evaluations."""
    champion_result = next(
        result
        for result in evaluation_results
        if int(result["opponent_id"]) == champion_id
    )
    candidate_score = float(champion_result["model_a_score"])
    weighted_population_score = sum(
        float(result["sampling_probability"])
        * float(result["model_a_score"])
        for result in evaluation_results
    )
    minimum_historical_score = min(
        float(result["model_a_score"])
        for result in evaluation_results
    )
    champion_lower_95 = float(champion_result["score_lower_95"])
    promoted = (
        candidate_score >= promotion_score
        and champion_lower_95 >= promotion_lower_bound
        and weighted_population_score >= population_score
        and minimum_historical_score >= historical_score_floor
    )
    return {
        "candidate_score": candidate_score,
        "champion_lower_95": champion_lower_95,
        "population_score": weighted_population_score,
        "minimum_historical_score": minimum_historical_score,
        "promoted": promoted,
    }


def _challenger_continuation_action(
    *,
    valid: bool,
    candidate_score: float,
    population_score: float,
    minimum_historical_score: float,
    historical_score_floor: float,
    has_previous_challenger: bool,
) -> str:
    """Choose the checkpoint from which the next league round should learn.

    Promotion remains deliberately strict, but a noisy score just below 0.50
    must not erase a useful multi-round challenger lineage.  A valid candidate
    may therefore continue while it stays above the league's existing safety
    floor and remains competitive with the population.  If it violates that
    continuation gate, fall back to the previous challenger instead of
    restarting from the champion.
    """
    candidate_is_safe = (
        valid
        and candidate_score >= historical_score_floor
        and population_score >= 0.50
        and minimum_historical_score >= historical_score_floor
    )
    if candidate_is_safe:
        return "candidate"
    if has_previous_challenger:
        return "previous"
    return "none"


def _select_exploiter_checkpoint(
    main_summary: Mapping[str, Any],
    exploiter_summary: Mapping[str, Any],
) -> bool:
    """Use the free branch only when it cleared the same development floor.

    The final promotion gate remains independent.  This is only a cheap,
    population-held-out decision about which lineage deserves that expensive
    gate, so a spectacular but brittle exploiter cannot replace the main line.
    """
    if not bool(exploiter_summary.get("direct_population_full_gate_selected", 0.0)):
        return False
    if not bool(main_summary.get("direct_population_full_gate_selected", 0.0)):
        return True
    exploiter_score = float(
        exploiter_summary.get("direct_population_full_gate_score", 0.0)
    )
    main_score = float(main_summary.get("direct_population_full_gate_score", 0.0))
    if exploiter_score > main_score + 1e-8:
        return True
    if abs(exploiter_score - main_score) > 1e-8:
        return False
    return float(
        exploiter_summary.get("direct_population_full_gate_worst_score", 0.0)
    ) > float(
        main_summary.get("direct_population_full_gate_worst_score", 0.0)
    ) + 1e-8


def _mcts_probe_passed(
    raw_score: float,
    search_score: float,
    *,
    minimum_search_score: float,
    minimum_improvement: float,
    valid: bool = True,
) -> bool:
    """Return whether search is useful enough to unlock later MCTS training."""
    return bool(valid) and search_score >= minimum_search_score and (
        search_score > raw_score + minimum_improvement
    )


def _round_champion_mode(
    requested_mode: str,
    capability: Mapping[str, Any] | None,
) -> str:
    """Resolve one league round without losing an earned MCTS transition.

    ``auto`` remains a raw-policy curriculum until its protected champion has
    a persisted, successful MCTS probe.  MCTS promotions write that capability
    to their own checkpoint too, so a resumed league run stays in absorption
    mode rather than silently returning to raw-only training.
    """
    if requested_mode == "auto":
        return "mcts" if bool(capability and capability.get("mcts_ready")) else "flat"
    if requested_mode in {"flat", "mcts"}:
        return requested_mode
    raise ValueError("requested_mode must be 'flat', 'mcts' or 'auto'")


def _flat_distillation_gate_decision(
    before_score: float,
    after_score: float,
    *,
    minimum_score: float,
    valid: bool = True,
) -> Dict[str, float | bool]:
    """Keep search imitation only after raw policy improves on paired maps."""
    improved = after_score > before_score + 1e-8
    accepted = bool(valid) and improved and after_score >= minimum_score
    return {
        "before_score": float(before_score),
        "after_score": float(after_score),
        "improvement": float(after_score - before_score),
        "accepted": accepted,
    }


def _champion_focused_probabilities(
    population_ids: Sequence[int],
    probabilities: Sequence[float],
    champion_id: int,
    *,
    champion_probability: float,
    probability_floor: float,
) -> list[float]:
    """Raise the current champion's training share without dropping a rival.

    Once a model has earned a promotion, learning has to be driven mainly by
    the policy it must now beat.  Historical models retain at least the PFSP
    floor, so this is focused self-play rather than forgetting the population.
    """
    result = [float(value) for value in probabilities]
    if len(population_ids) != len(result):
        raise ValueError("population ids and probabilities must have equal length")
    try:
        champion_index = list(population_ids).index(champion_id)
    except ValueError as error:
        raise ValueError("current champion is missing from population") from error
    other_indices = [
        index for index in range(len(result)) if index != champion_index
    ]
    if not other_indices:
        return [1.0]
    if not 0.0 <= probability_floor <= 1.0:
        raise ValueError("probability_floor must be between zero and one")
    maximum_champion_probability = 1.0 - len(other_indices) * probability_floor
    if champion_probability > maximum_champion_probability + 1e-8:
        raise ValueError(
            "post-bootstrap champion probability leaves no PFSP floor for rivals"
        )
    if result[champion_index] >= champion_probability:
        return result
    remaining = 1.0 - champion_probability
    free_mass = remaining - len(other_indices) * probability_floor
    old_other_mass = sum(result[index] for index in other_indices)
    if old_other_mass <= 0.0:
        shares = [1.0 / len(other_indices)] * len(other_indices)
    else:
        shares = [result[index] / old_other_mass for index in other_indices]
    result[champion_index] = champion_probability
    for index, share in zip(other_indices, shares):
        result[index] = probability_floor + free_mass * share
    return result


def _curriculum_controls(
    *,
    champion_status: str,
    direct_teacher_steps: int,
    direct_actor_steps: int,
    direct_actor_max_steps: int,
    direct_population_steps: int,
    direct_population_max_steps: int,
    direct_policy_anchor_weight: float,
    direct_population_teacher_weight: float,
    direct_population_opponent_policy_weight: float,
    post_bootstrap_population_steps: int,
    post_bootstrap_population_max_steps: int,
    post_bootstrap_policy_anchor_weight: float,
    post_bootstrap_teacher_weight: float,
    post_bootstrap_opponent_policy_weight: float,
    direct_objective_steps: int = 0,
    post_bootstrap_objective_steps: int = 0,
    refresh_promoted_curriculum: bool = False,
) -> Dict[str, float | int | str | bool]:
    """Select bootstrap or champion-focused raw-policy training.

    The teacher is valuable only until a model has earned its first league
    promotion.  Reapplying it to an established champion lineage repeatedly
    trains the candidate to beat the teacher rather than its current parent.
    """
    post_bootstrap = champion_status == "promoted"
    # A promoted checkpoint is normally mature enough that replaying the
    # heuristic curriculum would only pull it back toward the teacher.  A
    # weak-but-promoted checkpoint is the important exception: its policy and
    # value are not reliable priors for MCTS yet.  Recovery deliberately
    # re-enables the dense, visible-only curriculum while keeping the frozen
    # champion as the anchor in the calling code.
    if post_bootstrap and refresh_promoted_curriculum:
        return {
            "phase": "promoted_recovery",
            "post_bootstrap": True,
            "teacher_steps": direct_teacher_steps,
            "objective_steps": direct_objective_steps,
            "actor_steps": direct_actor_steps,
            "actor_max_steps": direct_actor_max_steps,
            "population_steps": direct_population_steps,
            "population_max_steps": direct_population_max_steps,
            "policy_anchor_weight": direct_policy_anchor_weight,
            "population_teacher_weight": direct_population_teacher_weight,
            "population_opponent_policy_weight": (
                direct_population_opponent_policy_weight
            ),
        }
    if post_bootstrap:
        return {
            "phase": "champion_population",
            "post_bootstrap": True,
            "teacher_steps": 0,
            # The old short-horizon objective stage was not event-selected:
            # it replayed arbitrary eight-decision windows and consumed about
            # an hour per round without a reliable lift. Established lineages
            # now use event-triggered prefix replay inside population play.
            # Keep the legacy argument for old commands, but never run it.
            "objective_steps": 0,
            "actor_steps": 0,
            "actor_max_steps": 0,
            "population_steps": post_bootstrap_population_steps,
            "population_max_steps": post_bootstrap_population_max_steps,
            "policy_anchor_weight": post_bootstrap_policy_anchor_weight,
            "population_teacher_weight": post_bootstrap_teacher_weight,
            "population_opponent_policy_weight": (
                post_bootstrap_opponent_policy_weight
            ),
        }
    return {
        "phase": "bootstrap",
        "post_bootstrap": False,
        "teacher_steps": direct_teacher_steps,
        "objective_steps": direct_objective_steps,
        "actor_steps": direct_actor_steps,
        "actor_max_steps": direct_actor_max_steps,
        "population_steps": direct_population_steps,
        "population_max_steps": direct_population_max_steps,
        "policy_anchor_weight": direct_policy_anchor_weight,
        "population_teacher_weight": direct_population_teacher_weight,
        "population_opponent_policy_weight": (
            direct_population_opponent_policy_weight
        ),
    }


def _validate_options(
    *,
    rounds: int,
    promotion_games: int,
    historical_games: int,
    promotion_score: float,
    promotion_lower_bound: float,
    population_score: float,
    historical_score_floor: float,
    evaluation_seed_stride: int,
    round_seed_stride: int,
    population_size: int,
    population_games: int,
    population_updates_per_game: int,
    population_workers: int,
    population_max_envs: int,
    population_mcts_critical_only: bool,
    population_mcts_critical_fraction: float,
    population_mcts_critical_minimum_score: float,
    population_mcts_critical_replay_fraction: float,
    population_mcts_reanalyse_capacity: int,
    population_mcts_reanalyse_positions: int,
    population_mcts_reanalyse_replay_fraction: float,
    pfsp_alpha: float,
    champion_probability: float,
    minimum_sampling_probability: float,
    champion_anchor_weight: float,
    elo_k: float,
) -> None:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if promotion_games < 2 or promotion_games % 2:
        raise ValueError("promotion_games must be positive and even")
    if historical_games < 2 or historical_games % 2:
        raise ValueError("historical_games must be positive and even")
    for name, value in {
        "promotion_score": promotion_score,
        "promotion_lower_bound": promotion_lower_bound,
        "population_score": population_score,
        "historical_score_floor": historical_score_floor,
        "champion_probability": champion_probability,
        "minimum_sampling_probability": minimum_sampling_probability,
        "champion_anchor_weight": champion_anchor_weight,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if population_games < 0 or population_updates_per_game < 0:
        raise ValueError("population games and updates must not be negative")
    if population_games and population_workers < 1:
        raise ValueError("population_workers must be positive")
    if population_max_envs < 1:
        raise ValueError("population_max_envs must be positive")
    if not 0.0 <= population_mcts_critical_fraction <= 1.0:
        raise ValueError(
            "population_mcts_critical_fraction must be between zero and one"
        )
    if population_mcts_critical_minimum_score < 0.0:
        raise ValueError(
            "population_mcts_critical_minimum_score must be non-negative"
        )
    if not 0.0 <= population_mcts_critical_replay_fraction <= 1.0:
        raise ValueError(
            "population_mcts_critical_replay_fraction must be between zero and one"
        )
    if population_mcts_critical_only and population_mcts_critical_fraction <= 0.0:
        raise ValueError(
            "population_mcts_critical_only requires positive critical fraction"
        )
    if population_mcts_reanalyse_capacity < 1:
        raise ValueError("population_mcts_reanalyse_capacity must be positive")
    if population_mcts_reanalyse_positions < 0:
        raise ValueError("population_mcts_reanalyse_positions must not be negative")
    if not 0.0 <= population_mcts_reanalyse_replay_fraction <= 1.0:
        raise ValueError(
            "population_mcts_reanalyse_replay_fraction must be between zero and one"
        )
    if pfsp_alpha <= 0.0:
        raise ValueError("pfsp_alpha must be positive")
    if elo_k <= 0.0:
        raise ValueError("elo_k must be positive")
    if round_seed_stride == 0:
        raise ValueError("round_seed_stride must not be zero")
    if rounds > 1 and evaluation_seed_stride == 0:
        raise ValueError("evaluation_seed_stride must rotate held-out maps")


def train_promotion_league(
    training: TrainingConfig,
    *,
    champion_in: Path,
    challenger_in: Path | None = None,
    checkpoint_dir: Path,
    rounds: int,
    promotion_games: int,
    promotion_score: float,
    promotion_lower_bound: float,
    evaluation_seed: int,
    evaluation_seed_stride: int,
    round_seed_stride: int,
    checkpoint_prefix: str,
    mcts: MCTSConfig,
    num_envs: int,
    steps: int,
    updates: int,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    database_path: Path | None = None,
    population_model_paths: Sequence[Path] = (),
    population_size: int = 6,
    population_games: int = 0,
    population_updates_per_game: int = 0,
    population_workers: int = 16,
    population_max_envs: int = 16,
    population_offline_collection: bool = True,
    population_mcts_critical_only: bool = True,
    population_mcts_critical_fraction: float = 0.25,
    population_mcts_critical_minimum_score: float = 0.25,
    population_mcts_critical_replay_fraction: float = 1.0,
    population_mcts_reanalyse_path: Path | None = None,
    population_mcts_reanalyse_capacity: int = 4_096,
    population_mcts_reanalyse_positions: int = 8,
    population_mcts_reanalyse_replay_fraction: float = 0.50,
    pfsp_alpha: float = 2.0,
    champion_probability: float = 0.40,
    minimum_sampling_probability: float = 0.05,
    historical_games: int = 20,
    population_score: float = 0.52,
    historical_score_floor: float = 0.45,
    elo_k: float = 32.0,
    champion_anchor_weight: float = 0.30,
    champion_value_anchor_weight: float = 0.25,
    direct_teacher_envs: int = 128,
    direct_teacher_steps: int = 0,
    direct_objective_envs: int = 128,
    direct_objective_steps: int = 0,
    direct_objective_block_steps: int = 500,
    direct_objective_rollout_horizon: int = 8,
    direct_objective_updates_per_rollout: int = 2,
    direct_objective_teacher_probability: float = 0.35,
    direct_actor_envs: int = 128,
    direct_actor_steps: int = 0,
    direct_actor_block_steps: int = 250,
    direct_actor_max_steps: int = 3_000,
    direct_gate_games: int = 20,
    direct_gate_max_moves: int = 700,
    direct_gate_score: float = 0.50,
    direct_policy_anchor_weight: float = 0.75,
    direct_value_anchor_weight: float = 0.50,
    direct_teacher_weight: float = 0.50,
    direct_shaping_weight: float = 0.20,
    direct_readiness_score: float = 0.50,
    direct_readiness_city_captures_per_game: float = 0.05,
    direct_readiness_completion_rate: float = 0.05,
    direct_readiness_patience: int = 2,
    direct_regression_patience: int = 2,
    direct_population_envs: int = 128,
    direct_population_steps: int = 0,
    direct_population_max_steps: int = 10_000,
    direct_population_block_steps: int = 1_000,
    direct_population_branch_count: int = 1,
    direct_population_branch_steps: int = 2_000,
    direct_population_branch_gate_games: int = 20,
    direct_population_branch_min_champion_score: float = 0.50,
    direct_population_gate_games: int = 100,
    direct_population_diagnostic_gate_games: int = 20,
    direct_population_gate_score: float = 0.50,
    direct_population_historical_floor: float = 0.45,
    direct_population_readiness_patience: int = 2,
    direct_population_regression_patience: int = 2,
    direct_population_opponent_policy_weight: float = 0.50,
    direct_population_teacher_weight: float = 0.25,
    direct_population_shaping_weight: float = 0.20,
    direct_population_shaping_final_weight: float = 0.0,
    direct_population_ppo_rollout_steps: int = 32,
    direct_population_ppo_epochs: int = 2,
    direct_population_ppo_minibatch_size: int = 256,
    direct_population_ppo_clip_range: float = 0.15,
    direct_population_ppo_gamma: float = 1.0,
    direct_population_ppo_gae_lambda: float = 0.95,
    direct_population_ppo_terminal_advantage_weight: float = 0.75,
    direct_population_ppo_top_advantage_fraction: float = 0.50,
    direct_population_ppo_target_kl: float = 0.010,
    direct_population_ppo_hard_kl: float = 0.040,
    direct_population_ppo_rollout_temperature: float = 0.85,
    direct_population_plateau_patience: int = 2,
    direct_population_plateau_temperature_increment: float = 0.10,
    direct_population_plateau_max_temperature: float = 1.20,
    direct_population_ppo_max_pending_examples: int = 32_768,
    direct_population_ppo_min_learning_rate: float = 1.0e-5,
    direct_population_ppo_lr_backoff: float = 0.5,
    direct_population_objective_replay_horizon: int = 16,
    direct_population_objective_replay_capacity: int = 512,
    direct_population_objective_replay_batch_size: int = 32,
    direct_population_objective_replay_interval: int = 16,
    direct_population_objective_replay_weight: float = 0.35,
    direct_population_objective_replay_recency: float = 0.92,
    direct_population_objective_replay_max_epochs: int = 4,
    direct_population_ppg_enabled: bool = True,
    direct_population_ppg_interval: int = 8,
    direct_population_ppg_buffer_capacity: int = 8_192,
    direct_population_ppg_samples_per_rollout: int = 512,
    direct_population_ppg_phase_samples: int = 4_096,
    direct_population_ppg_epochs: int = 2,
    direct_population_ppg_minibatch_size: int = 256,
    direct_population_ppg_policy_kl_weight: float = 1.0,
    direct_population_ppg_value_weight: float = 1.0,
    direct_population_ema_enabled: bool = True,
    direct_population_ema_decay: float = 0.999,
    breakthrough_exploiter: bool = False,
    exploiter_champion_probability: float = 0.75,
    exploiter_policy_anchor_weight: float = 0.0,
    exploiter_value_anchor_weight: float = 0.0,
    exploiter_ppo_target_kl: float = 0.010,
    exploiter_ppo_hard_kl: float = 0.040,
    exploiter_ppo_rollout_temperature: float = 1.0,
    post_bootstrap_population_steps: int = 6_000,
    post_bootstrap_population_max_steps: int = 12_000,
    post_bootstrap_objective_steps: int = 0,
    post_bootstrap_champion_probability: float = 0.65,
    post_bootstrap_policy_anchor_weight: float = 0.35,
    post_bootstrap_teacher_weight: float = 0.02,
    post_bootstrap_opponent_policy_weight: float = 0.65,
    refresh_promoted_curriculum: bool = False,
    champion_mode: str = "flat",
    flat_mcts_distill_games: int = 0,
    flat_mcts_distill_simulations: int = 32,
    flat_mcts_distill_updates_per_game: int = 2,
    flat_mcts_distill_gate_games: int = 20,
    flat_mcts_distill_policy_anchor_weight: float = 0.20,
    flat_mcts_distill_value_anchor_weight: float = 0.25,
    flat_mcts_distill_max_envs: int = 16,
    flat_mcts_distill_max_moves: int = 600,
    flat_mcts_distill_critical_fraction: float = 0.25,
    flat_mcts_distill_critical_minimum_score: float = 0.15,
    flat_mcts_distill_critical_replay_fraction: float = 0.50,
    mcts_probe_games: int = 10,
    mcts_probe_simulations: int = 32,
    mcts_probe_score: float = 0.55,
    mcts_probe_minimum_improvement: float = 0.0,
    mcts_probe_patience: int = 2,
    mcts_simulation_ladder: Sequence[int] = (),
    mcts_gate_games: int = 20,
    mcts_gate_score: float = 0.50,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train candidates against a PFSP population and promote robust winners.

    Flat mode promotes only the raw policy against a weighted mixture of the
    champion and historical checkpoints. Optional MCTS expert distillation may
    teach that raw policy before the gate, but is rolled back unless the raw
    apprentice improves on paired development maps. Legacy MCTS mode keeps the
    full search ladder. SQLite stores game/metric lineage; tensors remain in
    ``.pt`` files referenced by path and SHA-256 rather than database BLOBs.
    """
    _validate_options(
        rounds=rounds,
        promotion_games=promotion_games,
        historical_games=historical_games,
        promotion_score=promotion_score,
        promotion_lower_bound=promotion_lower_bound,
        population_score=population_score,
        historical_score_floor=historical_score_floor,
        evaluation_seed_stride=evaluation_seed_stride,
        round_seed_stride=round_seed_stride,
        population_size=population_size,
        population_games=population_games,
        population_updates_per_game=population_updates_per_game,
        population_workers=population_workers,
        population_max_envs=population_max_envs,
        population_mcts_critical_only=population_mcts_critical_only,
        population_mcts_critical_fraction=population_mcts_critical_fraction,
        population_mcts_critical_minimum_score=(
            population_mcts_critical_minimum_score
        ),
        population_mcts_critical_replay_fraction=(
            population_mcts_critical_replay_fraction
        ),
        population_mcts_reanalyse_capacity=population_mcts_reanalyse_capacity,
        population_mcts_reanalyse_positions=population_mcts_reanalyse_positions,
        population_mcts_reanalyse_replay_fraction=(
            population_mcts_reanalyse_replay_fraction
        ),
        pfsp_alpha=pfsp_alpha,
        champion_probability=champion_probability,
        minimum_sampling_probability=minimum_sampling_probability,
        champion_anchor_weight=champion_anchor_weight,
        elo_k=elo_k,
    )
    if not checkpoint_prefix:
        raise ValueError("checkpoint_prefix must not be empty")
    if champion_mode not in {"flat", "mcts", "auto"}:
        raise ValueError("champion_mode must be 'flat', 'mcts' or 'auto'")
    if champion_mode == "auto" and population_games < 1:
        raise ValueError(
            "champion_mode='auto' requires positive population_games for MCTS absorption"
        )
    if champion_mode == "auto" and mcts_probe_games < 2:
        raise ValueError(
            "champion_mode='auto' requires at least two MCTS probe games"
        )
    if flat_mcts_distill_games < 0:
        raise ValueError("flat_mcts_distill_games must not be negative")
    if flat_mcts_distill_games and not mcts_probe_games:
        raise ValueError(
            "flat MCTS distillation requires --mcts-probe-games so an "
            "untrusted search policy cannot teach the raw model"
        )
    if flat_mcts_distill_games and flat_mcts_distill_updates_per_game < 1:
        raise ValueError(
            "flat MCTS distillation requires positive updates per game"
        )
    if flat_mcts_distill_simulations < 1:
        raise ValueError("flat_mcts_distill_simulations must be positive")
    if flat_mcts_distill_max_envs < 1:
        raise ValueError("flat_mcts_distill_max_envs must be positive")
    if flat_mcts_distill_max_moves < 1:
        raise ValueError("flat_mcts_distill_max_moves must be positive")
    if not 0.0 <= flat_mcts_distill_critical_fraction <= 1.0:
        raise ValueError(
            "flat_mcts_distill_critical_fraction must be between zero and one"
        )
    if flat_mcts_distill_critical_minimum_score < 0.0:
        raise ValueError(
            "flat_mcts_distill_critical_minimum_score must be non-negative"
        )
    if not 0.0 <= flat_mcts_distill_critical_replay_fraction <= 1.0:
        raise ValueError(
            "flat_mcts_distill_critical_replay_fraction must be between zero and one"
        )
    if (
        flat_mcts_distill_gate_games < 2
        or flat_mcts_distill_gate_games % 2
    ):
        raise ValueError("flat_mcts_distill_gate_games must be positive and even")
    if num_envs < 2:
        raise ValueError("num_envs must be at least two for paired SelfPlayPool gates")
    if not 0 <= visible_action_history <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    if steps < 0 or updates < 0:
        raise ValueError("same-network MCTS steps and updates must not be negative")
    if steps == 0 and updates:
        raise ValueError("updates require a positive same-network MCTS step count")
    if direct_gate_games < 2 or direct_gate_games % 2:
        raise ValueError("direct_gate_games must be positive and even")
    if direct_population_gate_games < 2 or direct_population_gate_games % 2:
        raise ValueError(
            "direct_population_gate_games must be positive and even"
        )
    if (
        direct_population_diagnostic_gate_games < 2
        or direct_population_diagnostic_gate_games % 2
        or direct_population_diagnostic_gate_games
        > direct_population_gate_games
    ):
        raise ValueError(
            "direct_population_diagnostic_gate_games must be even, positive and no larger than the full gate"
        )
    if direct_gate_max_moves < 1:
        raise ValueError("direct_gate_max_moves must be positive")
    if mcts_gate_games < 2 or mcts_gate_games % 2:
        raise ValueError("mcts_gate_games must be positive and even")
    if mcts_probe_games < 0 or mcts_probe_games % 2:
        raise ValueError("mcts_probe_games must be non-negative and even")
    if mcts_probe_games and mcts_probe_games < 2:
        raise ValueError("mcts_probe_games must be zero or at least two")
    if mcts_probe_simulations < 1:
        raise ValueError("mcts_probe_simulations must be positive")
    if mcts_probe_patience < 1:
        raise ValueError("mcts_probe_patience must be positive")
    for name, value in {
        "direct_gate_score": direct_gate_score,
        "direct_policy_anchor_weight": direct_policy_anchor_weight,
        "direct_value_anchor_weight": direct_value_anchor_weight,
        "direct_teacher_weight": direct_teacher_weight,
        "direct_shaping_weight": direct_shaping_weight,
        "direct_objective_teacher_probability": (
            direct_objective_teacher_probability
        ),
        "direct_readiness_score": direct_readiness_score,
        "direct_readiness_completion_rate": direct_readiness_completion_rate,
        "direct_population_gate_score": direct_population_gate_score,
        "direct_population_historical_floor": (
            direct_population_historical_floor
        ),
        "direct_population_opponent_policy_weight": (
            direct_population_opponent_policy_weight
        ),
        "direct_population_teacher_weight": direct_population_teacher_weight,
        "direct_population_shaping_weight": direct_population_shaping_weight,
        "direct_population_shaping_final_weight": (
            direct_population_shaping_final_weight
        ),
        "direct_population_ppo_clip_range": (
            direct_population_ppo_clip_range
        ),
        "direct_population_ppo_gamma": direct_population_ppo_gamma,
        "direct_population_ppo_gae_lambda": (
            direct_population_ppo_gae_lambda
        ),
        "direct_population_ppo_terminal_advantage_weight": (
            direct_population_ppo_terminal_advantage_weight
        ),
        "direct_population_ppo_top_advantage_fraction": (
            direct_population_ppo_top_advantage_fraction
        ),
        "direct_population_objective_replay_weight": (
            direct_population_objective_replay_weight
        ),
        "direct_population_objective_replay_recency": (
            direct_population_objective_replay_recency
        ),
        "mcts_gate_score": mcts_gate_score,
        "mcts_probe_score": mcts_probe_score,
        "champion_value_anchor_weight": champion_value_anchor_weight,
        "flat_mcts_distill_policy_anchor_weight": (
            flat_mcts_distill_policy_anchor_weight
        ),
        "flat_mcts_distill_value_anchor_weight": (
            flat_mcts_distill_value_anchor_weight
        ),
        "flat_mcts_distill_critical_fraction": (
            flat_mcts_distill_critical_fraction
        ),
        "flat_mcts_distill_critical_replay_fraction": (
            flat_mcts_distill_critical_replay_fraction
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if not 0.0 <= mcts_probe_minimum_improvement <= 1.0:
        raise ValueError(
            "mcts_probe_minimum_improvement must be between zero and one"
        )
    if direct_readiness_city_captures_per_game < 0.0:
        raise ValueError(
            "direct_readiness_city_captures_per_game must not be negative"
        )
    if (
        direct_population_ppo_rollout_steps < 1
        or direct_population_ppo_epochs < 1
        or direct_population_ppo_minibatch_size < 1
        or direct_population_ppo_max_pending_examples
        < direct_population_ppo_minibatch_size
    ):
        raise ValueError("direct population PPO sizes must be positive")
    if not 0.0 < direct_population_ppo_target_kl <= (
        direct_population_ppo_hard_kl
    ):
        raise ValueError("direct population PPO requires 0 < target KL <= hard KL")
    if not 0.0 < direct_population_ppo_gamma <= 1.0:
        raise ValueError("direct population PPO gamma must be in (0, 1]")
    if not 0.0 <= direct_population_ppo_terminal_advantage_weight <= 1.0:
        raise ValueError(
            "direct population PPO terminal advantage weight must be between zero and one"
        )
    if not 0.0 < direct_population_ppo_top_advantage_fraction <= 1.0:
        raise ValueError(
            "direct population PPO top advantage fraction must be in (0, 1]"
        )
    if not 0.0 <= direct_population_shaping_final_weight <= 1.0:
        raise ValueError(
            "direct population final shaping weight must be between zero and one"
        )
    if not 0.1 <= direct_population_ppo_rollout_temperature <= 2.0:
        raise ValueError("direct population PPO rollout temperature is invalid")
    if direct_population_plateau_patience < 1:
        raise ValueError("direct population plateau patience must be positive")
    if direct_population_plateau_temperature_increment < 0.0:
        raise ValueError(
            "direct population plateau temperature increment must not be negative"
        )
    if not (
        direct_population_ppo_rollout_temperature
        <= direct_population_plateau_max_temperature
        <= 2.0
    ):
        raise ValueError("direct population plateau maximum temperature is invalid")
    if direct_population_ppo_min_learning_rate <= 0.0:
        raise ValueError("direct population PPO minimum learning rate must be positive")
    if not 0.0 < direct_population_ppo_lr_backoff < 1.0:
        raise ValueError("direct population PPO learning-rate backoff is invalid")
    for name, value in {
        "direct_population_objective_replay_horizon": (
            direct_population_objective_replay_horizon
        ),
        "direct_population_objective_replay_capacity": (
            direct_population_objective_replay_capacity
        ),
        "direct_population_objective_replay_batch_size": (
            direct_population_objective_replay_batch_size
        ),
        "direct_population_objective_replay_interval": (
            direct_population_objective_replay_interval
        ),
        "direct_population_objective_replay_max_epochs": (
            direct_population_objective_replay_max_epochs
        ),
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if (
        direct_population_objective_replay_capacity
        < direct_population_objective_replay_batch_size
    ):
        raise ValueError(
            "direct population objective replay capacity must be at least "
            "its batch size"
        )
    for name, value in {
        "direct_population_ppg_interval": direct_population_ppg_interval,
        "direct_population_ppg_buffer_capacity": (
            direct_population_ppg_buffer_capacity
        ),
        "direct_population_ppg_samples_per_rollout": (
            direct_population_ppg_samples_per_rollout
        ),
        "direct_population_ppg_phase_samples": (
            direct_population_ppg_phase_samples
        ),
        "direct_population_ppg_epochs": direct_population_ppg_epochs,
        "direct_population_ppg_minibatch_size": (
            direct_population_ppg_minibatch_size
        ),
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if direct_population_ppg_enabled and (
        direct_population_ppg_buffer_capacity
        < direct_population_ppg_phase_samples
        or direct_population_ppg_phase_samples
        < direct_population_ppg_minibatch_size
        or direct_population_ppg_buffer_capacity
        < direct_population_ppg_samples_per_rollout
    ):
        raise ValueError("invalid direct population PPG buffer sizes")
    if (
        direct_population_ppg_policy_kl_weight < 0.0
        or direct_population_ppg_value_weight < 0.0
    ):
        raise ValueError("direct population PPG weights must not be negative")
    if direct_population_ema_enabled and not 0.0 <= (
        direct_population_ema_decay
    ) < 1.0:
        raise ValueError("direct population EMA decay must be in [0, 1)")
    for name, value in {
        "exploiter_champion_probability": exploiter_champion_probability,
        "exploiter_policy_anchor_weight": exploiter_policy_anchor_weight,
        "exploiter_value_anchor_weight": exploiter_value_anchor_weight,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if not 0.0 < exploiter_ppo_target_kl <= exploiter_ppo_hard_kl:
        raise ValueError("exploiter PPO requires 0 < target KL <= hard KL")
    if not 0.1 <= exploiter_ppo_rollout_temperature <= 2.0:
        raise ValueError("exploiter PPO rollout temperature is invalid")
    if direct_objective_steps < 0 or post_bootstrap_objective_steps < 0:
        raise ValueError("objective curriculum steps must not be negative")
    if (direct_objective_steps or post_bootstrap_objective_steps) and (
        direct_objective_envs < 1
        or direct_objective_block_steps < 1
        or direct_objective_rollout_horizon < 2
        or direct_objective_updates_per_rollout < 1
    ):
        raise ValueError(
            "objective curriculum requires positive envs/blocks/updates and horizon >= 2"
        )
    if direct_actor_max_steps < direct_actor_steps:
        raise ValueError(
            "direct_actor_max_steps must not be smaller than direct_actor_steps"
        )
    if direct_readiness_patience < 1:
        raise ValueError("direct_readiness_patience must be positive")
    if direct_regression_patience < 1:
        raise ValueError("direct_regression_patience must be positive")
    if direct_population_steps < 0:
        raise ValueError("direct_population_steps must not be negative")
    if direct_population_max_steps < direct_population_steps:
        raise ValueError(
            "direct_population_max_steps must not be smaller than its minimum"
        )
    if direct_population_branch_count < 1:
        raise ValueError("direct_population_branch_count must be positive")
    if direct_population_branch_count > 1:
        if direct_population_branch_steps < 1:
            raise ValueError("direct_population_branch_steps must be positive")
        if (
            direct_population_branch_gate_games < 2
            or direct_population_branch_gate_games % 2
        ):
            raise ValueError(
                "direct_population_branch_gate_games must be positive and even"
            )
    if not 0.0 <= direct_population_branch_min_champion_score <= 1.0:
        raise ValueError(
            "direct_population_branch_min_champion_score must be between zero and one"
        )
    if post_bootstrap_population_steps < 0:
        raise ValueError("post_bootstrap_population_steps must not be negative")
    if post_bootstrap_population_max_steps < post_bootstrap_population_steps:
        raise ValueError(
            "post_bootstrap_population_max_steps must not be smaller than its minimum"
        )
    for name, value in {
        "post_bootstrap_champion_probability": (
            post_bootstrap_champion_probability
        ),
        "post_bootstrap_policy_anchor_weight": (
            post_bootstrap_policy_anchor_weight
        ),
        "post_bootstrap_teacher_weight": post_bootstrap_teacher_weight,
        "post_bootstrap_opponent_policy_weight": (
            post_bootstrap_opponent_policy_weight
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if (direct_population_steps or post_bootstrap_population_steps) and (
        direct_population_envs < 2 or direct_population_block_steps < 1
    ):
        raise ValueError(
            "direct population envs and block steps must be positive"
        )
    if direct_population_readiness_patience < 1:
        raise ValueError(
            "direct_population_readiness_patience must be positive"
        )
    if direct_population_regression_patience < 1:
        raise ValueError(
            "direct_population_regression_patience must be positive"
        )
    ladder = tuple(int(value) for value in mcts_simulation_ladder)
    if any(value < 1 for value in ladder) or tuple(sorted(set(ladder))) != ladder:
        raise ValueError("mcts_simulation_ladder must be strictly increasing")
    population_rungs = ladder or ((int(mcts.simulations),) if population_games else ())
    population_game_allocations = tuple(
        population_games // max(1, len(population_rungs))
        + int(index < population_games % max(1, len(population_rungs)))
        for index in range(len(population_rungs))
    )
    # Flat-only runs without a readiness probe do not need the native routed
    # MCTS binding at all.
    if (
        champion_mode in {"mcts", "auto"}
        or mcts_probe_games
        or steps
        or population_games
        or flat_mcts_distill_games
    ):
        require_routed_self_play_pool()
    if not champion_in.is_file():
        raise FileNotFoundError(f"champion checkpoint does not exist: {champion_in}")
    if challenger_in is not None and not challenger_in.is_file():
        raise FileNotFoundError(
            f"challenger checkpoint does not exist: {challenger_in}"
        )
    for path in population_model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"population checkpoint does not exist: {path}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    population_mcts_reanalyse_path = (
        population_mcts_reanalyse_path
        if population_mcts_reanalyse_path is not None
        else checkpoint_dir / "mcts-reanalyse.pkl"
    )
    database_path = database_path or checkpoint_dir / "league.sqlite3"
    resolved_evaluation_seed = _offset_seed(evaluation_seed, 0)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "league")
    evaluation_mcts = replace(
        mcts,
        temperature=0.0,
        strategic_prior_weight=0.0,
        teacher_prior_weight=0.0,
    )
    league_config: Dict[str, Any] = {
        "training": training,
        "mcts": mcts,
        "champion_in": champion_in,
        "challenger_in": challenger_in,
        "checkpoint_dir": checkpoint_dir,
        "database_path": database_path,
        "population_model_paths": list(population_model_paths),
        "rounds": rounds,
        "promotion_games": promotion_games,
        "historical_games": historical_games,
        "promotion_score": promotion_score,
        "promotion_lower_bound": promotion_lower_bound,
        "population_score": population_score,
        "historical_score_floor": historical_score_floor,
        "evaluation_seed": resolved_evaluation_seed,
        "evaluation_seed_stride": evaluation_seed_stride,
        "round_seed_stride": round_seed_stride,
        "population_size": population_size,
        "population_games": population_games,
        "population_updates_per_game": population_updates_per_game,
        "population_workers": population_workers,
        "population_max_envs": population_max_envs,
        "population_offline_collection": population_offline_collection,
        "population_mcts_critical_only": population_mcts_critical_only,
        "population_mcts_critical_fraction": population_mcts_critical_fraction,
        "population_mcts_critical_minimum_score": (
            population_mcts_critical_minimum_score
        ),
        "population_mcts_critical_replay_fraction": (
            population_mcts_critical_replay_fraction
        ),
        "population_mcts_reanalyse_path": population_mcts_reanalyse_path,
        "population_mcts_reanalyse_capacity": population_mcts_reanalyse_capacity,
        "population_mcts_reanalyse_positions": population_mcts_reanalyse_positions,
        "population_mcts_reanalyse_replay_fraction": (
            population_mcts_reanalyse_replay_fraction
        ),
        "pfsp_alpha": pfsp_alpha,
        "champion_probability": champion_probability,
        "minimum_sampling_probability": minimum_sampling_probability,
        "champion_anchor_weight": champion_anchor_weight,
        "champion_value_anchor_weight": champion_value_anchor_weight,
        "elo_k": elo_k,
        "checkpoint_prefix": checkpoint_prefix,
        "num_envs": num_envs,
        "steps": steps,
        "updates": updates,
        "num_threads": num_threads,
        "max_pending_leaves": max_pending_leaves,
        "visible_action_history": visible_action_history,
        "belief_retries": belief_retries,
        "direct_teacher_envs": direct_teacher_envs,
        "direct_teacher_steps": direct_teacher_steps,
        "direct_objective_envs": direct_objective_envs,
        "direct_objective_steps": direct_objective_steps,
        "direct_objective_block_steps": direct_objective_block_steps,
        "direct_objective_rollout_horizon": direct_objective_rollout_horizon,
        "direct_objective_updates_per_rollout": (
            direct_objective_updates_per_rollout
        ),
        "direct_objective_teacher_probability": (
            direct_objective_teacher_probability
        ),
        "direct_actor_envs": direct_actor_envs,
        "direct_actor_steps": direct_actor_steps,
        "direct_actor_block_steps": direct_actor_block_steps,
        "direct_actor_max_steps": direct_actor_max_steps,
        "direct_gate_games": direct_gate_games,
        "direct_gate_max_moves": direct_gate_max_moves,
        "direct_gate_score": direct_gate_score,
        "direct_policy_anchor_weight": direct_policy_anchor_weight,
        "direct_value_anchor_weight": direct_value_anchor_weight,
        "direct_teacher_weight": direct_teacher_weight,
        "direct_shaping_weight": direct_shaping_weight,
        "direct_readiness_score": direct_readiness_score,
        "direct_readiness_city_captures_per_game": (
            direct_readiness_city_captures_per_game
        ),
        "direct_readiness_completion_rate": direct_readiness_completion_rate,
        "direct_readiness_patience": direct_readiness_patience,
        "direct_regression_patience": direct_regression_patience,
        "direct_population_envs": direct_population_envs,
        "direct_population_steps": direct_population_steps,
        "direct_population_max_steps": direct_population_max_steps,
        "direct_population_block_steps": direct_population_block_steps,
        "direct_population_branch_count": direct_population_branch_count,
        "direct_population_branch_steps": direct_population_branch_steps,
        "direct_population_branch_gate_games": direct_population_branch_gate_games,
        "direct_population_branch_min_champion_score": (
            direct_population_branch_min_champion_score
        ),
        "direct_population_gate_games": direct_population_gate_games,
        "direct_population_diagnostic_gate_games": (
            direct_population_diagnostic_gate_games
        ),
        "direct_population_gate_score": direct_population_gate_score,
        "direct_population_historical_floor": (
            direct_population_historical_floor
        ),
        "direct_population_readiness_patience": (
            direct_population_readiness_patience
        ),
        "direct_population_regression_patience": (
            direct_population_regression_patience
        ),
        "direct_population_opponent_policy_weight": (
            direct_population_opponent_policy_weight
        ),
        "direct_population_teacher_weight": direct_population_teacher_weight,
        "direct_population_shaping_weight": direct_population_shaping_weight,
        "direct_population_ppo_rollout_steps": (
            direct_population_ppo_rollout_steps
        ),
        "direct_population_ppo_epochs": direct_population_ppo_epochs,
        "direct_population_ppo_minibatch_size": (
            direct_population_ppo_minibatch_size
        ),
        "direct_population_ppo_clip_range": (
            direct_population_ppo_clip_range
        ),
        "direct_population_ppo_gamma": direct_population_ppo_gamma,
        "direct_population_ppo_gae_lambda": (
            direct_population_ppo_gae_lambda
        ),
        "direct_population_ppo_terminal_advantage_weight": (
            direct_population_ppo_terminal_advantage_weight
        ),
        "direct_population_ppo_top_advantage_fraction": (
            direct_population_ppo_top_advantage_fraction
        ),
        "direct_population_ppo_target_kl": (
            direct_population_ppo_target_kl
        ),
        "direct_population_ppo_hard_kl": (
            direct_population_ppo_hard_kl
        ),
        "direct_population_ppo_rollout_temperature": (
            direct_population_ppo_rollout_temperature
        ),
        "direct_population_plateau_patience": direct_population_plateau_patience,
        "direct_population_plateau_temperature_increment": (
            direct_population_plateau_temperature_increment
        ),
        "direct_population_plateau_max_temperature": (
            direct_population_plateau_max_temperature
        ),
        "direct_population_ppo_max_pending_examples": (
            direct_population_ppo_max_pending_examples
        ),
        "direct_population_ppo_min_learning_rate": (
            direct_population_ppo_min_learning_rate
        ),
        "direct_population_ppo_lr_backoff": (
            direct_population_ppo_lr_backoff
        ),
        "direct_population_objective_replay_horizon": (
            direct_population_objective_replay_horizon
        ),
        "direct_population_objective_replay_capacity": (
            direct_population_objective_replay_capacity
        ),
        "direct_population_objective_replay_batch_size": (
            direct_population_objective_replay_batch_size
        ),
        "direct_population_objective_replay_interval": (
            direct_population_objective_replay_interval
        ),
        "direct_population_objective_replay_weight": (
            direct_population_objective_replay_weight
        ),
        "direct_population_objective_replay_recency": (
            direct_population_objective_replay_recency
        ),
        "direct_population_objective_replay_max_epochs": (
            direct_population_objective_replay_max_epochs
        ),
        "direct_population_ppg_enabled": direct_population_ppg_enabled,
        "direct_population_ppg_interval": direct_population_ppg_interval,
        "direct_population_ppg_buffer_capacity": (
            direct_population_ppg_buffer_capacity
        ),
        "direct_population_ppg_samples_per_rollout": (
            direct_population_ppg_samples_per_rollout
        ),
        "direct_population_ppg_phase_samples": (
            direct_population_ppg_phase_samples
        ),
        "direct_population_ppg_epochs": direct_population_ppg_epochs,
        "direct_population_ppg_minibatch_size": (
            direct_population_ppg_minibatch_size
        ),
        "direct_population_ppg_policy_kl_weight": (
            direct_population_ppg_policy_kl_weight
        ),
        "direct_population_ppg_value_weight": (
            direct_population_ppg_value_weight
        ),
        "direct_population_ema_enabled": direct_population_ema_enabled,
        "direct_population_ema_decay": direct_population_ema_decay,
        "breakthrough_exploiter": breakthrough_exploiter,
        "exploiter_champion_probability": exploiter_champion_probability,
        "exploiter_policy_anchor_weight": exploiter_policy_anchor_weight,
        "exploiter_value_anchor_weight": exploiter_value_anchor_weight,
        "exploiter_ppo_target_kl": exploiter_ppo_target_kl,
        "exploiter_ppo_hard_kl": exploiter_ppo_hard_kl,
        "exploiter_ppo_rollout_temperature": (
            exploiter_ppo_rollout_temperature
        ),
        "post_bootstrap_population_steps": post_bootstrap_population_steps,
        "post_bootstrap_population_max_steps": (
            post_bootstrap_population_max_steps
        ),
        "post_bootstrap_objective_steps": post_bootstrap_objective_steps,
        "post_bootstrap_champion_probability": (
            post_bootstrap_champion_probability
        ),
        "post_bootstrap_policy_anchor_weight": (
            post_bootstrap_policy_anchor_weight
        ),
        "post_bootstrap_teacher_weight": post_bootstrap_teacher_weight,
        "post_bootstrap_opponent_policy_weight": (
            post_bootstrap_opponent_policy_weight
        ),
        "champion_mode": champion_mode,
        "flat_mcts_distill_games": flat_mcts_distill_games,
        "flat_mcts_distill_simulations": flat_mcts_distill_simulations,
        "flat_mcts_distill_max_envs": flat_mcts_distill_max_envs,
        "flat_mcts_distill_max_moves": flat_mcts_distill_max_moves,
        "flat_mcts_distill_updates_per_game": (
            flat_mcts_distill_updates_per_game
        ),
        "flat_mcts_distill_gate_games": flat_mcts_distill_gate_games,
        "flat_mcts_distill_policy_anchor_weight": (
            flat_mcts_distill_policy_anchor_weight
        ),
        "flat_mcts_distill_value_anchor_weight": (
            flat_mcts_distill_value_anchor_weight
        ),
        "flat_mcts_distill_critical_fraction": (
            flat_mcts_distill_critical_fraction
        ),
        "flat_mcts_distill_critical_minimum_score": (
            flat_mcts_distill_critical_minimum_score
        ),
        "flat_mcts_distill_critical_replay_fraction": (
            flat_mcts_distill_critical_replay_fraction
        ),
        "mcts_probe_games": mcts_probe_games,
        "mcts_probe_simulations": mcts_probe_simulations,
        "mcts_probe_score": mcts_probe_score,
        "mcts_probe_minimum_improvement": (
            mcts_probe_minimum_improvement
        ),
        "mcts_probe_patience": mcts_probe_patience,
        "mcts_simulation_ladder": list(ladder),
        "population_game_allocations": list(population_game_allocations),
        "mcts_gate_games": mcts_gate_games,
        "mcts_gate_score": mcts_gate_score,
    }
    run.write_config(**league_config)

    champion_path = champion_in
    challenger_path = challenger_in
    challenger_id: int | None = None
    promotions = 0
    last_candidate_score = 0.0
    last_population_score = 0.0
    last_training: Dict[str, Any] = {}
    current_round_id: list[int | None] = [None]

    with LeagueDatabase(database_path) as database:
        database_run_id = database.start_run(run.directory, league_config)

        def metric_sink(stage_path: Path, row: Mapping[str, Any]) -> None:
            try:
                relative_stage = stage_path.relative_to(run.directory)
                stage = "league" if relative_stage == Path(".") else str(relative_stage)
            except ValueError:
                stage = str(stage_path)
            database.record_metric_row(database_run_id, current_round_id[0], stage, row)

        run.set_metric_sink(metric_sink)
        try:
            initial_model, initial_config = load_model(champion_path, device)
            champion_id = database.register_checkpoint(
                champion_path,
                initial_config,
                run_id=database_run_id,
                status="initial",
                protected=True,
            )
            del initial_model
            if challenger_path is not None:
                challenger_model, challenger_config = load_model(
                    challenger_path, device
                )
                if challenger_config != initial_config:
                    raise RuntimeError(
                        "challenger checkpoint has a different ModelConfig"
                    )
                challenger_id = database.register_checkpoint(
                    challenger_path,
                    challenger_config,
                    run_id=database_run_id,
                    parent_model_id=champion_id,
                    status="challenger",
                    protected=False,
                    in_population=False,
                )
                if challenger_id == champion_id:
                    challenger_path = None
                    challenger_id = None
                del challenger_model
            for population_path in population_model_paths:
                population_model, population_config = load_model(population_path, device)
                if population_config != initial_config:
                    raise RuntimeError(
                        f"population checkpoint has a different ModelConfig: {population_path}"
                    )
                database.register_checkpoint(
                    population_path,
                    population_config,
                    run_id=database_run_id,
                    status="initial",
                    protected=True,
                    protect_existing=True,
                )
                del population_model
        except BaseException as error:
            database.finish_run(
                database_run_id,
                "failed",
                {"error": repr(error), "champion_path": str(champion_path)},
            )
            raise

        try:
            for round_index in range(1, rounds + 1):
                round_training = replace(
                    training,
                    seed=_offset_seed(training.seed, (round_index - 1) * round_seed_stride),
                )
                round_evaluation_seed = _offset_seed(
                    resolved_evaluation_seed,
                    (round_index - 1) * evaluation_seed_stride,
                )
                round_id = database.start_round(
                    database_run_id,
                    round_index,
                    champion_id,
                    round_training.seed,
                    round_evaluation_seed,
                )
                current_round_id[0] = round_id
                candidate_id: int | None = None
                round_training_summary: Dict[str, Any] = {}
                try:
                    print(
                        f"league round {round_index}/{rounds} | champion: {champion_path} | "
                        f"collection seed: {round_training.seed}",
                        flush=True,
                    )
                    previous_challenger_path = challenger_path
                    previous_challenger_id = challenger_id
                    champion_capability = database.model_capability(champion_id)
                    round_champion_mode = _round_champion_mode(
                        champion_mode, champion_capability
                    )
                    round_training_summary["champion_mode"] = {
                        "requested": champion_mode,
                        "effective": round_champion_mode,
                        "mcts_ready": float(
                            bool(
                                champion_capability
                                and champion_capability.get("mcts_ready")
                            )
                        ),
                    }
                    if champion_mode == "auto":
                        print(
                            "league auto mode | "
                            f"effective={round_champion_mode} | "
                            f"mcts_ready={int(round_training_summary['champion_mode']['mcts_ready'])}",
                            flush=True,
                        )
                    # Independent branch selection deliberately starts every
                    # round from the frozen champion.  A merely safe
                    # challenger (for example 0.46 against the champion) is
                    # not allowed to drift further away for the next round.
                    fresh_branch_round = (
                        direct_population_branch_count > 1
                        or round_champion_mode == "mcts"
                    )
                    candidate_source = (
                        champion_path
                        if fresh_branch_round
                        else previous_challenger_path or champion_path
                    )
                    candidate_parent_id = (
                        champion_id
                        if fresh_branch_round
                        else previous_challenger_id or champion_id
                    )
                    candidate_source_kind = (
                        "challenger"
                        if (
                            previous_challenger_path is not None
                            and not fresh_branch_round
                        )
                        else "champion"
                    )
                    candidate_evaluation_label = (
                        f"pretendent {candidate_source.stem}"
                    )
                    print(
                        "league candidate source | "
                        f"{candidate_source_kind}: {candidate_source}",
                        flush=True,
                    )
                    candidate, model_config = load_model(candidate_source, device)
                    champion, champion_config = load_model(champion_path, device)
                    if champion_config != model_config:
                        raise RuntimeError("candidate and champion ModelConfig differ")
                    candidate_id = database.create_candidate(
                        database_run_id,
                        candidate_parent_id,
                        f"candidate-round-{round_index:03d}",
                        model_config,
                    )
                    database.attach_candidate(round_id, candidate_id)

                    population_rows = database.select_population(champion_id, population_size)
                    population_ids = [int(row["id"]) for row in population_rows]
                    population_paths = [Path(str(row["checkpoint_path"])) for row in population_rows]
                    population_evaluation_labels = [
                        path.stem for path in population_paths
                    ]
                    probabilities, reference_scores = database.opponent_probabilities(
                        champion_id,
                        population_ids,
                        pfsp_alpha=pfsp_alpha,
                        champion_probability=champion_probability,
                        probability_floor=minimum_sampling_probability,
                    )
                    champion_record = database.model_record(champion_id)
                    controls = _curriculum_controls(
                        champion_status=str(champion_record["status"]),
                        direct_teacher_steps=direct_teacher_steps,
                        direct_actor_steps=direct_actor_steps,
                        direct_actor_max_steps=direct_actor_max_steps,
                        direct_population_steps=direct_population_steps,
                        direct_population_max_steps=direct_population_max_steps,
                        direct_policy_anchor_weight=direct_policy_anchor_weight,
                        direct_population_teacher_weight=(
                            direct_population_teacher_weight
                        ),
                        direct_population_opponent_policy_weight=(
                            direct_population_opponent_policy_weight
                        ),
                        post_bootstrap_population_steps=(
                            post_bootstrap_population_steps
                        ),
                        post_bootstrap_population_max_steps=(
                            post_bootstrap_population_max_steps
                        ),
                        post_bootstrap_policy_anchor_weight=(
                            post_bootstrap_policy_anchor_weight
                        ),
                        post_bootstrap_teacher_weight=(
                            post_bootstrap_teacher_weight
                        ),
                        post_bootstrap_opponent_policy_weight=(
                            post_bootstrap_opponent_policy_weight
                        ),
                        direct_objective_steps=direct_objective_steps,
                        post_bootstrap_objective_steps=(
                            post_bootstrap_objective_steps
                        ),
                        refresh_promoted_curriculum=(
                            refresh_promoted_curriculum
                            and not (
                                champion_mode == "auto"
                                and round_champion_mode == "mcts"
                            )
                        ),
                    )
                    if champion_mode == "auto" and round_champion_mode == "mcts":
                        # The preceding flat rounds already established the
                        # raw backbone and MCTS capability.  Start the next
                        # generation from a frozen champion immediately;
                        # repeating PPO here would change the collector before
                        # it has produced one fixed MCTS replay generation.
                        controls = {
                            **controls,
                            "phase": "mcts_absorption",
                            "teacher_steps": 0,
                            "objective_steps": 0,
                            "actor_steps": 0,
                            "actor_max_steps": 0,
                            "population_steps": 0,
                            "population_max_steps": 0,
                            "population_teacher_weight": 0.0,
                            "population_opponent_policy_weight": 0.0,
                        }
                    post_bootstrap = bool(controls["post_bootstrap"])
                    if post_bootstrap:
                        probabilities = _champion_focused_probabilities(
                            population_ids,
                            probabilities,
                            champion_id,
                            champion_probability=(
                                post_bootstrap_champion_probability
                            ),
                            probability_floor=minimum_sampling_probability,
                        )
                    effective_teacher_steps = int(controls["teacher_steps"])
                    effective_objective_steps = int(controls["objective_steps"])
                    effective_actor_steps = int(controls["actor_steps"])
                    effective_actor_max_steps = int(
                        controls["actor_max_steps"]
                    )
                    effective_population_steps = int(
                        controls["population_steps"]
                    )
                    effective_population_max_steps = int(
                        controls["population_max_steps"]
                    )
                    effective_policy_anchor_weight = float(
                        controls["policy_anchor_weight"]
                    )
                    effective_population_teacher_weight = float(
                        controls["population_teacher_weight"]
                    )
                    effective_population_opponent_policy_weight = float(
                        controls["population_opponent_policy_weight"]
                    )
                    database.record_opponent_selection(
                        round_id, population_ids, probabilities, reference_scores
                    )
                    curriculum_phase = str(controls["phase"])
                    round_training_summary["curriculum"] = {
                        "phase": curriculum_phase,
                        "teacher_steps": float(effective_teacher_steps),
                        "objective_steps": float(effective_objective_steps),
                        "actor_steps": float(effective_actor_steps),
                        "population_min_steps": float(
                            effective_population_steps
                        ),
                        "population_max_steps": float(
                            effective_population_max_steps
                        ),
                        "policy_anchor_weight": float(
                            effective_policy_anchor_weight
                        ),
                        "population_teacher_weight": float(
                            effective_population_teacher_weight
                        ),
                        "population_opponent_policy_weight": float(
                            effective_population_opponent_policy_weight
                        ),
                    }
                    population_training_seed = _offset_seed(
                        round_training.seed, 5_000_003
                    )
                    direct_population_training_seed = _offset_seed(
                        round_training.seed, 2_500_009
                    )
                    flat_mcts_distill_training_seed = _offset_seed(
                        round_training.seed, 7_500_029
                    )
                    evaluation_map_seeds: set[int] = set()
                    for opponent_index, opponent_id in enumerate(population_ids):
                        games_for_opponent = (
                            promotion_games if opponent_id == champion_id else historical_games
                        )
                        match_seed = _offset_seed(
                            round_evaluation_seed, opponent_index * 1_000_003
                        )
                        opponent_map_seeds = {
                            match_seed + game_index // 2
                            for game_index in range(games_for_opponent)
                        }
                        if evaluation_map_seeds & opponent_map_seeds:
                            raise ValueError("held-out evaluation schedules overlap")
                        evaluation_map_seeds.update(opponent_map_seeds)
                    direct_gate_seed = _offset_seed(
                        round_evaluation_seed, 300_000_007
                    )
                    evaluation_map_seeds.update(
                        direct_gate_seed + game_index // 2
                        for game_index in range(direct_gate_games)
                    )
                    direct_population_gate_seed = _offset_seed(
                        round_evaluation_seed, 350_000_009
                    )
                    for opponent_index in range(len(population_ids)):
                        opponent_gate_seed = _offset_seed(
                            direct_population_gate_seed,
                            opponent_index * 1_000_003,
                        )
                        evaluation_map_seeds.update(
                            opponent_gate_seed + game_index // 2
                            for game_index in range(
                                direct_population_gate_games
                            )
                        )
                    mcts_probe_seed = _offset_seed(
                        round_evaluation_seed, 375_000_011
                    )
                    evaluation_map_seeds.update(
                        mcts_probe_seed + game_index // 2
                        for game_index in range(mcts_probe_games)
                    )
                    flat_mcts_distill_gate_seed = _offset_seed(
                        round_evaluation_seed, 425_000_009
                    )
                    evaluation_map_seeds.update(
                        flat_mcts_distill_gate_seed + game_index // 2
                        for game_index in range(flat_mcts_distill_gate_games)
                    )
                    # Reuse one development arena across search depths so a
                    # later rung cannot silently erase a score earned by an
                    # earlier rung. The final promotion arena remains separate.
                    mcts_gate_seeds = [
                        _offset_seed(round_evaluation_seed, 400_000_007)
                    ] * len(population_rungs)
                    for seed in mcts_gate_seeds:
                        evaluation_map_seeds.update(
                            seed + game_index // 2
                            for game_index in range(mcts_gate_games)
                        )
                    known_training_map_seeds: set[int] = set()
                    for rung_index, rung_games in enumerate(
                        population_game_allocations
                    ):
                        rung_seed = _offset_seed(
                            population_training_seed,
                            rung_index * 2_000_003,
                        )
                        known_training_map_seeds.update(
                            rung_seed + game_index
                            for game_index in range(rung_games)
                        )
                    known_training_map_seeds.update(
                        flat_mcts_distill_training_seed + game_index
                        for game_index in range(flat_mcts_distill_games)
                    )
                    if evaluation_map_seeds & known_training_map_seeds:
                        raise ValueError(
                            "population training maps overlap held-out evaluation maps"
                        )
                    if round_training.seed in evaluation_map_seeds:
                        raise ValueError(
                            "SelfPlayPool collection seed overlaps held-out evaluation maps"
                        )
                    if direct_population_training_seed in evaluation_map_seeds:
                        raise ValueError(
                            "direct population seed overlaps held-out evaluation maps"
                        )
                    print(
                        "league population | "
                        + ", ".join(
                            f"{path.stem}: {probability:.1%}"
                            for path, probability in zip(population_paths, probabilities)
                        ),
                        flush=True,
                    )
                    if curriculum_phase == "champion_population":
                        print(
                            "league curriculum | champion-population | "
                            "teacher/actor skipped | "
                            f"objectives={effective_objective_steps} | "
                            f"population={effective_population_steps}-"
                            f"{effective_population_max_steps} steps",
                            flush=True,
                        )
                    elif curriculum_phase == "promoted_recovery":
                        print(
                            "league curriculum | promoted recovery | "
                            "teacher/objectives/actor restored before PPO | "
                            f"population={effective_population_steps}-"
                            f"{effective_population_max_steps} steps",
                            flush=True,
                        )
                    elif curriculum_phase == "mcts_absorption":
                        print(
                            "league curriculum | auto MCTS absorption | "
                            "frozen generator collection before optimisation",
                            flush=True,
                        )

                    direct_summary: Dict[str, Any] = {}
                    if (
                        effective_teacher_steps
                        or effective_objective_steps
                        or effective_actor_steps
                    ):
                        def record_direct_gate_game(
                            stage: str,
                            block: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            row = dict(game)
                            row["curriculum_stage"] = stage
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=f"direct_curriculum_gate_{stage}_{block}",
                                model_a_id=candidate_id,
                                model_b_id=champion_id,
                                game=row,
                            )

                        direct_summary = train_direct_policy_curriculum(
                            round_training,
                            model_config,
                            candidate,
                            champion,
                            teacher_envs=direct_teacher_envs,
                            teacher_steps=effective_teacher_steps,
                            objective_envs=direct_objective_envs,
                            objective_steps=effective_objective_steps,
                            objective_block_steps=direct_objective_block_steps,
                            objective_rollout_horizon=(
                                direct_objective_rollout_horizon
                            ),
                            objective_updates_per_rollout=(
                                direct_objective_updates_per_rollout
                            ),
                            objective_teacher_probability=(
                                min(
                                    direct_objective_teacher_probability,
                                    0.15,
                                )
                                if post_bootstrap
                                else direct_objective_teacher_probability
                            ),
                            actor_envs=direct_actor_envs,
                            actor_steps=effective_actor_steps,
                            actor_block_steps=direct_actor_block_steps,
                            actor_max_steps=effective_actor_max_steps,
                            num_threads=num_threads,
                            max_actions=mcts.native_max_actions,
                            gate_games=direct_gate_games,
                            gate_seed=direct_gate_seed,
                            gate_max_moves=direct_gate_max_moves,
                            gate_score=direct_gate_score,
                            policy_anchor_weight=effective_policy_anchor_weight,
                            value_anchor_weight=direct_value_anchor_weight,
                            teacher_weight=(
                                min(direct_teacher_weight, 0.10)
                                if post_bootstrap
                                else direct_teacher_weight
                            ),
                            shaping_weight=direct_shaping_weight,
                            readiness_score=direct_readiness_score,
                            readiness_city_captures_per_game=(
                                direct_readiness_city_captures_per_game
                            ),
                            readiness_completion_rate=(
                                direct_readiness_completion_rate
                            ),
                            readiness_patience=direct_readiness_patience,
                            regression_patience=direct_regression_patience,
                            gate_game_callback=record_direct_gate_game,
                            candidate_label=candidate_evaluation_label,
                            champion_label=champion_path.stem,
                            run=run.child(
                                f"round_{round_index:03d}/direct_curriculum"
                            ),
                        )
                    round_training_summary["direct"] = direct_summary

                    direct_enabled = bool(
                        effective_teacher_steps
                        or effective_objective_steps
                        or effective_actor_steps
                    )
                    direct_ready = (
                        not direct_enabled
                        or bool(direct_summary.get("direct_ready", 0.0))
                    )
                    if not direct_ready and round_champion_mode == "mcts":
                        # MCTS is deliberately unavailable until the raw
                        # network can expand and finish games on held-out maps.
                        # Preserve a non-regressing challenger so the next
                        # round continues the direct curriculum instead of
                        # restarting from the teacher.
                        best_direct_score = float(
                            direct_summary.get("direct_best_score", 0.0)
                        )
                        retain_direct_challenger = best_direct_score >= 0.50
                        if retain_direct_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        round_training_summary["pool"] = {}
                        round_training_summary["direct_population"] = {}
                        round_training_summary["population"] = {
                            "mcts_skipped_not_ready": 1.0,
                            "mcts_ladder": [],
                        }
                        last_training = dict(round_training_summary)
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        direct_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 0.0,
                            "mcts_started": 0.0,
                            "retained_challenger": float(
                                retain_direct_challenger
                            ),
                            "direct_best_score": best_direct_score,
                            "direct_best_city_captures_per_game": float(
                                direct_summary.get(
                                    "direct_best_city_captures_per_game", 0.0
                                )
                            ),
                            "direct_best_completion_rate": float(
                                direct_summary.get(
                                    "direct_best_completion_rate", 0.0
                                )
                            ),
                            "reason": "direct_policy_not_ready_for_mcts",
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            direct_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            direct_ready=0.0,
                            mcts_started=0.0,
                            retained_challenger=float(
                                retain_direct_challenger
                            ),
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league direct policy not ready | MCTS skipped | "
                            f"score={best_direct_score:.4f} | "
                            + (
                                "training will continue from the retained "
                                "challenger in the next round"
                                if retain_direct_challenger
                                else "candidate rejected; next round returns "
                                "to the champion"
                            ),
                            flush=True,
                        )
                        del candidate, champion
                        continue
                    if not direct_ready:
                        print(
                            "league direct curriculum did not reach its early "
                            "readiness target | continuing to authoritative "
                            "flat promotion gate",
                            flush=True,
                        )

                    frozen_models = []
                    for population_path in population_paths:
                        if population_path.resolve() == champion_path.resolve():
                            frozen_model, frozen_config = champion, champion_config
                        else:
                            frozen_model, frozen_config = load_model(
                                population_path, device
                            )
                        if frozen_config != model_config:
                            raise RuntimeError(
                                "population checkpoint has a different "
                                f"ModelConfig: {population_path}"
                            )
                        frozen_models.append(frozen_model.eval())

                    # The free exploiter starts from exactly the same point as
                    # the main line, before the latter's champion-anchored PPO
                    # updates.  It is evaluated on the same development arena
                    # and can only earn the later independent promotion gate.
                    exploiter_source_state = (
                        model_state_on_cpu(candidate)
                        if breakthrough_exploiter and effective_population_steps
                        else None
                    )
                    direct_population_summary: Dict[str, Any] = {}
                    direct_population_main_summary: Dict[str, Any] = {}
                    direct_population_exploiter_summary: Dict[str, Any] = {}
                    selected_population_branch = "main"
                    if effective_population_steps:
                        def record_direct_population_gate_game(
                            stage: str,
                            block: int,
                            opponent_index: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            row = dict(game)
                            row["curriculum_stage"] = stage
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=(
                                    "direct_population_gate_"
                                    f"{stage}_{block}"
                                ),
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=row,
                            )

                        def record_direct_population_training_game(
                            block: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            opponent_index = int(
                                game.get("opponent_model_index", 0)
                            )
                            row = dict(game)
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=f"direct_population_training_{block}",
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=row,
                            )

                        def train_population_stage(
                            stage_candidate: PolicyValueBeliefNet,
                            *,
                            seed: int,
                            minimum_steps: int,
                            maximum_steps: int,
                            gate_games: int,
                            diagnostic_gate_games: int,
                            gate_seed: int,
                            stage_name: str,
                            label: str,
                        ) -> Dict[str, Any]:
                            def record_stage_gate_game(
                                stage: str,
                                block: int,
                                opponent_index: int,
                                game: Mapping[str, Any],
                            ) -> None:
                                decorated = dict(game)
                                decorated["curriculum_stage"] = stage
                                decorated["curriculum_block"] = int(block)
                                phase_prefix = (
                                    "direct_population_gate"
                                    if stage_name == "main"
                                    else f"direct_population_{stage_name}_gate"
                                )
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=f"{phase_prefix}_{stage}_{block}",
                                    model_a_id=candidate_id,
                                    model_b_id=population_ids[opponent_index],
                                    game=decorated,
                                )

                            def record_stage_training_game(
                                block: int, game: Mapping[str, Any]
                            ) -> None:
                                opponent_index = int(
                                    game.get("opponent_model_index", 0)
                                )
                                decorated = dict(game)
                                decorated["curriculum_block"] = int(block)
                                if stage_name != "main":
                                    decorated["training_branch"] = stage_name
                                phase_prefix = (
                                    "direct_population_training"
                                    if stage_name == "main"
                                    else f"direct_population_{stage_name}_training"
                                )
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=f"{phase_prefix}_{block}",
                                    model_a_id=candidate_id,
                                    model_b_id=population_ids[opponent_index],
                                    game=decorated,
                                )

                            return train_direct_population_curriculum(
                                replace(
                                    round_training,
                                    seed=seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                model_config,
                                stage_candidate,
                                champion,
                                frozen_models,
                                probabilities,
                                num_envs=direct_population_envs,
                                minimum_steps=minimum_steps,
                                maximum_steps=maximum_steps,
                                block_steps=min(
                                    direct_population_block_steps,
                                    maximum_steps,
                                ),
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                                visible_action_history=visible_action_history,
                                gate_games=gate_games,
                                diagnostic_gate_games=diagnostic_gate_games,
                                gate_seed=gate_seed,
                                gate_score=direct_population_gate_score,
                                historical_score_floor=(
                                    direct_population_historical_floor
                                ),
                                readiness_city_captures_per_game=(
                                    direct_readiness_city_captures_per_game
                                ),
                                readiness_completion_rate=(
                                    direct_readiness_completion_rate
                                ),
                                readiness_patience=(
                                    direct_population_readiness_patience
                                ),
                                regression_patience=(
                                    direct_population_regression_patience
                                ),
                                policy_anchor_weight=(
                                    effective_policy_anchor_weight
                                ),
                                value_anchor_weight=direct_value_anchor_weight,
                                opponent_policy_weight=(
                                    effective_population_opponent_policy_weight
                                ),
                                teacher_weight=(
                                    effective_population_teacher_weight
                                ),
                                shaping_weight=(
                                    direct_population_shaping_weight
                                ),
                                shaping_final_weight=(
                                    direct_population_shaping_final_weight
                                ),
                                ppo_rollout_steps=(
                                    direct_population_ppo_rollout_steps
                                ),
                                ppo_epochs=direct_population_ppo_epochs,
                                ppo_minibatch_size=(
                                    direct_population_ppo_minibatch_size
                                ),
                                ppo_clip_range=(
                                    direct_population_ppo_clip_range
                                ),
                                ppo_gamma=direct_population_ppo_gamma,
                                ppo_gae_lambda=(
                                    direct_population_ppo_gae_lambda
                                ),
                                ppo_terminal_advantage_weight=(
                                    direct_population_ppo_terminal_advantage_weight
                                ),
                                ppo_top_advantage_fraction=(
                                    direct_population_ppo_top_advantage_fraction
                                ),
                                ppo_target_kl=(
                                    direct_population_ppo_target_kl
                                ),
                                ppo_hard_kl=(
                                    direct_population_ppo_hard_kl
                                ),
                                ppo_rollout_temperature=(
                                    direct_population_ppo_rollout_temperature
                                ),
                                plateau_patience=(
                                    direct_population_plateau_patience
                                ),
                                plateau_temperature_increment=(
                                    direct_population_plateau_temperature_increment
                                ),
                                plateau_max_temperature=(
                                    direct_population_plateau_max_temperature
                                ),
                                ppo_max_pending_examples=(
                                    direct_population_ppo_max_pending_examples
                                ),
                                ppo_min_learning_rate=(
                                    direct_population_ppo_min_learning_rate
                                ),
                                ppo_lr_backoff=direct_population_ppo_lr_backoff,
                                objective_replay_horizon=(
                                    direct_population_objective_replay_horizon
                                ),
                                objective_replay_capacity=(
                                    direct_population_objective_replay_capacity
                                ),
                                objective_replay_batch_size=(
                                    direct_population_objective_replay_batch_size
                                ),
                                objective_replay_interval=(
                                    direct_population_objective_replay_interval
                                ),
                                objective_replay_weight=(
                                    direct_population_objective_replay_weight
                                ),
                                objective_replay_recency=(
                                    direct_population_objective_replay_recency
                                ),
                                objective_replay_max_epochs=(
                                    direct_population_objective_replay_max_epochs
                                ),
                                ppg_enabled=direct_population_ppg_enabled,
                                ppg_interval=direct_population_ppg_interval,
                                ppg_buffer_capacity=(
                                    direct_population_ppg_buffer_capacity
                                ),
                                ppg_samples_per_rollout=(
                                    direct_population_ppg_samples_per_rollout
                                ),
                                ppg_phase_samples=(
                                    direct_population_ppg_phase_samples
                                ),
                                ppg_epochs=direct_population_ppg_epochs,
                                ppg_minibatch_size=(
                                    direct_population_ppg_minibatch_size
                                ),
                                ppg_policy_kl_weight=(
                                    direct_population_ppg_policy_kl_weight
                                ),
                                ppg_value_weight=(
                                    direct_population_ppg_value_weight
                                ),
                                ema_enabled=direct_population_ema_enabled,
                                ema_decay=direct_population_ema_decay,
                                gate_game_callback=record_stage_gate_game,
                                training_game_callback=record_stage_training_game,
                                candidate_label=label,
                                opponent_labels=population_evaluation_labels,
                                run=run.child(
                                    f"round_{round_index:03d}/"
                                    f"direct_population_{stage_name}"
                                ),
                            )

                        branch_records: list[Dict[str, float | int | bool]] = []
                        if direct_population_branch_count > 1:
                            source_state = model_state_on_cpu(candidate)
                            branch_winner: Dict[str, Any] | None = None
                            for branch_index in range(
                                direct_population_branch_count
                            ):
                                branch = PolicyValueBeliefNet(model_config).to(
                                    device
                                ).eval()
                                restore_model_state(branch, source_state)
                                branch_seed = _offset_seed(
                                    direct_population_training_seed,
                                    (branch_index + 1) * 1_500_007,
                                )
                                branch_summary = train_population_stage(
                                    branch,
                                    seed=branch_seed,
                                    minimum_steps=direct_population_branch_steps,
                                    maximum_steps=direct_population_branch_steps,
                                    gate_games=direct_population_branch_gate_games,
                                    diagnostic_gate_games=(
                                        direct_population_branch_gate_games
                                    ),
                                    gate_seed=_offset_seed(
                                        direct_population_gate_seed,
                                        (branch_index + 1) * 10_000_019,
                                    ),
                                    stage_name=f"branch_{branch_index + 1:02d}",
                                    label=(
                                        f"{candidate_evaluation_label} "
                                        f"branch-{branch_index + 1}"
                                    ),
                                )
                                champion_match = evaluate_direct_population_matches(
                                    replace(
                                        round_training,
                                        seed=branch_seed,
                                        max_moves=direct_gate_max_moves,
                                    ),
                                    branch,
                                    [champion],
                                    [1.0],
                                    games=direct_population_branch_gate_games,
                                    gate_seed=_offset_seed(
                                        direct_population_gate_seed,
                                        900_000_001 + branch_index,
                                    ),
                                    visible_action_history=(
                                        visible_action_history
                                    ),
                                    candidate_label=(
                                        f"{candidate_evaluation_label} "
                                        f"branch-{branch_index + 1}"
                                    ),
                                    opponent_labels=[champion_path.stem],
                                    num_threads=num_threads,
                                    max_actions=mcts.native_max_actions,
                                )
                                champion_score = float(
                                    champion_match["model_a_score"]
                                )
                                population_branch_score = float(
                                    branch_summary[
                                        "direct_population_full_gate_score"
                                    ]
                                )
                                eligible = bool(
                                    branch_summary[
                                        "direct_population_full_gate_selected"
                                    ]
                                ) and (
                                    champion_score
                                    >= direct_population_branch_min_champion_score
                                )
                                record = {
                                    "branch": branch_index + 1,
                                    "champion_score": champion_score,
                                    "population_score": population_branch_score,
                                    "eligible": eligible,
                                }
                                branch_records.append(record)
                                print(
                                    "league branch "
                                    f"{branch_index + 1}/"
                                    f"{direct_population_branch_count} | "
                                    f"champion={champion_score:.4f} | "
                                    f"population={population_branch_score:.4f} | "
                                    f"eligible={int(eligible)}",
                                    flush=True,
                                )
                                if eligible and (
                                    branch_winner is None
                                    or (
                                        champion_score,
                                        population_branch_score,
                                    )
                                    > (
                                        branch_winner["champion_score"],
                                        branch_winner["population_score"],
                                    )
                                ):
                                    branch_winner = {
                                        "state": model_state_on_cpu(branch),
                                        "branch": branch_index + 1,
                                        "champion_score": champion_score,
                                        "population_score": population_branch_score,
                                    }
                                del branch
                            round_training_summary[
                                "direct_population_branches"
                            ] = branch_records
                            if branch_winner is None:
                                print(
                                    "league branch selection | no branch met "
                                    "the champion development floor; skipping "
                                    "long PPO/PPG phase",
                                    flush=True,
                                )
                                direct_population_summary = {
                                    "direct_population_branch_selected": 0.0,
                                    "direct_population_full_gate_selected": 0.0,
                                }
                            else:
                                restore_model_state(candidate, branch_winner["state"])
                                print(
                                    "league branch selection | winner="
                                    f"{branch_winner['branch']} | champion="
                                    f"{branch_winner['champion_score']:.4f} | "
                                    "starting long PPO/PPG phase",
                                    flush=True,
                                )
                                direct_population_summary = train_population_stage(
                                    candidate,
                                    seed=direct_population_training_seed,
                                    minimum_steps=effective_population_steps,
                                    maximum_steps=(
                                        effective_population_max_steps
                                    ),
                                    gate_games=direct_population_gate_games,
                                    diagnostic_gate_games=(
                                        direct_population_diagnostic_gate_games
                                    ),
                                    gate_seed=direct_population_gate_seed,
                                    stage_name="main",
                                    label=candidate_evaluation_label,
                                )
                                direct_population_summary[
                                    "direct_population_branch_selected"
                                ] = float(branch_winner["branch"])
                        else:
                            direct_population_summary = train_population_stage(
                                candidate,
                                seed=direct_population_training_seed,
                                minimum_steps=effective_population_steps,
                                maximum_steps=effective_population_max_steps,
                                gate_games=direct_population_gate_games,
                                diagnostic_gate_games=(
                                    direct_population_diagnostic_gate_games
                                ),
                                gate_seed=direct_population_gate_seed,
                                stage_name="main",
                                label=candidate_evaluation_label,
                            )
                        direct_population_main_summary = dict(
                            direct_population_summary
                        )
                        if exploiter_source_state is not None:
                            # The main candidate was loaded through
                            # ``load_model(..., device)``.  A fresh exploiter
                            # has no such loader call, so put it on the same
                            # device before its baseline gate creates CUDA
                            # observation tensors.
                            exploiter = PolicyValueBeliefNet(model_config).to(
                                device
                            ).eval()
                            restore_model_state(exploiter, exploiter_source_state)
                            exploiter_probabilities = (
                                _champion_focused_probabilities(
                                    population_ids,
                                    probabilities,
                                    champion_id,
                                    champion_probability=(
                                        min(
                                            exploiter_champion_probability,
                                            1.0
                                            - (len(population_ids) - 1)
                                            * minimum_sampling_probability,
                                        )
                                    ),
                                    probability_floor=(
                                        minimum_sampling_probability
                                    ),
                                )
                            )

                            def record_exploiter_gate_game(
                                stage: str,
                                block: int,
                                opponent_index: int,
                                game: Mapping[str, Any],
                            ) -> None:
                                record_direct_population_gate_game(
                                    f"exploiter_{stage}",
                                    block,
                                    opponent_index,
                                    game,
                                )

                            def record_exploiter_training_game(
                                block: int,
                                game: Mapping[str, Any],
                            ) -> None:
                                opponent_index = int(
                                    game.get("opponent_model_index", 0)
                                )
                                row = dict(game)
                                row["curriculum_block"] = int(block)
                                row["training_branch"] = "exploiter"
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=(
                                        "direct_population_exploiter_training_"
                                        f"{block}"
                                    ),
                                    model_a_id=candidate_id,
                                    model_b_id=population_ids[opponent_index],
                                    game=row,
                                )

                            direct_population_exploiter_summary = (
                                train_direct_population_curriculum(
                                    replace(
                                        round_training,
                                        seed=_offset_seed(
                                            direct_population_training_seed,
                                            1_250_003,
                                        ),
                                        max_moves=direct_gate_max_moves,
                                    ),
                                    model_config,
                                    exploiter,
                                    champion,
                                    frozen_models,
                                    exploiter_probabilities,
                                    num_envs=direct_population_envs,
                                    minimum_steps=effective_population_steps,
                                    maximum_steps=effective_population_max_steps,
                                    block_steps=direct_population_block_steps,
                                    num_threads=num_threads,
                                    max_actions=mcts.native_max_actions,
                                    visible_action_history=(
                                        visible_action_history
                                    ),
                                    gate_games=direct_population_gate_games,
                                    diagnostic_gate_games=(
                                        direct_population_diagnostic_gate_games
                                    ),
                                    gate_seed=direct_population_gate_seed,
                                    gate_score=direct_population_gate_score,
                                    historical_score_floor=(
                                        direct_population_historical_floor
                                    ),
                                    readiness_city_captures_per_game=(
                                        direct_readiness_city_captures_per_game
                                    ),
                                    readiness_completion_rate=(
                                        direct_readiness_completion_rate
                                    ),
                                    readiness_patience=(
                                        direct_population_readiness_patience
                                    ),
                                    regression_patience=(
                                        direct_population_regression_patience
                                    ),
                                    policy_anchor_weight=(
                                        exploiter_policy_anchor_weight
                                    ),
                                    value_anchor_weight=(
                                        exploiter_value_anchor_weight
                                    ),
                                    opponent_policy_weight=0.0,
                                    teacher_weight=0.0,
                                    shaping_weight=(
                                        direct_population_shaping_weight
                                    ),
                                    shaping_final_weight=(
                                        direct_population_shaping_final_weight
                                    ),
                                    ppo_rollout_steps=(
                                        direct_population_ppo_rollout_steps
                                    ),
                                    ppo_epochs=direct_population_ppo_epochs,
                                    ppo_minibatch_size=(
                                        direct_population_ppo_minibatch_size
                                    ),
                                    ppo_clip_range=(
                                        direct_population_ppo_clip_range
                                    ),
                                    ppo_gamma=direct_population_ppo_gamma,
                                    ppo_gae_lambda=(
                                        direct_population_ppo_gae_lambda
                                    ),
                                    ppo_terminal_advantage_weight=(
                                        direct_population_ppo_terminal_advantage_weight
                                    ),
                                    ppo_top_advantage_fraction=(
                                        direct_population_ppo_top_advantage_fraction
                                    ),
                                    ppo_target_kl=exploiter_ppo_target_kl,
                                    ppo_hard_kl=exploiter_ppo_hard_kl,
                                    ppo_rollout_temperature=(
                                        exploiter_ppo_rollout_temperature
                                    ),
                                    ppo_max_pending_examples=(
                                        direct_population_ppo_max_pending_examples
                                    ),
                                    ppo_min_learning_rate=(
                                        direct_population_ppo_min_learning_rate
                                    ),
                                    ppo_lr_backoff=(
                                        direct_population_ppo_lr_backoff
                                    ),
                                    objective_replay_horizon=(
                                        direct_population_objective_replay_horizon
                                    ),
                                    objective_replay_capacity=(
                                        direct_population_objective_replay_capacity
                                    ),
                                    objective_replay_batch_size=(
                                        direct_population_objective_replay_batch_size
                                    ),
                                    objective_replay_interval=(
                                        direct_population_objective_replay_interval
                                    ),
                                    objective_replay_weight=(
                                        direct_population_objective_replay_weight
                                    ),
                                    objective_replay_recency=(
                                        direct_population_objective_replay_recency
                                    ),
                                    objective_replay_max_epochs=(
                                        direct_population_objective_replay_max_epochs
                                    ),
                                    ppg_enabled=direct_population_ppg_enabled,
                                    ppg_interval=direct_population_ppg_interval,
                                    ppg_buffer_capacity=(
                                        direct_population_ppg_buffer_capacity
                                    ),
                                    ppg_samples_per_rollout=(
                                        direct_population_ppg_samples_per_rollout
                                    ),
                                    ppg_phase_samples=(
                                        direct_population_ppg_phase_samples
                                    ),
                                    ppg_epochs=direct_population_ppg_epochs,
                                    ppg_minibatch_size=(
                                        direct_population_ppg_minibatch_size
                                    ),
                                    ppg_policy_kl_weight=(
                                        direct_population_ppg_policy_kl_weight
                                    ),
                                    ppg_value_weight=(
                                        direct_population_ppg_value_weight
                                    ),
                                    ema_enabled=direct_population_ema_enabled,
                                    ema_decay=direct_population_ema_decay,
                                    gate_game_callback=(
                                        record_exploiter_gate_game
                                    ),
                                    training_game_callback=(
                                        record_exploiter_training_game
                                    ),
                                    candidate_label=(
                                        f"exploiter {candidate_source.stem}"
                                    ),
                                    opponent_labels=(
                                        population_evaluation_labels
                                    ),
                                    run=run.child(
                                        f"round_{round_index:03d}/"
                                        "direct_population_exploiter"
                                    ),
                                )
                            )
                            if _select_exploiter_checkpoint(
                                direct_population_main_summary,
                                direct_population_exploiter_summary,
                            ):
                                candidate = exploiter
                                candidate_evaluation_label = (
                                    f"exploiter {candidate_source.stem}"
                                )
                                direct_population_summary = dict(
                                    direct_population_exploiter_summary
                                )
                                selected_population_branch = "exploiter"
                                print(
                                    "league breakthrough exploiter selected | "
                                    f"score={float(direct_population_summary.get('direct_population_full_gate_score', 0.0)):.4f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    "league breakthrough exploiter kept in diagnostics | "
                                    f"score={float(direct_population_exploiter_summary.get('direct_population_full_gate_score', 0.0)):.4f}",
                                    flush=True,
                                )
                    round_training_summary["direct_population"] = (
                        {
                            **direct_population_summary,
                            "selected_branch": selected_population_branch,
                            "main": direct_population_main_summary,
                            "exploiter": direct_population_exploiter_summary,
                        }
                    )
                    direct_population_ready = (
                        not bool(effective_population_steps)
                        or bool(
                            direct_population_summary.get(
                                "direct_population_ready", 0.0
                            )
                        )
                    )
                    if (
                        not direct_population_ready
                        and round_champion_mode == "mcts"
                    ):
                        best_population_score = float(
                            direct_population_summary.get(
                                "direct_population_best_score", 0.0
                            )
                        )
                        best_population_worst = float(
                            direct_population_summary.get(
                                "direct_population_best_worst_score", 0.0
                            )
                        )
                        retain_population_challenger = (
                            best_population_score >= 0.50
                            and best_population_worst
                            >= direct_population_historical_floor
                        )
                        if retain_population_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        round_training_summary["pool"] = {}
                        round_training_summary["population"] = {
                            "mcts_skipped_not_ready": 1.0,
                            "mcts_ladder": [],
                        }
                        last_training = dict(round_training_summary)
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        population_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 1.0,
                            "direct_population_ready": 0.0,
                            "mcts_started": 0.0,
                            "retained_challenger": float(
                                retain_population_challenger
                            ),
                            "direct_population_best_score": (
                                best_population_score
                            ),
                            "direct_population_best_worst_score": (
                                best_population_worst
                            ),
                            "reason": (
                                "direct_population_not_ready_for_mcts"
                            ),
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            population_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            direct_ready=1.0,
                            direct_population_ready=0.0,
                            mcts_started=0.0,
                            retained_challenger=float(
                                retain_population_challenger
                            ),
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league direct population not ready | "
                            "MCTS skipped | "
                            f"score={best_population_score:.4f} | "
                            f"worst={best_population_worst:.4f}",
                            flush=True,
                        )
                        del candidate, champion, frozen_models
                        continue
                    if not direct_population_ready:
                        print(
                            "league direct population did not reach its early "
                            "readiness target | continuing to authoritative "
                            "flat promotion gate",
                            flush=True,
                        )

                    flat_distillation_summary: Dict[str, Any] = {}
                    mcts_distillation_ready = False
                    if (
                        round_champion_mode == "flat"
                        and flat_mcts_distill_games
                    ):
                        # Search is a teacher only when it already improves on
                        # the raw candidate on exactly the same held-out maps.
                        # Without this guard a weak policy/value/belief stack
                        # feeds its own noise back into the raw policy.
                        raw_probe_id = database.start_evaluation(
                            round_id,
                            candidate_id,
                            champion_id,
                            "flat_mcts_distillation_raw_readiness",
                            mcts_probe_seed,
                            mcts_probe_games,
                        )
                        raw_probe = evaluate_direct_matches(
                            candidate,
                            champion,
                            device,
                            replace(
                                round_training,
                                seed=mcts_probe_seed,
                                max_moves=direct_gate_max_moves,
                            ),
                            mcts_probe_games,
                            model_a_label=candidate_evaluation_label,
                            model_b_label=champion_path.stem,
                            visible_action_history=visible_action_history,
                            num_threads=num_threads,
                            max_actions=mcts.native_max_actions,
                        )
                        database.finish_evaluation_unrated(raw_probe_id, raw_probe)
                        search_probe_id = database.start_evaluation(
                            round_id,
                            candidate_id,
                            champion_id,
                            "flat_mcts_distillation_search_readiness",
                            mcts_probe_seed,
                            mcts_probe_games,
                        )
                        try:
                            search_probe = evaluate_matches_pool(
                                candidate,
                                champion,
                                replace(
                                    round_training,
                                    seed=mcts_probe_seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                replace(
                                    evaluation_mcts,
                                    simulations=mcts_probe_simulations,
                                ),
                                mcts_probe_games,
                                num_threads=num_threads,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=min(num_envs, mcts_probe_games),
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                            )
                            database.finish_evaluation_unrated(
                                search_probe_id, search_probe
                            )
                        except (BeliefQualityError, BeliefRecoveryRequired) as error:
                            search_probe = {
                                "games": 0.0,
                                "model_a_score": 0.0,
                                "valid": 0.0,
                                "invalid_reason": str(error),
                            }
                            database.invalidate_evaluation(
                                search_probe_id, search_probe
                            )
                        raw_probe_score = float(raw_probe["model_a_score"])
                        search_probe_score = float(
                            search_probe.get("model_a_score", 0.0)
                        )
                        mcts_distillation_ready = _mcts_probe_passed(
                            raw_probe_score,
                            search_probe_score,
                            minimum_search_score=mcts_probe_score,
                            minimum_improvement=mcts_probe_minimum_improvement,
                            valid=bool(search_probe.get("valid", 1.0)),
                        )
                        flat_distillation_summary = {
                            "enabled": 1.0,
                            "readiness_passed": float(mcts_distillation_ready),
                            "raw_readiness": raw_probe,
                            "search_readiness": search_probe,
                            "raw_readiness_score": raw_probe_score,
                            "search_readiness_score": search_probe_score,
                            "minimum_search_score": float(mcts_probe_score),
                        }
                        run.log(
                            round=round_index,
                            flat_mcts_distill_readiness_raw=raw_probe_score,
                            flat_mcts_distill_readiness_search=search_probe_score,
                            flat_mcts_distill_readiness_passed=float(
                                mcts_distillation_ready
                            ),
                        )
                        if not mcts_distillation_ready:
                            print(
                                "league flat MCTS distillation skipped | "
                                f"raw={raw_probe_score:.4f} | "
                                f"search={search_probe_score:.4f} | "
                                "search is not a reliable teacher yet",
                                flush=True,
                            )
                    if (
                        round_champion_mode == "flat"
                        and flat_mcts_distill_games
                        and mcts_distillation_ready
                    ):
                        pre_distillation_state = model_state_on_cpu(candidate)

                        def evaluate_flat_distillation(
                            label: str,
                        ) -> Dict[str, Any]:
                            evaluation_id = database.start_evaluation(
                                round_id,
                                candidate_id,
                                champion_id,
                                f"flat_mcts_distillation_{label}",
                                flat_mcts_distill_gate_seed,
                                flat_mcts_distill_gate_games,
                            )

                            def record_gate_game(
                                game: Mapping[str, Any],
                            ) -> None:
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    evaluation_id=evaluation_id,
                                    phase=f"flat_mcts_distillation_{label}",
                                    model_a_id=candidate_id,
                                    model_b_id=champion_id,
                                    game=game,
                                )

                            result = evaluate_direct_matches(
                                candidate,
                                champion,
                                device,
                                replace(
                                    training,
                                    seed=flat_mcts_distill_gate_seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                flat_mcts_distill_gate_games,
                                game_callback=record_gate_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                                visible_action_history=(
                                    visible_action_history
                                ),
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                            if bool(result.get("valid", 1.0)):
                                database.finish_evaluation_unrated(
                                    evaluation_id, result
                                )
                            else:
                                database.invalidate_evaluation(
                                    evaluation_id, result
                                )
                            return dict(result)

                        before_distillation = evaluate_flat_distillation(
                            "before"
                        )

                        def record_flat_distillation_game(
                            game: Mapping[str, Any],
                        ) -> None:
                            opponent_index = int(
                                game.get("opponent_model_index", 0)
                            )
                            row = dict(game)
                            row["mcts_simulations"] = int(
                                flat_mcts_distill_simulations
                            )
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="flat_mcts_distillation_training",
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=row,
                            )

                        try:
                            # MCTS leaf tensors scale with the number of live
                            # roots.  The flat-policy PPO stage has just used
                            # CUDA heavily, so release its cached blocks before
                            # starting the deliberately smaller search batch.
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            distillation_mcts = replace(
                                mcts,
                                simulations=flat_mcts_distill_simulations,
                                strategic_prior_weight=min(
                                    mcts.strategic_prior_weight, 0.50
                                ),
                                teacher_prior_weight=min(
                                    mcts.teacher_prior_weight, 0.10
                                ),
                            )
                            distillation_training = train_population_pool(
                                replace(
                                    round_training,
                                    seed=flat_mcts_distill_training_seed,
                                    max_moves=flat_mcts_distill_max_moves,
                                ),
                                model_config,
                                distillation_mcts,
                                games=flat_mcts_distill_games,
                                updates_per_game=(
                                    flat_mcts_distill_updates_per_game
                                ),
                                candidate=candidate,
                                opponent_models=frozen_models,
                                opponent_weights=probabilities,
                                anchor_model=champion,
                                anchor_policy_weight=(
                                    flat_mcts_distill_policy_anchor_weight
                                ),
                                anchor_value_weight=(
                                    flat_mcts_distill_value_anchor_weight
                                ),
                                critical_decision_fraction=(
                                    flat_mcts_distill_critical_fraction
                                ),
                                critical_minimum_score=(
                                    flat_mcts_distill_critical_minimum_score
                                ),
                                critical_replay_fraction=(
                                    flat_mcts_distill_critical_replay_fraction
                                ),
                                critical_only=True,
                                reanalysis_path=population_mcts_reanalyse_path,
                                reanalysis_capacity=(
                                    population_mcts_reanalyse_capacity
                                ),
                                reanalysis_positions=(
                                    population_mcts_reanalyse_positions
                                ),
                                reanalysis_replay_fraction=(
                                    population_mcts_reanalyse_replay_fraction
                                ),
                                num_threads=population_workers,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=min(
                                    num_envs, flat_mcts_distill_max_envs
                                ),
                                offline_collection=True,
                                game_callback=(
                                    record_flat_distillation_game
                                ),
                                reserved_seeds=evaluation_map_seeds,
                                run=run.child(
                                    f"round_{round_index:03d}/"
                                    "flat_mcts_distillation"
                                ),
                            )
                            after_distillation = evaluate_flat_distillation(
                                "after"
                            )
                            distillation_decision = (
                                _flat_distillation_gate_decision(
                                    float(
                                        before_distillation[
                                            "model_a_score"
                                        ]
                                    ),
                                    float(
                                        after_distillation[
                                            "model_a_score"
                                        ]
                                    ),
                                    minimum_score=(
                                        direct_population_historical_floor
                                    ),
                                    valid=bool(
                                        after_distillation.get("valid", 1.0)
                                    ),
                                )
                            )
                            accepted_distillation = bool(
                                distillation_decision["accepted"]
                            )
                            if not accepted_distillation:
                                restore_model_state(
                                    candidate, pre_distillation_state
                                )
                            flat_distillation_summary = {
                                **flat_distillation_summary,
                                "enabled": 1.0,
                                "training": distillation_training,
                                "before": before_distillation,
                                "after": after_distillation,
                                **distillation_decision,
                            }
                            run.log(
                                round=round_index,
                                flat_mcts_distill_before=float(
                                    distillation_decision["before_score"]
                                ),
                                flat_mcts_distill_after=float(
                                    distillation_decision["after_score"]
                                ),
                                flat_mcts_distill_improvement=float(
                                    distillation_decision["improvement"]
                                ),
                                flat_mcts_distill_accepted=float(
                                    accepted_distillation
                                ),
                            )
                            print(
                                "league flat MCTS distillation | "
                                f"before={float(distillation_decision['before_score']):.4f} | "
                                f"after={float(distillation_decision['after_score']):.4f} | "
                                "critical="
                                f"{int(distillation_training.get('population_pool_critical_mcts_decisions', 0.0))} | "
                                "critical_mean="
                                f"{float(distillation_training.get('population_pool_critical_mcts_mean_score', 0.0)):.4f} | "
                                f"accepted={int(accepted_distillation)}",
                                flush=True,
                            )
                        except (
                            BeliefQualityError,
                            BeliefRecoveryRequired,
                        ) as error:
                            restore_model_state(
                                candidate, pre_distillation_state
                            )
                            flat_distillation_summary = {
                                **flat_distillation_summary,
                                "enabled": 1.0,
                                "accepted": False,
                                "rolled_back": 1.0,
                                "reason": str(error),
                                "before": before_distillation,
                            }
                            run.log(
                                round=round_index,
                                flat_mcts_distill_accepted=0.0,
                                flat_mcts_distill_belief_failure=1.0,
                            )
                            print(
                                "league flat MCTS distillation rolled back | "
                                f"reason={error}",
                                flush=True,
                            )
                        round_training_summary[
                            "flat_mcts_distillation"
                        ] = flat_distillation_summary

                    if round_champion_mode == "flat":
                        # Flat-policy strength is the only promotion authority
                        # in this mode. Optional search distillation above must
                        # first improve the raw apprentice; the later probe can
                        # neither block nor overwrite a champion.
                        flat_evaluations: list[Dict[str, Any]] = []
                        flat_invalid = False
                        champion_before_id = champion_id
                        for opponent_index, (
                            opponent_id,
                            opponent_path,
                            opponent_model,
                            probability,
                        ) in enumerate(zip(
                            population_ids,
                            population_paths,
                            frozen_models,
                            probabilities,
                        )):
                            is_champion = opponent_id == champion_before_id
                            games = (
                                promotion_games if is_champion else historical_games
                            )
                            purpose = (
                                "flat_champion_gate"
                                if is_champion
                                else "flat_historical_gate"
                            )
                            match_seed = _offset_seed(
                                round_evaluation_seed,
                                opponent_index * 1_000_003,
                            )
                            evaluation_id = database.start_evaluation(
                                round_id,
                                candidate_id,
                                opponent_id,
                                purpose,
                                match_seed,
                                games,
                            )

                            def record_flat_gate_game(
                                game: Mapping[str, Any],
                                *,
                                selected_evaluation_id: int = evaluation_id,
                                selected_opponent_id: int = opponent_id,
                            ) -> None:
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    evaluation_id=selected_evaluation_id,
                                    phase="flat_evaluation",
                                    model_a_id=candidate_id,
                                    model_b_id=selected_opponent_id,
                                    game=game,
                                )

                            match = evaluate_direct_matches(
                                candidate,
                                opponent_model,
                                device,
                                replace(
                                    training,
                                    seed=match_seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                games,
                                game_callback=record_flat_gate_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=opponent_path.stem,
                                visible_action_history=(
                                    visible_action_history
                                ),
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                            result = {
                                "opponent_id": opponent_id,
                                "opponent_path": str(opponent_path),
                                "sampling_probability": probability,
                                "purpose": purpose,
                                **match,
                            }
                            flat_evaluations.append(result)
                            if not bool(match.get("valid", 1.0)):
                                flat_invalid = True
                                database.invalidate_evaluation(
                                    evaluation_id, match
                                )
                                break
                            database.finish_evaluation(
                                evaluation_id, match, elo_k=elo_k
                            )
                            print(
                                f"league flat gate vs {opponent_path.stem} | "
                                f"score={float(match['model_a_score']):.4f} | "
                                f"lower95={float(match['score_lower_95']):.4f}",
                                flush=True,
                            )

                        if flat_invalid:
                            decision: Dict[str, float | bool] = {
                                "candidate_score": 0.0,
                                "champion_lower_95": 0.0,
                                "population_score": 0.0,
                                "minimum_historical_score": 0.0,
                                "promoted": False,
                            }
                        else:
                            decision = _promotion_gate_decision(
                                flat_evaluations,
                                champion_before_id,
                                promotion_score=promotion_score,
                                promotion_lower_bound=promotion_lower_bound,
                                population_score=population_score,
                                historical_score_floor=historical_score_floor,
                            )
                        last_candidate_score = float(
                            decision["candidate_score"]
                        )
                        last_population_score = float(
                            decision["population_score"]
                        )
                        minimum_historical_score = float(
                            decision["minimum_historical_score"]
                        )
                        champion_lower_95 = float(
                            decision["champion_lower_95"]
                        )
                        flat_promoted = bool(decision["promoted"])
                        promoted = flat_promoted
                        continuation_action = (
                            "none"
                            if fresh_branch_round
                            else _challenger_continuation_action(
                                valid=not flat_invalid,
                                candidate_score=last_candidate_score,
                                population_score=last_population_score,
                                minimum_historical_score=(
                                    minimum_historical_score
                                ),
                                historical_score_floor=historical_score_floor,
                                has_previous_challenger=(
                                    previous_challenger_path is not None
                                    and previous_challenger_id is not None
                                ),
                            )
                        )
                        retain_challenger = continuation_action == "candidate"
                        continue_previous_challenger = (
                            continuation_action == "previous"
                        )
                        parent_capability = database.model_capability(
                            candidate_parent_id
                        )
                        # Raw policy is the sole champion representation.
                        # A successful MCTS@32 probe is only evidence that a
                        # later search-teacher experiment is allowed; it must
                        # never defer or veto a flat-policy promotion.
                        mcts_gate_active = False
                        pending_flat_path: Path | None = None

                        # Persist the raw-policy outcome before touching MCTS.
                        # A Ctrl+C or belief failure in the diagnostic probe
                        # must never lose a flat promotion already earned.
                        if promoted and mcts_gate_active:
                            # The lineage already proved that search@32 helps.
                            # Preserve this flat winner as a resumable
                            # challenger, then require the short search probe
                            # before it can replace an MCTS-aware champion.
                            pending_flat_path = _next_checkpoint_path(
                                checkpoint_dir, checkpoint_prefix
                            )
                            save_model(
                                pending_flat_path, candidate, model_config
                            )
                            database.retain_candidate(
                                candidate_id, pending_flat_path
                            )
                            challenger_path = pending_flat_path
                            challenger_id = candidate_id
                            promoted = False
                            print(
                                "league flat gate passed | MCTS-aware lineage "
                                "requires search@32 confirmation",
                                flush=True,
                            )
                        elif promoted:
                            promotions += 1
                            promoted_path = _next_checkpoint_path(
                                checkpoint_dir, checkpoint_prefix
                            )
                            save_model(promoted_path, candidate, model_config)
                            database.finalise_candidate(
                                candidate_id, promoted_path=promoted_path
                            )
                            champion_path = promoted_path
                            champion_id = candidate_id
                            challenger_path = None
                            challenger_id = None
                            print(
                                f"league flat promotion {promotions}: saved "
                                f"{champion_path}",
                                flush=True,
                            )
                        elif retain_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(
                                candidate_id, retained_path
                            )
                            challenger_path = retained_path
                            challenger_id = candidate_id
                            print(
                                "league challenger advanced | saved "
                                f"{challenger_path} | continuation floor="
                                f"{historical_score_floor:.4f}",
                                flush=True,
                            )
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            if continue_previous_challenger:
                                challenger_path = previous_challenger_path
                                challenger_id = previous_challenger_id
                                print(
                                    "league challenger update rejected | "
                                    "continuing next round from previous "
                                    f"challenger: {challenger_path}",
                                    flush=True,
                                )
                            else:
                                challenger_path = None
                                challenger_id = None
                                print(
                                    "league candidate rejected | next round "
                                    "starts from the champion",
                                    flush=True,
                                )

                        probe_summary: Dict[str, Any] = {
                            "enabled": float(bool(mcts_probe_games)),
                            "performed": 0.0,
                            "passed": 0.0,
                            "ready": 0.0,
                            "streak": 0.0,
                            "promotion_gate_active": float(
                                mcts_gate_active
                            ),
                            "evaluation_active": float(mcts_gate_active),
                        }
                        probe_passed = False
                        # Search quality is a property of a model that has
                        # already earned a flat promotion.  Probing merely
                        # retained challengers consumed expensive belief-MCTS
                        # games without changing the champion or producing a
                        # useful readiness signal for its lineage.
                        if mcts_probe_games and flat_promoted:
                            raw_probe_id = database.start_evaluation(
                                round_id,
                                candidate_id,
                                champion_before_id,
                                "mcts_readiness_raw_probe",
                                mcts_probe_seed,
                                mcts_probe_games,
                            )

                            def record_raw_probe_game(
                                game: Mapping[str, Any],
                            ) -> None:
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    evaluation_id=raw_probe_id,
                                    phase="mcts_readiness_raw_probe",
                                    model_a_id=candidate_id,
                                    model_b_id=champion_before_id,
                                    game=game,
                                )

                            raw_probe = evaluate_direct_matches(
                                candidate,
                                champion,
                                device,
                                replace(
                                    training,
                                    seed=mcts_probe_seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                mcts_probe_games,
                                game_callback=record_raw_probe_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                                visible_action_history=(
                                    visible_action_history
                                ),
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                            database.finish_evaluation_unrated(
                                raw_probe_id, raw_probe
                            )
                            search_probe_id = database.start_evaluation(
                                round_id,
                                candidate_id,
                                champion_before_id,
                                "mcts_readiness_search_probe",
                                mcts_probe_seed,
                                mcts_probe_games,
                            )

                            def record_search_probe_game(
                                game: Mapping[str, Any],
                            ) -> None:
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    evaluation_id=search_probe_id,
                                    phase="mcts_readiness_search_probe",
                                    model_a_id=candidate_id,
                                    model_b_id=champion_before_id,
                                    game=game,
                                )

                            probe_mcts = replace(
                                evaluation_mcts,
                                simulations=mcts_probe_simulations,
                            )
                            try:
                                search_probe = evaluate_matches_pool(
                                    candidate,
                                    champion,
                                    replace(training, seed=mcts_probe_seed),
                                    probe_mcts,
                                    mcts_probe_games,
                                    num_threads=num_threads,
                                    max_pending_leaves=max_pending_leaves,
                                    visible_action_history=(
                                        visible_action_history
                                    ),
                                    belief_retries=belief_retries,
                                    max_envs=min(
                                        num_envs, mcts_probe_games
                                    ),
                                    game_callback=record_search_probe_game,
                                    model_a_label=candidate_evaluation_label,
                                    model_b_label=champion_path.stem,
                                )
                            except (
                                BeliefQualityError,
                                BeliefRecoveryRequired,
                            ) as error:
                                search_probe = {
                                    "games": 0.0,
                                    "model_a_wins": 0.0,
                                    "model_a_losses": 0.0,
                                    "draws": 0.0,
                                    "model_a_score": 0.0,
                                    "score_lower_95": 0.0,
                                    "valid": 0.0,
                                    "invalid_evaluations": 1.0,
                                    "invalid_reason": str(error),
                                }
                            probe_valid = bool(search_probe.get("valid", 1.0))
                            if probe_valid:
                                database.finish_evaluation_unrated(
                                    search_probe_id, search_probe
                                )
                            else:
                                database.invalidate_evaluation(
                                    search_probe_id, search_probe
                                )
                            raw_probe_score = float(
                                raw_probe["model_a_score"]
                            )
                            search_probe_score = float(
                                search_probe.get("model_a_score", 0.0)
                            )
                            probe_passed = _mcts_probe_passed(
                                raw_probe_score,
                                search_probe_score,
                                minimum_search_score=mcts_probe_score,
                                minimum_improvement=(
                                    mcts_probe_minimum_improvement
                                ),
                                valid=probe_valid,
                            )
                            parent_streak = (
                                int(parent_capability["mcts_probe_streak"])
                                if parent_capability is not None
                                else 0
                            )
                            probe_streak = (
                                parent_streak + 1 if probe_passed else 0
                            )
                            mcts_ready = (
                                probe_streak >= mcts_probe_patience
                            )
                            probe_summary = {
                                "enabled": 1.0,
                                "performed": 1.0,
                                "games": float(mcts_probe_games),
                                "simulations": float(
                                    mcts_probe_simulations
                                ),
                                "raw_score": raw_probe_score,
                                "search_score": search_probe_score,
                                "minimum_search_score": float(
                                    mcts_probe_score
                                ),
                                "minimum_improvement": float(
                                    mcts_probe_minimum_improvement
                                ),
                                "valid": float(probe_valid),
                                "passed": float(probe_passed),
                                "streak": float(probe_streak),
                                "patience": float(mcts_probe_patience),
                                "ready": float(mcts_ready),
                                "promotion_gate_active": float(
                                    mcts_gate_active
                                ),
                                "evaluation_active": float(
                                    mcts_gate_active or mcts_ready
                                ),
                                "raw": raw_probe,
                                "search": search_probe,
                            }
                            database.record_mcts_capability(
                                candidate_id,
                                games=mcts_probe_games,
                                simulations=mcts_probe_simulations,
                                raw_score=raw_probe_score,
                                search_score=search_probe_score,
                                streak=probe_streak,
                                ready=mcts_ready,
                                evaluation_active=(
                                    mcts_gate_active or mcts_ready
                                ),
                                summary=probe_summary,
                            )
                            print(
                                "league MCTS readiness probe | "
                                f"raw={raw_probe_score:.4f} | "
                                f"search={search_probe_score:.4f} | "
                                f"passed={int(probe_passed)} | "
                                f"ready={int(mcts_ready)} "
                                f"({probe_streak}/{mcts_probe_patience})",
                                flush=True,
                            )

                        if mcts_gate_active and flat_promoted:
                            if probe_passed and pending_flat_path is not None:
                                database.finalise_candidate(
                                    candidate_id,
                                    promoted_path=pending_flat_path,
                                )
                                promotions += 1
                                champion_path = pending_flat_path
                                champion_id = candidate_id
                                challenger_path = None
                                challenger_id = None
                                retain_challenger = False
                                promoted = True
                                print(
                                    "league MCTS-aware promotion "
                                    f"{promotions}: saved {champion_path}",
                                    flush=True,
                                )
                            else:
                                promoted = False
                                retain_challenger = True
                                print(
                                    "league MCTS-aware promotion deferred | "
                                    "flat checkpoint retained as challenger",
                                    flush=True,
                                )

                        round_training_summary["pool"] = {}
                        round_training_summary["population"] = {
                            "mcts_training_skipped_flat_champion": 1.0,
                            "mcts_probe": probe_summary,
                        }
                        last_training = dict(round_training_summary)
                        gate_summary = {
                            "mode": "flat",
                            "flat_promoted": float(flat_promoted),
                            "promoted": float(promoted),
                            "valid": float(not flat_invalid),
                            "candidate_score": last_candidate_score,
                            "champion_lower_95": champion_lower_95,
                            "population_score": last_population_score,
                            "minimum_historical_score": (
                                minimum_historical_score
                            ),
                            "retained_challenger": float(
                                retain_challenger and not promoted
                            ),
                            "continued_previous_challenger": float(
                                continue_previous_challenger and not promoted
                            ),
                            "evaluations": flat_evaluations,
                            "mcts_probe": probe_summary,
                            "mcts_ready": float(
                                probe_summary.get("ready", 0.0)
                            ),
                            "mcts_promotion_gate_active": float(
                                mcts_gate_active
                            ),
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            promoted,
                            round_training_summary,
                            gate_summary,
                        )
                        run.log(
                            round=round_index,
                            champion_mode_flat=1.0,
                            champion_mode_auto_flat=float(champion_mode == "auto"),
                            candidate_score=last_candidate_score,
                            population_score=last_population_score,
                            promoted=float(promoted),
                            retained_challenger=float(
                                retain_challenger and not promoted
                            ),
                            continued_previous_challenger=float(
                                continue_previous_challenger and not promoted
                            ),
                            mcts_probe_passed=float(
                                probe_summary.get("passed", 0.0)
                            ),
                            mcts_ready=float(
                                probe_summary.get("ready", 0.0)
                            ),
                            mcts_evaluation_active=float(
                                probe_summary.get(
                                    "evaluation_active", 0.0
                                )
                            ),
                            mcts_promotion_gate_active=float(
                                mcts_gate_active
                            ),
                            promotions=promotions,
                        )
                        database.flush()
                        next_candidate_source = (
                            "champion"
                            if challenger_path is None
                            else challenger_path.stem
                        )
                        print(
                            f"league flat decision {round_index}/{rounds} | "
                            f"champion={last_candidate_score:.4f} | "
                            f"population={last_population_score:.4f} | "
                            f"worst={minimum_historical_score:.4f} | "
                            f"promoted={int(promoted)} | "
                            f"next={next_candidate_source}",
                            flush=True,
                        )
                        del candidate, champion, frozen_models
                        continue

                    direct_population_state = model_state_on_cpu(candidate)
                    pool_summary: Dict[str, Any] = {}
                    if steps:
                        # This legacy same-network phase is now optional. It is
                        # disabled in the recommended curriculum because weak
                        # candidate-vs-candidate MCTS was the main regression
                        # source observed in long league runs.
                        pool_summary = train_self_play_pool(
                            round_training,
                            model_config,
                            mcts,
                            num_envs=num_envs,
                            steps=steps,
                            updates=updates,
                            num_threads=num_threads,
                            max_pending_leaves=max_pending_leaves,
                            visible_action_history=visible_action_history,
                            belief_retries=belief_retries,
                            model_out=None,
                            model=candidate,
                            anchor_model=champion,
                            anchor_policy_weight=champion_anchor_weight,
                            anchor_value_weight=champion_value_anchor_weight,
                            reserved_seeds=evaluation_map_seeds,
                            run=run.child(
                                f"round_{round_index:03d}/pool_self_play"
                            ),
                        )
                        # Any failed ladder rung still returns to the raw
                        # population checkpoint captured before this phase.
                        pool_summary["rollback_state_available"] = 1.0
                    round_training_summary["pool"] = pool_summary

                    population_summary: Dict[str, Any] = {
                        "mcts_ladder": [],
                        "mcts_ladder_accepted": 0.0,
                        "mcts_ladder_rollbacks": 0.0,
                    }
                    # A failed first MCTS rung must return to the last state
                    # proven safe against the raw-policy population gate, not
                    # to an optional ungated legacy self-play snapshot.
                    best_mcts_state = direct_population_state
                    best_mcts_score = 0.50
                    if direct_population_summary:
                        raw_population_baseline = {
                            "model_a_score": float(
                                direct_population_summary.get(
                                    "direct_population_best_score", 0.0
                                )
                            ),
                            "worst_score": float(
                                direct_population_summary.get(
                                    "direct_population_best_worst_score", 0.0
                                )
                            ),
                            "model_a_city_captures_per_game": float(
                                direct_population_summary.get(
                                    "direct_population_best_city_captures_per_game",
                                    0.0,
                                )
                            ),
                            "completion_rate": float(
                                direct_population_summary.get(
                                    "direct_population_best_completion_rate", 0.0
                                )
                            ),
                        }
                    else:
                        def record_raw_population_baseline_game(
                            opponent_index: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="mcts_absorption_baseline",
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=game,
                            )

                        raw_population_baseline = (
                            evaluate_direct_population_matches(
                                replace(
                                    training,
                                    max_moves=direct_gate_max_moves,
                                ),
                                candidate,
                                frozen_models,
                                probabilities,
                                games=direct_population_gate_games,
                                gate_seed=direct_population_gate_seed,
                                game_callback=(
                                    record_raw_population_baseline_game
                                ),
                                candidate_label=candidate_evaluation_label,
                                opponent_labels=population_evaluation_labels,
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                        )
                    population_summary["raw_population_baseline"] = (
                        raw_population_baseline
                    )
                    best_raw_policy_score = float(
                        raw_population_baseline["model_a_score"]
                    )
                    best_raw_worst_score = float(
                        raw_population_baseline["worst_score"]
                    )
                    best_raw_city_capture_rate = float(
                        raw_population_baseline[
                            "model_a_city_captures_per_game"
                        ]
                    )
                    best_raw_completion_rate = float(
                        raw_population_baseline["completion_rate"]
                    )
                    if population_games:
                        for rung_index, (simulations, rung_games) in enumerate(
                            zip(population_rungs, population_game_allocations)
                        ):
                            if rung_games <= 0:
                                continue
                            rung_training = replace(
                                round_training,
                                seed=_offset_seed(
                                    population_training_seed,
                                    rung_index * 2_000_003,
                                ),
                            )
                            rung_mcts = replace(
                                mcts,
                                simulations=int(simulations),
                                # The direct curriculum must first prove that
                                # the neural policy/value help search. Manual
                                # priors are therefore reduced rung by rung.
                                strategic_prior_weight=min(
                                    mcts.strategic_prior_weight,
                                    0.50 if rung_index == 0 else 0.20,
                                ),
                                teacher_prior_weight=min(
                                    mcts.teacher_prior_weight,
                                    0.10,
                                ),
                            )

                            def record_training_game(
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                opponent_index = game.get("opponent_model_index")
                                opponent_id = (
                                    population_ids[int(opponent_index)]
                                    if opponent_index is not None else None
                                )
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=f"population_training_rung_{selected_rung + 1}",
                                    model_a_id=candidate_id,
                                    model_b_id=opponent_id,
                                    game=row,
                                )

                            rung_summary = train_population_pool(
                                rung_training,
                                model_config,
                                rung_mcts,
                                games=rung_games,
                                updates_per_game=population_updates_per_game,
                                candidate=candidate,
                                opponent_models=frozen_models,
                                opponent_weights=probabilities,
                                anchor_model=champion,
                                anchor_policy_weight=champion_anchor_weight,
                                anchor_value_weight=champion_value_anchor_weight,
                                critical_decision_fraction=(
                                    population_mcts_critical_fraction
                                ),
                                critical_minimum_score=(
                                    population_mcts_critical_minimum_score
                                ),
                                critical_replay_fraction=(
                                    population_mcts_critical_replay_fraction
                                ),
                                critical_only=population_mcts_critical_only,
                                reanalysis_path=population_mcts_reanalyse_path,
                                reanalysis_capacity=(
                                    population_mcts_reanalyse_capacity
                                ),
                                reanalysis_positions=(
                                    population_mcts_reanalyse_positions
                                ),
                                reanalysis_replay_fraction=(
                                    population_mcts_reanalyse_replay_fraction
                                ),
                                num_threads=population_workers,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=min(num_envs, population_max_envs),
                                offline_collection=population_offline_collection,
                                game_callback=record_training_game,
                                reserved_seeds=evaluation_map_seeds,
                                run=run.child(
                                    f"round_{round_index:03d}/mcts_rung_{rung_index + 1:02d}"
                                ),
                            )
                            gate_mcts = replace(
                                evaluation_mcts,
                                simulations=int(simulations),
                            )

                            def record_rung_gate_game(
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=(
                                        f"mcts_curriculum_gate_rung_"
                                        f"{selected_rung + 1}"
                                    ),
                                    model_a_id=candidate_id,
                                    model_b_id=champion_id,
                                    game=row,
                                )

                            rung_gate = evaluate_matches_pool(
                                candidate,
                                champion,
                                replace(
                                    training,
                                    seed=mcts_gate_seeds[rung_index],
                                ),
                                gate_mcts,
                                mcts_gate_games,
                                num_threads=num_threads,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=min(num_envs, population_max_envs),
                                game_callback=record_rung_gate_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                            )
                            rung_score = float(rung_gate.get("model_a_score", 0.0))

                            def record_absorption_gate_game(
                                opponent_index: int,
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=(
                                        f"mcts_absorption_gate_rung_"
                                        f"{selected_rung + 1}"
                                    ),
                                    model_a_id=candidate_id,
                                    model_b_id=population_ids[opponent_index],
                                    game=row,
                                )

                            absorption_gate = evaluate_direct_population_matches(
                                replace(
                                    training,
                                    max_moves=direct_gate_max_moves,
                                ),
                                candidate,
                                frozen_models,
                                probabilities,
                                games=direct_population_gate_games,
                                gate_seed=direct_population_gate_seed,
                                game_callback=record_absorption_gate_game,
                                candidate_label=candidate_evaluation_label,
                                opponent_labels=population_evaluation_labels,
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                            raw_policy_score = float(
                                absorption_gate.get("model_a_score", 0.0)
                            )
                            raw_worst_score = float(
                                absorption_gate.get("worst_score", 0.0)
                            )
                            raw_city_capture_rate = float(
                                absorption_gate.get(
                                    "model_a_city_captures_per_game", 0.0
                                )
                            )
                            raw_completion_rate = float(
                                absorption_gate.get("completion_rate", 0.0)
                            )
                            search_ready = bool(
                                rung_gate.get("valid", 1.0)
                            ) and (
                                rung_score
                                >= max(float(mcts_gate_score), best_mcts_score)
                            )
                            network_absorbed_search = (
                                raw_policy_score >= best_raw_policy_score
                                and raw_worst_score >= best_raw_worst_score
                                and raw_city_capture_rate
                                >= best_raw_city_capture_rate
                                and raw_completion_rate
                                >= best_raw_completion_rate
                            )
                            accepted = search_ready and network_absorbed_search
                            if accepted:
                                best_mcts_state = model_state_on_cpu(candidate)
                                best_mcts_score = rung_score
                                best_raw_policy_score = raw_policy_score
                                best_raw_worst_score = raw_worst_score
                                best_raw_city_capture_rate = (
                                    raw_city_capture_rate
                                )
                                best_raw_completion_rate = raw_completion_rate
                                population_summary["mcts_ladder_accepted"] += 1.0
                            else:
                                restore_model_state(candidate, best_mcts_state)
                                population_summary["mcts_ladder_rollbacks"] += 1.0
                            rung_record = {
                                "simulations": float(simulations),
                                "games": float(rung_games),
                                "accepted": float(accepted),
                                "gate_score": rung_score,
                                "best_gate_score": best_mcts_score,
                                "gate_lower_95": float(
                                    rung_gate.get("score_lower_95", 0.0)
                                ),
                                "search_ready": float(search_ready),
                                "network_absorbed_search": float(
                                    network_absorbed_search
                                ),
                                "raw_policy_score": raw_policy_score,
                                "raw_worst_score": raw_worst_score,
                                "raw_city_captures_per_game": (
                                    raw_city_capture_rate
                                ),
                                "raw_completion_rate": raw_completion_rate,
                                "training": rung_summary,
                                "gate": rung_gate,
                                "absorption_gate": absorption_gate,
                            }
                            population_summary["mcts_ladder"].append(rung_record)
                            run.log(
                                round=round_index,
                                mcts_rung=rung_index + 1,
                                mcts_simulations=simulations,
                                mcts_rung_score=rung_score,
                                mcts_raw_policy_score=raw_policy_score,
                                mcts_network_absorbed_search=float(
                                    network_absorbed_search
                                ),
                                mcts_rung_accepted=float(accepted),
                            )
                            print(
                                f"league MCTS rung {simulations} | "
                                f"search={rung_score:.4f} | "
                                f"raw={raw_policy_score:.4f} | "
                                f"absorbed={int(network_absorbed_search)} | "
                                f"accepted={int(accepted)}",
                                flush=True,
                            )
                            if not accepted:
                                break
                    expected_mcts_rungs = sum(
                        int(rung_games > 0)
                        for rung_games in population_game_allocations
                    )
                    completed_mcts_curriculum = (
                        not bool(population_games)
                        or int(population_summary["mcts_ladder_accepted"])
                        == expected_mcts_rungs
                    )
                    population_summary["mcts_ladder_required"] = float(
                        expected_mcts_rungs
                    )
                    population_summary["mcts_ready_for_promotion_gate"] = float(
                        completed_mcts_curriculum
                    )
                    round_training_summary["population"] = population_summary
                    database.flush()
                    last_training = dict(round_training_summary)

                    if not completed_mcts_curriculum:
                        # A failed MCTS generation is not evidence for a
                        # useful challenger.  The rung has already restored
                        # ``best_mcts_state``; discard the candidate instead
                        # of serialising that rollback as a misleading new
                        # challenger for the next generation.
                        database.finalise_candidate(
                            candidate_id, promoted_path=None
                        )
                        challenger_path = None
                        challenger_id = None
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        mcts_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 1.0,
                            "direct_population_ready": 1.0,
                            "mcts_started": 1.0,
                            "mcts_ready_for_promotion_gate": 0.0,
                            "mcts_ladder_accepted": float(
                                population_summary["mcts_ladder_accepted"]
                            ),
                            "mcts_ladder_required": float(
                                expected_mcts_rungs
                            ),
                            "retained_challenger": 0.0,
                            "reason": "mcts_curriculum_incomplete",
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            mcts_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            mcts_ready_for_promotion_gate=0.0,
                            retained_challenger=0.0,
                            promoted=0.0,
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league MCTS curriculum incomplete | final gate "
                            "skipped | accepted="
                            f"{int(population_summary['mcts_ladder_accepted'])}/"
                            f"{expected_mcts_rungs}",
                            flush=True,
                        )
                        del candidate, champion, frozen_models
                        continue

                    evaluation_results: list[Dict[str, Any]] = []
                    invalid_evaluation = False
                    for opponent_index, (
                        opponent_id,
                        opponent_path,
                        opponent_model,
                        probability,
                    ) in enumerate(zip(
                        population_ids, population_paths, frozen_models, probabilities
                    )):
                        is_champion = opponent_id == champion_id
                        games = promotion_games if is_champion else historical_games
                        purpose = "champion_gate" if is_champion else "historical_gate"
                        match_seed = _offset_seed(
                            round_evaluation_seed,
                            opponent_index * 1_000_003,
                        )
                        evaluation_id = database.start_evaluation(
                            round_id,
                            candidate_id,
                            opponent_id,
                            purpose,
                            match_seed,
                            games,
                        )

                        def record_evaluation_game(
                            game: Mapping[str, Any],
                            *,
                            selected_evaluation_id: int = evaluation_id,
                            selected_opponent_id: int = opponent_id,
                        ) -> None:
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="evaluation",
                                model_a_id=candidate_id,
                                model_b_id=selected_opponent_id,
                                game=game,
                                evaluation_id=selected_evaluation_id,
                            )

                        match = evaluate_matches_pool(
                            candidate,
                            opponent_model,
                            replace(training, seed=match_seed),
                            evaluation_mcts,
                            games,
                            num_threads=num_threads,
                            max_pending_leaves=max_pending_leaves,
                            visible_action_history=visible_action_history,
                            belief_retries=belief_retries,
                            max_envs=min(num_envs, population_max_envs),
                            game_callback=record_evaluation_game,
                            model_a_label=candidate_evaluation_label,
                            model_b_label=opponent_path.stem,
                        )
                        result = {
                            "opponent_id": opponent_id,
                            "opponent_path": str(opponent_path),
                            "sampling_probability": probability,
                            "purpose": purpose,
                            **match,
                        }
                        evaluation_results.append(result)
                        if not bool(match.get("valid", 1.0)):
                            invalid_evaluation = True
                            database.invalidate_evaluation(evaluation_id, match)
                            run.log(
                                round=round_index,
                                invalid_evaluations=1,
                                invalid_evaluation_envs=match.get("invalid_envs", 0),
                                promoted=0,
                            )
                            print(
                                f"league gate vs {opponent_path.stem} invalid | "
                                f"candidate rejected | reason={match.get('invalid_reason', 'belief recovery exhausted')}",
                                flush=True,
                            )
                            break
                        database.finish_evaluation(evaluation_id, match, elo_k=elo_k)
                        print(
                            f"league gate vs {opponent_path.stem} | "
                            f"score={float(match['model_a_score']):.4f} | "
                            f"lower95={float(match['score_lower_95']):.4f}",
                            flush=True,
                        )

                    if invalid_evaluation:
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        minimum_historical_score = 0.0
                        champion_lower_95 = 0.0
                        promoted = False
                    else:
                        champion_result = next(
                            result for result in evaluation_results
                            if int(result["opponent_id"]) == champion_id
                        )
                        last_candidate_score = float(champion_result["model_a_score"])
                        last_population_score = sum(
                            float(result["sampling_probability"]) * float(result["model_a_score"])
                            for result in evaluation_results
                        )
                        minimum_historical_score = min(
                            float(result["model_a_score"]) for result in evaluation_results
                        )
                        champion_lower_95 = float(champion_result["score_lower_95"])
                        promoted = (
                            last_candidate_score >= promotion_score
                            and champion_lower_95 >= promotion_lower_bound
                            and last_population_score >= population_score
                            and minimum_historical_score >= historical_score_floor
                        )
                    gate_summary = {
                        "promoted": float(promoted),
                        "valid": float(not invalid_evaluation),
                        "invalid_evaluations": float(invalid_evaluation),
                        "candidate_score": last_candidate_score,
                        "champion_lower_95": champion_lower_95,
                        "population_score": last_population_score,
                        "minimum_historical_score": minimum_historical_score,
                        "evaluations": evaluation_results,
                    }
                    print(
                        f"league decision {round_index}/{rounds} | "
                        f"champion={last_candidate_score:.4f} | "
                        f"population={last_population_score:.4f} | "
                        f"worst={minimum_historical_score:.4f} | promoted={int(promoted)}",
                        flush=True,
                    )

                    if promoted:
                        promotions += 1
                        promoted_path = _next_checkpoint_path(checkpoint_dir, checkpoint_prefix)
                        save_model(promoted_path, candidate, model_config)
                        database.finalise_candidate(candidate_id, promoted_path=promoted_path)
                        if champion_mode == "auto":
                            # The accepted absorption ladder is the evidence
                            # that this new checkpoint may itself generate the
                            # next frozen MCTS generation. Persist it on the
                            # promoted child so the mode survives restarts.
                            database.record_mcts_capability(
                                candidate_id,
                                games=mcts_gate_games,
                                simulations=int(population_rungs[-1]),
                                raw_score=best_raw_policy_score,
                                search_score=best_mcts_score,
                                streak=mcts_probe_patience,
                                ready=True,
                                evaluation_active=True,
                                summary={
                                    "source": "auto_mcts_absorption",
                                    "accepted_rungs": float(
                                        population_summary[
                                            "mcts_ladder_accepted"
                                        ]
                                    ),
                                    "required_rungs": float(
                                        expected_mcts_rungs
                                    ),
                                    "raw_policy_score": best_raw_policy_score,
                                    "mcts_score": best_mcts_score,
                                },
                            )
                        champion_path = promoted_path
                        champion_id = candidate_id
                        challenger_path = None
                        challenger_id = None
                        print(
                            f"league promotion {promotions}: saved {champion_path}",
                            flush=True,
                        )
                    else:
                        # A statistically inconclusive 0.50-0.55 challenger is
                        # useful learning progress, not a reason to restart
                        # from the teacher. Keep it only when every completed
                        # held-out gate is non-regressing; the protected
                        # champion and league population remain unchanged.
                        retain_challenger = (
                            not invalid_evaluation
                            and last_candidate_score >= 0.50
                            and last_population_score >= 0.50
                            and minimum_historical_score
                            >= historical_score_floor
                        )
                        if retain_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                            print(
                                "league challenger retained for the next "
                                f"round: {challenger_path}",
                                flush=True,
                            )
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                        gate_summary["retained_challenger"] = float(
                            retain_challenger
                        )

                    round_summary = {
                        "direct": direct_summary,
                        "direct_population": direct_population_summary,
                        "pool": pool_summary,
                        "population": population_summary,
                    }
                    database.finish_round(
                        round_id,
                        champion_id,
                        promoted,
                        round_summary,
                        gate_summary,
                    )
                    run.log(
                        round=round_index,
                        candidate_score=last_candidate_score,
                        population_score=last_population_score,
                        minimum_historical_score=minimum_historical_score,
                        invalid_evaluations=float(invalid_evaluation),
                        promoted=float(promoted),
                        promotions=promotions,
                    )
                    database.flush()
                    del candidate, champion, frozen_models
                except (BeliefQualityError, BeliefRecoveryRequired) as error:
                    if candidate_id is not None:
                        database.finalise_candidate(candidate_id, promoted_path=None)
                    diagnostics = (
                        error.metrics
                        if isinstance(error, BeliefQualityError)
                        else {
                            "belief_recovery_env_ids": error.env_ids.tolist(),
                            "belief_recovery_diagnostics": list(error.diagnostics),
                        }
                    )
                    rejection_summary = {
                        "promoted": 0.0,
                        "valid": 0.0,
                        "belief_quality_failed": 1.0,
                        "error": str(error),
                        "diagnostics": diagnostics,
                    }
                    database.finish_round(
                        round_id,
                        champion_id,
                        False,
                        round_training_summary,
                        rejection_summary,
                    )
                    run.log(
                        round=round_index,
                        promoted=0.0,
                        belief_quality_failed=1.0,
                        promotions=promotions,
                    )
                    database.flush()
                    print(
                        f"league round {round_index} rejected by belief circuit breaker: {error}",
                        flush=True,
                    )
                except BaseException as error:
                    if candidate_id is not None:
                        database.finalise_candidate(candidate_id, promoted_path=None)
                    database.fail_round(round_id, error)
                    raise
                finally:
                    current_round_id[0] = None

            summary: Dict[str, float | str] = {
                "league_rounds": float(rounds),
                "league_promotions": float(promotions),
                "league_last_candidate_score": last_candidate_score,
                "league_last_population_score": last_population_score,
                "league_promotion_threshold": promotion_score,
                "league_population_threshold": population_score,
                "league_historical_score_floor": historical_score_floor,
                "league_champion_path": str(champion_path),
                "league_challenger_path": (
                    str(challenger_path) if challenger_path is not None else ""
                ),
                "league_database_path": str(database_path.resolve()),
                "league_population_size": float(population_size),
                "league_champion_mode": champion_mode,
                "league_mcts_probe_games": float(mcts_probe_games),
                "league_mcts_probe_simulations": float(
                    mcts_probe_simulations
                ),
                "league_champion_anchor_weight": float(champion_anchor_weight),
                "league_champion_value_anchor_weight": float(
                    champion_value_anchor_weight
                ),
                "league_evaluation_seed": float(resolved_evaluation_seed),
                "league_evaluation_seed_stride": float(evaluation_seed_stride),
            }
            if last_training:
                summary["league_last_pool_updates"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_optimisation_updates", 0.0
                    )
                )
                population_rung_rows = last_training.get("population", {}).get(
                    "mcts_ladder", []
                )
                summary["league_last_population_games"] = float(
                    sum(
                        float(row.get("training", {}).get("population_pool_games", 0.0))
                        for row in population_rung_rows
                    )
                )
                summary["league_last_mcts_rungs_accepted"] = float(
                    last_training.get("population", {}).get(
                        "mcts_ladder_accepted", 0.0
                    )
                )
                summary["league_last_mcts_rung_rollbacks"] = float(
                    last_training.get("population", {}).get(
                        "mcts_ladder_rollbacks", 0.0
                    )
                )
                summary["league_last_direct_best_score"] = float(
                    last_training.get("direct", {}).get("direct_best_score", 0.0)
                )
                summary["league_last_direct_ready"] = float(
                    last_training.get("direct", {}).get("direct_ready", 0.0)
                )
                summary["league_last_direct_population_steps"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_steps", 0.0
                    )
                )
                summary["league_last_direct_population_ready"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_ready", 0.0
                    )
                )
                summary["league_last_direct_population_score"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_best_score", 0.0
                    )
                )
                summary["league_last_direct_population_worst"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_best_worst_score", 0.0
                    )
                )
                summary["league_last_mcts_started"] = float(
                    bool(population_rung_rows)
                )
                probe = last_training.get("population", {}).get(
                    "mcts_probe", {}
                )
                summary["league_last_mcts_probe_performed"] = float(
                    probe.get("performed", 0.0)
                )
                summary["league_last_mcts_probe_passed"] = float(
                    probe.get("passed", 0.0)
                )
                summary["league_last_mcts_ready"] = float(
                    probe.get("ready", 0.0)
                )
                summary["league_last_mcts_evaluation_active"] = float(
                    probe.get("evaluation_active", 0.0)
                )
                summary["league_last_pool_anchor_mean_l1"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_anchor_mean_l1", 0.0
                    )
                )
                summary["league_last_population_anchor_mean_l1"] = float(
                    population_rung_rows[-1]
                    .get("training", {})
                    .get("population_pool_anchor_mean_l1", 0.0)
                    if population_rung_rows
                    else 0.0
                )
            run_directory = run.finish(summary)
            summary["run_dir"] = str(run_directory)
            database.finish_run(database_run_id, "complete", summary)
            return summary
        except BaseException as error:
            database.finish_run(
                database_run_id,
                "failed",
                {"error": repr(error), "champion_path": str(champion_path)},
            )
            raise
