"""Population-based self-play league with regression-safe promotion gates."""

from __future__ import annotations

import json
import pickle
import hashlib
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
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
from .replay_buffer import TrainingExample
from .self_play_pool_train import train_self_play_pool
from .vector_population import (
    _flat_pfsp_weights,
    evaluate_direct_population_matches,
    train_direct_population_curriculum,
)


_MAX_SEED = 2**31 - 1
_MAX_CONSECUTIVE_MCTS_BELIEF_REPAIRS = 2


def _updated_mcts_gate_map_schedule(
    current_schedule: Sequence[int],
    evaluation: Mapping[str, Any],
    games: int,
) -> tuple[int, ...]:
    """Keep every MCTS ladder rung on the last valid paired arena.

    Evaluation may replace a pair whose belief reconstruction failed.  The
    replacement is part of the arena, so subsequent search depths must reuse
    it rather than going back to the original seed schedule.
    """
    current = tuple(int(seed) for seed in current_schedule)
    if not bool(evaluation.get("valid", 1.0)):
        return current
    observed = tuple(
        int(seed) for seed in evaluation.get("evaluated_map_seeds", ())
    )
    if len(observed) != int(games):
        raise RuntimeError("MCTS gate evaluation returned an incomplete map schedule")
    if any(
        observed[index] != observed[index + 1]
        for index in range(0, len(observed), 2)
    ):
        raise RuntimeError("MCTS gate evaluation returned an unpaired map schedule")
    return observed


def _checkpoint_fingerprint(path: Path) -> str | None:
    """Return a content identity for a checkpoint, not merely its pathname."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _belief_validation_signature(
    training: TrainingConfig,
    mcts: MCTSConfig,
    *,
    visible_action_history: int,
    belief_retries: int,
    evaluation_seed: int,
    evaluation_games: int,
) -> dict[str, Any]:
    """Describe every setting that changes a belief held-out measurement."""
    return {
        "map_size": int(training.map_size),
        "map_type": str(training.map_type),
        "particles": int(mcts.particles),
        "belief_particle_temperature": float(mcts.belief_particle_temperature),
        "belief_retries": int(belief_retries),
        "native_max_actions": int(mcts.native_max_actions),
        "visible_action_history": int(visible_action_history),
        "max_moves": int(training.max_moves),
        # Exploration changes the actual public-action trajectories sampled
        # during held-out games.  A measurement collected with a mostly raw
        # policy must not certify a model that will be admitted under a much
        # more exploratory data-generation regime.
        "exploration_rate": float(training.exploration_rate),
        "mcts_temperature": float(mcts.temperature),
        "root_dirichlet_epsilon": float(mcts.root_dirichlet_epsilon),
        "root_dirichlet_concentration": float(mcts.root_dirichlet_concentration),
        "temperature_drop_move": int(mcts.temperature_drop_move),
        "late_temperature": float(mcts.late_temperature),
        "evaluation_seed": int(evaluation_seed),
        "evaluation_games": int(evaluation_games),
    }


def _value_bootstrap_required(
    *,
    champion_mode: str,
    belief_calibration_required: bool,
    value_bootstrap_games: int,
    champion_fingerprint: str | None,
    apprentice_state: Mapping[str, Any],
) -> bool:
    """Run the value bootstrap once for each frozen champion checkpoint.

    Bootstrap updates the shared/belief-side heads and therefore has to be
    followed by a fresh held-out calibration.  Its completion is persisted by
    champion content identity, rather than by round number, so ``auto`` can
    subsequently enter MCTS instead of alternating bootstrap/calibration
    forever.  Replacing the champion at the same path intentionally enables
    one bootstrap again because its fingerprint changes.
    """
    return bool(
        champion_mode == "mcts"
        and not belief_calibration_required
        and value_bootstrap_games > 0
        and champion_fingerprint is not None
        and apprentice_state.get("value_bootstrap_champion_fingerprint")
        != champion_fingerprint
    )


def _belief_calibration_can_run(
    *,
    population_games: int,
    replay_updates: int,
    evaluation_games: int,
) -> bool:
    """Whether a required calibration can at least produce held-out proof.

    A bootstrap invalidates the belief marker even when the caller requested
    no fresh collection and no replay optimisation.  In that case the fixed
    held-out arena is still mandatory: it measures the newly saved decoder
    and keeps the round on the calibration exit path, rather than letting the
    outer MCTS path mistake an invalid marker for permission to search.
    """
    return bool(population_games > 0 or replay_updates > 0 or evaluation_games > 0)


def _load_belief_apprentice_state(path: Path) -> Dict[str, Any]:
    """Load the small, non-authoritative calibration marker safely."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _save_belief_apprentice_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(state), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _belief_replay_example_matches_map_size(
    example: TrainingExample,
    map_size: int | None,
) -> bool:
    """Reject replay rows whose board shape cannot match this training run."""
    if map_size is None:
        return True
    expected_tiles = int(map_size) * int(map_size)
    try:
        return bool(
            example.map_tokens.ndim == 2
            and example.full_map_target.ndim == 2
            and example.hidden_mask.ndim == 1
            and int(example.map_tokens.shape[0]) == expected_tiles
            and int(example.full_map_target.shape[0]) == expected_tiles
            and int(example.hidden_mask.shape[0]) == expected_tiles
        )
    except (AttributeError, IndexError, TypeError):
        return False


def _load_belief_calibration_replay(
    path: Path,
    capacity: int,
    *,
    map_size: int | None = None,
) -> list[TrainingExample] | None:
    """Load a bounded local reservoir of supervised fog observations.

    This contains only PolyBot replay examples. It is never an inference
    input; it prevents a short new seed batch from making the belief decoder
    forget the maps seen in prior calibration runs.
    """
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
            if isinstance(raw, Mapping) and raw.get("format") == "belief-replay-v2":
                # V2 stores examples as independent pickle records. Retain a
                # bounded tail while reading, rather than materialising a
                # giant list in RAM before applying ``capacity``.
                window: deque[TrainingExample] = deque(maxlen=capacity)
                while True:
                    try:
                        example = pickle.load(handle)
                    except EOFError:
                        break
                    if (
                        isinstance(example, TrainingExample)
                        and _belief_replay_example_matches_map_size(
                            example, map_size
                        )
                    ):
                        window.append(example)
                return list(window)
    except MemoryError:
        # Legacy replay files were stored as one pickle list. A very large
        # list may not fit in RAM even when the requested live capacity is
        # small, because pickle must materialise it before slicing. ``None``
        # lets the caller choose a new path instead of overwriting the old
        # artifact with an empty reservoir.
        return None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
        return []
    if not isinstance(raw, list):
        return []
    return [
        example
        for example in raw
        if isinstance(example, TrainingExample)
        and _belief_replay_example_matches_map_size(example, map_size)
    ][-capacity:]


def _save_belief_calibration_replay(
    path: Path,
    examples: Sequence[TrainingExample],
    capacity: int,
) -> None:
    """Atomically retain the most recent, diverse calibration window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"format": "belief-replay-v2"}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for example in list(examples)[-capacity:]:
            pickle.dump(example, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


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


def _mcts_preflight_diagnostics(
    raw: Mapping[str, Any],
    search: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a preflight without turning an invalid result into a loss.

    A score from an incomplete paired arena is useful for debugging, but it is
    not evidence that search is worse than raw policy.  Keep the distinction
    explicit in both run artifacts and the human-readable console line.
    """
    raw_valid = bool(raw.get("valid", 1.0))
    search_valid = bool(search.get("valid", 1.0))
    raw_completed = int(raw.get("completed_games", raw.get("games", 0)))
    search_completed = int(search.get("completed_games", search.get("games", 0)))
    raw_planned = int(raw.get("planned_games", raw.get("games", 0)))
    search_planned = int(search.get("planned_games", search.get("games", 0)))
    if not search_valid:
        status = "search_evaluation_invalid"
        reason = str(search.get("invalid_reason", "search evaluation is invalid"))
    elif not raw_valid:
        status = "raw_evaluation_invalid"
        reason = str(raw.get("invalid_reason", "raw evaluation is invalid"))
    else:
        status = "valid"
        reason = ""
    return {
        "status": status,
        "reason": reason,
        "raw_valid": float(raw_valid),
        "search_valid": float(search_valid),
        "raw_completed_games": float(raw_completed),
        "raw_planned_games": float(raw_planned),
        "search_completed_games": float(search_completed),
        "search_planned_games": float(search_planned),
    }


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
    flat_enabled: bool = False,
) -> Dict[str, float | int | str | bool]:
    """Choose either the authoritative MCTS or explicit flat curriculum.

    The direct teacher, objective, flat actor-critic and raw-population PPO
    curricula remain available as standalone tools for a new model.  They are
    deliberately not part of ``train_league``: an existing checkpoint must
    receive policy targets from MCTS visit counts and value targets from game
    outcomes, rather than repeatedly being pulled toward a shaped local
    reward.  Keeping the parameters accepts old command lines without letting
    a legacy bootstrap silently block MCTS collection.
    """
    if flat_enabled:
        # ``train_flat.py`` is an intentionally separate Stage-0 workflow.
        # It needs the old direct VectorGameEnv curriculum, but must never
        # leak that curriculum into a normal MCTS league run.
        post_bootstrap = champion_status == "promoted"
        if post_bootstrap:
            population_steps = post_bootstrap_population_steps
            population_max_steps = post_bootstrap_population_max_steps
            policy_anchor_weight = post_bootstrap_policy_anchor_weight
            population_teacher_weight = post_bootstrap_teacher_weight
            population_opponent_policy_weight = (
                post_bootstrap_opponent_policy_weight
            )
            objective_steps = post_bootstrap_objective_steps
        else:
            population_steps = direct_population_steps
            population_max_steps = direct_population_max_steps
            policy_anchor_weight = direct_policy_anchor_weight
            population_teacher_weight = direct_population_teacher_weight
            population_opponent_policy_weight = (
                direct_population_opponent_policy_weight
            )
            objective_steps = direct_objective_steps
        return {
            "phase": "flat_population",
            "post_bootstrap": post_bootstrap,
            "teacher_steps": direct_teacher_steps,
            "objective_steps": objective_steps,
            "actor_steps": direct_actor_steps,
            "actor_max_steps": direct_actor_max_steps,
            "population_steps": population_steps,
            "population_max_steps": population_max_steps,
            "policy_anchor_weight": policy_anchor_weight,
            "population_teacher_weight": population_teacher_weight,
            "population_opponent_policy_weight": population_opponent_policy_weight,
        }

    # Parameters remain for backwards-compatible callers and run metadata,
    # but a normal league loop never executes their raw-policy stages.
    _ = (
        direct_teacher_steps, direct_objective_steps, direct_actor_steps,
        direct_actor_max_steps, direct_population_steps,
        direct_population_max_steps, direct_policy_anchor_weight,
        direct_population_teacher_weight,
        direct_population_opponent_policy_weight,
        post_bootstrap_population_steps, post_bootstrap_population_max_steps,
        post_bootstrap_policy_anchor_weight, post_bootstrap_teacher_weight,
        post_bootstrap_opponent_policy_weight, refresh_promoted_curriculum,
    )
    return {
        "phase": "mcts_self_play",
        "post_bootstrap": champion_status == "promoted",
        "teacher_steps": 0,
        "objective_steps": 0,
        "actor_steps": 0,
        "actor_max_steps": 0,
        "population_steps": 0,
        "population_max_steps": 0,
        "policy_anchor_weight": 0.0,
        "population_teacher_weight": 0.0,
        "population_opponent_policy_weight": 0.0,
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
    population_optimisation_block_updates: int,
    population_optimisation_screen_games: int,
    population_optimisation_screen_max_regression: float,
    population_workers: int,
    population_max_envs: int,
    population_mcts_critical_only: bool,
    population_mcts_critical_fraction: float,
    population_mcts_critical_minimum_score: float,
    population_mcts_critical_replay_fraction: float,
    population_selective_mcts_fraction: float,
    population_selective_mcts_minimum_envs: int,
    population_selective_mcts_audit_fraction: float,
    population_mcts_reanalyse_capacity: int,
    population_mcts_reanalyse_max_bytes: int,
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
    if population_optimisation_block_updates < 1:
        raise ValueError("population_optimisation_block_updates must be positive")
    if (
        population_optimisation_screen_games < 0
        or population_optimisation_screen_games % 2
    ):
        raise ValueError(
            "population_optimisation_screen_games must be non-negative and even"
        )
    if population_optimisation_screen_max_regression < 0.0:
        raise ValueError(
            "population_optimisation_screen_max_regression must be non-negative"
        )
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
    if not 0.0 < population_selective_mcts_fraction <= 1.0:
        raise ValueError("population_selective_mcts_fraction must be in (0, 1]")
    if population_selective_mcts_minimum_envs < 0:
        raise ValueError("population_selective_mcts_minimum_envs must not be negative")
    if not 0.0 <= population_selective_mcts_audit_fraction <= 1.0:
        raise ValueError(
            "population_selective_mcts_audit_fraction must be in [0, 1]"
        )
    if population_mcts_reanalyse_capacity < 1:
        raise ValueError("population_mcts_reanalyse_capacity must be positive")
    if population_mcts_reanalyse_max_bytes < 1:
        raise ValueError("population_mcts_reanalyse_max_bytes must be positive")
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
    training_stage: str = "auto",
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
    belief_apprentice_path: Path | None = None,
    belief_action_equivalence_target: float = 0.10,
    belief_action_equivalence_min_particles: int = 512,
    belief_calibration_eval_games: int = 8,
    belief_calibration_replay_path: Path | None = None,
    belief_calibration_replay_capacity: int = 50_000,
    belief_replay_updates: int = 512,
    database_path: Path | None = None,
    population_model_paths: Sequence[Path] = (),
    population_size: int = 6,
    population_games: int = 0,
    population_updates_per_game: int = 0,
    population_optimisation_block_updates: int = 32,
    population_optimisation_screen_games: int = 4,
    population_optimisation_screen_max_regression: float = 0.10,
    population_workers: int = 16,
    population_max_envs: int = 16,
    population_offline_collection: bool = True,
    population_belief_search_mode: str = "strict",
    population_mcts_critical_only: bool = False,
    population_mcts_critical_fraction: float = 0.25,
    population_mcts_critical_minimum_score: float = 0.25,
    population_mcts_critical_replay_fraction: float = 1.0,
    population_selective_mcts: bool = True,
    population_selective_mcts_fraction: float = 0.50,
    population_selective_mcts_minimum_envs: int = 2,
    population_selective_mcts_audit_fraction: float = 0.10,
    population_mcts_reanalyse_path: Path | None = None,
    population_mcts_reanalyse_capacity: int = 4_096,
    population_mcts_reanalyse_max_bytes: int = 64 * 1024**2,
    population_mcts_reanalyse_positions: int = 8,
    population_mcts_reanalyse_replay_fraction: float = 0.50,
    value_bootstrap_games: int = 64,
    value_bootstrap_updates_per_game: int = 8,
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
    flat_population_persistent_optimizer: bool = False,
    flat_population_terminal_replay_capacity: int = 0,
    flat_population_terminal_replay_batch_size: int = 0,
    flat_population_terminal_replay_interval: int = 1,
    flat_population_terminal_replay_value_weight: float = 0.0,
    flat_population_ppo_max_policy_age: int | None = None,
    flat_population_policy_anchor_final_weight: float | None = None,
    flat_population_snapshot_archive_size: int = 0,
    flat_population_pfsp_alpha: float = 2.0,
    flat_population_pfsp_probability_floor: float = 0.05,
    flat_population_pfsp_score_ema_decay: float = 0.80,
    flat_main_exploiter_probability: float = 0.70,
    flat_main_exploiter_archive_score: float = 0.55,
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
        population_optimisation_block_updates=(
            population_optimisation_block_updates
        ),
        population_optimisation_screen_games=(
            population_optimisation_screen_games
        ),
        population_optimisation_screen_max_regression=(
            population_optimisation_screen_max_regression
        ),
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
        population_selective_mcts_fraction=population_selective_mcts_fraction,
        population_selective_mcts_minimum_envs=(
            population_selective_mcts_minimum_envs
        ),
        population_selective_mcts_audit_fraction=(
            population_selective_mcts_audit_fraction
        ),
        population_mcts_reanalyse_capacity=population_mcts_reanalyse_capacity,
        population_mcts_reanalyse_max_bytes=population_mcts_reanalyse_max_bytes,
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
    if training_stage not in {"auto", "flat", "belief", "mcts"}:
        raise ValueError("training_stage must be 'auto', 'flat', 'belief' or 'mcts'")
    if training_stage in {"belief", "mcts"} and champion_mode != "mcts":
        raise ValueError(
            "--training-stage belief/mcts requires --champion-mode mcts"
        )
    if training_stage == "flat" and champion_mode != "flat":
        raise ValueError("--training-stage flat requires --champion-mode flat")
    if training_stage == "flat" and population_games:
        raise ValueError(
            "flat-only training must use population_games=0; use direct-population PPO instead"
        )
    if training_stage == "flat" and value_bootstrap_games:
        raise ValueError("flat-only training must use value_bootstrap_games=0")
    if training_stage == "flat" and flat_mcts_distill_games:
        raise ValueError("flat-only training must use flat_mcts_distill_games=0")
    if training_stage == "flat" and mcts_probe_games:
        raise ValueError("flat-only training must use mcts_probe_games=0")
    if training_stage == "mcts" and value_bootstrap_games:
        raise ValueError(
            "MCTS-only stage must use value_bootstrap_games=0; run value/belief "
            "calibration in the separate belief stage"
        )
    if training_stage != "auto" and (steps or updates):
        raise ValueError(
            "explicit flat/belief/mcts stages require steps=0 and updates=0; use "
            "the stage-specific population/direct options instead"
        )
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
    if value_bootstrap_games < 0 or value_bootstrap_games % 2:
        raise ValueError("value_bootstrap_games must be non-negative and even")
    if value_bootstrap_games and value_bootstrap_updates_per_game < 1:
        raise ValueError(
            "value_bootstrap_updates_per_game must be positive when bootstrap is enabled"
        )
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
    if not 0.0 <= flat_population_pfsp_score_ema_decay < 1.0:
        raise ValueError("flat PFSP score EMA decay must be in [0, 1)")
    if not 0.0 <= flat_main_exploiter_probability <= 1.0:
        raise ValueError("flat main-exploiter probability must be in [0, 1]")
    if not 0.0 <= flat_main_exploiter_archive_score <= 1.0:
        raise ValueError("flat main-exploiter archive score must be in [0, 1]")
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
    if not 0.0 <= belief_action_equivalence_target <= 1.0:
        raise ValueError("belief_action_equivalence_target must be between zero and one")
    if belief_action_equivalence_min_particles < 1:
        raise ValueError("belief_action_equivalence_min_particles must be positive")
    if belief_calibration_eval_games < 0:
        raise ValueError("belief_calibration_eval_games must not be negative")
    if belief_calibration_replay_capacity < 1:
        raise ValueError("belief_calibration_replay_capacity must be positive")
    if belief_replay_updates < 0:
        raise ValueError("belief_replay_updates must not be negative")
    if population_belief_search_mode not in {"strict", "root_shield"}:
        raise ValueError(
            "population_belief_search_mode must be 'strict' or 'root_shield'"
        )
    ladder = tuple(int(value) for value in mcts_simulation_ladder)
    if any(value < 1 for value in ladder) or tuple(sorted(set(ladder))) != ladder:
        raise ValueError("mcts_simulation_ladder must be strictly increasing")
    # Belief calibration is raw-policy data collection, not an MCTS
    # curriculum.  Splitting its game budget across the search ladder and
    # breaking after the first rung silently discarded half the requested
    # examples (96 requested games became 48 with the default 32,64 ladder).
    population_rungs = (
        (int(mcts.simulations),)
        if training_stage == "belief" and population_games
        else ladder or ((int(mcts.simulations),) if population_games else ())
    )
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
    belief_apprentice_path = (
        belief_apprentice_path
        if belief_apprentice_path is not None
        else checkpoint_dir / f"{checkpoint_prefix}-belief-apprentice.pt"
    )
    belief_apprentice_state_path = belief_apprentice_path.with_suffix(
        belief_apprentice_path.suffix + ".json"
    )
    belief_calibration_replay_path = (
        belief_calibration_replay_path
        if belief_calibration_replay_path is not None
        else checkpoint_dir / f"{checkpoint_prefix}-belief-calibration-replay.pkl"
    )
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
        "population_belief_search_mode": population_belief_search_mode,
        "population_mcts_critical_only": population_mcts_critical_only,
        "population_mcts_critical_fraction": population_mcts_critical_fraction,
        "population_mcts_critical_minimum_score": (
            population_mcts_critical_minimum_score
        ),
        "population_mcts_critical_replay_fraction": (
            population_mcts_critical_replay_fraction
        ),
        "population_selective_mcts": population_selective_mcts,
        "population_selective_mcts_fraction": population_selective_mcts_fraction,
        "population_selective_mcts_minimum_envs": (
            population_selective_mcts_minimum_envs
        ),
        "population_selective_mcts_audit_fraction": (
            population_selective_mcts_audit_fraction
        ),
        "population_mcts_reanalyse_path": population_mcts_reanalyse_path,
        "population_mcts_reanalyse_capacity": population_mcts_reanalyse_capacity,
        "population_mcts_reanalyse_positions": population_mcts_reanalyse_positions,
        "population_mcts_reanalyse_replay_fraction": (
            population_mcts_reanalyse_replay_fraction
        ),
        "belief_calibration_eval_games": belief_calibration_eval_games,
        "belief_calibration_replay_path": str(belief_calibration_replay_path),
        "belief_calibration_replay_capacity": belief_calibration_replay_capacity,
        "belief_replay_updates": belief_replay_updates,
        "value_bootstrap_games": value_bootstrap_games,
        "value_bootstrap_updates_per_game": value_bootstrap_updates_per_game,
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
        "flat_population_persistent_optimizer": flat_population_persistent_optimizer,
        "flat_population_terminal_replay_capacity": flat_population_terminal_replay_capacity,
        "flat_population_terminal_replay_batch_size": flat_population_terminal_replay_batch_size,
        "flat_population_terminal_replay_interval": flat_population_terminal_replay_interval,
        "flat_population_terminal_replay_value_weight": flat_population_terminal_replay_value_weight,
        "flat_population_ppo_max_policy_age": flat_population_ppo_max_policy_age,
        "flat_population_policy_anchor_final_weight": flat_population_policy_anchor_final_weight,
        "flat_population_snapshot_archive_size": flat_population_snapshot_archive_size,
        "flat_population_pfsp_alpha": flat_population_pfsp_alpha,
        "flat_population_pfsp_probability_floor": flat_population_pfsp_probability_floor,
        "flat_population_pfsp_score_ema_decay": flat_population_pfsp_score_ema_decay,
        "flat_main_exploiter_probability": flat_main_exploiter_probability,
        "flat_main_exploiter_archive_score": flat_main_exploiter_archive_score,
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
        "training_stage": training_stage,
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
                    # The normal curriculum chooses this automatically.  The
                    # explicit modes deliberately make the two optimisation
                    # problems independently runnable: no policy/value update
                    # can occur in a belief job, and no fresh belief
                    # calibration can be silently mixed into an MCTS job.
                    if training_stage in {"belief", "mcts"}:
                        round_champion_mode = "mcts"
                    elif training_stage == "flat":
                        round_champion_mode = "flat"
                    round_training_summary["champion_mode"] = {
                        "requested": champion_mode,
                        "effective": round_champion_mode,
                        "training_stage": training_stage,
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
                    # MCTS still evaluates every update against the frozen
                    # champion, but a challenger which cleared the retention
                    # gate must remain the starting point for the next round.
                    # Otherwise a multi-round MCTS run silently performs one
                    # update per fresh apprentice and never accumulates its
                    # accepted learning.
                    belief_apprentice_state = _load_belief_apprentice_state(
                        belief_apprentice_state_path
                    )
                    champion_fingerprint = _checkpoint_fingerprint(champion_path)
                    apprentice_fingerprint = _checkpoint_fingerprint(
                        belief_apprentice_path
                    )
                    belief_validation_signature = _belief_validation_signature(
                        round_training,
                        mcts,
                        visible_action_history=visible_action_history,
                        belief_retries=belief_retries,
                        evaluation_seed=_offset_seed(
                            resolved_evaluation_seed, 17_000_003
                        ),
                        evaluation_games=belief_calibration_eval_games,
                    )
                    # Checkpoint provenance and measurement validity are
                    # separate: changing validation settings must remeasure
                    # the trained apprentice rather than discard its weights.
                    apprentice_matches_champion = (
                        belief_apprentice_path.exists()
                        and belief_apprentice_state.get("champion_path")
                        == str(champion_path.resolve())
                        and champion_fingerprint is not None
                        and belief_apprentice_state.get("champion_fingerprint")
                        == champion_fingerprint
                        and belief_apprentice_state.get("apprentice_fingerprint")
                        == apprentice_fingerprint
                    )
                    apprentice_validation_matches = (
                        apprentice_matches_champion
                        and int(belief_apprentice_state.get("held_out_games", 0)) > 0
                        and belief_apprentice_state.get("validation_signature")
                        == belief_validation_signature
                    )
                    apprentice_rejection_rate = float(
                        belief_apprentice_state.get("rejection_rate", 1.0)
                    )
                    apprentice_checked_particles = int(
                        belief_apprentice_state.get("checked_particles", 0)
                    )
                    # A marker created under a loose old threshold must not
                    # bypass a later, stricter calibration request. Conversely
                    # a checkpoint measured at 2.7% must be eligible when the
                    # user deliberately selects a safe 4% operational gate;
                    # the old stored ``ready`` bit only described the old
                    # threshold and is not an independent measurement.
                    apprentice_meets_current_belief_gate = (
                        apprentice_validation_matches
                        and apprentice_checked_particles
                        >= belief_action_equivalence_min_particles
                        and apprentice_rejection_rate
                        <= belief_action_equivalence_target
                    )
                    # A retained challenger has different weights from the
                    # belief apprentice, even when both descend from the same
                    # champion.  Its own checkpoint therefore needs its own
                    # fingerprinted held-out marker; an apprentice result
                    # must never authorise it to enter MCTS.
                    continue_mcts_challenger = (
                        round_champion_mode == "mcts"
                        and previous_challenger_path is not None
                        and previous_challenger_id is not None
                    )
                    challenger_belief_state: Dict[str, Any] = {}
                    challenger_matches_champion = False
                    challenger_validation_matches = False
                    challenger_checked_particles = 0
                    challenger_rejection_rate = 1.0
                    challenger_meets_current_belief_gate = False
                    if continue_mcts_challenger:
                        assert previous_challenger_path is not None
                        challenger_state_path = previous_challenger_path.with_suffix(
                            previous_challenger_path.suffix + ".json"
                        )
                        challenger_belief_state = _load_belief_apprentice_state(
                            challenger_state_path
                        )
                        challenger_fingerprint = _checkpoint_fingerprint(
                            previous_challenger_path
                        )
                        challenger_matches_champion = (
                            challenger_belief_state.get("champion_path")
                            == str(champion_path.resolve())
                            and champion_fingerprint is not None
                            and challenger_belief_state.get(
                                "champion_fingerprint"
                            ) == champion_fingerprint
                            and challenger_belief_state.get(
                                "apprentice_fingerprint"
                            ) == challenger_fingerprint
                        )
                        challenger_validation_matches = (
                            challenger_matches_champion
                            and int(
                                challenger_belief_state.get(
                                    "held_out_games", 0
                                )
                            ) > 0
                            and challenger_belief_state.get(
                                "validation_signature"
                            ) == belief_validation_signature
                        )
                        challenger_checked_particles = int(
                            challenger_belief_state.get(
                                "checked_particles", 0
                            )
                        )
                        challenger_rejection_rate = float(
                            challenger_belief_state.get(
                                "rejection_rate", 1.0
                            )
                        )
                        challenger_meets_current_belief_gate = (
                            challenger_validation_matches
                            and challenger_checked_particles
                            >= belief_action_equivalence_min_particles
                            and challenger_rejection_rate
                            <= belief_action_equivalence_target
                        )
                    active_belief_matches_champion = (
                        challenger_matches_champion
                        if continue_mcts_challenger
                        else apprentice_matches_champion
                    )
                    active_belief_validation_matches = (
                        challenger_validation_matches
                        if continue_mcts_challenger
                        else apprentice_validation_matches
                    )
                    active_belief_meets_current_gate = (
                        challenger_meets_current_belief_gate
                        if continue_mcts_challenger
                        else apprentice_meets_current_belief_gate
                    )
                    active_belief_checked_particles = (
                        challenger_checked_particles
                        if continue_mcts_challenger
                        else apprentice_checked_particles
                    )
                    active_belief_rejection_rate = (
                        challenger_rejection_rate
                        if continue_mcts_challenger
                        else apprentice_rejection_rate
                    )
                    # An MCTS-only invalid evaluation is evidence about the
                    # hidden-world path, not a defeat of the raw policy.  A
                    # fresh apprentice can safely return to supervised belief
                    # calibration on the next auto round.  A retained MCTS
                    # challenger has its own checkpoint/marker lifecycle, so
                    # we report it for diagnosis instead of silently writing
                    # calibration output into the apprentice path.
                    mcts_repair_required = bool(
                        not continue_mcts_challenger
                        and belief_apprentice_state.get(
                            "mcts_belief_repair_required", False
                        )
                    )
                    mcts_repair_attempts = int(
                        belief_apprentice_state.get(
                            "mcts_belief_repair_attempts", 0
                        )
                    )
                    if (
                        training_stage == "auto"
                        and mcts_repair_required
                        and mcts_repair_attempts
                        >= _MAX_CONSECUTIVE_MCTS_BELIEF_REPAIRS
                    ):
                        raise BeliefQualityError(
                            "MCTS auto-recovery stopped after repeated invalid "
                            "belief evaluations; inspect the stored "
                            "mcts_belief_repair_reason before retrying.",
                            {
                                "attempts": float(mcts_repair_attempts),
                                "last_reason": str(
                                    belief_apprentice_state.get(
                                        "mcts_belief_repair_reason", ""
                                    )
                                ),
                            },
                        )
                    round_training_summary["belief_checkpoint"] = {
                        "path": str(
                            previous_challenger_path
                            if continue_mcts_challenger
                            else belief_apprentice_path
                        ),
                        "is_retained_challenger": float(
                            continue_mcts_challenger
                        ),
                        "matches_champion": float(
                            active_belief_matches_champion
                        ),
                        "ready": float(active_belief_meets_current_gate),
                        "checked_particles": float(
                            active_belief_checked_particles
                        ),
                        "rejection_rate": active_belief_rejection_rate,
                        "mcts_belief_repair_required": float(
                            mcts_repair_required
                        ),
                        "mcts_belief_repair_attempts": float(
                            mcts_repair_attempts
                        ),
                    }
                    bootstrap_champion_fingerprint = (
                        belief_apprentice_state.get(
                            "value_bootstrap_champion_fingerprint"
                        )
                        if belief_apprentice_state.get(
                            "value_bootstrap_champion_fingerprint"
                        ) == champion_fingerprint
                        else None
                    )
                    if training_stage == "belief":
                        belief_calibration_required = True
                    elif training_stage == "mcts":
                        if not (
                            active_belief_matches_champion
                            and active_belief_meets_current_gate
                        ):
                            raise BeliefQualityError(
                                "MCTS-only training requires an accepted belief "
                                "checkpoint for this exact champion. Run the belief "
                                "stage until held-out action mismatch reaches the gate.",
                                {
                                    "belief_checkpoint_path": str(
                                        previous_challenger_path
                                        if continue_mcts_challenger
                                        else belief_apprentice_path
                                    ),
                                    "checkpoint_matches_champion": float(
                                        active_belief_matches_champion
                                    ),
                                    "checked_particles": float(
                                        active_belief_checked_particles
                                    ),
                                    "rejection_rate": active_belief_rejection_rate,
                                    "target_rejection_rate": (
                                        belief_action_equivalence_target
                                    ),
                                },
                            )
                        belief_calibration_required = False
                    else:
                        belief_calibration_required = (
                            round_champion_mode == "mcts"
                            and (
                                mcts_repair_required
                                or not (
                                    active_belief_matches_champion
                                    and active_belief_meets_current_gate
                                )
                            )
                        )
                    if (
                        belief_calibration_required
                        and belief_calibration_eval_games < 1
                    ):
                        raise ValueError(
                            "belief calibration requires at least one held-out "
                            "evaluation game after optimisation"
                        )
                    fresh_branch_round = (
                        direct_population_branch_count > 1
                        or round_champion_mode == "mcts"
                    )
                    candidate_source = (
                        previous_challenger_path
                        if continue_mcts_challenger
                        else belief_apprentice_path
                        if round_champion_mode == "mcts" and apprentice_matches_champion
                        else (
                            champion_path
                            if fresh_branch_round
                            else previous_challenger_path or champion_path
                        )
                    )
                    candidate_parent_id = (
                        previous_challenger_id
                        if continue_mcts_challenger
                        else champion_id
                        if fresh_branch_round
                        else previous_challenger_id or champion_id
                    )
                    candidate_source_kind = (
                        "challenger"
                        if continue_mcts_challenger
                        else "belief-apprentice"
                        if candidate_source == belief_apprentice_path
                        else (
                            "challenger"
                            if (
                                previous_challenger_path is not None
                                and not fresh_branch_round
                            )
                            else "champion"
                        )
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

                    population_rows = database.select_population(
                        champion_id,
                        population_size,
                        include_archived_counters=(training_stage == "flat"),
                    )
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
                        flat_enabled=training_stage == "flat",
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
                    # Belief calibration has its own fixed post-update arena.
                    # Include it in the held-out reservation now, otherwise a
                    # collection seed can generate precisely the maps later
                    # used to decide whether the decoder generalised.
                    belief_validation_seed = _offset_seed(
                        resolved_evaluation_seed, 17_000_003
                    )
                    evaluation_map_seeds.update(
                        belief_validation_seed + game_index
                        for game_index in range(belief_calibration_eval_games)
                    )
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
                    # earlier rung.  If recovery replaces an incompatible
                    # pair, the schedule below is updated with that exact
                    # replacement and carried into every later rung.
                    mcts_gate_seed = _offset_seed(
                        round_evaluation_seed, 400_000_007
                    )
                    mcts_gate_map_seeds = tuple(
                        mcts_gate_seed + game_index // 2
                        for game_index in range(mcts_gate_games)
                    )
                    evaluation_map_seeds.update(mcts_gate_map_seeds)
                    known_training_map_seeds: set[int] = set()
                    # Bootstrap is collection too. Reserve its complete seed
                    # range before checking train/evaluation disjointness.
                    bootstrap_will_collect = _value_bootstrap_required(
                        champion_mode=round_champion_mode,
                        belief_calibration_required=belief_calibration_required,
                        value_bootstrap_games=value_bootstrap_games,
                        champion_fingerprint=champion_fingerprint,
                        apprentice_state=belief_apprentice_state,
                    )
                    if bootstrap_will_collect:
                        bootstrap_seed = _offset_seed(
                            population_training_seed, 1_000_003
                        )
                        known_training_map_seeds.update(
                            bootstrap_seed + game_index
                            for game_index in range(value_bootstrap_games)
                        )
                    if belief_calibration_required and population_games:
                        known_training_map_seeds.update(
                            population_training_seed + game_index
                            for game_index in range(population_games)
                        )
                    else:
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
                    if curriculum_phase == "mcts_self_play":
                        print(
                            "league curriculum | MCTS self-play | "
                            "legacy direct stages skipped",
                            flush=True,
                        )
                    elif curriculum_phase == "flat_population":
                        print(
                            "league curriculum | flat VectorGameEnv PPO/PPG | "
                            f"teacher={effective_teacher_steps} | "
                            f"objectives={effective_objective_steps} | "
                            f"actor={effective_actor_steps} | "
                            f"population={effective_population_steps}-"
                            f"{effective_population_max_steps} steps",
                            flush=True,
                        )
                    elif curriculum_phase == "champion_population":
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

                    population_id_set = set(population_ids)

                    def recorded_population_opponent_id(
                        game: Mapping[str, Any], fallback_index: int = 0
                    ) -> int | None:
                        """Map a callback's stable opponent identity to DB ID.

                        Flat curriculum may temporarily add in-memory
                        snapshots. They retain their string identity in the
                        match JSON, but are not database models until they are
                        archived.  Never reinterpret one as whichever model
                        now occupies the same list offset.
                        """
                        stable_id = game.get("opponent_id")
                        if isinstance(stable_id, int) and stable_id in population_id_set:
                            return stable_id
                        if stable_id is None and 0 <= fallback_index < len(population_ids):
                            return population_ids[fallback_index]
                        return None

                    # The free exploiter starts from exactly the same point as
                    # the main line, before the latter's champion-anchored PPO
                    # updates.  It is evaluated on the same development arena
                    # and can only earn the later independent promotion gate.
                    exploiter_source_state = (
                        model_state_on_cpu(candidate)
                        if (
                            training_stage == "flat"
                            and breakthrough_exploiter
                            and effective_population_steps
                        )
                        else None
                    )
                    direct_population_summary: Dict[str, Any] = {}
                    direct_population_main_summary: Dict[str, Any] = {}
                    direct_population_exploiter_summary: Dict[str, Any] = {}
                    main_exploiter_archive_summary: Dict[str, Any] = {
                        "enabled": float(exploiter_source_state is not None),
                        "main_score": 0.0,
                        "eligible": 0.0,
                        "archived": 0.0,
                    }
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
                                model_b_id=recorded_population_opponent_id(
                                    row, opponent_index
                                ),
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
                                model_b_id=recorded_population_opponent_id(
                                    row, opponent_index
                                ),
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
                                    model_b_id=recorded_population_opponent_id(
                                        decorated, opponent_index
                                    ),
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
                                    model_b_id=recorded_population_opponent_id(
                                        decorated, opponent_index
                                    ),
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
                                flat_persistent_optimizer=(
                                    training_stage == "flat"
                                    and flat_population_persistent_optimizer
                                ),
                                flat_terminal_replay_capacity=(
                                    flat_population_terminal_replay_capacity
                                    if training_stage == "flat" else 0
                                ),
                                flat_terminal_replay_batch_size=(
                                    flat_population_terminal_replay_batch_size
                                    if training_stage == "flat" else 0
                                ),
                                flat_terminal_replay_interval=(
                                    flat_population_terminal_replay_interval
                                ),
                                flat_terminal_replay_value_weight=(
                                    flat_population_terminal_replay_value_weight
                                    if training_stage == "flat" else 0.0
                                ),
                                flat_ppo_max_policy_age=(
                                    flat_population_ppo_max_policy_age
                                    if training_stage == "flat" else None
                                ),
                                flat_policy_anchor_final_weight=(
                                    flat_population_policy_anchor_final_weight
                                    if training_stage == "flat" else None
                                ),
                                flat_snapshot_archive_size=(
                                    flat_population_snapshot_archive_size
                                    if training_stage == "flat" else 0
                                ),
                                flat_pfsp_alpha=flat_population_pfsp_alpha,
                                flat_pfsp_probability_floor=(
                                    flat_population_pfsp_probability_floor
                                ),
                                flat_pfsp_score_ema_decay=(
                                    flat_population_pfsp_score_ema_decay
                                ),
                                flat_role="main",
                                gate_game_callback=record_stage_gate_game,
                                training_game_callback=record_stage_training_game,
                                candidate_label=label,
                                opponent_labels=population_evaluation_labels,
                                opponent_ids=population_ids,
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
                            # Freeze the main after its PPO block.  The
                            # exploiter collects chiefly against this exact
                            # policy, so it can discover a concrete counter
                            # rather than optimizing against an old champion.
                            current_main = PolicyValueBeliefNet(model_config)
                            restore_model_state(
                                current_main, model_state_on_cpu(candidate)
                            )
                            current_main.to(device).eval()
                            for parameter in current_main.parameters():
                                parameter.requires_grad_(False)
                            current_main_id = "flat-current-main"
                            exploiter_models = [current_main, *frozen_models]
                            exploiter_labels = [
                                f"current main {candidate_source.stem}",
                                *population_evaluation_labels,
                            ]
                            exploiter_ids: list[int | str] = [
                                current_main_id, *population_ids
                            ]
                            exploiter_probabilities = _flat_pfsp_weights(
                                np.full(
                                    len(exploiter_models), 0.5,
                                    dtype=np.float64,
                                ),
                                alpha=flat_population_pfsp_alpha,
                                probability_floor=(
                                    flat_population_pfsp_probability_floor
                                ),
                                role="main_exploiter",
                                primary_index=0,
                                primary_probability=(
                                    flat_main_exploiter_probability
                                ),
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
                                    model_b_id=recorded_population_opponent_id(
                                        row, opponent_index
                                    ),
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
                                    exploiter_models,
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
                                    flat_persistent_optimizer=(
                                        training_stage == "flat"
                                        and flat_population_persistent_optimizer
                                    ),
                                    flat_terminal_replay_capacity=(
                                        flat_population_terminal_replay_capacity
                                        if training_stage == "flat" else 0
                                    ),
                                    flat_terminal_replay_batch_size=(
                                        flat_population_terminal_replay_batch_size
                                        if training_stage == "flat" else 0
                                    ),
                                    flat_terminal_replay_interval=(
                                        flat_population_terminal_replay_interval
                                    ),
                                    flat_terminal_replay_value_weight=(
                                        flat_population_terminal_replay_value_weight
                                        if training_stage == "flat" else 0.0
                                    ),
                                    flat_ppo_max_policy_age=(
                                        flat_population_ppo_max_policy_age
                                        if training_stage == "flat" else None
                                    ),
                                    flat_policy_anchor_final_weight=(
                                        flat_population_policy_anchor_final_weight
                                        if training_stage == "flat" else None
                                    ),
                                    flat_snapshot_archive_size=(
                                        flat_population_snapshot_archive_size
                                        if training_stage == "flat" else 0
                                    ),
                                    flat_pfsp_alpha=flat_population_pfsp_alpha,
                                    flat_pfsp_probability_floor=(
                                        flat_population_pfsp_probability_floor
                                    ),
                                    flat_pfsp_score_ema_decay=(
                                        flat_population_pfsp_score_ema_decay
                                    ),
                                    flat_role="main_exploiter",
                                    flat_primary_opponent_id=current_main_id,
                                    flat_primary_opponent_probability=(
                                        flat_main_exploiter_probability
                                    ),
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
                                        exploiter_labels
                                    ),
                                    opponent_ids=exploiter_ids,
                                    # Main and exploiter may collect against
                                    # different PFSP mixtures, but promotion
                                    # and rollback must use one fixed arena.
                                    benchmark_opponent_models=frozen_models,
                                    benchmark_opponent_weights=probabilities,
                                    benchmark_opponent_labels=(
                                        population_evaluation_labels
                                    ),
                                    benchmark_opponent_ids=population_ids,
                                    run=run.child(
                                        f"round_{round_index:03d}/"
                                        "direct_population_exploiter"
                                    ),
                                )
                            )
                            # Its PFSP benchmark intentionally excludes the
                            # live main.  Verify the actual exploit on a
                            # separate paired arena before spending a durable
                            # archive slot on it.
                            exploiter_main_match = (
                                evaluate_direct_population_matches(
                                    replace(
                                        round_training,
                                        seed=_offset_seed(
                                            direct_population_training_seed,
                                            2_750_003,
                                        ),
                                        max_moves=direct_gate_max_moves,
                                    ),
                                    exploiter,
                                    [current_main],
                                    [1.0],
                                    games=direct_population_gate_games,
                                    gate_seed=_offset_seed(
                                        direct_population_gate_seed,
                                        2_750_003,
                                    ),
                                    visible_action_history=(
                                        visible_action_history
                                    ),
                                    candidate_label=(
                                        f"main exploiter {candidate_source.stem}"
                                    ),
                                    opponent_labels=[
                                        f"current main {candidate_source.stem}"
                                    ],
                                    num_threads=num_threads,
                                    max_actions=mcts.native_max_actions,
                                )
                            )
                            exploiter_main_score = float(
                                exploiter_main_match["model_a_score"]
                            )
                            exploiter_archive_eligible = bool(
                                exploiter_main_match.get("valid", 1.0)
                            ) and (
                                exploiter_main_score
                                >= flat_main_exploiter_archive_score
                            )
                            main_exploiter_archive_summary = {
                                "enabled": 1.0,
                                "main_score": exploiter_main_score,
                                "eligible": float(exploiter_archive_eligible),
                                "archived": 0.0,
                                "threshold": flat_main_exploiter_archive_score,
                                "games": float(
                                    exploiter_main_match.get("games", 0.0)
                                ),
                            }
                            if exploiter_archive_eligible:
                                exploiter_model_id = database.create_candidate(
                                    database_run_id,
                                    candidate_id,
                                    f"main-exploiter-round-{round_index:03d}",
                                    model_config,
                                    role="flat_main_exploiter",
                                )
                                archive_path = _next_checkpoint_path(
                                    checkpoint_dir,
                                    f"{checkpoint_prefix}-main-exploiter",
                                )
                                save_model(
                                    archive_path, exploiter, model_config
                                )
                                database.archive_candidate(
                                    exploiter_model_id,
                                    archive_path,
                                    role="flat_main_exploiter",
                                )
                                main_exploiter_archive_summary.update(
                                    archived=1.0,
                                    model_id=exploiter_model_id,
                                    checkpoint_path=str(archive_path),
                                )
                                print(
                                    "league main exploiter archived | "
                                    f"vs-main={exploiter_main_score:.4f} | "
                                    f"saved {archive_path}",
                                    flush=True,
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
                            "main_exploiter_archive": (
                                main_exploiter_archive_summary
                            ),
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
                                excluded_map_seeds=known_training_map_seeds,
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
                                reanalysis_max_bytes=(
                                    population_mcts_reanalyse_max_bytes
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
                        champion_counter_score = next(
                            (
                                float(result.get("model_a_score", 0.0))
                                for result in flat_evaluations
                                if int(result.get("opponent_id", -1))
                                == champion_before_id
                            ),
                            0.0,
                        )
                        # A model which has found a non-losing response to
                        # the current champion is strategically useful even
                        # when it misses the global, all-opponent promotion
                        # gate.  Keep it as a durable training counter, not
                        # merely as an in-memory PFSP snapshot.
                        archive_counterstrategy = bool(
                            training_stage == "flat"
                            and not flat_invalid
                            and champion_counter_score >= 0.50
                            and minimum_historical_score
                            >= historical_score_floor
                        )
                        archived_counterstrategy = False
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
                                (
                                    f"{checkpoint_prefix}-counter"
                                    if archive_counterstrategy
                                    else f"{checkpoint_prefix}-challenger"
                                ),
                            )
                            save_model(retained_path, candidate, model_config)
                            if archive_counterstrategy:
                                database.archive_candidate(
                                    candidate_id,
                                    retained_path,
                                    role="flat_counter",
                                )
                                archived_counterstrategy = True
                            else:
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
                            if archive_counterstrategy:
                                archived_path = _next_checkpoint_path(
                                    checkpoint_dir,
                                    f"{checkpoint_prefix}-counter",
                                )
                                save_model(
                                    archived_path, candidate, model_config
                                )
                                database.archive_candidate(
                                    candidate_id,
                                    archived_path,
                                    role="flat_counter",
                                )
                                archived_counterstrategy = True
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
                        round_training_summary["flat_counterstrategy"] = {
                            "champion_score": champion_counter_score,
                            "eligible": float(archive_counterstrategy),
                            "archived": float(archived_counterstrategy),
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

                            # The probe measures the *candidate's* marginal
                            # value from search. Keep the opponent raw in both
                            # legs, keep max_moves identical, and reuse the
                            # exact post-recovery map schedule below.
                            probe_mcts = replace(
                                evaluation_mcts,
                                simulations=mcts_probe_simulations,
                            )
                            raw_probe = evaluate_direct_matches(
                                candidate,
                                champion,
                                device,
                                replace(round_training, seed=mcts_probe_seed),
                                mcts_probe_games,
                                game_callback=record_raw_probe_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                                visible_action_history=visible_action_history,
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                            )
                            raw_probe_valid = bool(raw_probe.get("valid", 1.0))
                            if raw_probe_valid:
                                database.finish_evaluation_unrated(
                                    raw_probe_id, raw_probe
                                )
                            else:
                                database.invalidate_evaluation(
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

                            try:
                                if not raw_probe_valid:
                                    search_probe = {
                                        "games": 0.0,
                                        "model_a_score": 0.0,
                                        "score_lower_95": 0.0,
                                        "valid": 0.0,
                                        "invalid_evaluations": 1.0,
                                        "invalid_reason": "raw readiness probe is invalid",
                                    }
                                else:
                                    search_probe = evaluate_matches_pool(
                                        candidate,
                                        champion,
                                        replace(
                                            round_training,
                                            seed=mcts_probe_seed,
                                        ),
                                        probe_mcts,
                                        mcts_probe_games,
                                        num_threads=num_threads,
                                        max_pending_leaves=max_pending_leaves,
                                        visible_action_history=visible_action_history,
                                        belief_retries=belief_retries,
                                        belief_action_mode=(
                                            population_belief_search_mode
                                        ),
                                        model_a_uses_mcts=True,
                                        model_b_uses_mcts=False,
                                        map_seeds=raw_probe[
                                            "evaluated_map_seeds"
                                        ],
                                        excluded_map_seeds=known_training_map_seeds,
                                        replace_failed_pairs=False,
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
                            probe_valid = raw_probe_valid and bool(
                                search_probe.get("valid", 1.0)
                            )
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
                        "mcts_preflight": [],
                    }
                    belief_calibration_summary: Dict[str, Any] | None = None
                    # Belief calibration is a supervised optimisation loop of
                    # its own.  Keep an in-memory rollback point so a noisy
                    # collection round can never overwrite a decoder which
                    # was better on the fixed held-out maps.
                    belief_calibration_start_state = (
                        model_state_on_cpu(candidate)
                        if belief_calibration_required
                        else None
                    )
                    previous_held_out_rate: float | None = None
                    if (
                        belief_calibration_required
                        and active_belief_matches_champion
                        and active_belief_validation_matches
                        and active_belief_checked_particles
                        >= belief_action_equivalence_min_particles
                    ):
                        previous_held_out_rate = float(
                            (
                                challenger_belief_state
                                if continue_mcts_challenger
                                else belief_apprentice_state
                            ).get(
                                "held_out_rejection_rate",
                                active_belief_rejection_rate,
                            )
                        )
                    value_bootstrap_summary: Dict[str, Any] = {}
                    bootstrap_required = _value_bootstrap_required(
                        champion_mode=round_champion_mode,
                        belief_calibration_required=belief_calibration_required,
                        value_bootstrap_games=value_bootstrap_games,
                        champion_fingerprint=champion_fingerprint,
                        apprentice_state=belief_apprentice_state,
                    )
                    if bootstrap_required and not _belief_calibration_can_run(
                        population_games=population_games,
                        replay_updates=belief_replay_updates,
                        evaluation_games=belief_calibration_eval_games,
                    ):
                        # Do not alter a certified apprentice when this run
                        # has no possible post-bootstrap measurement.  In
                        # particular, falling through here would leave a
                        # ready=False marker and allow the outer MCTS path to
                        # use unvalidated particles.
                        raise ValueError(
                            "value bootstrap requires population games, belief "
                            "replay updates, or held-out belief evaluation "
                            "before MCTS can run"
                        )
                    if bootstrap_required:
                        # This is not a legacy actor-critic/PPO stage. Both
                        # players execute their frozen raw policy; every
                        # collected policy target is masked. Only terminal
                        # outcome supervision calibrates value, while the
                        # belief heads keep learning from full-map labels.
                        def record_value_bootstrap_game(
                            game: Mapping[str, Any],
                        ) -> None:
                            opponent_index = game.get("opponent_model_index")
                            opponent_id = (
                                population_ids[int(opponent_index)]
                                if opponent_index is not None else None
                            )
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="value_belief_bootstrap",
                                model_a_id=candidate_id,
                                model_b_id=opponent_id,
                                game=game,
                            )

                        bootstrap_training = replace(
                            round_training,
                            seed=_offset_seed(
                                population_training_seed, 1_000_003
                            ),
                            policy_weight=0.0,
                            value_weight=1.0,
                        )
                        # No MCTS leaves are selected in this stage. One
                        # accepted particle is nevertheless required by
                        # PolyEnv before a checked raw-policy step.
                        bootstrap_mcts = replace(mcts, particles=1)
                        print(
                            "league value/belief bootstrap | "
                            f"raw games={value_bootstrap_games} | "
                            "policy loss=0",
                            flush=True,
                        )
                        value_bootstrap_summary = train_population_pool(
                            bootstrap_training,
                            model_config,
                            bootstrap_mcts,
                            games=value_bootstrap_games,
                            updates_per_game=value_bootstrap_updates_per_game,
                            candidate=candidate,
                            opponent_models=frozen_models,
                            opponent_weights=probabilities,
                            anchor_model=None,
                            anchor_policy_weight=0.0,
                            anchor_value_weight=0.0,
                            critical_decision_fraction=0.0,
                            critical_replay_fraction=0.0,
                            critical_only=False,
                            selective_mcts=False,
                            reanalysis_path=None,
                            reanalysis_positions=0,
                            reanalysis_replay_fraction=0.0,
                            offline_collection=True,
                            value_belief_heads_only=True,
                            raw_policy_only=True,
                            num_threads=population_workers,
                            max_pending_leaves=max_pending_leaves,
                            visible_action_history=visible_action_history,
                            belief_retries=belief_retries,
                            belief_action_mode="root_shield",
                            max_envs=min(num_envs, population_max_envs),
                            game_callback=record_value_bootstrap_game,
                            reserved_seeds=evaluation_map_seeds,
                            run=run.child(
                                f"round_{round_index:03d}/value_belief_bootstrap"
                            ),
                        )
                        # The raw policy and encoder were frozen, so this is
                        # a safe persistent calibration checkpoint even when
                        # MCTS still fails its preflight. Without saving it,
                        # every rejected rung would throw away the very value
                        # evidence needed to make the next preflight better.
                        save_model(
                            belief_apprentice_path, candidate, model_config
                        )
                        # Bootstrap changes decoder weights. Its old held-out
                        # number therefore cannot certify this newly saved
                        # checkpoint; force a fresh strict calibration before
                        # MCTS can use it.
                        bootstrap_marker = {
                            "champion_path": str(champion_path.resolve()),
                            "champion_fingerprint": _checkpoint_fingerprint(champion_path),
                            "apprentice_fingerprint": _checkpoint_fingerprint(
                                belief_apprentice_path
                            ),
                            "ready": False,
                            "checked_particles": 0,
                            "rejected_particles": 0,
                            "rejection_rate": 1.0,
                            "held_out_games": 0,
                            "validation_signature": None,
                            # Retain this identity after the mandatory
                            # post-bootstrap held-out calibration succeeds.
                            # It is the guard that keeps auto from doing the
                            # same destructive bootstrap every round.
                            "value_bootstrap_champion_fingerprint": champion_fingerprint,
                            "value_bootstrap_round": round_index,
                            "value_bootstrap_games": value_bootstrap_games,
                            "value_bootstrap_terminals": int(
                                value_bootstrap_summary.get(
                                    "population_pool_terminals", 0.0
                                )
                            ),
                            "value_bootstrap_updates": int(
                                value_bootstrap_summary.get(
                                    "population_pool_updates", 0.0
                                )
                            ),
                        }
                        _save_belief_apprentice_state(
                            belief_apprentice_state_path, bootstrap_marker
                        )
                        # The in-memory candidate now contains new decoder
                        # weights. It must complete strict held-out belief
                        # calibration before any later code in this round can
                        # enter an MCTS rung.
                        belief_calibration_required = True
                        bootstrap_champion_fingerprint = champion_fingerprint
                        belief_calibration_start_state = model_state_on_cpu(candidate)
                        previous_held_out_rate = None
                        population_summary["value_belief_bootstrap"] = (
                            value_bootstrap_summary
                        )
                        print(
                            "league value/belief bootstrap | "
                            f"terminal={int(value_bootstrap_summary.get('population_pool_terminals', 0.0))}/"
                            f"{value_bootstrap_games} | "
                            f"updates={int(value_bootstrap_summary.get('population_pool_updates', 0.0))} | "
                            f"value_mae={float(value_bootstrap_summary.get('value_mae', 0.0)):.4f}",
                            flush=True,
                        )
                    round_training_summary["value_belief_bootstrap"] = (
                        value_bootstrap_summary
                    )
                    if (
                        not belief_calibration_required
                        and value_bootstrap_summary
                        and not population_games
                    ):
                        # A bootstrap-only invocation is a diagnostic, not a
                        # promotion attempt.  Do the same small asymmetric
                        # test used before an MCTS rung, then stop.  Without
                        # this branch, population_games=0 would incorrectly
                        # fall through to the expensive 100-game MCTS gate.
                        bootstrap_probe_mcts = replace(
                            evaluation_mcts,
                            simulations=int(population_rungs[-1])
                            if population_rungs else int(mcts.simulations),
                        )
                        raw_probe = evaluate_direct_matches(
                            candidate,
                            champion,
                            device,
                            replace(round_training, seed=mcts_probe_seed),
                            mcts_probe_games,
                            model_a_label=candidate_evaluation_label,
                            model_b_label=champion_path.stem,
                            visible_action_history=visible_action_history,
                            num_threads=num_threads,
                            max_actions=mcts.native_max_actions,
                        )
                        if bool(raw_probe.get("valid", 1.0)):
                            search_probe = evaluate_matches_pool(
                                candidate,
                                champion,
                                replace(round_training, seed=mcts_probe_seed),
                                bootstrap_probe_mcts,
                                mcts_probe_games,
                                num_threads=num_threads,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                belief_action_mode=population_belief_search_mode,
                                model_a_uses_mcts=True,
                                model_b_uses_mcts=False,
                                map_seeds=raw_probe["evaluated_map_seeds"],
                                excluded_map_seeds=known_training_map_seeds,
                                replace_failed_pairs=False,
                                max_envs=min(num_envs, population_max_envs),
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                            )
                        else:
                            search_probe = {
                                "model_a_score": 0.0,
                                "valid": 0.0,
                                "invalid_reason": "raw bootstrap probe is invalid",
                            }
                        raw_score = float(raw_probe.get("model_a_score", 0.0))
                        search_score = float(
                            search_probe.get("model_a_score", 0.0)
                        )
                        passed = _mcts_probe_passed(
                            raw_score,
                            search_score,
                            minimum_search_score=mcts_probe_score,
                            minimum_improvement=mcts_probe_minimum_improvement,
                            valid=bool(raw_probe.get("valid", 1.0))
                            and bool(search_probe.get("valid", 1.0)),
                        )
                        bootstrap_probe = {
                            "raw": raw_probe,
                            "search": search_probe,
                            "raw_score": raw_score,
                            "search_score": search_score,
                            "improvement": search_score - raw_score,
                            "passed": float(passed),
                            "simulations": float(
                                population_rungs[-1]
                                if population_rungs else mcts.simulations
                            ),
                        }
                        round_training_summary["value_belief_bootstrap_probe"] = (
                            bootstrap_probe
                        )
                        database.finalise_candidate(candidate_id, promoted_path=None)
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            {
                                "promoted": 0.0,
                                "valid": float(search_probe.get("valid", 1.0)),
                                "reason": "value_belief_bootstrap_probe",
                                "value_belief_bootstrap_probe": bootstrap_probe,
                            },
                        )
                        run.log(
                            round=round_index,
                            value_bootstrap_probe=1.0,
                            value_bootstrap_probe_raw_score=raw_score,
                            value_bootstrap_probe_search_score=search_score,
                            value_bootstrap_probe_improvement=(
                                search_score - raw_score
                            ),
                            value_bootstrap_probe_passed=float(passed),
                            promoted=0.0,
                        )
                        database.flush()
                        print(
                            "league value/belief bootstrap probe | "
                            f"search={search_score:.4f} | raw={raw_score:.4f} | "
                            f"passed={int(passed)} | no promotion",
                            flush=True,
                        )
                        last_training = dict(round_training_summary)
                        last_candidate_score = search_score
                        last_population_score = 0.0
                        challenger_path = None
                        challenger_id = None
                        del candidate, champion, frozen_models
                        continue
                    # Roll back a rejected MCTS rung to the freshly calibrated
                    # raw-policy checkpoint, never to its pre-bootstrap value.
                    direct_population_state = model_state_on_cpu(candidate)
                    # A failed first MCTS rung must return to the last state
                    # proven safe against the raw-policy population gate, not
                    # to an optional ungated legacy self-play snapshot.
                    best_mcts_state = direct_population_state
                    best_mcts_score = 0.50
                    # Persisted only after a rung passes its paired search
                    # and absorption gates.  It is deliberately distinct
                    # from the champion: a later rung or promotion gate can
                    # still reject it without losing this verified progress.
                    verified_mcts_path: Path | None = None
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
                    if population_games or (
                        belief_calibration_required
                        and _belief_calibration_can_run(
                            population_games=population_games,
                            replay_updates=belief_replay_updates,
                            evaluation_games=belief_calibration_eval_games,
                        )
                    ):
                        # Calibration is a decoder dataset pass, not a search
                        # ladder. This applies both to an explicit belief run
                        # and to auto mode when it discovers that calibration
                        # is still required.
                        calibration_rungs = (
                            ((int(mcts.simulations), population_games),)
                            if belief_calibration_required and population_games
                            else (
                                zip(population_rungs, population_game_allocations)
                                if population_games
                                else ((int(mcts.simulations), 0),)
                            )
                        )
                        for rung_index, (simulations, rung_games) in enumerate(
                            calibration_rungs
                        ):
                            if rung_games <= 0 and not (
                                belief_calibration_required
                                and _belief_calibration_can_run(
                                    population_games=population_games,
                                    replay_updates=belief_replay_updates,
                                    evaluation_games=(
                                        belief_calibration_eval_games
                                    ),
                                )
                            ):
                                continue
                            rung_training = replace(
                                round_training,
                                seed=_offset_seed(
                                    population_training_seed,
                                    rung_index * 2_000_003,
                                ),
                            )
                            # Until a model can reconstruct particles whose
                            # action set matches its public observation, MCTS
                            # visits are not a trustworthy policy teacher.
                            # Calibrate only the belief-specific decoder and
                            # public-action conditioning; policy/value and
                            # the shared encoder remain frozen.
                            if belief_calibration_required:
                                rung_training = replace(
                                    rung_training,
                                    policy_weight=0.0,
                                    value_weight=0.0,
                                    belief_weight=max(1.0, rung_training.belief_weight),
                                    engine_action_constraint_weight=max(
                                        1.0,
                                        rung_training.engine_action_constraint_weight,
                                    ),
                                    bridge_constraint_weight=max(
                                        0.25,
                                        rung_training.bridge_constraint_weight,
                                    ),
                                    engine_action_constraint_replay_fraction=max(
                                        0.40,
                                        rung_training.engine_action_constraint_replay_fraction,
                                    ),
                                    bridge_replay_fraction=max(
                                        0.20,
                                        rung_training.bridge_replay_fraction,
                                    ),
                                )
                            rung_mcts = replace(
                                mcts,
                                simulations=int(simulations),
                                # A calibration generation needs diverse,
                                # real legal observations rather than a
                                # policy-biased subset.  It does not train a
                                # policy target, so no top-K speed shortcut.
                                search_top_actions=(
                                    0
                                    if belief_calibration_required
                                    else mcts.search_top_actions
                                ),
                                search_random_actions=(
                                    0
                                    if belief_calibration_required
                                    else mcts.search_random_actions
                                ),
                                search_min_prior_mass=(
                                    0.0
                                    if belief_calibration_required
                                    else mcts.search_min_prior_mass
                                ),
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

                            # Search must prove that it improves on the raw
                            # policy *before* it is allowed to create policy
                            # targets.  A post-update rollback protects the
                            # checkpoint, but still wastes a full generation
                            # learning from a weak tree.  Use paired maps and
                            # the same frozen champion for both measurements.
                            preflight: Dict[str, Any] = {}
                            if not belief_calibration_required and mcts_probe_games:
                                try:
                                    search_preflight = evaluate_matches_pool(
                                        candidate,
                                        champion,
                                        replace(
                                            rung_training,
                                            seed=mcts_probe_seed,
                                            max_moves=direct_gate_max_moves,
                                        ),
                                        replace(
                                            evaluation_mcts,
                                            simulations=int(simulations),
                                        ),
                                        mcts_probe_games,
                                        num_threads=num_threads,
                                        max_pending_leaves=max_pending_leaves,
                                        visible_action_history=visible_action_history,
                                        belief_retries=belief_retries,
                                        belief_action_mode=(
                                            population_belief_search_mode
                                        ),
                                        model_b_uses_mcts=False,
                                        excluded_map_seeds=known_training_map_seeds,
                                        max_envs=min(
                                            num_envs, population_max_envs
                                        ),
                                        model_a_label=candidate_evaluation_label,
                                        model_b_label=champion_path.stem,
                                    )
                                    # Recovery may replace an action-incompatible
                                    # map pair.  Measure the raw policy on the
                                    # exact final map schedule, not on the
                                    # abandoned original pair.
                                    if bool(search_preflight.get("valid", 1.0)):
                                        raw_preflight = evaluate_direct_matches(
                                            candidate,
                                            champion,
                                            device,
                                            replace(
                                                rung_training,
                                                seed=mcts_probe_seed,
                                                max_moves=direct_gate_max_moves,
                                            ),
                                            mcts_probe_games,
                                            map_seeds=search_preflight[
                                                "evaluated_map_seeds"
                                            ],
                                            model_a_label=candidate_evaluation_label,
                                            model_b_label=champion_path.stem,
                                            visible_action_history=visible_action_history,
                                            num_threads=num_threads,
                                            max_actions=mcts.native_max_actions,
                                        )
                                    else:
                                        raw_preflight = dict(search_preflight)
                                except (
                                    BeliefQualityError,
                                    BeliefRecoveryRequired,
                                ) as error:
                                    search_preflight = {
                                        "games": 0.0,
                                        "model_a_score": 0.0,
                                        "valid": 0.0,
                                        "invalid_reason": str(error),
                                    }
                                    raw_preflight = dict(search_preflight)
                                raw_preflight_score = float(
                                    raw_preflight.get("model_a_score", 0.0)
                                )
                                search_preflight_score = float(
                                    search_preflight.get("model_a_score", 0.0)
                                )
                                preflight_diagnostics = _mcts_preflight_diagnostics(
                                    raw_preflight, search_preflight
                                )
                                passed_preflight = _mcts_probe_passed(
                                    raw_preflight_score,
                                    search_preflight_score,
                                    minimum_search_score=mcts_probe_score,
                                    minimum_improvement=(
                                        mcts_probe_minimum_improvement
                                    ),
                                    valid=(
                                        bool(search_preflight.get("valid", 1.0))
                                        and bool(raw_preflight.get("valid", 1.0))
                                    ),
                                )
                                preflight = {
                                    "performed": 1.0,
                                    "passed": float(passed_preflight),
                                    "simulations": float(simulations),
                                    "raw": raw_preflight,
                                    "search": search_preflight,
                                    "raw_score": raw_preflight_score,
                                    "search_score": search_preflight_score,
                                    "improvement": (
                                        search_preflight_score
                                        - raw_preflight_score
                                    ),
                                    **preflight_diagnostics,
                                }
                                population_summary["mcts_preflight"].append(
                                    preflight
                                )
                                run.log(
                                    round=round_index,
                                    mcts_preflight=1.0,
                                    mcts_preflight_simulations=simulations,
                                    mcts_preflight_raw_score=raw_preflight_score,
                                    mcts_preflight_search_score=(
                                        search_preflight_score
                                    ),
                                    mcts_preflight_improvement=(
                                        search_preflight_score
                                        - raw_preflight_score
                                    ),
                                    mcts_preflight_passed=float(passed_preflight),
                                    mcts_preflight_raw_valid=float(
                                        preflight_diagnostics["raw_valid"]
                                    ),
                                    mcts_preflight_search_valid=float(
                                        preflight_diagnostics["search_valid"]
                                    ),
                                    mcts_preflight_raw_completed_games=float(
                                        preflight_diagnostics["raw_completed_games"]
                                    ),
                                    mcts_preflight_search_completed_games=float(
                                        preflight_diagnostics["search_completed_games"]
                                    ),
                                )
                                print(
                                    f"league MCTS preflight {simulations} | "
                                    f"search={search_preflight_score:.4f} | "
                                    f"raw={raw_preflight_score:.4f} | "
                                    f"passed={int(passed_preflight)} | "
                                    f"status={preflight_diagnostics['status']} | "
                                    f"search_games="
                                    f"{int(preflight_diagnostics['search_completed_games'])}/"
                                    f"{int(preflight_diagnostics['search_planned_games'])} | "
                                    f"raw_games="
                                    f"{int(preflight_diagnostics['raw_completed_games'])}/"
                                    f"{int(preflight_diagnostics['raw_planned_games'])}"
                                    + (
                                        f" | reason={preflight_diagnostics['reason']}"
                                        if preflight_diagnostics["reason"]
                                        else ""
                                    ),
                                    flush=True,
                                )
                                if not passed_preflight:
                                    invalid_preflight = (
                                        preflight_diagnostics["status"] != "valid"
                                    )
                                    if (
                                        invalid_preflight
                                        and training_stage == "auto"
                                        and not continue_mcts_challenger
                                        and preflight_diagnostics["status"]
                                        == "search_evaluation_invalid"
                                    ):
                                        # Persist a narrowly-scoped repair
                                        # request.  The next auto round will
                                        # enter the supervised belief stage;
                                        # an invalid raw-only/environment
                                        # measurement never triggers blind
                                        # decoder updates.
                                        repair_marker = dict(
                                            belief_apprentice_state
                                        )
                                        repair_marker.update({
                                            "mcts_belief_repair_required": True,
                                            "mcts_belief_repair_reason": str(
                                                preflight_diagnostics["reason"]
                                            ),
                                            "mcts_belief_repair_round": int(
                                                round_index
                                            ),
                                            "mcts_belief_repair_attempts": int(
                                                belief_apprentice_state.get(
                                                    "mcts_belief_repair_attempts",
                                                    0,
                                                )
                                            ) + 1,
                                        })
                                        _save_belief_apprentice_state(
                                            belief_apprentice_state_path,
                                            repair_marker,
                                        )
                                        population_summary[
                                            "mcts_repair_action"
                                        ] = "belief_calibration_next_auto_round"
                                    population_summary["mcts_ladder"].append({
                                        "simulations": float(simulations),
                                        "games": float(rung_games),
                                        "accepted": 0.0,
                                        "preflight": preflight,
                                        "reason": (
                                            str(preflight_diagnostics["status"])
                                            if invalid_preflight
                                            else "mcts_preflight_not_better_than_raw"
                                        ),
                                    })
                                    print(
                                        (
                                            "league MCTS preflight invalid | "
                                            "rung skipped without optimisation; "
                                            "belief/engine diagnostics required"
                                            if invalid_preflight
                                            else "league MCTS preflight failed | "
                                            "rung skipped without optimisation; "
                                            "trying the next search depth"
                                        ),
                                        flush=True,
                                    )
                                    continue

                            calibration_replay: Sequence[TrainingExample] = ()
                            calibration_replay_saved = [0]
                            replay_snapshot_callback = None
                            if belief_calibration_required:
                                calibration_replay = _load_belief_calibration_replay(
                                    belief_calibration_replay_path,
                                    belief_calibration_replay_capacity,
                                    map_size=round_training.map_size,
                                )
                                if calibration_replay is None:
                                    legacy_path = belief_calibration_replay_path
                                    recovered_path = legacy_path.with_name(
                                        legacy_path.stem + "-recovered" + legacy_path.suffix
                                    )
                                    recovered_replay = _load_belief_calibration_replay(
                                        recovered_path,
                                        belief_calibration_replay_capacity,
                                        map_size=round_training.map_size,
                                    )
                                    if recovered_replay is not None:
                                        belief_calibration_replay_path = recovered_path
                                        calibration_replay = recovered_replay
                                        print(
                                            "league belief replay | legacy file is too large "
                                            f"to load; preserving {legacy_path} and recovering "
                                            f"the bounded reservoir at {recovered_path}",
                                            flush=True,
                                        )
                                    else:
                                        # Do not overwrite either unreadable
                                        # file. A unique successor protects
                                        # every previously recovered sample.
                                        suffix = 2
                                        while True:
                                            candidate_path = legacy_path.with_name(
                                                legacy_path.stem
                                                + f"-recovered-{suffix}"
                                                + legacy_path.suffix
                                            )
                                            if not candidate_path.exists():
                                                belief_calibration_replay_path = candidate_path
                                                break
                                            suffix += 1
                                        calibration_replay = []
                                        print(
                                            "league belief replay | existing replay files are too "
                                            f"large to load; preserving them and starting a bounded "
                                            f"reservoir at {belief_calibration_replay_path}",
                                            flush=True,
                                        )

                                def persist_calibration_replay(
                                    examples: Sequence[TrainingExample],
                                ) -> None:
                                    _save_belief_calibration_replay(
                                        belief_calibration_replay_path,
                                        examples,
                                        belief_calibration_replay_capacity,
                                    )
                                    calibration_replay_saved[0] = min(
                                        len(examples),
                                        belief_calibration_replay_capacity,
                                    )

                                replay_snapshot_callback = persist_calibration_replay
                                print(
                                    "league belief replay | "
                                    f"loaded={len(calibration_replay)} | "
                                    f"capacity={belief_calibration_replay_capacity}",
                                    flush=True,
                                )

                            rung_summary = train_population_pool(
                                rung_training,
                                model_config,
                                rung_mcts,
                                games=rung_games,
                                updates_per_game=population_updates_per_game,
                                optimisation_block_updates=(
                                    population_optimisation_block_updates
                                ),
                                optimisation_screen_games=(
                                    0
                                    if belief_calibration_required
                                    else population_optimisation_screen_games
                                ),
                                optimisation_screen_max_regression=(
                                    population_optimisation_screen_max_regression
                                ),
                                candidate=candidate,
                                opponent_models=frozen_models,
                                opponent_weights=probabilities,
                                anchor_model=champion,
                                anchor_policy_weight=(
                                    0.0
                                    if belief_calibration_required
                                    else champion_anchor_weight
                                ),
                                anchor_value_weight=(
                                    0.0
                                    if belief_calibration_required
                                    else champion_value_anchor_weight
                                ),
                                critical_decision_fraction=(
                                    0.0
                                    if belief_calibration_required
                                    else population_mcts_critical_fraction
                                ),
                                critical_minimum_score=(
                                    population_mcts_critical_minimum_score
                                ),
                                critical_replay_fraction=(
                                    0.0
                                    if belief_calibration_required
                                    else population_mcts_critical_replay_fraction
                                ),
                                critical_only=(
                                    False
                                    if belief_calibration_required
                                    else population_mcts_critical_only
                                ),
                                selective_mcts=(
                                    False
                                    if belief_calibration_required
                                    else population_selective_mcts
                                ),
                                selective_mcts_fraction=(
                                    population_selective_mcts_fraction
                                ),
                                selective_mcts_minimum_envs=(
                                    population_selective_mcts_minimum_envs
                                ),
                                selective_mcts_audit_fraction=(
                                    population_selective_mcts_audit_fraction
                                ),
                                reanalysis_path=population_mcts_reanalyse_path,
                                reanalysis_capacity=(
                                    population_mcts_reanalyse_capacity
                                ),
                                reanalysis_max_bytes=(
                                    population_mcts_reanalyse_max_bytes
                                ),
                                reanalysis_positions=(
                                    0
                                    if belief_calibration_required
                                    else population_mcts_reanalyse_positions
                                ),
                                reanalysis_replay_fraction=(
                                    0.0
                                    if belief_calibration_required
                                    else population_mcts_reanalyse_replay_fraction
                                ),
                                num_threads=population_workers,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                # Calibration is deliberately strict: it
                                # measures whether the apprentice can model
                                # public action evidence. The fast shield is
                                # for ordinary policy/value data generation.
                                belief_action_mode=(
                                    "strict"
                                    if belief_calibration_required
                                    else population_belief_search_mode
                                ),
                                max_envs=min(num_envs, population_max_envs),
                                offline_collection=population_offline_collection,
                                belief_only=belief_calibration_required,
                                # Calibration learns whether particles respect
                                # public action evidence. It does not need a
                                # tree or MCTS visit targets, so execute raw
                                # actions after the same strict particle
                                # validation and keep this stage affordable.
                                raw_policy_only=belief_calibration_required,
                                game_callback=record_training_game,
                                reserved_seeds=evaluation_map_seeds,
                                replay_examples=calibration_replay,
                                replay_snapshot_callback=replay_snapshot_callback,
                                extra_replay_updates=(
                                    belief_replay_updates
                                    if belief_calibration_required
                                    else 0
                                ),
                                run=run.child(
                                    f"round_{round_index:03d}/mcts_rung_{rung_index + 1:02d}"
                                ),
                            )
                            if not belief_calibration_required:
                                value_metrics = [
                                    f"value_loss={float(rung_summary[key]):.4f}"
                                    for key in ("value_loss",)
                                    if key in rung_summary
                                ]
                                value_metrics.extend(
                                    f"{label}={float(rung_summary[key]):.4f}"
                                    for label, key in (
                                        ("value_mae", "value_mae"),
                                        ("value_bias", "value_bias"),
                                        ("anchor_value", "value_anchor_loss"),
                                    )
                                    if key in rung_summary
                                )
                                if value_metrics:
                                    print(
                                        "league MCTS value diagnostics | "
                                        + " | ".join(value_metrics),
                                        flush=True,
                                    )
                            if belief_calibration_required:
                                rung_summary["belief_calibration_replay_loaded"] = float(
                                    len(calibration_replay)
                                )
                                rung_summary["belief_calibration_replay_saved"] = float(
                                    calibration_replay_saved[0]
                                )
                                collection_checked = int(
                                    rung_summary.get(
                                        "belief_action_equivalence_checked_particles",
                                        0.0,
                                    )
                                )
                                collection_rejected = int(
                                    rung_summary.get(
                                        "belief_action_equivalence_rejected_particles",
                                        0.0,
                                    )
                                )
                                collection_rate = float(
                                    collection_rejected / max(1, collection_checked)
                                )
                                held_out_summary: Dict[str, Any] = {}
                                if belief_calibration_eval_games:
                                    # This is deliberately after optimisation
                                    # and uses a fixed seed schedule. The
                                    # collection mismatch is measured before
                                    # its updates, so it cannot say whether
                                    # this calibration round generalised.
                                    held_out_summary = train_population_pool(
                                        replace(
                                            round_training,
                                            # The promotion arena rotates by
                                            # round, but a calibration rollback
                                            # must compare the same held-out
                                            # belief maps as the prior model.
                                            seed=belief_validation_seed,
                                            policy_weight=0.0,
                                            value_weight=0.0,
                                        ),
                                        model_config,
                                        mcts,
                                        games=belief_calibration_eval_games,
                                        updates_per_game=0,
                                        candidate=candidate,
                                        opponent_models=frozen_models,
                                        opponent_weights=probabilities,
                                        anchor_model=None,
                                        anchor_policy_weight=0.0,
                                        anchor_value_weight=0.0,
                                        critical_decision_fraction=0.0,
                                        critical_replay_fraction=0.0,
                                        critical_only=False,
                                        selective_mcts=False,
                                        reanalysis_path=None,
                                        reanalysis_positions=0,
                                        reanalysis_replay_fraction=0.0,
                                        offline_collection=True,
                                        belief_only=True,
                                        raw_policy_only=True,
                                        num_threads=population_workers,
                                        max_pending_leaves=max_pending_leaves,
                                        visible_action_history=visible_action_history,
                                        belief_retries=belief_retries,
                                        belief_action_mode="strict",
                                        max_envs=min(
                                            num_envs, population_max_envs
                                        ),
                                        reserved_seeds=evaluation_map_seeds,
                                        run=run.child(
                                            f"round_{round_index:03d}/belief_held_out"
                                        ),
                                    )
                                checked = int(held_out_summary.get(
                                    "belief_action_equivalence_checked_particles",
                                    collection_checked,
                                ))
                                rejected = int(held_out_summary.get(
                                    "belief_action_equivalence_rejected_particles",
                                    collection_rejected,
                                ))
                                rate = float(rejected / max(1, checked))
                                ready = bool(
                                    checked >= belief_action_equivalence_min_particles
                                    and rate <= belief_action_equivalence_target
                                )
                                # This is a strict admission checkpoint. Do
                                # not overwrite a better apprentice merely
                                # because a worse score is close to it.
                                rolled_back = bool(
                                    previous_held_out_rate is not None
                                    and rate >= previous_held_out_rate
                                )
                                if rolled_back:
                                    if belief_calibration_start_state is None:
                                        raise RuntimeError(
                                            "belief calibration rollback state is missing"
                                        )
                                    restore_model_state(
                                        candidate, belief_calibration_start_state
                                    )
                                else:
                                    save_model(
                                        belief_apprentice_path, candidate, model_config
                                    )
                                    _save_belief_apprentice_state(
                                        belief_apprentice_state_path,
                                        {
                                            "champion_path": str(champion_path.resolve()),
                                            "champion_fingerprint": _checkpoint_fingerprint(champion_path),
                                            "apprentice_fingerprint": _checkpoint_fingerprint(
                                                belief_apprentice_path
                                            ),
                                            "validation_signature": belief_validation_signature,
                                            "ready": ready,
                                            "checked_particles": checked,
                                            "rejected_particles": rejected,
                                            "rejection_rate": rate,
                                            "held_out_rejection_rate": rate,
                                            "collection_checked_particles": collection_checked,
                                            "collection_rejected_particles": collection_rejected,
                                            "collection_rejection_rate": collection_rate,
                                            "held_out_games": belief_calibration_eval_games,
                                            "target_rejection_rate": (
                                                belief_action_equivalence_target
                                            ),
                                            "minimum_particles": (
                                                belief_action_equivalence_min_particles
                                            ),
                                            # A successful calibration does
                                            # not erase the fact that this
                                            # champion has already received
                                            # its one value bootstrap.
                                            "value_bootstrap_champion_fingerprint": (
                                                bootstrap_champion_fingerprint
                                            ),
                                            "round": round_index,
                                        },
                                    )
                                belief_calibration_summary = {
                                    "path": str(belief_apprentice_path),
                                    "ready": float(ready),
                                    "checked_particles": float(checked),
                                    "rejected_particles": float(rejected),
                                    "rejection_rate": rate,
                                    "collection_checked_particles": float(collection_checked),
                                    "collection_rejected_particles": float(collection_rejected),
                                    "collection_rejection_rate": collection_rate,
                                    "held_out": held_out_summary,
                                    "previous_held_out_rejection_rate": (
                                        previous_held_out_rate
                                        if previous_held_out_rate is not None
                                        else -1.0
                                    ),
                                    "held_out_rollback": float(rolled_back),
                                    "target_rejection_rate": (
                                        belief_action_equivalence_target
                                    ),
                                    "training": rung_summary,
                                }
                                population_summary["belief_calibration"] = (
                                    belief_calibration_summary
                                )
                                print(
                                    "league belief calibration | "
                                    f"collection={collection_rate:.2%} | "
                                    f"held-out mismatch={rate:.2%} "
                                    f"({rejected}/{checked}) | "
                                    f"ready={int(ready)} | "
                                    + (
                                        "regression: restored prior apprentice"
                                        if rolled_back
                                        else f"saved {belief_apprentice_path}"
                                    ),
                                    flush=True,
                                )
                                # The particle mismatch is the admission
                                # metric, but it alone cannot identify whether
                                # a plateau comes from terrain, rare entities,
                                # or the hidden action endpoints.  Report the
                                # optimisation diagnostics from this frozen
                                # collection before asking for another costly
                                # calibration generation.
                                diagnostic_fields = (
                                    (
                                        "replay_loaded",
                                        "belief_calibration_replay_loaded",
                                        ".0f",
                                    ),
                                    (
                                        "replay_saved",
                                        "belief_calibration_replay_saved",
                                        ".0f",
                                    ),
                                    ("updates", "population_pool_updates", ".0f"),
                                    (
                                        "offline_updates",
                                        "population_pool_offline_replay_updates",
                                        ".0f",
                                    ),
                                    ("belief_loss", "belief_loss", ".4f"),
                                    (
                                        "action_loss",
                                        "belief_engine_action_constraint_loss",
                                        ".4f",
                                    ),
                                    (
                                        "missing_actions",
                                        "belief_engine_action_constraint_missing_actions",
                                        ".0f",
                                    ),
                                    (
                                        "extra_actions",
                                        "belief_engine_action_constraint_extra_actions",
                                        ".0f",
                                    ),
                                    (
                                        "terrain_acc",
                                        "belief_hidden_terrain_accuracy",
                                        ".1%",
                                    ),
                                    (
                                        "action_terrain_acc",
                                        "belief_action_constraint_terrain_accuracy",
                                        ".1%",
                                    ),
                                    (
                                        "city_recall",
                                        "belief_hidden_city_recall",
                                        ".1%",
                                    ),
                                    (
                                        "unit_recall",
                                        "belief_hidden_unit_recall",
                                        ".1%",
                                    ),
                                )
                                diagnostics = [
                                    f"{label}={float(rung_summary[key]):{format_spec}}"
                                    for label, key, format_spec in diagnostic_fields
                                    if key in rung_summary
                                ]
                                if diagnostics:
                                    print(
                                        "league belief diagnostics | "
                                        + " | ".join(diagnostics),
                                        flush=True,
                                    )
                                break
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

                            prior_mcts_gate_map_seeds = mcts_gate_map_seeds
                            rung_gate = evaluate_matches_pool(
                                candidate,
                                champion,
                                replace(
                                    training,
                                    seed=mcts_gate_seed,
                                ),
                                gate_mcts,
                                mcts_gate_games,
                                num_threads=num_threads,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                belief_action_mode=population_belief_search_mode,
                                map_seeds=mcts_gate_map_seeds,
                                excluded_map_seeds=known_training_map_seeds,
                                max_envs=min(num_envs, population_max_envs),
                                game_callback=record_rung_gate_game,
                                model_a_label=candidate_evaluation_label,
                                model_b_label=champion_path.stem,
                            )
                            mcts_gate_map_seeds = _updated_mcts_gate_map_schedule(
                                mcts_gate_map_seeds,
                                rung_gate,
                                mcts_gate_games,
                            )
                            # Future collection has not happened yet.  Reserve
                            # newly substituted arena maps before the next
                            # rung so they cannot become MCTS training data.
                            evaluation_map_seeds.update(mcts_gate_map_seeds)
                            rung_score = float(rung_gate.get("model_a_score", 0.0))
                            arena_rebased = (
                                mcts_gate_map_seeds
                                != prior_mcts_gate_map_seeds
                            )
                            baseline_gate: Dict[str, Any] | None = None
                            baseline_valid = True
                            # A replacement pair changes the arena itself.
                            # The accepted model from the prior rung must be
                            # measured again on that same recovered schedule;
                            # comparing against its score on the abandoned
                            # maps can reject a genuinely equal or better
                            # candidate merely because the new map is harder.
                            if arena_rebased:
                                candidate_after_rung = model_state_on_cpu(candidate)
                                try:
                                    restore_model_state(candidate, best_mcts_state)
                                    baseline_gate = evaluate_matches_pool(
                                        candidate,
                                        champion,
                                        replace(training, seed=mcts_gate_seed),
                                        gate_mcts,
                                        mcts_gate_games,
                                        num_threads=num_threads,
                                        max_pending_leaves=max_pending_leaves,
                                        visible_action_history=(
                                            visible_action_history
                                        ),
                                        belief_retries=belief_retries,
                                        belief_action_mode=(
                                            population_belief_search_mode
                                        ),
                                        map_seeds=mcts_gate_map_seeds,
                                        excluded_map_seeds=(
                                            known_training_map_seeds
                                        ),
                                        replace_failed_pairs=False,
                                        max_envs=min(
                                            num_envs, population_max_envs
                                        ),
                                        model_a_label=(
                                            f"{candidate_evaluation_label} baseline"
                                        ),
                                        model_b_label=champion_path.stem,
                                    )
                                finally:
                                    restore_model_state(
                                        candidate, candidate_after_rung
                                    )
                                baseline_valid = bool(
                                    baseline_gate.get("valid", 1.0)
                                )
                                if baseline_valid:
                                    best_mcts_score = float(
                                        baseline_gate.get("model_a_score", 0.0)
                                    )

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
                            ) and baseline_valid and (
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
                                verified_mcts_path = _next_checkpoint_path(
                                    checkpoint_dir,
                                    f"{checkpoint_prefix}-mcts-verified",
                                )
                                save_model(
                                    verified_mcts_path, candidate, model_config
                                )
                                # The MCTS update changes the shared belief
                                # decoder.  Make the next round revalidate it
                                # instead of accidentally inheriting the
                                # previous apprentice's calibration marker.
                                _save_belief_apprentice_state(
                                    verified_mcts_path.with_suffix(
                                        verified_mcts_path.suffix + ".json"
                                    ),
                                    {
                                        "champion_path": str(
                                            champion_path.resolve()
                                        ),
                                        "champion_fingerprint": (
                                            champion_fingerprint
                                        ),
                                        "apprentice_fingerprint": (
                                            _checkpoint_fingerprint(
                                                verified_mcts_path
                                            )
                                        ),
                                        "validation_signature": None,
                                        "ready": False,
                                        "held_out_games": 0,
                                        "stage": "mcts_verified_development",
                                        "round": round_index,
                                        "mcts_simulations": int(simulations),
                                        "mcts_score": rung_score,
                                        "raw_policy_score": raw_policy_score,
                                    },
                                )
                                event = getattr(run, "event", None)
                                if callable(event):
                                    event(
                                        "save",
                                        kind="mcts_verified_development",
                                        path=str(verified_mcts_path),
                                        simulations=int(simulations),
                                        search_score=rung_score,
                                        raw_policy_score=raw_policy_score,
                                    )
                            else:
                                restore_model_state(candidate, best_mcts_state)
                                population_summary["mcts_ladder_rollbacks"] += 1.0
                            rung_record = {
                                "simulations": float(simulations),
                                "games": float(rung_games),
                                "accepted": float(accepted),
                                "gate_score": rung_score,
                                "best_gate_score": best_mcts_score,
                                "arena_rebased": float(arena_rebased),
                                "arena_baseline": baseline_gate,
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
                    if belief_calibration_summary is not None:
                        round_training_summary["population"] = population_summary
                        database.finalise_candidate(candidate_id, promoted_path=None)
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            {
                                "promoted": 0.0,
                                "valid": 1.0,
                                "reason": "belief_calibration",
                                "belief_calibration": belief_calibration_summary,
                            },
                        )
                        run.log(
                            round=round_index,
                            belief_calibration=1.0,
                            belief_action_equivalence_rate=(
                                belief_calibration_summary["rejection_rate"]
                            ),
                            belief_action_equivalence_ready=(
                                belief_calibration_summary["ready"]
                            ),
                            promoted=0.0,
                        )
                        database.flush()
                        last_training = dict(round_training_summary)
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        # A calibration regression restores the candidate to
                        # the retained challenger weights. That checkpoint
                        # remains the strongest known continuation and must
                        # stay attached to the next round; clearing it here
                        # would make the next MCTS round restart from an
                        # older generic belief apprentice.
                        preserve_restored_challenger = bool(
                            continue_mcts_challenger
                            and belief_calibration_summary.get(
                                "held_out_rollback", 0.0
                            )
                        )
                        if preserve_restored_challenger:
                            challenger_path = previous_challenger_path
                            challenger_id = previous_challenger_id
                            print(
                                "league belief calibration regression | "
                                "continuing next round from restored challenger: "
                                f"{challenger_path}",
                                flush=True,
                            )
                        else:
                            challenger_path = None
                            challenger_id = None
                        del candidate, champion, frozen_models
                        continue

                    # A rung whose preflight proved that search is not better
                    # than raw policy is deliberately skipped.  It must not
                    # later become an impossible requirement for a deeper
                    # rung which did pass its own preflight and absorption
                    # gate.
                    admitted_mcts_rungs = [
                        rung
                        for rung in population_summary["mcts_ladder"]
                        if rung.get("reason")
                        not in {
                            "mcts_preflight_not_better_than_raw",
                            "search_evaluation_invalid",
                            "raw_evaluation_invalid",
                        }
                    ]
                    expected_mcts_rungs = len(admitted_mcts_rungs)
                    completed_mcts_curriculum = (
                        not bool(population_games)
                        or (
                            expected_mcts_rungs > 0
                            and int(population_summary["mcts_ladder_accepted"])
                            == expected_mcts_rungs
                        )
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
                        retained_verified_rung = verified_mcts_path is not None
                        if retained_verified_rung:
                            database.retain_candidate(
                                candidate_id, verified_mcts_path
                            )
                            challenger_path = verified_mcts_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        mcts_collected_games = sum(
                            float(row.get("training", {}).get(
                                "population_pool_games", 0.0
                            ))
                            for row in population_summary["mcts_ladder"]
                        )
                        mcts_optimisation_updates = sum(
                            float(row.get("training", {}).get(
                                "population_pool_updates", 0.0
                            ))
                            for row in population_summary["mcts_ladder"]
                        )
                        mcts_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 1.0,
                            "direct_population_ready": 1.0,
                            # Preflight is evaluation only.  Keep its state
                            # distinct from collection and optimiser work so
                            # a skipped run never looks like training.
                            "mcts_started": float(mcts_collected_games > 0),
                            "mcts_preflight_performed": float(
                                bool(population_summary["mcts_ladder"])
                            ),
                            "mcts_collection_games": mcts_collected_games,
                            "mcts_optimisation_updates": mcts_optimisation_updates,
                            "mcts_ready_for_promotion_gate": 0.0,
                            "mcts_ladder_accepted": float(
                                population_summary["mcts_ladder_accepted"]
                            ),
                            "mcts_ladder_required": float(
                                expected_mcts_rungs
                            ),
                            "retained_challenger": float(retained_verified_rung),
                            "verified_mcts_path": (
                                str(verified_mcts_path)
                                if verified_mcts_path is not None else ""
                            ),
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
                            retained_challenger=float(retained_verified_rung),
                            promoted=0.0,
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league MCTS curriculum incomplete | final gate "
                            "skipped | accepted="
                            f"{int(population_summary['mcts_ladder_accepted'])}/"
                            f"{expected_mcts_rungs}"
                            + (
                                f" | retained {verified_mcts_path}"
                                if retained_verified_rung else ""
                            ),
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
                            belief_action_mode=population_belief_search_mode,
                            excluded_map_seeds=known_training_map_seeds,
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
                        challenger_belief_summary: Dict[str, Any] = {}
                        if retain_challenger and round_champion_mode == "mcts":
                            # MCTS updates the shared/belief parameters. A
                            # marker inherited from the pre-rung apprentice
                            # would certify different weights, so measure this
                            # exact candidate before retaining it for a later
                            # MCTS-only round.
                            challenger_held_out = train_population_pool(
                                replace(
                                    round_training,
                                    seed=belief_validation_seed,
                                    policy_weight=0.0,
                                    value_weight=0.0,
                                ),
                                model_config,
                                mcts,
                                games=belief_calibration_eval_games,
                                updates_per_game=0,
                                candidate=candidate,
                                opponent_models=frozen_models,
                                opponent_weights=probabilities,
                                anchor_model=None,
                                anchor_policy_weight=0.0,
                                anchor_value_weight=0.0,
                                critical_decision_fraction=0.0,
                                critical_replay_fraction=0.0,
                                critical_only=False,
                                selective_mcts=False,
                                reanalysis_path=None,
                                reanalysis_positions=0,
                                reanalysis_replay_fraction=0.0,
                                offline_collection=True,
                                belief_only=True,
                                raw_policy_only=True,
                                num_threads=population_workers,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                belief_action_mode="strict",
                                max_envs=min(num_envs, population_max_envs),
                                reserved_seeds=evaluation_map_seeds,
                                run=run.child(
                                    f"round_{round_index:03d}/challenger_belief_held_out"
                                ),
                            )
                            challenger_checked = int(
                                challenger_held_out.get(
                                    "belief_action_equivalence_checked_particles",
                                    0.0,
                                )
                            )
                            challenger_rejected = int(
                                challenger_held_out.get(
                                    "belief_action_equivalence_rejected_particles",
                                    0.0,
                                )
                            )
                            challenger_rate = float(
                                challenger_rejected / max(1, challenger_checked)
                            )
                            challenger_belief_ready = bool(
                                challenger_checked
                                >= belief_action_equivalence_min_particles
                                and challenger_rate
                                <= belief_action_equivalence_target
                            )
                            challenger_belief_summary = {
                                "checked_particles": float(challenger_checked),
                                "rejected_particles": float(challenger_rejected),
                                "rejection_rate": challenger_rate,
                                "ready": float(challenger_belief_ready),
                                "held_out": challenger_held_out,
                            }
                            retain_challenger = challenger_belief_ready
                            if not retain_challenger:
                                print(
                                    "league challenger belief validation failed | "
                                    f"mismatch={challenger_rate:.2%} | "
                                    f"required<={belief_action_equivalence_target:.2%}",
                                    flush=True,
                                )
                        if retain_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            retained_marker_path = retained_path.with_suffix(
                                retained_path.suffix + ".json"
                            )
                            if round_champion_mode == "mcts":
                                _save_belief_apprentice_state(
                                    retained_marker_path,
                                    {
                                        "champion_path": str(
                                            champion_path.resolve()
                                        ),
                                        "champion_fingerprint": (
                                            _checkpoint_fingerprint(champion_path)
                                        ),
                                        "apprentice_fingerprint": (
                                            _checkpoint_fingerprint(retained_path)
                                        ),
                                        "validation_signature": (
                                            belief_validation_signature
                                        ),
                                        "ready": True,
                                        "checked_particles": int(
                                            challenger_belief_summary[
                                                "checked_particles"
                                            ]
                                        ),
                                        "rejected_particles": int(
                                            challenger_belief_summary[
                                                "rejected_particles"
                                            ]
                                        ),
                                        "rejection_rate": float(
                                            challenger_belief_summary[
                                                "rejection_rate"
                                            ]
                                        ),
                                        "held_out_rejection_rate": float(
                                            challenger_belief_summary[
                                                "rejection_rate"
                                            ]
                                        ),
                                        "held_out_games": (
                                            belief_calibration_eval_games
                                        ),
                                        "target_rejection_rate": (
                                            belief_action_equivalence_target
                                        ),
                                        "minimum_particles": (
                                            belief_action_equivalence_min_particles
                                        ),
                                        "value_bootstrap_champion_fingerprint": (
                                            bootstrap_champion_fingerprint
                                        ),
                                        "round": round_index,
                                    },
                                )
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
                        if challenger_belief_summary:
                            gate_summary["retained_challenger_belief"] = (
                                challenger_belief_summary
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
                "league_training_stage": training_stage,
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
                summary["league_last_mcts_preflight_performed"] = float(
                    bool(population_rung_rows)
                )
                summary["league_last_mcts_started"] = float(
                    any(
                        float(row.get("training", {}).get(
                            "population_pool_games", 0.0
                        )) > 0.0
                        for row in population_rung_rows
                    )
                )
                summary["league_last_mcts_optimisation_updates"] = float(
                    sum(
                        float(row.get("training", {}).get(
                            "population_pool_updates", 0.0
                        ))
                        for row in population_rung_rows
                    )
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
