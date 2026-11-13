"""Multi-checkpoint training and evaluation on one native ``SelfPlayPool``.

The pool owns every live game and MCTS tree.  Python only routes dense belief
and leaf rows through the model selected by PolyEnv's ``agent_id`` field, then
scatters neural outputs back to the original native order.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.full_actions import exact_self_play_pool
from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..evaluation.matches import evaluate_direct_matches
from ..data.player_context import flatten_player_context, polyenv_map_type
from ..data.polyenv_features import (
    native_batch_to_tensors,
    native_global_features,
    native_capture_flags,
    stack_examples,
)
from ..infrastructure.device import select_device
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.belief_mcts import BeliefMCTS
from ..planning.action_pruning import (
    ActionPruningStats,
    prune_search_logits,
    restrict_root_policy_to_candidates,
)
from .global_mcts_train import _native_leaf_bias
from .belief_recovery import (
    BeliefQualityError,
    BeliefRecoveryRequired,
    capped_update_target,
    effective_replay_epochs,
    ismcts_action_difference_rejections,
    ismcts_rejected_envs,
    particle_temperature,
    partition_recovery_envs,
    recovery_rate_exceeded,
    replace_rejected_particles,
    replacement_seeds,
)
from .optimisation import train_batch
from .mcts_exploration import add_dirichlet_noise_to_root_logits, self_play_temperature
from .persistent_reanalysis import PersistentReanalyseBuffer, ReanalysePosition
from .replay_buffer import ReplayBuffer, TrainingExample
from .run_artifacts import TrainingRun
from .self_play_pool_train import (
    _allocate_native_buffers,
    _belief_rejection_kind,
    _finish_trajectory,
    _root_examples,
)
from .value_targets import Transition


GameCallback = Callable[[Mapping[str, Any]], None]

_EVALUATION_MIN_BELIEF_RETRIES = 64
_EVALUATION_POSITION_RECOVERY_CYCLES = 2
# A failed belief position must not turn an evaluation gate into an unbounded
# stream of freshly generated maps.  A gate with this many failed pairs is
# invalid evidence, not a reason to keep running forever.
_MAX_EVALUATION_PAIR_REPLACEMENTS = 8


def _mcts_leaf_wave_capacity(
    simulations: int,
    max_pending_leaves: int,
    wave: int,
) -> int:
    """Return the per-root leaf budget for one native MCTS wave."""
    if wave < 0 or simulations < 1 or max_pending_leaves < 1:
        return 0
    if wave == 0:
        return 1
    return max(
        0,
        min(
            int(max_pending_leaves),
            int(simulations) - 1 - (int(wave) - 1) * int(max_pending_leaves),
        ),
    )


def _sample_reanalysis_positions_excluding_reserved(
    positions: Sequence[ReanalysePosition],
    count: int,
    rng: np.random.Generator,
    reserved_seeds: set[int],
) -> tuple[tuple[ReanalysePosition, ...], int]:
    """Sample old MCTS targets without contaminating held-out map seeds.

    Persistent reanalysis survives across league invocations, while the
    current invocation may reserve a different evaluation schedule.  Keep the
    retained row on disk for a future non-overlapping round, but never replay
    it into this round's optimiser.
    """
    eligible = tuple(
        position
        for position in positions
        if int(position.map_seed) not in reserved_seeds
    )
    excluded = len(positions) - len(eligible)
    if count <= 0 or not eligible:
        return (), excluded
    chosen = rng.choice(
        len(eligible), size=min(int(count), len(eligible)), replace=False
    )
    return tuple(eligible[int(index)] for index in np.asarray(chosen)), excluded


def _filter_replay_examples_excluding_reserved(
    examples: Sequence[TrainingExample],
    reserved_seeds: set[int],
) -> tuple[list[TrainingExample], int, int]:
    """Keep replay provenance disjoint from this run's held-out arenas.

    Legacy rows do not have ``map_seed``.  Once a run has a held-out arena we
    must reject them rather than assume they are safe: a previous training
    seed can become an evaluation seed after a CLI seed change.
    """
    if not reserved_seeds:
        return list(examples), 0, 0
    eligible: list[TrainingExample] = []
    reserved_excluded = 0
    unknown_excluded = 0
    for example in examples:
        map_seed = getattr(example, "map_seed", None)
        if map_seed is None:
            unknown_excluded += 1
        elif int(map_seed) in reserved_seeds:
            reserved_excluded += 1
        else:
            eligible.append(example)
    return eligible, reserved_excluded, unknown_excluded


def _compact_belief_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> str:
    """Format fog-safe rejection details for logs and SQLite summaries."""
    rows: list[str] = []
    for row in diagnostics[:max(0, int(limit))]:
        text = (
            f"env={int(row.get('env_id', -1))}"
            f"/player={int(row.get('player', -1))}"
            f"/particle={int(row.get('particle', -1))}"
            f"/{row.get('reason', 'unknown')}"
        )
        if row.get("error"):
            text += f"/{row['error']}"
        missing = row.get("missing_actions", ())
        extra = row.get("extra_actions", ())
        if missing:
            first = missing[0]
            text += (
                "/missing="
                f"{first.get('type_fullname', 'unknown')}"
                f"({int(first.get('source_index', -1))}->{int(first.get('target_index', -1))})"
            )
        if extra:
            first = extra[0]
            text += (
                "/extra="
                f"{first.get('type_fullname', 'unknown')}"
                f"({int(first.get('source_index', -1))}->{int(first.get('target_index', -1))})"
            )
        rows.append(text)
    return "; ".join(rows)


def _evaluation_search_with_recovery(
    search: "RoutedPoolSearch",
    request: Mapping[str, np.ndarray],
    active: np.ndarray,
    *,
    mcts_mask: np.ndarray | None = None,
    recovery_cycles: int = _EVALUATION_POSITION_RECOVERY_CYCLES,
) -> tuple[Dict[str, np.ndarray], tuple[BeliefRecoveryRequired, ...]]:
    """Retry one held-out position without changing its state, map, or seed."""
    if recovery_cycles < 1:
        raise ValueError("evaluation recovery_cycles must be positive")
    failures: list[BeliefRecoveryRequired] = []
    for _ in range(int(recovery_cycles)):
        try:
            root = search.search(
                request,
                active,
                root_temperature=0.0,
                belief_temperature=0.0,
                prior_scale=0.0,
                mcts_mask=mcts_mask,
            )
            return root, tuple(failures)
        except BeliefRecoveryRequired as error:
            failures.append(error)
    final = failures[-1]
    # Preserve all fog-safe diagnostics seen across independent stochastic
    # cycles. This makes the final invalid evaluation actionable in SQLite.
    combined = tuple(
        row for failure in failures for row in failure.diagnostics
    )[:64]
    raise BeliefRecoveryRequired(
        final.env_ids,
        f"{final}; position recovery exhausted after {len(failures)} cycles",
        combined,
    ) from final


def _pool_types() -> tuple[Any, Any, Any, Any]:
    try:
        from PolyEnv import Bardur, Imperius, MapType, SelfPlayPool
    except ImportError as error:  # pragma: no cover - depends on local binding
        raise RuntimeError("PolyEnv with the multi-agent SelfPlayPool API is required") from error
    required = (
        "all_player_belief_requests",
        "completed_batch_spec",
        "drain_completed_into",
        "reset_slots_into",
        "select_leaves_masked",
        "select_leaves_masked_into",
    )
    missing = [name for name in required if not hasattr(SelfPlayPool, name)]
    if missing:
        raise RuntimeError(
            "Installed PolyEnv is too old for routed league play; missing: "
            + ", ".join(missing)
        )
    return Bardur, Imperius, MapType, SelfPlayPool


def require_routed_self_play_pool() -> None:
    """Fail before a long league round when the installed binding is stale."""
    _pool_types()


def _slice_rows(batch: Mapping[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice a dense native packet while preserving its field names."""
    return {name: np.asarray(value)[rows] for name, value in batch.items()}


def _freeze_reference_model(
    model: PolicyValueBeliefNet,
    device: torch.device,
) -> PolicyValueBeliefNet:
    """Move a league reference to the device and make mutation impossible."""
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _reference_policy(
    model: PolicyValueBeliefNet,
    batch: Mapping[str, np.ndarray],
    device: torch.device,
    training: TrainingConfig,
) -> np.ndarray:
    """Return a frozen checkpoint's legal policy for public root packets."""
    tensors = native_batch_to_tensors(
        dict(batch), device, training.map_size, training.map_type
    )
    with torch.inference_mode():
        output = model(
            tensors.map_tokens,
            tensors.global_features,
            tensors.action_categorical,
            tensors.action_continuous,
            tensors.action_mask,
            tensors.event_features,
            action_history=tensors.action_history,
            action_history_mask=tensors.action_history_mask,
            include_belief=False,
        )
        policy = torch.softmax(output.policy_logits, dim=-1).float().cpu().numpy()
    return policy


def _reference_policy_value(
    model: PolicyValueBeliefNet,
    batch: Mapping[str, np.ndarray],
    device: torch.device,
    training: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen policy and value in one inference pass."""
    tensors = native_batch_to_tensors(
        dict(batch), device, training.map_size, training.map_type
    )
    with torch.inference_mode():
        output = model(
            tensors.map_tokens,
            tensors.global_features,
            tensors.action_categorical,
            tensors.action_continuous,
            tensors.action_mask,
            tensors.event_features,
            action_history=tensors.action_history,
            action_history_mask=tensors.action_history_mask,
            include_belief=False,
        )
        policy = torch.softmax(
            output.policy_logits, dim=-1
        ).float().cpu().numpy()
        value = output.value.float().cpu().numpy()
    return policy, value


def _policy_distance(
    search_policy: np.ndarray,
    reference_policy: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[int, int, float]:
    """Return decisions, top-action agreements and summed L1 distance."""
    decisions = 0
    agreements = 0
    l1_sum = 0.0
    for index in range(search_policy.shape[0]):
        count = int(action_mask[index].sum())
        if count <= 0:
            continue
        search_row = np.asarray(search_policy[index, :count], dtype=np.float32)
        reference_row = np.asarray(reference_policy[index, :count], dtype=np.float32)
        search_row = search_row / max(float(search_row.sum()), 1e-8)
        reference_row = reference_row / max(float(reference_row.sum()), 1e-8)
        agreements += int(search_row.argmax() == reference_row.argmax())
        l1_sum += float(np.abs(search_row - reference_row).sum())
        decisions += 1
    return decisions, agreements, l1_sum


def _mark_critical_mcts_examples(
    examples: Sequence[TrainingExample],
    mcts_policy: np.ndarray,
    raw_policy: np.ndarray,
    action_mask: np.ndarray,
    *,
    fraction: float,
    minimum_score: float,
) -> tuple[int, float]:
    """Mark the strongest public disagreements between MCTS and the policy.

    The score rewards a confident MCTS recommendation that the raw policy
    assigns little probability to.  Agreement with MCTS is deliberately not
    replay-prioritised: it is already represented by ordinary MCTS examples.
    Selecting a bounded top fraction per vector root avoids filling replay
    with many near-identical end-turn decisions from one long game.
    """
    if len(examples) != int(mcts_policy.shape[0]):
        raise ValueError("critical MCTS examples must match policy rows")
    if raw_policy.shape != mcts_policy.shape or action_mask.shape != mcts_policy.shape:
        raise ValueError("critical MCTS policy tensors must have matching shapes")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("critical MCTS fraction must be between zero and one")
    if minimum_score < 0.0:
        raise ValueError("critical MCTS minimum score must be non-negative")
    scores = np.zeros(len(examples), dtype=np.float32)
    for index in range(len(examples)):
        count = int(action_mask[index].sum())
        if count <= 1:
            continue
        search_row = np.asarray(
            mcts_policy[index, :count], dtype=np.float32
        ).copy()
        raw_row = np.asarray(raw_policy[index, :count], dtype=np.float32).copy()
        search_row /= max(float(search_row.sum()), 1e-8)
        raw_row /= max(float(raw_row.sum()), 1e-8)
        search_top = int(search_row.argmax())
        if search_top == int(raw_row.argmax()):
            continue
        # A score of 0.15 requires, for example, MCTS to put 50% probability
        # on a move that the raw policy assigns at most 20%.
        scores[index] = search_row[search_top] * max(
            0.0, float(search_row[search_top] - raw_row[search_top])
        )
    for example, score in zip(examples, scores):
        example.critical_priority = False
        example.critical_score = float(score)
    eligible = np.flatnonzero(scores >= minimum_score)
    if not eligible.size or fraction <= 0.0:
        return 0, 0.0
    keep = min(
        int(eligible.size),
        max(1, int(np.ceil(len(examples) * fraction))),
    )
    selected = eligible[np.argsort(scores[eligible], kind="stable")[-keep:]]
    for index in selected:
        examples[int(index)].critical_priority = True
    return int(selected.size), float(scores[selected].sum())


def _blend_reanalysis_policy_target(
    visits: np.ndarray,
    anchor_policy: np.ndarray | None,
    anchor_policy_weight: float,
) -> np.ndarray:
    """Apply the same frozen-policy anchor used by fresh MCTS examples."""
    target = np.asarray(visits, dtype=np.float32).copy()
    if anchor_policy is None or anchor_policy_weight <= 0.0:
        return target
    anchor = np.asarray(anchor_policy, dtype=np.float32)
    if anchor.shape != target.shape:
        raise ValueError("reanalysis anchor policy shape differs from MCTS visits")
    total = float(anchor.sum())
    if not np.isfinite(total) or total <= 0.0:
        return target
    anchor /= total
    weight = float(np.clip(anchor_policy_weight, 0.0, 1.0))
    target = (1.0 - weight) * target + weight * anchor
    target /= max(float(target.sum()), 1e-8)
    return target


def _reanalyse_critical_positions(
    positions: Sequence[ReanalysePosition],
    *,
    candidate: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    device: torch.device,
    minimum_score: float,
    visible_action_history: int,
    anchor_model: PolicyValueBeliefNet | None = None,
    anchor_policy_weight: float = 0.0,
    anchor_value_weight: float = 0.0,
) -> tuple[list[TrainingExample], int]:
    """Refresh old MCTS targets by replaying their public game prefixes.

    PolyEnv game states are not serialisable through ``SelfPlayPool``.  Its
    scalar ``GameEnv`` is deterministic from map seed plus legal action ids,
    however, so a retained critical position can be reconstructed exactly and
    searched again by the current apprentice.  The terminal value target is
    intentionally retained from the original completed game; only the stale
    MCTS policy target is reanalysed.
    """
    if not positions:
        return [], 0
    compatible_positions = tuple(
        position
        for position in positions
        if int(position.map_size) == int(training.map_size)
        and str(position.map_type) == str(training.map_type)
        and int(position.visible_action_history) == int(visible_action_history)
    )
    # Configuration changes invalidate native ids by design. Silently
    # skipping them is safer than calling PolyEnv with a prefix whose action
    # semantics belonged to a different map generator.
    if not compatible_positions:
        return [], 0
    try:
        from PolyEnv import Bardur, GameEnv, Imperius
    except ImportError as error:  # pragma: no cover - depends on local binding
        raise RuntimeError("PolyEnv GameEnv is required for MCTS reanalysis") from error

    planner = BeliefMCTS(
        candidate,
        device,
        replace(mcts, temperature=0.0),
        map_size=training.map_size,
        map_type=training.map_type,
        visible_action_history=visible_action_history,
        event_summary_enabled=False,
    )
    refreshed: list[TrainingExample] = []
    failed = 0
    candidate.eval()
    for position in compatible_positions:
        try:
            env = GameEnv(
                seed=int(position.map_seed),
                map_size=int(training.map_size),
                players=(Bardur, Imperius),
                map_type=training.map_type,
            )
            for action_id in position.action_prefix:
                result = env.step_fast(int(action_id))
                if not bool(result[0]):
                    raise RuntimeError("stored MCTS action was rejected during replay")
            if env.is_done() or int(env.current_player()) != int(position.player_id):
                raise RuntimeError("stored MCTS position no longer has its candidate turn")
            decision = planner.decide(env)
            previous = position.example
            visits = np.asarray(decision.visit_counts, dtype=np.float32)
            if visits.shape != previous.policy_target.shape:
                raise RuntimeError("replayed MCTS action layout differs from retained target")
            # Priors are a raw-policy fallback, not an MCTS target. Persisting
            # them as a refreshed critical example would claim that search
            # succeeded even when every belief particle was rejected.
            if (
                int(decision.valid_particles) <= 0
                or float(visits.sum()) <= 0.0
            ):
                raise RuntimeError(
                    "reanalysis produced no valid particle-MCTS visits"
                )
            visits /= max(float(visits.sum()), 1e-8)
            tensors = stack_examples([previous], device)
            with torch.inference_mode():
                output = candidate(
                    tensors["map_tokens"],
                    tensors["global_features"],
                    tensors["action_categorical"],
                    tensors["action_continuous"],
                    tensors["action_mask"],
                    tensors["event_features"],
                    action_history=tensors["action_history"],
                    action_history_mask=tensors["action_history_mask"],
                    include_belief=False,
                )
                # A reanalysed row must not keep the old anchor target or
                # loss coefficient: those belonged to the checkpoint and
                # command-line settings that created the stale row.  Anchor
                # supervision is a current-run regulariser, so regenerate it
                # from the current frozen anchor (or disable it explicitly).
                anchor_value_target = 0.0
                anchor_policy: np.ndarray | None = None
                if anchor_model is not None and anchor_value_weight > 0.0:
                    anchor_output = anchor_model(
                        tensors["map_tokens"],
                        tensors["global_features"],
                        tensors["action_categorical"],
                        tensors["action_continuous"],
                        tensors["action_mask"],
                        tensors["event_features"],
                        action_history=tensors["action_history"],
                        action_history_mask=tensors["action_history_mask"],
                        include_belief=False,
                    )
                    anchor_value_target = float(anchor_output.value[0].item())
                elif anchor_model is not None and anchor_policy_weight > 0.0:
                    anchor_output = anchor_model(
                        tensors["map_tokens"],
                        tensors["global_features"],
                        tensors["action_categorical"],
                        tensors["action_continuous"],
                        tensors["action_mask"],
                        tensors["event_features"],
                        action_history=tensors["action_history"],
                        action_history_mask=tensors["action_history_mask"],
                        include_belief=False,
                    )
                else:
                    anchor_output = None
                if anchor_output is not None and anchor_policy_weight > 0.0:
                    anchor_policy = torch.softmax(
                        anchor_output.policy_logits[0, : visits.size].float(), dim=-1
                    ).cpu().numpy()
            raw = torch.softmax(
                output.policy_logits[0, : visits.size].float(), dim=-1
            ).cpu().numpy()
            top = int(visits.argmax())
            score = float(visits[top] * max(0.0, visits[top] - raw[top]))
            if score < minimum_score or top == int(raw.argmax()):
                continue
            policy_target = _blend_reanalysis_policy_target(
                visits, anchor_policy, anchor_policy_weight
            )
            updated = copy.deepcopy(previous)
            updated.map_seed = int(position.map_seed)
            updated.policy_target = policy_target.astype(np.float32, copy=False)
            updated.critical_priority = True
            updated.critical_score = score
            updated.reanalysis_priority = True
            updated.anchor_value_target = anchor_value_target
            updated.anchor_value_loss_weight = (
                float(anchor_value_weight)
                if anchor_model is not None and anchor_value_weight > 0.0
                else 0.0
            )
            position.example = copy.deepcopy(updated)
            position.refreshes += 1
            refreshed.append(updated)
        except Exception:
            # One replay can become invalid after a PolyEnv action-id format
            # change.  Keep the rest of the persistent buffer usable.
            failed += 1
    return refreshed, failed


def _validate_agents(agent_ids: np.ndarray, model_count: int, label: str) -> None:
    if agent_ids.size == 0:
        return
    smallest = int(agent_ids.min())
    largest = int(agent_ids.max())
    if smallest < 0 or largest >= model_count:
        raise RuntimeError(
            f"{label} contains agent_id outside [0, {model_count - 1}]: "
            f"min={smallest}, max={largest}"
        )


@dataclass
class SearchCounters:
    leaf_rows: int = 0
    leaf_waves: int = 0
    belief_retries: int = 0
    belief_fallback_rows: int = 0
    belief_rejections: int = 0
    belief_rejected_particles: int = 0
    belief_rejected_envs: int = 0
    belief_resampled_envs: int = 0
    belief_retry_successes: int = 0
    belief_conservative_attempts: int = 0
    belief_slot_resets: int = 0
    belief_discarded_prefixes: int = 0
    engine_action_constraint_queries: int = 0
    engine_action_constraint_particles: int = 0
    engine_action_constraint_missing_actions: int = 0
    engine_action_constraint_extra_actions: int = 0
    engine_action_constraint_tiles: int = 0
    engine_action_constraint_score_sum: float = 0.0
    belief_action_equivalence_checked_particles: int = 0
    belief_action_equivalence_rejected_particles: int = 0
    belief_action_equivalence_rejected_envs: int = 0
    belief_particle_diagnostics: Counter[str] = field(default_factory=Counter)
    belief_build_errors: dict[str, dict[str, int]] = field(default_factory=dict)
    # ``root_shield`` keeps public action legality authoritative even when a
    # learned determinisation has a different private action set. These are
    # separate from action-pruning counters: shielding corrects a belief
    # mismatch, whereas pruning is an optional policy-speed heuristic.
    root_shielded_roots: int = 0
    root_shield_missing_public_actions: int = 0
    root_shield_hallucinated_actions: int = 0
    root_shield_uniform_fallbacks: int = 0
    root_env_decisions: int = 0
    # Selective MCTS is decided from public root packets before native tree
    # traversal.  Keep this separate from leaf rows: one selected decision
    # can produce several native leaves and several search waves.
    selective_mcts_decisions: int = 0
    direct_policy_decisions: int = 0
    selective_mcts_audit_decisions: int = 0
    # A random stratum of public strategic decisions is searched even when a
    # weak policy is confidently wrong.  Entropy alone cannot discover those.
    selective_mcts_strategic_available: int = 0
    selective_mcts_strategic_coverage_decisions: int = 0
    selective_raw_entropy_sum: float = 0.0
    selective_direct_entropy_sum: float = 0.0
    search_policy_rows: int = 0
    search_pruned_rows: int = 0
    search_legal_actions: int = 0
    search_selected_actions: int = 0
    search_mandatory_actions: int = 0
    search_random_actions: int = 0
    search_prior_mass_retained_sum: float = 0.0
    search_restricted_roots: int = 0
    search_missing_root_candidates: int = 0
    search_root_visits: int = 0
    search_outside_candidate_visits: int = 0
    search_unfiltered_choices_outside: int = 0
    search_root_policy_rows: int = 0
    search_root_policy_entropy_sum: float = 0.0
    search_root_top_visit_share_sum: float = 0.0

    def add_action_pruning(self, stats: ActionPruningStats) -> None:
        self.search_policy_rows += int(stats.rows)
        self.search_pruned_rows += int(stats.pruned_rows)
        self.search_legal_actions += int(stats.legal_actions)
        self.search_selected_actions += int(stats.selected_actions)
        self.search_mandatory_actions += int(stats.mandatory_actions)
        self.search_random_actions += int(stats.random_actions)
        self.search_prior_mass_retained_sum += float(stats.prior_mass_retained_sum)

    def add(self, other: "SearchCounters") -> None:
        """Merge counters from one short-lived routed pool."""
        for name in self.__dataclass_fields__:
            if name == "belief_particle_diagnostics":
                self.belief_particle_diagnostics.update(other.belief_particle_diagnostics)
            elif name == "belief_build_errors":
                for message, example in other.belief_build_errors.items():
                    if message in self.belief_build_errors:
                        self.belief_build_errors[message]["count"] += example["count"]
                    else:
                        self.belief_build_errors[message] = example.copy()
            elif name in {
                "search_prior_mass_retained_sum",
                "search_root_policy_entropy_sum",
                "search_root_top_visit_share_sum",
                "engine_action_constraint_score_sum",
                "selective_raw_entropy_sum",
                "selective_direct_entropy_sum",
            }:
                setattr(self, name, float(getattr(self, name)) + float(getattr(other, name)))
            else:
                setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))


def _action_pruning_metrics(counters: SearchCounters) -> Dict[str, float]:
    """Return stable aggregate metrics for logs, summaries and SQLite."""
    rows = max(1, int(counters.search_policy_rows))
    legal = max(1, int(counters.search_legal_actions))
    return {
        "search_policy_rows": float(counters.search_policy_rows),
        "search_pruned_rows": float(counters.search_pruned_rows),
        "search_mean_legal_actions": float(counters.search_legal_actions / rows),
        "search_mean_selected_actions": float(counters.search_selected_actions / rows),
        "search_action_reduction": float(
            1.0 - counters.search_selected_actions / legal
        ),
        "search_prior_mass_retained": float(
            counters.search_prior_mass_retained_sum / rows
        ),
        "search_mandatory_actions": float(counters.search_mandatory_actions),
        "search_random_actions": float(counters.search_random_actions),
        "search_restricted_roots": float(counters.search_restricted_roots),
        "search_missing_root_candidates": float(
            counters.search_missing_root_candidates
        ),
        "search_outside_candidate_visit_rate": float(
            counters.search_outside_candidate_visits
            / max(1, counters.search_root_visits)
        ),
        "search_unfiltered_choices_outside_rate": float(
            counters.search_unfiltered_choices_outside
            / max(1, counters.search_restricted_roots)
        ),
        "search_root_policy_entropy": float(
            counters.search_root_policy_entropy_sum
            / max(1, counters.search_root_policy_rows)
        ),
        "search_root_top_visit_share": float(
            counters.search_root_top_visit_share_sum
            / max(1, counters.search_root_policy_rows)
        ),
        "root_shielded_roots": float(counters.root_shielded_roots),
        "root_shield_missing_public_actions": float(
            counters.root_shield_missing_public_actions
        ),
        "root_shield_hallucinated_actions": float(
            counters.root_shield_hallucinated_actions
        ),
        "root_shield_uniform_fallbacks": float(
            counters.root_shield_uniform_fallbacks
        ),
        "selective_mcts_decisions": float(counters.selective_mcts_decisions),
        "direct_policy_decisions": float(counters.direct_policy_decisions),
        "selective_mcts_fraction": float(
            counters.selective_mcts_decisions / max(1, counters.root_env_decisions)
        ),
        "selective_mcts_audit_decisions": float(
            counters.selective_mcts_audit_decisions
        ),
        "selective_mcts_strategic_available": float(
            counters.selective_mcts_strategic_available
        ),
        "selective_mcts_strategic_coverage_decisions": float(
            counters.selective_mcts_strategic_coverage_decisions
        ),
        "selective_mcts_strategic_coverage_rate": float(
            counters.selective_mcts_strategic_coverage_decisions
            / max(1, counters.selective_mcts_decisions)
        ),
        "selective_mcts_raw_entropy": float(
            counters.selective_raw_entropy_sum
            / max(1, counters.selective_mcts_decisions)
        ),
        "direct_policy_raw_entropy": float(
            counters.selective_direct_entropy_sum
            / max(1, counters.direct_policy_decisions)
        ),
    }


@dataclass(frozen=True)
class RootShieldStats:
    """Additive diagnostics from projecting an MCTS root onto public actions."""

    roots: int = 0
    missing_public_actions: int = 0
    hallucinated_actions: int = 0
    uniform_fallbacks: int = 0


def shield_root_policy_with_public_actions(
    root: Mapping[str, np.ndarray],
    request: Mapping[str, np.ndarray],
    active: np.ndarray,
) -> RootShieldStats:
    """Make the executed root policy exactly match PolyEnv's public actions.

    An ISMCTS particle is allowed to be wrong about hidden state, but it must
    never add or remove an action available to the real player. This routine
    runs *after* native search, maps retained visit mass by stable action id,
    and restores the public action order used by replay and ``step_into``.
    Missing public actions receive zero visit mass; a fully incompatible root
    falls back to a public uniform distribution. The chosen action is thus
    always legal without regenerating a particle.
    """
    required = ("action_id", "action_mask", "policy")
    if any(name not in root for name in required) or any(
        name not in request for name in required[:2]
    ):
        raise ValueError("root shield requires action_id, action_mask and policy packets")
    root_ids = np.asarray(root["action_id"])
    # ``uint8 -> bool`` allocates a copy. Preserve the backing native field
    # so shielding changes the mask seen by later root selection/stepping.
    root_mask_storage = np.asarray(root["action_mask"])
    root_mask = np.asarray(root_mask_storage, dtype=bool)
    policy = np.asarray(root["policy"])
    visit_count = (
        None
        if "visit_count" not in root
        else np.asarray(root["visit_count"])
    )
    public_ids = np.asarray(request["action_id"])
    public_mask = np.asarray(request["action_mask"], dtype=bool)
    enabled = np.asarray(active, dtype=bool)
    if (
        root_ids.shape != root_mask.shape
        or root_ids.shape != policy.shape
        or (visit_count is not None and visit_count.shape != root_ids.shape)
        or public_ids.shape != public_mask.shape
        or root_ids.shape[0] != public_ids.shape[0]
        or enabled.shape != (root_ids.shape[0],)
    ):
        raise ValueError("root shield packets must agree on batch rows and action shapes")

    roots = missing = hallucinated = fallbacks = 0
    for env in np.flatnonzero(enabled):
        public_positions = np.flatnonzero(public_mask[env])
        if public_positions.size == 0:
            raise RuntimeError(f"live root {env} has no public legal actions")
        if public_positions.size > root_ids.shape[1]:
            raise RuntimeError("public action packet exceeds root-policy capacity")
        native_positions = np.flatnonzero(root_mask[env])
        native_ids = root_ids[env, native_positions]
        native_policy = policy[env, native_positions]
        native_visits = (
            None if visit_count is None else visit_count[env, native_positions]
        )
        public_action_ids = public_ids[env, public_positions]
        # Action ids are stable and unique inside one native packet. ``dict``
        # avoids assuming that an incompatible particle kept the same order.
        visit_mass = {
            int(action_id): float(probability)
            for action_id, probability in zip(native_ids, native_policy)
        }
        visits_by_id = (
            {}
            if native_visits is None
            else {
                int(action_id): int(visits)
                for action_id, visits in zip(native_ids, native_visits)
            }
        )
        public_id_set = {int(action_id) for action_id in public_action_ids}
        missing += sum(int(action_id) not in visit_mass for action_id in public_id_set)
        hallucinated += sum(int(action_id) not in public_id_set for action_id in native_ids)

        # Replay rows assume root policy column j corresponds to public action
        # column j. Compact the root packet into that public order instead of
        # leaving holes in ``action_mask``.
        root_ids[env].fill(-1)
        root_mask[env].fill(False)
        root_mask_storage[env].fill(0)
        policy[env].fill(0.0)
        if visit_count is not None:
            visit_count[env].fill(0)
        count = int(public_action_ids.size)
        root_ids[env, :count] = public_action_ids
        root_mask[env, :count] = True
        root_mask_storage[env, :count] = 1
        if visit_count is not None:
            visit_count[env, :count] = np.asarray(
                [visits_by_id.get(int(action_id), 0) for action_id in public_action_ids],
                dtype=visit_count.dtype,
            )
        masses = np.asarray(
            [visit_mass.get(int(action_id), 0.0) for action_id in public_action_ids],
            dtype=policy.dtype,
        )
        total = float(masses.sum())
        if not np.isfinite(total) or total <= 0.0:
            masses.fill(1.0 / count)
            fallbacks += 1
        else:
            masses /= total
        policy[env, :count] = masses
        roots += 1
    return RootShieldStats(roots, missing, hallucinated, fallbacks)


class RoutedPoolSearch:
    """Persistent native pool whose neural rows are routed by ``agent_id``."""

    def __init__(
        self,
        training: TrainingConfig,
        mcts: MCTSConfig,
        models: Sequence[PolicyValueBeliefNet],
        *,
        num_envs: int,
        num_threads: int | None,
        max_pending_leaves: int,
        visible_action_history: int,
        belief_retries: int,
        seed: int,
        belief_action_mode: str = "strict",
        collect_engine_action_constraints: bool = False,
        public_raw_only: bool = False,
        search_top_actions_by_agent: Sequence[int] | None = None,
        search_random_actions_by_agent: Sequence[int] | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not models:
            raise ValueError("at least one routed model is required")
        if not 1 <= max_pending_leaves <= 8:
            raise ValueError("max_pending_leaves must be between one and eight")
        if belief_retries < 0:
            raise ValueError("belief_retries must not be negative")
        if belief_action_mode not in {"strict", "root_shield"}:
            raise ValueError("belief_action_mode must be 'strict' or 'root_shield'")
        if mcts.search_top_actions < 0 or mcts.search_random_actions < 0:
            raise ValueError("MCTS action candidate counts must not be negative")
        Bardur, Imperius, MapType, SelfPlayPool = _pool_types()
        self.training = training
        self.mcts = mcts
        self.models = list(models)
        self.search_top_actions_by_agent = tuple(
            int(value) for value in (
                search_top_actions_by_agent
                if search_top_actions_by_agent is not None
                else (mcts.search_top_actions,) * len(self.models)
            )
        )
        self.search_random_actions_by_agent = tuple(
            int(value) for value in (
                search_random_actions_by_agent
                if search_random_actions_by_agent is not None
                else (mcts.search_random_actions,) * len(self.models)
            )
        )
        if (
            len(self.search_top_actions_by_agent) != len(self.models)
            or len(self.search_random_actions_by_agent) != len(self.models)
        ):
            raise ValueError("per-agent MCTS candidate counts must match routed models")
        if min(self.search_top_actions_by_agent, default=0) < 0 or min(
            self.search_random_actions_by_agent, default=0
        ) < 0:
            raise ValueError("per-agent MCTS candidate counts must not be negative")
        self.device = select_device(training.device)
        self.num_envs = int(num_envs)
        self.max_pending_leaves = int(max_pending_leaves)
        self.belief_retries = int(belief_retries)
        self.belief_action_mode = str(belief_action_mode)
        self.collect_engine_action_constraints = bool(collect_engine_action_constraints)
        # A SelfPlayPool is an ISMCTS state machine: every native step requires
        # submitted beliefs. Fully raw evaluations are routed through
        # VectorGameEnv/GameEnv by the league instead of this pool.
        if public_raw_only:
            raise ValueError(
                "raw-only evaluation must use evaluate_direct_matches, not SelfPlayPool"
            )
        self.counters = SearchCounters()
        self._action_generator: torch.Generator | None = None
        if self.device.type in ("cpu", "cuda"):
            self._action_generator = torch.Generator(device=self.device)
            self._action_generator.manual_seed(int(seed) ^ 0x5EA2C11)
        self._capture_root_candidates = False
        self._root_candidate_action_ids: dict[int, np.ndarray] = {}
        self._root_noise_enabled = False
        self._root_noise_rng = np.random.default_rng(int(seed) ^ 0xA17C0DE)
        for model in self.models:
            model.to(self.device).eval()
        map_type = polyenv_map_type(training.map_type, MapType)
        native_threads = 0 if num_threads is None else max(1, int(num_threads))
        self.pool = exact_self_play_pool(SelfPlayPool,
            num_envs=self.num_envs,
            seed=int(seed),
            map_size=training.map_size,
            tribes=(Bardur, Imperius),
            map_type=map_type,
            num_threads=native_threads,
            max_actions=mcts.native_max_actions,
            auto_reset=False,
            c_puct=mcts.c_puct,
            visible_action_history=visible_action_history,
            require_all_actions=True,
        )
        self.pool.configure_search(self.max_pending_leaves, 1.0, 0, 0)
        if self.collect_engine_action_constraints and not callable(
            getattr(self.pool, "ismcts_belief_action_differences", None)
        ):
            raise RuntimeError(
                "Installed PolyEnv is missing SelfPlayPool."
                "ismcts_belief_action_differences required by "
                "--engine-action-constraint-weight"
            )
        self.request, request_owners = _allocate_native_buffers(
            self.pool, "belief_batch_spec", self.device
        )
        self.root, root_owners = _allocate_native_buffers(
            self.pool, "root_policy_spec", self.device
        )
        self.all_players, all_player_owners = _allocate_native_buffers(
            self.pool, "all_player_belief_batch_spec", self.device
        )
        leaf_capacity = self.num_envs * self.max_pending_leaves
        self.leaves, leaf_owners = _allocate_native_buffers(
            self.pool, "leaf_batch_spec", self.device, leaf_capacity
        )
        self.completed, completed_owners = _allocate_native_buffers(
            self.pool, "completed_batch_spec", self.device
        )
        self.leaf_capacity = leaf_capacity
        # ``leaf_batch_spec(max_leaves=...)`` is part of PolyEnv's ABI.  It
        # may contain capacity-dependent metadata in addition to the leading
        # leaf dimension, so a view into one maximum-size packet is not a
        # portable substitute for asking the native binding for the precise
        # packet required by a smaller masked wave.
        self._leaf_buffer_cache: dict[int, Dict[str, np.ndarray]] = {
            leaf_capacity: self.leaves
        }
        self._leaf_buffer_owner_cache: dict[int, Dict[str, torch.Tensor]] = {
            leaf_capacity: leaf_owners
        }
        # Pinned tensors own the memory exported through their NumPy views.
        self._buffer_owners = (
            request_owners,
            root_owners,
            all_player_owners,
            leaf_owners,
            completed_owners,
        )
        self.last_engine_action_constraint_tiles = np.zeros(
            (self.num_envs, training.map_size * training.map_size), dtype=bool
        )
        self.last_engine_action_constraint_scores = np.zeros(
            self.num_envs, dtype=np.float32
        )

    def _leaf_buffers_for_capacity(self, capacity: int) -> Dict[str, np.ndarray]:
        """Return buffers matching PolyEnv's exact native leaf-packet ABI."""
        capacity = int(capacity)
        if not 1 <= capacity <= self.leaf_capacity:
            raise ValueError("leaf buffer capacity is outside the configured pool limit")
        cached = self._leaf_buffer_cache.get(capacity)
        if cached is not None:
            return cached
        buffers, owners = _allocate_native_buffers(
            self.pool, "leaf_batch_spec", self.device, capacity
        )
        self._leaf_buffer_cache[capacity] = buffers
        self._leaf_buffer_owner_cache[capacity] = owners
        return buffers

    def reset(self, seeds: np.ndarray, assignments: np.ndarray) -> Dict[str, np.ndarray]:
        seeds = np.ascontiguousarray(seeds, dtype=np.uint32)
        assignments = np.ascontiguousarray(assignments, dtype=np.int32)
        if seeds.shape != (self.num_envs,):
            raise ValueError(f"seeds must have shape ({self.num_envs},)")
        if assignments.shape != (self.num_envs, 2):
            raise ValueError(f"assignments must have shape ({self.num_envs}, 2)")
        _validate_agents(assignments, len(self.models), "assignments")
        self.pool.reset_into(self.request, seeds=seeds, agent_ids=assignments)
        return self.request

    @torch.inference_mode()
    def direct_policy_value(
        self,
        request: Mapping[str, np.ndarray],
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one fog-safe raw policy/value pass, routed by ``agent_id``.

        The returned arrays retain the full pool row layout so callers can
        safely merge direct decisions with native MCTS roots.  Padded actions
        are always zeroed and every legal row is renormalised explicitly;
        this keeps direct execution anchored to PolyEnv's public action list.
        """
        rows = np.asarray(rows, dtype=np.int32)
        action_mask = np.asarray(request["action_mask"], dtype=bool)
        policies = np.zeros(action_mask.shape, dtype=np.float32)
        values = np.zeros(self.num_envs, dtype=np.float32)
        for model in self.models:
            model.eval()
        if rows.size == 0:
            return policies, values
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self.num_envs):
            raise ValueError("direct-policy rows must be valid pool indices")
        agents = np.asarray(request["agent_id"], dtype=np.int32)
        _validate_agents(agents[rows], len(self.models), "direct-policy packet")
        for agent in np.unique(agents[rows]):
            local = np.flatnonzero(agents[rows] == agent)
            selected_rows = rows[local]
            batch = _slice_rows(request, selected_rows)
            tensors = native_batch_to_tensors(
                batch, self.device, self.training.map_size, self.training.map_type
            )
            output = self.models[int(agent)](
                tensors.map_tokens,
                tensors.global_features,
                tensors.action_categorical,
                tensors.action_continuous,
                tensors.action_mask,
                tensors.event_features,
                action_history=tensors.action_history,
                action_history_mask=tensors.action_history_mask,
                include_belief=False,
            )
            logits = output.policy_logits.masked_fill(~tensors.action_mask, float("-inf"))
            policy = torch.softmax(logits, dim=-1).float().cpu().numpy()
            policy *= np.asarray(batch["action_mask"], dtype=np.float32)
            totals = policy.sum(axis=1, keepdims=True)
            empty = ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0.0)
            if np.any(empty):
                legal = np.asarray(batch["action_mask"], dtype=bool)
                policy[empty] = legal[empty].astype(np.float32)
                totals = policy.sum(axis=1, keepdims=True)
            policy /= np.maximum(totals, 1.0e-8)
            policies[selected_rows] = policy
            values[selected_rows] = output.value.float().reshape(-1).cpu().numpy()
        return policies, values

    def selective_mcts_mask(
        self,
        request: Mapping[str, np.ndarray],
        active: np.ndarray,
        *,
        fraction: float,
        minimum_envs: int,
        audit_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose a bounded, public-information subset for expensive MCTS.

        Entropy and top-action ambiguity are cheap uncertainty proxies.  A
        small random audit prevents the policy from avoiding search merely by
        becoming confidently wrong.  No full-map labels or belief particles
        participate in this decision.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("selective MCTS fraction must be in (0, 1]")
        if minimum_envs < 0:
            raise ValueError("selective MCTS minimum_envs must be non-negative")
        if not 0.0 <= audit_fraction <= 1.0:
            raise ValueError("selective MCTS audit_fraction must be in [0, 1]")
        active = np.asarray(active, dtype=bool)
        if active.shape != (self.num_envs,):
            raise ValueError("active mask must have shape [num_envs]")
        active_rows = np.flatnonzero(active).astype(np.int32)
        policies, values = self.direct_policy_value(request, active_rows)
        selected = np.zeros(self.num_envs, dtype=bool)
        if active_rows.size == 0:
            return selected, policies, values
        if fraction >= 1.0:
            selected[active_rows] = True
        else:
            action_mask = np.asarray(request["action_mask"], dtype=bool)
            features = np.asarray(request["action_features"], dtype=np.int64)
            legal = action_mask[active_rows]
            raw = policies[active_rows]
            legal_counts = legal.sum(axis=1)
            entropy = np.zeros(active_rows.size, dtype=np.float64)
            top_mass = np.ones(active_rows.size, dtype=np.float64)
            for index, count in enumerate(legal_counts):
                count = int(count)
                if count <= 1:
                    continue
                probabilities = raw[index, :count]
                entropy[index] = -float(
                    np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))
                ) / float(np.log(count))
                top_mass[index] = float(probabilities.max())
            # Search a stratified subset of genuinely strategic roots.  An
            # entropy-only scheduler misses a model which is confidently
            # wrong; choosing these rows uniformly provides correction data
            # for combat, city and construction choices without peeking under
            # fog.  ``end turn`` is intentionally excluded because it is
            # legal almost everywhere and would defeat this coverage stratum.
            action_types = features[active_rows]
            action_kind = action_types[:, :, 0]
            tile_action = action_types[:, :, 8]
            damage = action_types[:, :, 15]
            strategic = (
                ((action_kind == 1) & (damage > 0))
                | ((action_kind == 8) & np.isin(tile_action, (10, 11, 12, 14)))
            ) & legal
            strategic_rows = active_rows[strategic.any(axis=1)]
            self.counters.selective_mcts_strategic_available += int(
                strategic_rows.size
            )
            strategic_bonus = strategic.any(axis=1).astype(np.float64) * 0.20
            action_width = np.minimum(1.0, np.log1p(legal_counts) / np.log(16.0))
            score = entropy + 0.35 * (1.0 - top_mass) + 0.10 * action_width + strategic_bonus
            target = min(
                active_rows.size,
                max(int(minimum_envs), int(np.ceil(active_rows.size * fraction))),
            )
            audit_count = min(target, int(np.ceil(active_rows.size * audit_fraction)))
            if audit_count:
                audited = np.asarray(
                    rng.choice(active_rows, size=audit_count, replace=False), dtype=np.int32
                )
                selected[audited] = True
            # Half of the bounded MCTS budget is stratified public coverage.
            # The rest remains available for uncertainty ranking, which keeps
            # the original throughput advantage of selective search.
            coverage_target = min(
                target - int(selected[active_rows].sum()),
                int(np.ceil(target * 0.50)),
                int((~selected[strategic_rows]).sum()),
            )
            if coverage_target:
                coverage_rows = strategic_rows[~selected[strategic_rows]]
                covered = np.asarray(
                    rng.choice(coverage_rows, size=coverage_target, replace=False),
                    dtype=np.int32,
                )
                selected[covered] = True
                self.counters.selective_mcts_strategic_coverage_decisions += int(
                    covered.size
                )
            remaining = active_rows[~selected[active_rows]]
            take = min(target - int(selected[active_rows].sum()), remaining.size)
            if take:
                local_score = score[~selected[active_rows]]
                chosen = np.argsort(-local_score, kind="stable")[:take]
                selected[remaining[chosen]] = True
            # Candidate policy targets are the reason this collection pass
            # exists.  A global uncertainty ranking can otherwise spend a
            # small B=8 budget entirely on frozen-opponent turns.  Reserve
            # one candidate root by replacing the weakest selected opponent
            # root, without increasing the configured MCTS budget.
            agents = np.asarray(request["agent_id"], dtype=np.int32)
            candidate_rows = active_rows[agents[active_rows] == 0]
            if candidate_rows.size and not selected[candidate_rows].any():
                candidate_scores = score[agents[active_rows] == 0]
                candidate = int(candidate_rows[np.argmax(candidate_scores)])
                selected[candidate] = True
                selected_active = active_rows[selected[active_rows]]
                if selected_active.size > target:
                    removable = selected_active[
                        agents[selected_active] != 0
                    ]
                    if removable.size:
                        removable_scores = score[
                            np.isin(active_rows, removable, assume_unique=True)
                        ]
                        selected[int(removable[np.argmin(removable_scores)])] = False
            self.counters.selective_mcts_audit_decisions += int(audit_count)

        entropy_by_row = np.zeros(self.num_envs, dtype=np.float64)
        action_mask = np.asarray(request["action_mask"], dtype=bool)
        for env in active_rows:
            count = int(action_mask[env].sum())
            if count > 1:
                p = policies[env, :count]
                entropy_by_row[env] = -float(
                    np.sum(p * np.log(np.maximum(p, 1.0e-12))) / np.log(count)
                )
        self.counters.selective_mcts_decisions += int(selected[active_rows].sum())
        self.counters.direct_policy_decisions += int((~selected & active).sum())
        self.counters.selective_raw_entropy_sum += float(entropy_by_row[selected].sum())
        self.counters.selective_direct_entropy_sum += float(
            entropy_by_row[(~selected) & active].sum()
        )
        return selected, policies, values

    def _capture_engine_action_constraints(
        self,
        differences: Sequence[Mapping[str, Any]],
        active: np.ndarray,
        actors: np.ndarray,
        request: Mapping[str, np.ndarray],
    ) -> None:
        """Turn PolyEnv action differences into fog-safe replay focus masks.

        PolyEnv only returns public action signatures.  The normal supervised
        belief target remains the sole source of map labels, while this method
        tells PolyBot which hidden source/target endpoints deserve the extra
        loss.  It intentionally runs once for the original learned particles,
        before acceptance retries erase the useful mismatch signal.
        """
        maps = np.asarray(request["map_tokens"], dtype=np.int32)
        self.last_engine_action_constraint_tiles = np.zeros(
            maps.shape[:2], dtype=bool
        )
        self.last_engine_action_constraint_scores = np.zeros(
            maps.shape[0], dtype=np.float32
        )
        if not self.collect_engine_action_constraints:
            return
        candidate_agents = np.asarray(request["agent_id"], dtype=np.int32) == 0
        tiles = maps.shape[1]
        self.counters.engine_action_constraint_queries += 1
        for raw in differences:
            env = int(raw.get("env_id", -1))
            player = int(raw.get("player", -1))
            if (
                env < 0 or env >= maps.shape[0] or not active[env]
                or not candidate_agents[env] or player != int(actors[env])
            ):
                continue
            missing = tuple(raw.get("missing_actions", ()) or ())
            extra = tuple(raw.get("extra_actions", ()) or ())
            if not missing and not extra:
                continue
            self.counters.engine_action_constraint_particles += 1
            self.counters.engine_action_constraint_missing_actions += len(missing)
            self.counters.engine_action_constraint_extra_actions += len(extra)
            # Extra actions are more dangerous: MCTS can otherwise choose a
            # move unavailable in the real public state.
            score = float(len(missing) + 2 * len(extra))
            self.last_engine_action_constraint_scores[env] += score
            self.counters.engine_action_constraint_score_sum += score
            before = bool(self.last_engine_action_constraint_tiles[env].any())
            for action in (*missing, *extra):
                if not isinstance(action, Mapping):
                    continue
                for key in ("source_index", "target_index"):
                    index = int(action.get(key, -1))
                    if 0 <= index < tiles and maps[env, index, 0] == 0:
                        self.last_engine_action_constraint_tiles[env, index] = True
            if not self.last_engine_action_constraint_tiles[env].any() and not before:
                # Some actions depend on a path or a rule whose causal tile is
                # not one of the public endpoints. Keep the signal truthful by
                # falling back to all hidden tiles of this *mismatched* public
                # position, rather than inventing a local movement rule.
                self.last_engine_action_constraint_tiles[env] = maps[env, :, 0] == 0
        self.counters.engine_action_constraint_tiles += int(
            self.last_engine_action_constraint_tiles.sum()
        )

    def _record_particle_diagnostics(
        self,
        request: Mapping[str, np.ndarray],
        candidate_rows: np.ndarray,
        candidate_actors: np.ndarray,
        sampled_particles: np.ndarray,
        rejected: np.ndarray,
        action_diagnostics: Sequence[Mapping[str, Any]],
        native_diagnostics: Sequence[Mapping[str, Any]],
        moves_played: np.ndarray | None,
    ) -> None:
        """Measure submitted hypotheses before any recovery or optimisation."""
        counts = self.counters.belief_particle_diagnostics
        truth = self.pool.belief_targets_numpy(request["state_id"])[candidate_rows]
        hidden = request["map_tokens"][candidate_rows, None, :, 0] == 0
        actual_units = np.broadcast_to(
            hidden & (truth[:, None, :, 4] >= 1), sampled_particles.shape[:-1]
        )
        predicted_units = hidden & (sampled_particles[..., 4] >= 1)
        counts["unit_actual"] += int(actual_units.sum())
        counts["unit_predicted"] += int(predicted_units.sum())
        counts["unit_correct"] += int((actual_units & predicted_units).sum())
        counts["own_hidden_units"] += int(
            (predicted_units & (sampled_particles[..., 3] == candidate_actors[:, None, None])).sum()
        )

        rejected_keys = {
            (int(candidate_rows[row]), int(candidate_actors[row]), int(particle))
            for row, particle in np.argwhere(rejected)
        }
        details: dict[tuple[int, ...], Mapping[str, Any]] = {}
        for diagnostic in (*action_diagnostics, *native_diagnostics):
            key = tuple(int(diagnostic.get(name, -1)) for name in ("env_id", "player", "particle"))
            if key in rejected_keys:
                details[key] = diagnostic
        for key in sorted(rejected_keys):
            diagnostic = details.get(key, {})
            reason = str(diagnostic.get("reason", "unknown"))
            counts[f"rejected_{reason}"] += 1
            if reason == "belief_build_error" and moves_played is not None:
                env, player, particle = key
                # Keep the engine's message and omit any attached stack trace.
                message, _, _ = str(diagnostic.get("error") or "no engine message").strip().partition("\n")
                errors = self.counters.belief_build_errors
                if message not in errors:
                    errors[message] = {
                        "count": 0,
                        "map_seed": int(request["map_seed"][env]),
                        "move": int(moves_played[env]) + 1,
                        "player": player,
                        "particle": particle,
                    }
                errors[message]["count"] += 1
        for diagnostic in action_diagnostics:
            key = tuple(int(diagnostic.get(name, -1)) for name in ("env_id", "player", "particle"))
            if key not in rejected_keys:
                continue
            for direction in ("missing", "extra"):
                for action in diagnostic[f"{direction}_actions"]:
                    action_type = "unknown"
                    if isinstance(action, Mapping):
                        action_type = action.get("type_fullname") or action.get("type", "unknown")
                    counts[f"{direction}_action_{action_type}"] += 1

    @torch.inference_mode()
    def _actor_particles(
        self,
        request: Mapping[str, np.ndarray],
        rows: np.ndarray,
        particle_count: int,
        base_temperature: float,
        attempt: int,
    ) -> np.ndarray:
        maps = np.asarray(request["map_tokens"], dtype=np.int32)
        result = np.empty(
            (len(rows), particle_count, maps.shape[1], maps.shape[2]),
            dtype=np.int32,
        )
        agents = np.asarray(request["agent_id"], dtype=np.int32)
        _validate_agents(agents[rows], len(self.models), "root packet")
        for agent in np.unique(agents[rows]):
            local = np.flatnonzero(agents[rows] == agent)
            selected_rows = rows[local]
            batch = _slice_rows(request, selected_rows)
            tensors = native_batch_to_tensors(
                batch, self.device, self.training.map_size, self.training.map_type
            )
            model = self.models[int(agent)]
            output = model(
                tensors.map_tokens,
                tensors.global_features,
                tensors.action_categorical,
                tensors.action_continuous,
                tensors.action_mask,
                tensors.event_features,
                action_history=tensors.action_history,
                action_history_mask=tensors.action_history_mask,
            )
            perspectives = torch.as_tensor(
                np.asarray(batch["to_play"], dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )
            city_counts = torch.as_tensor(
                np.asarray(batch["state"], dtype=np.int32)[:, 6],
                dtype=torch.long,
                device=self.device,
            )
            for particle in range(particle_count):
                completed = model.sample_completed_map_batch_from_belief_logits(
                    tensors.map_tokens,
                    output.belief_logits,
                    temperature=particle_temperature(base_temperature, attempt, particle),
                    perspective_players=perspectives,
                    known_city_counts=city_counts,
                    action_categorical=tensors.action_categorical,
                    action_mask=tensors.action_mask,
                )
                result[local, particle] = completed.numpy().astype(np.int32, copy=False)
        return result

    @torch.inference_mode()
    def _all_player_particles(
        self,
        request: Mapping[str, np.ndarray],
        particle_count: int,
        base_temperature: float,
        attempt: int,
    ) -> np.ndarray:
        maps = np.asarray(request["map_tokens"], dtype=np.int32)
        states = np.asarray(request["state"], dtype=np.int32)
        perspectives = np.asarray(request["belief_player"], dtype=np.int64)
        history = np.asarray(request["visible_action_history"], dtype=np.int32)
        history_mask = np.asarray(request["visible_action_history_mask"], dtype=bool)
        agents = np.asarray(request["agent_id"], dtype=np.int32)
        envs, players, tiles, fields = maps.shape
        flat_maps = maps.reshape(envs * players, tiles, fields)
        flat_states = states.reshape(envs * players, states.shape[-1])
        flat_perspectives = perspectives.reshape(-1)
        flat_history = history.reshape(envs * players, history.shape[-2], history.shape[-1])
        flat_history_mask = history_mask.reshape(envs * players, history_mask.shape[-1])
        flat_agents = agents.reshape(-1)
        flat_context = flatten_player_context(request)["player_context"]
        _validate_agents(flat_agents, len(self.models), "all-player belief packet")
        result = np.empty(
            (envs * players, particle_count, tiles, fields), dtype=np.int32
        )
        for agent in np.unique(flat_agents):
            rows = np.flatnonzero(flat_agents == agent)
            count = len(rows)
            model = self.models[int(agent)]
            map_tensor = torch.as_tensor(
                flat_maps[rows], dtype=torch.long, device=self.device
            )
            categorical = torch.full(
                (count, 1, 12), -1, dtype=torch.long, device=self.device
            )
            continuous = torch.zeros(
                (count, 1, 4), dtype=torch.float32, device=self.device
            )
            action_mask = torch.zeros(
                (count, 1), dtype=torch.bool, device=self.device
            )
            output = model(
                map_tensor,
                torch.as_tensor(
                    native_global_features(
                        {"state": flat_states[rows], "player_context": flat_context[rows]},
                        self.training.map_size, self.training.map_type
                    ),
                    dtype=torch.float32,
                    device=self.device,
                ),
                categorical,
                continuous,
                action_mask,
                action_history=torch.as_tensor(
                    flat_history[rows], dtype=torch.float32, device=self.device
                ),
                action_history_mask=torch.as_tensor(
                    flat_history_mask[rows], dtype=torch.bool, device=self.device
                ),
            )
            perspective_tensor = torch.as_tensor(
                flat_perspectives[rows], dtype=torch.long, device=self.device
            )
            city_counts = torch.as_tensor(
                flat_states[rows, 6], dtype=torch.long, device=self.device
            )
            for particle in range(particle_count):
                completed = model.sample_completed_map_batch_from_belief_logits(
                    map_tensor,
                    output.belief_logits,
                    temperature=particle_temperature(base_temperature, attempt, particle),
                    perspective_players=perspective_tensor,
                    known_city_counts=city_counts,
                )
                result[rows, particle] = completed.numpy().astype(np.int32, copy=False)
        return result.reshape(envs, players, particle_count, tiles, fields)

    @torch.inference_mode()
    def _conservative_particles(
        self,
        request: Mapping[str, np.ndarray],
        particle_count: int,
    ) -> np.ndarray:
        maps = np.asarray(request["map_tokens"], dtype=np.int32)
        states = np.asarray(request["state"], dtype=np.int32)
        perspectives = np.asarray(request["belief_player"], dtype=np.int64)
        agents = np.asarray(request["agent_id"], dtype=np.int32)
        envs, players, tiles, fields = maps.shape
        flat_maps = maps.reshape(envs * players, tiles, fields)
        flat_states = states.reshape(envs * players, states.shape[-1])
        flat_perspectives = perspectives.reshape(-1)
        flat_agents = agents.reshape(-1)
        _validate_agents(flat_agents, len(self.models), "fallback belief packet")
        single = np.empty_like(flat_maps)
        for agent in np.unique(flat_agents):
            rows = np.flatnonzero(flat_agents == agent)
            completed = self.models[int(agent)].conservative_completed_map_batch(
                torch.as_tensor(flat_maps[rows], dtype=torch.long, device=self.device),
                player_count=players,
                perspective_players=torch.as_tensor(
                    flat_perspectives[rows], dtype=torch.long, device=self.device
                ),
                known_city_counts=torch.as_tensor(
                    flat_states[rows, 6], dtype=torch.long, device=self.device
                ),
            )
            single[rows] = completed.numpy().astype(np.int32, copy=False)
        single = single.reshape(envs, players, tiles, fields)
        return np.repeat(single[:, :, None], particle_count, axis=2)

    @torch.inference_mode()
    def _actor_conservative_particles(
        self,
        request: Mapping[str, np.ndarray],
        rows: np.ndarray,
        particle_count: int,
    ) -> np.ndarray:
        """Action-conditioned conservative fallback for the active player."""
        maps = np.asarray(request["map_tokens"], dtype=np.int32)
        result = np.empty(
            (len(rows), particle_count, maps.shape[1], maps.shape[2]), dtype=np.int32
        )
        agents = np.asarray(request["agent_id"], dtype=np.int32)
        _validate_agents(agents[rows], len(self.models), "fallback root packet")
        for agent in np.unique(agents[rows]):
            local = np.flatnonzero(agents[rows] == agent)
            selected_rows = rows[local]
            batch = _slice_rows(request, selected_rows)
            tensors = native_batch_to_tensors(
                batch, self.device, self.training.map_size, self.training.map_type
            )
            completed = self.models[int(agent)].conservative_completed_map_batch(
                tensors.map_tokens,
                player_count=2,
                perspective_players=torch.as_tensor(
                    np.asarray(batch["to_play"], dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                ),
                known_city_counts=torch.as_tensor(
                    np.asarray(batch["state"], dtype=np.int32)[:, 6],
                    dtype=torch.long,
                    device=self.device,
                ),
                action_categorical=tensors.action_categorical,
                action_mask=tensors.action_mask,
                suppress_unlisted_bridges=True,
            ).numpy().astype(np.int32, copy=False)
            result[local] = np.repeat(completed[:, None], particle_count, axis=1)
        return result

    @torch.inference_mode()
    def _expand_leaves(
        self,
        leaf_buffers: Mapping[str, np.ndarray],
        leaf_count: int,
        prior_scale: float,
    ) -> None:
        batch = {name: value[:leaf_count] for name, value in leaf_buffers.items()}
        agents = np.asarray(batch["agent_id"], dtype=np.int32)
        _validate_agents(agents, len(self.models), "leaf packet")
        max_actions = np.asarray(batch["action_mask"]).shape[1]
        logits = np.full((leaf_count, max_actions), -np.inf, dtype=np.float32)
        values = np.zeros(leaf_count, dtype=np.float32)
        # Dirichlet noise must be applied to the same action set which was
        # admitted by pruning.  Using the full native mask here would revive
        # pruned actions and spend MCTS visits that are discarded later.
        noise_mask = np.zeros((leaf_count, max_actions), dtype=bool)
        for agent in np.unique(agents):
            rows = np.flatnonzero(agents == agent)
            routed = _slice_rows(batch, rows)
            tensors = native_batch_to_tensors(
                routed, self.device, self.training.map_size, self.training.map_type
            )
            output = self.models[int(agent)](
                tensors.map_tokens,
                tensors.global_features,
                tensors.action_categorical,
                tensors.action_continuous,
                tensors.action_mask,
                tensors.event_features,
                action_history=tensors.action_history,
                action_history_mask=tensors.action_history_mask,
                include_belief=False,
            )
            routed_logits = output.policy_logits + _native_leaf_bias(
                tensors.action_categorical,
                tensors.action_continuous,
                tensors.action_mask,
                self.mcts.teacher_prior_weight * prior_scale,
                self.mcts.strategic_prior_weight * prior_scale,
                capture_city_bonus=self.mcts.capture_city_prior_bonus,
                capture_capital_bonus=self.mcts.capture_capital_prior_bonus,
                found_city_bonus=self.mcts.found_city_prior_bonus,
                attack_bonus=self.mcts.attack_prior_bonus,
                end_turn_penalty=self.mcts.end_turn_prior_penalty,
                map_tokens=tensors.map_tokens,
            ).to(output.policy_logits.dtype)
            routed_logits, selected, pruning = prune_search_logits(
                routed_logits,
                tensors.action_categorical,
                tensors.action_mask,
                top_actions=self.search_top_actions_by_agent[int(agent)],
                random_actions=self.search_random_actions_by_agent[int(agent)],
                min_prior_mass=self.mcts.search_min_prior_mass,
                generator=self._action_generator,
            )
            self.counters.add_action_pruning(pruning)
            if self._capture_root_candidates:
                noise_mask[rows] = selected.detach().cpu().numpy()
            if self._capture_root_candidates:
                env_ids = np.asarray(batch["env_id"], dtype=np.int32)[rows]
                action_ids = np.asarray(batch["action_id"], dtype=np.int32)[rows]
                selected_cpu = selected.detach().cpu().numpy()
                for local, env in enumerate(env_ids):
                    if self.search_top_actions_by_agent[int(agent)] > 0:
                        self._root_candidate_action_ids[int(env)] = action_ids[
                            local, selected_cpu[local]
                        ].copy()
            logits[rows] = routed_logits.float().cpu().numpy()
            values[rows] = output.value.float().cpu().numpy()
        if self._capture_root_candidates and self._root_noise_enabled:
            add_dirichlet_noise_to_root_logits(
                logits,
                noise_mask,
                self._root_noise_rng,
                epsilon=self.mcts.root_dirichlet_epsilon,
                concentration=self.mcts.root_dirichlet_concentration,
            )
        self.pool.expand_and_backup(
            np.asarray(batch["leaf_id"], dtype=np.uint64), logits, values
        )

    def search(
        self,
        request: Mapping[str, np.ndarray],
        active: np.ndarray,
        *,
        root_temperature: float,
        belief_temperature: float,
        prior_scale: float,
        root_noise: bool = False,
        mcts_mask: np.ndarray | None = None,
        direct_policy: np.ndarray | None = None,
        direct_value: np.ndarray | None = None,
        raw_policy_only: bool = False,
        moves_played: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """Run full or selective routed belief-MCTS for one vector decision.

        Rows outside ``mcts_mask`` execute the public raw policy. In raw-only
        collection, learned particles are measured for belief supervision and
        the public decision is returned without constructing a search forest.
        """
        for model in self.models:
            model.eval()
        active = np.asarray(active, dtype=bool)
        if active.shape != (self.num_envs,):
            raise ValueError("active mask must have shape [num_envs]")
        if mcts_mask is None:
            mcts_mask = np.zeros_like(active) if raw_policy_only else active.copy()
        else:
            mcts_mask = np.asarray(mcts_mask, dtype=bool)
            if mcts_mask.shape != (self.num_envs,):
                raise ValueError("mcts_mask must have shape [num_envs]")
            if np.any(mcts_mask & ~active):
                raise ValueError("mcts_mask must be a subset of active slots")
            mcts_mask = mcts_mask.copy()
        if raw_policy_only and mcts_mask.any():
            raise ValueError("raw-policy collection cannot run MCTS")
        direct_rows = active & ~mcts_mask
        if (raw_policy_only or direct_rows.any()) and (
            direct_policy is None or direct_value is None
        ):
            inferred_policy, inferred_value = self.direct_policy_value(
                request, np.flatnonzero(direct_rows).astype(np.int32)
            )
            direct_policy = inferred_policy
            direct_value = inferred_value
        if direct_policy is not None:
            direct_policy = np.asarray(direct_policy, dtype=np.float32)
            if direct_policy.shape != np.asarray(request["action_mask"]).shape:
                raise ValueError("direct_policy must match request action_mask")
        if direct_value is not None:
            direct_value = np.asarray(direct_value, dtype=np.float32)
            if direct_value.shape != (self.num_envs,):
                raise ValueError("direct_value must have shape [num_envs]")

        # Search objects are also used by promotion gates. Noise is opt-in so
        # those fixed-seed evaluations remain deterministic.
        self._root_noise_enabled = bool(root_noise)
        self.counters.root_env_decisions += int(active.sum())
        state_ids = np.asarray(request["state_id"], dtype=np.uint64)
        actors = np.asarray(request["to_play"], dtype=np.int64)
        particle_count = max(1, int(self.mcts.particles))
        self.pool.all_player_belief_requests_into(self.all_players)
        all_players = self.all_players

        def learned_rows(rows: np.ndarray, attempt: int) -> np.ndarray:
            rows = np.asarray(rows, dtype=np.int32)
            local_all_players = _slice_rows(all_players, rows)
            local_particles = self._all_player_particles(
                local_all_players, particle_count, belief_temperature, attempt
            )
            actor_local = np.flatnonzero(active[rows])
            if actor_local.size:
                actor_rows = rows[actor_local]
                completed = self._actor_particles(
                    request,
                    actor_rows,
                    particle_count,
                    belief_temperature,
                    attempt,
                )
                local_particles[actor_local, actors[actor_rows]] = completed
            return local_particles

        all_rows = np.arange(self.num_envs, dtype=np.int32)
        particles = np.ascontiguousarray(learned_rows(all_rows, 0))
        (
            initial_action_rejected_rows,
            initial_action_rejected_mask,
            initial_action_rejected_particles,
            initial_action_diagnostics,
        ) = ismcts_action_difference_rejections(self.pool, state_ids, particles)
        candidate_active = active & (
            np.asarray(request["agent_id"], dtype=np.int32) == 0
        )
        self.counters.belief_action_equivalence_checked_particles += int(
            candidate_active.sum() * particle_count
        )
        candidate_rows = np.flatnonzero(candidate_active)
        rejected_candidates = initial_action_rejected_mask[
            candidate_rows, actors[candidate_rows]
        ]
        if raw_policy_only and candidate_rows.size:
            # The action diff may be empty when a proposed world cannot even
            # be built. Count those proposals as failures of calibration too.
            _, native_rejected_mask, _, native_diagnostics = ismcts_rejected_envs(
                self.pool, state_ids, particles, submit_failed=False
            )
            rejected_candidates |= native_rejected_mask[
                candidate_rows, actors[candidate_rows]
            ]
            self._record_particle_diagnostics(
                request,
                candidate_rows,
                actors[candidate_rows],
                particles[candidate_rows, actors[candidate_rows]],
                rejected_candidates,
                initial_action_diagnostics,
                native_diagnostics,
                moves_played,
            )
        self.counters.belief_action_equivalence_rejected_particles += int(
            rejected_candidates.sum()
        )
        self._capture_engine_action_constraints(
            initial_action_diagnostics, active, actors, request
        )
        if raw_policy_only:
            assert direct_policy is not None and direct_value is not None
            self.counters.belief_action_equivalence_rejected_envs += int(
                rejected_candidates.any(axis=1).sum()
            )
            return {
                "action_id": request["action_id"],
                "action_mask": request["action_mask"],
                "state_id": state_ids,
                "policy": direct_policy,
                "root_value": direct_value,
            }
        # A particle which changes the active player's public action set is
        # not merely a root-display issue: its descendants would be searched
        # in an impossible world. ``root_shield`` may remap a completed root
        # policy, but cannot undo visits already backed up through that world.
        # Therefore every particle entering MCTS must be action-equivalent in
        # both modes. The modes differ only in the final public-root mapping.
        require_action_equivalence = True
        submitted = False
        last_error: ValueError | None = None
        rejected_rows = all_rows
        retry_pending_envs = 0
        diagnostics: tuple[Mapping[str, Any], ...] = ()
        has_acceptance_mask = callable(
            getattr(self.pool, "ismcts_belief_acceptance_mask", None)
        )
        for attempt in range(self.belief_retries + 1):
            if attempt == 0:
                action_rejected_rows = initial_action_rejected_rows
                action_rejected_mask = initial_action_rejected_mask
                action_rejected_particles = initial_action_rejected_particles
                action_diagnostics = initial_action_diagnostics
            else:
                (
                    action_rejected_rows,
                    action_rejected_mask,
                    action_rejected_particles,
                    action_diagnostics,
                ) = ismcts_action_difference_rejections(
                    self.pool, state_ids, particles
                )
            if require_action_equivalence and action_rejected_rows.size:
                retry_pending_envs = int(action_rejected_rows.size)
                diagnostics = action_diagnostics
                last_error = ValueError(
                    "ismcts_belief_action_differences rejected action-mismatched particles"
                )
                self.counters.belief_action_equivalence_rejected_envs += int(
                    action_rejected_rows.size
                )
                self.counters.belief_rejections += int(action_rejected_rows.size)
                self.counters.belief_rejected_envs += int(action_rejected_rows.size)
                self.counters.belief_rejected_particles += int(
                    action_rejected_particles
                )
                if attempt < self.belief_retries:
                    self.counters.belief_retries += int(action_rejected_rows.size)
                    self.counters.belief_resampled_envs += int(action_rejected_rows.size)
                    fresh = learned_rows(action_rejected_rows, attempt + 1)
                    replace_rejected_particles(
                        particles, action_rejected_rows, action_rejected_mask, fresh
                    )
                    continue
                rejected_rows = action_rejected_rows
                rejected_mask = action_rejected_mask
                break
            if has_acceptance_mask:
                rejected_rows, rejected_mask, rejected_particles, diagnostics = ismcts_rejected_envs(
                    self.pool, state_ids, particles, submit_failed=False
                )
                if rejected_rows.size:
                    retry_pending_envs = int(rejected_rows.size)
                    last_error = ValueError(
                        "ismcts_belief_acceptance_mask rejected learned particles"
                    )
                    self.counters.belief_rejections += int(rejected_rows.size)
                    self.counters.belief_rejected_envs += int(rejected_rows.size)
                    self.counters.belief_rejected_particles += int(rejected_particles)
                    if attempt < self.belief_retries:
                        self.counters.belief_retries += int(rejected_rows.size)
                        self.counters.belief_resampled_envs += int(rejected_rows.size)
                        fresh = learned_rows(rejected_rows, attempt + 1)
                        replace_rejected_particles(
                            particles, rejected_rows, rejected_mask, fresh
                        )
                        continue
                    break
            try:
                self.pool.submit_beliefs(state_ids, particles)
                submitted = True
                if attempt:
                    self.counters.belief_retry_successes += retry_pending_envs
                break
            except ValueError as error:
                last_error = error
                rejected_rows, rejected_mask, rejected_particles, diagnostics = ismcts_rejected_envs(
                    self.pool, state_ids, particles
                )
                retry_pending_envs = int(rejected_rows.size)
                self.counters.belief_rejections += int(rejected_rows.size)
                self.counters.belief_rejected_envs += int(rejected_rows.size)
                self.counters.belief_rejected_particles += int(rejected_particles)
                if attempt < self.belief_retries:
                    self.counters.belief_retries += int(rejected_rows.size)
                    self.counters.belief_resampled_envs += int(rejected_rows.size)
                    fresh = learned_rows(rejected_rows, attempt + 1)
                    replace_rejected_particles(
                        particles, rejected_rows, rejected_mask, fresh
                    )
        if not submitted:
            fallback_target_rows = rejected_rows.copy()
            fallback_target_count = int(fallback_target_rows.size)
            self.counters.belief_conservative_attempts += fallback_target_count
            fallback_rows = self._conservative_particles(
                _slice_rows(all_players, fallback_target_rows), particle_count
            )
            active_fallback = self._actor_conservative_particles(
                request, fallback_target_rows, particle_count
            )
            fallback_actors = np.asarray(request["to_play"], dtype=np.int64)[
                fallback_target_rows
            ]
            fallback_rows[
                np.arange(fallback_target_count), fallback_actors
            ] = active_fallback
            replace_rejected_particles(
                particles, fallback_target_rows, rejected_mask, fallback_rows
            )
            fallback_error: ValueError | None = None
            fallback_preflight_rejected = False
            if require_action_equivalence:
                (
                    rejected_rows,
                    rejected_mask,
                    rejected_particles,
                    diagnostics,
                ) = ismcts_action_difference_rejections(self.pool, state_ids, particles)
                if rejected_rows.size:
                    fallback_preflight_rejected = True
                    fallback_error = ValueError(
                        "ismcts_belief_action_differences rejected conservative particles"
                    )
                    self.counters.belief_action_equivalence_rejected_envs += int(
                        rejected_rows.size
                    )
                    self.counters.belief_rejections += int(rejected_rows.size)
                    self.counters.belief_rejected_envs += int(rejected_rows.size)
                    self.counters.belief_rejected_particles += int(rejected_particles)
            if has_acceptance_mask:
                if fallback_error is None:
                    rejected_rows, rejected_mask, rejected_particles, diagnostics = ismcts_rejected_envs(
                        self.pool, state_ids, particles, submit_failed=False
                    )
                    if rejected_rows.size:
                        fallback_preflight_rejected = True
                        fallback_error = ValueError(
                            "ismcts_belief_acceptance_mask rejected conservative particles"
                        )
                        self.counters.belief_rejections += int(rejected_rows.size)
                        self.counters.belief_rejected_envs += int(rejected_rows.size)
                        self.counters.belief_rejected_particles += int(rejected_particles)
            if fallback_error is None:
                try:
                    self.pool.submit_beliefs(state_ids, np.ascontiguousarray(particles))
                    submitted = True
                    self.counters.belief_fallback_rows += fallback_target_count
                except ValueError as error:
                    fallback_error = error
            if fallback_error is not None:
                error = fallback_error
                if not fallback_preflight_rejected:
                    rejected_rows, rejected_mask, rejected_particles, diagnostics = ismcts_rejected_envs(
                        self.pool, state_ids, particles
                    )
                    self.counters.belief_rejections += int(rejected_rows.size)
                    self.counters.belief_rejected_envs += int(rejected_rows.size)
                    self.counters.belief_rejected_particles += int(rejected_particles)
                kind = _belief_rejection_kind(error, diagnostics)
                original = f"; learned error={last_error}" if last_error is not None else ""
                diagnostic_text = _compact_belief_diagnostics(diagnostics)
                detail = f"; diagnostics={diagnostic_text}" if diagnostic_text else ""
                raise BeliefRecoveryRequired(
                    rejected_rows,
                    "Routed SelfPlayPool rejected both learned and conservative "
                    f"beliefs ({kind}): {error}{original}{detail}",
                    diagnostics,
                ) from error

        waves = 0
        if mcts_mask.any():
            waves = 1 + int(
                np.ceil(max(0, self.mcts.simulations - 1) / self.max_pending_leaves)
            )
        self._root_candidate_action_ids = {}
        for wave in range(waves):
            # The first native selection expands one root leaf.  Every later
            # selection may pipeline several leaves, but only up to the
            # remaining per-root simulation budget.  Previously the last
            # wave was always allowed a full ``max_pending_leaves`` batch:
            # e.g. simulations=2 and pending=8 could do 1 + 8 expansions.
            leaves_per_root = _mcts_leaf_wave_capacity(
                self.mcts.simulations,
                self.max_pending_leaves,
                wave,
            )
            if leaves_per_root <= 0:
                break
            self._capture_root_candidates = wave == 0
            # Get the exact packet layout for this masked wave from the
            # installed PolyEnv binding.  In particular, do not assume that
            # slicing a maximum packet preserves every ABI shape contract.
            wave_capacity = min(
                self.leaf_capacity,
                int(mcts_mask.sum()) * int(leaves_per_root),
            )
            leaf_buffers = self._leaf_buffers_for_capacity(wave_capacity)
            leaf_count = int(
                self.pool.select_leaves_masked_into(
                    leaf_buffers,
                    np.ascontiguousarray(mcts_mask, dtype=np.uint8),
                    max_leaves=wave_capacity,
                )
            )
            if leaf_count > wave_capacity:
                raise RuntimeError(
                    "PolyEnv returned more leaves than the requested MCTS wave"
                )
            self.counters.leaf_waves += 1
            self.counters.leaf_rows += leaf_count
            if leaf_count == 0:
                break
            self._expand_leaves(leaf_buffers, leaf_count, prior_scale)
        self._capture_root_candidates = False
        self.pool.root_policy_into(
            self.root, temperature=root_temperature, public_request=request,
            search_mask=mcts_mask,
        )
        if direct_rows.any():
            assert direct_policy is not None and direct_value is not None
            self.root["policy"][direct_rows] = direct_policy[direct_rows]
            self.root["root_value"][direct_rows] = direct_value[direct_rows]
        if not waves:
            return self.root
        if self.belief_action_mode == "root_shield":
            shield = shield_root_policy_with_public_actions(
                self.root, request, mcts_mask
            )
            self.counters.root_shielded_roots += shield.roots
            self.counters.root_shield_missing_public_actions += (
                shield.missing_public_actions
            )
            self.counters.root_shield_hallucinated_actions += (
                shield.hallucinated_actions
            )
            self.counters.root_shield_uniform_fallbacks += shield.uniform_fallbacks
        action_counts = np.asarray(self.root["action_mask"], dtype=np.uint8).sum(axis=1)
        empty_live = np.flatnonzero(mcts_mask & (action_counts == 0))
        if empty_live.size:
            raise RuntimeError(
                "SelfPlayPool returned unexpanded live roots: "
                f"{empty_live[:8].tolist()}; stats={self.pool.search_stats()}"
            )
        root_agents = np.asarray(self.root["agent_id"], dtype=np.int32)
        enabled = np.zeros(self.num_envs, dtype=bool)
        for env in np.flatnonzero(mcts_mask):
            agent = int(root_agents[env])
            if 0 <= agent < len(self.search_top_actions_by_agent):
                enabled[env] = self.search_top_actions_by_agent[agent] > 0
        restriction = restrict_root_policy_to_candidates(
            self.root, self._root_candidate_action_ids, enabled
        )
        self.counters.search_restricted_roots += restriction.roots
        self.counters.search_missing_root_candidates += restriction.missing_roots
        self.counters.search_root_visits += restriction.visits
        self.counters.search_outside_candidate_visits += (
            restriction.outside_candidate_visits
        )
        self.counters.search_unfiltered_choices_outside += (
            restriction.unfiltered_choices_outside
        )
        root_policy = np.asarray(self.root["policy"], dtype=np.float64)
        root_mask = np.asarray(self.root["action_mask"], dtype=bool)
        for env in np.flatnonzero(mcts_mask):
            probabilities = root_policy[env, root_mask[env]]
            total = float(probabilities.sum())
            if probabilities.size == 0 or total <= 0.0 or not np.isfinite(total):
                continue
            probabilities = probabilities / total
            # Normalize entropy so 0 means one clear action and 1 means a
            # uniform root, independently of the number of legal actions.
            if probabilities.size > 1:
                entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
                entropy /= float(np.log(probabilities.size))
            else:
                entropy = 0.0
            self.counters.search_root_policy_rows += 1
            self.counters.search_root_policy_entropy_sum += entropy
            self.counters.search_root_top_visit_share_sum += float(
                probabilities.max()
            )
        return self.root

    def step_public_policy(
        self,
        actions: np.ndarray,
        state_ids: np.ndarray,
        request: dict[str, np.ndarray],
    ) -> None:
        """Advance already chosen public actions in the pinned SelfPlayPool."""
        # The binding requires a submitted root even for a checked live step.
        # Install a single exact snapshot only after the raw actions have been
        # chosen, preserve the actor's view, then step immediately. Never use
        # this snapshot for inference, particle diagnostics or tree expansion.
        public_map = request["map_tokens"]
        visible = public_map[..., 0] == 1
        snapshot = self.pool.belief_targets_numpy(state_ids)
        snapshot[..., 0] = public_map[..., 0]
        snapshot[visible] = public_map[visible]
        self.pool.submit_beliefs(state_ids, np.ascontiguousarray(snapshot))
        self.pool.step_into_checked(actions, state_ids, request)

    def drain_completed(self) -> Dict[str, np.ndarray]:
        count = int(self.pool.drain_completed_into(self.completed))
        return {name: np.asarray(value[:count]).copy() for name, value in self.completed.items()}


def _sample_active_rows(
    root: Mapping[str, np.ndarray],
    request: Mapping[str, np.ndarray],
    active_rows: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float | np.ndarray,
    exploration_rate: float,
) -> np.ndarray:
    """Choose MCTS actions, with optional visible strategic exploration."""
    policies = np.asarray(root["policy"], dtype=np.float64)
    mask = np.asarray(root["action_mask"], dtype=bool)
    features = np.asarray(request["action_features"], dtype=np.int64)
    selected = np.full(policies.shape[0], -1, dtype=np.int64)
    for env in active_rows:
        count = int(mask[env].sum())
        if count <= 0:
            raise RuntimeError(f"live SelfPlayPool root {env} has no legal action")
        weights = policies[env, :count].copy()
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            weights.fill(1.0 / count)
        else:
            weights /= total
        row_temperature = (
            float(np.asarray(temperature)[env])
            if np.asarray(temperature).ndim
            else float(temperature)
        )
        if row_temperature <= 1e-5:
            row = int(weights.argmax())
        else:
            sampled = np.power(weights, 1.0 / row_temperature)
            sampled_total = float(sampled.sum())
            # Do not divide by an arbitrary epsilon: at low positive
            # temperatures a valid distribution can have a sum below 1e-8,
            # and NumPy correctly rejects the resulting non-distribution.
            row = (
                int(weights.argmax())
                if not np.isfinite(sampled_total) or sampled_total <= 0.0
                else int(rng.choice(count, p=sampled / sampled_total))
            )
        if exploration_rate > 0.0 and rng.random() < exploration_rate:
            action = features[env, :count]
            bias = (
                ((action[:, 0] == 8) & (action[:, 8] == 14)).astype(np.float64) * 4.0
                + ((action[:, 0] == 8) & (action[:, 8] == 11)).astype(np.float64) * 2.0
                + (action[:, 0] == 1).astype(np.float64) * 1.0
                + (action[:, 0] == 0).astype(np.float64) * 0.25
                - (action[:, 0] == 3).astype(np.float64) * 0.75
            )
            explore = np.exp(np.clip(bias - bias.max(), -20.0, 0.0))
            row = int(rng.choice(count, p=explore / explore.sum()))
        selected[env] = row
    return selected


def _dense_shaped_rewards(
    request: Mapping[str, np.ndarray], rows: np.ndarray, action_rows: np.ndarray
) -> np.ndarray:
    features = np.asarray(request["action_features"], dtype=np.int64)
    chosen = features[rows, action_rows[rows]]
    return (
        ((chosen[:, 0] == 8) & (chosen[:, 8] == 14)).astype(np.float32) * 1.0
        + ((chosen[:, 0] == 8) & (chosen[:, 8] == 11)).astype(np.float32) * 0.45
        + (chosen[:, 0] == 1).astype(np.float32)
        * (0.08 + np.clip(chosen[:, 15], 0, 10) * 0.012)
        - (chosen[:, 0] == 3).astype(np.float32) * 0.05
    )


def _paired_uncertainty(totals: Dict[str, float], scores: Sequence[float]) -> None:
    totals["model_a_win_rate"] = totals["model_a_wins"] / max(1.0, totals["games"])
    totals["model_a_score"] = float(np.mean(scores)) if scores else 0.0
    map_scores = [float(np.mean(scores[index:index + 2])) for index in range(0, len(scores), 2)]
    totals["evaluation_maps"] = float(len(map_scores))
    if len(map_scores) < 2:
        # One paired map has no empirical uncertainty estimate. Treating its
        # standard error as zero turned two wins into a fictitious 100% lower
        # confidence bound and could open a promotion gate by chance.
        totals["score_standard_error"] = float("inf")
        totals["score_lower_95"] = 0.0
    else:
        totals["score_standard_error"] = float(
            np.std(map_scores, ddof=1) / np.sqrt(len(map_scores))
        )
        totals["score_lower_95"] = max(
            0.0,
            totals["model_a_score"] - 1.96 * totals["score_standard_error"],
        )
    totals["mean_moves"] = totals["moves"] / max(1.0, totals["games"])
    totals["mean_model_a_particles"] = (
        totals["model_a_particles"] / max(1.0, totals["model_a_decisions"])
    )


def _invalid_evaluation_result(
    totals: Dict[str, Any],
    scores: Sequence[float],
    *,
    reason: str,
    invalid_attempts: int,
    discarded_pairs: int,
    replacement_events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a complete, explicitly invalid gate result without hanging."""
    _paired_uncertainty(totals, scores)
    totals["valid"] = 0.0
    # ``games`` is the requested arena size.  Keep completed games separate:
    # a score calculated from a partial paired arena is diagnostic evidence,
    # never a zero-score match result.
    totals["completed_games"] = float(len(scores))
    totals["planned_games"] = float(totals.get("games", 0.0))
    totals["invalid_evaluations"] = 1.0
    totals["invalid_envs"] = float(max(0, invalid_attempts))
    totals["invalid_reason"] = reason
    totals["discarded_evaluation_pairs"] = float(discarded_pairs)
    totals["invalid_evaluation_attempts"] = float(invalid_attempts)
    totals["evaluation_replacement_events"] = list(replacement_events)
    return totals


def _paired_pool_schedule(
    base_seed: int, start: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return duplicate map seeds and slot-swapped agent assignments."""
    if start < 0 or count < 2 or start % 2 or count % 2:
        raise ValueError("paired pool chunks must start and end on map-pair boundaries")
    indices = np.arange(start, start + count, dtype=np.int64)
    seeds = (int(base_seed) + indices // 2).astype(np.uint32)
    assignments = np.where(
        (indices % 2)[:, None] == 0,
        np.array([[0, 1]], dtype=np.int32),
        np.array([[1, 0]], dtype=np.int32),
    ).astype(np.int32)
    return seeds, assignments


def _evaluation_pair_rows(
    env_ids: Sequence[int] | np.ndarray,
    count: int,
) -> np.ndarray:
    """Expand rejected slots to complete even/odd paired-evaluation rows."""
    valid = np.unique(np.asarray([
        int(env) for env in env_ids if 0 <= int(env) < int(count)
    ], dtype=np.int32))
    if valid.size == 0:
        return np.empty(0, dtype=np.int32)
    starts = np.unique((valid // 2) * 2)
    return np.column_stack((starts, starts + 1)).reshape(-1).astype(np.int32)


def _next_evaluation_replacement_seed(
    base_seed: int,
    cursor: int,
    used: set[int],
) -> tuple[int, int]:
    """Choose a deterministic held-out seed outside the ordinary gate range."""
    max_seed = 2**31 - 1
    next_cursor = int(cursor)
    while True:
        value = (int(base_seed) - 1 + 500_000 + next_cursor) % max_seed + 1
        next_cursor += 1
        if value not in used:
            used.add(value)
            return value, next_cursor


def evaluate_matches_pool(
    model_a: PolicyValueBeliefNet,
    model_b: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    games: int,
    *,
    num_threads: int | None = None,
    max_pending_leaves: int = 4,
    visible_action_history: int = 0,
    belief_retries: int = 4,
    belief_action_mode: str = "strict",
    model_a_uses_mcts: bool = True,
    model_b_uses_mcts: bool = True,
    map_seeds: Sequence[int] | None = None,
    excluded_map_seeds: Sequence[int] = (),
    replace_failed_pairs: bool = True,
    max_envs: int = 128,
    game_callback: GameCallback | None = None,
    model_b_search_top_actions: int | None = None,
    model_b_search_random_actions: int | None = None,
    model_a_label: str = "model A",
    model_b_label: str = "model B",
) -> Dict[str, Any]:
    """Run deterministic paired evaluation on a supplied or generated map set.

    ``model_a_uses_mcts=True, model_b_uses_mcts=False`` measures whether the
    candidate's tree search is itself stronger than the frozen opponent's raw
    policy.  This asymmetric probe is required before search visit counts can
    be trusted as policy targets; MCTS on both sides can otherwise hide an
    equal search gain behind an unchanged win rate.

    ``excluded_map_seeds`` reserves maps which must never be used as a
    recovery replacement.  Callers use it to keep an invalid evaluation pair
    from being silently replaced with a map used by any training collection.
    """
    if games < 2 or games % 2:
        raise ValueError("games must be positive and even for paired evaluation")
    if max_envs < 2:
        raise ValueError("max_envs must be at least two")
    if belief_action_mode not in {"strict", "root_shield"}:
        raise ValueError("belief_action_mode must be 'strict' or 'root_shield'")
    fixed_map_seeds = None
    if map_seeds is not None:
        fixed_map_seeds = np.asarray(map_seeds, dtype=np.uint32)
        if fixed_map_seeds.shape != (games,):
            raise ValueError("map_seeds must contain exactly one seed per paired game")
        if np.any(fixed_map_seeds[0::2] != fixed_map_seeds[1::2]):
            raise ValueError("each paired evaluation map must use the same seed twice")
    model_b_top = (
        mcts.search_top_actions
        if model_b_search_top_actions is None
        else int(model_b_search_top_actions)
    )
    model_b_random = (
        mcts.search_random_actions
        if model_b_search_random_actions is None
        else int(model_b_search_random_actions)
    )
    if model_b_top < 0 or model_b_random < 0:
        raise ValueError("model B MCTS candidate counts must not be negative")
    seed_everything(training.seed)
    # Keep the paired map and player slots fixed during recovery. Evaluation
    # gets extra independent stochastic belief attempts before an exhausted
    # pair is discarded and replaced by another deterministic held-out map.
    # A strict 100-game gate cannot have the same small retry budget as data
    # collection: even a rare 0.1% exhausted position would invalidate most
    # long evaluations. Only rejected particles are regenerated, so this high
    # ceiling has negligible cost on normal positions.
    evaluation_belief_retries = max(
        _EVALUATION_MIN_BELIEF_RETRIES,
        16 * int(belief_retries),
    )
    totals: Dict[str, Any] = {
        "games": float(games),
        "model_a_wins": 0.0,
        "model_a_losses": 0.0,
        "draws": 0.0,
        "model_a_city_captures": 0.0,
        "model_a_capital_captures": 0.0,
        "opponent_city_captures": 0.0,
        "opponent_capital_captures": 0.0,
        "model_a_particles": 0.0,
        "model_a_decisions": 0.0,
        "moves": 0.0,
        "evaluation_seed": float(training.seed),
    }
    scores: list[float] = []
    completed_game_rows: list[Dict[str, Any]] = []
    routing = SearchCounters()
    evaluation_position_recovery_cycles = 0
    discarded_evaluation_pairs = 0
    invalid_evaluation_attempts = 0
    replacement_pair_cursor = 0
    replacement_events: list[Dict[str, Any]] = []
    used_evaluation_seeds = (
        set(int(seed) for seed in fixed_map_seeds)
        if fixed_map_seeds is not None
        else {int(training.seed) + game_index // 2 for game_index in range(games)}
    )
    # Keep the forbidden training arena in the same set used by the
    # deterministic replacement allocator.  It must not appear in completed
    # evaluation rows, but it must block a collision when recovery replaces a
    # failed paired map.
    used_evaluation_seeds.update(int(seed) for seed in excluded_map_seeds)
    chunk_limit = max(2, int(max_envs) - int(max_envs) % 2)
    with tqdm(
        total=games,
        desc=f"evaluate MCTS {model_a_label} vs {model_b_label}",
        unit="game",
        dynamic_ncols=True,
    ) as progress:
        for start in range(0, games, chunk_limit):
            count = min(chunk_limit, games - start)
            if count % 2:
                raise RuntimeError("paired evaluation chunk split a map pair")
            indices = np.arange(start, start + count, dtype=np.int64)
            seeds, assignments = _paired_pool_schedule(training.seed, start, count)
            if fixed_map_seeds is not None:
                seeds = fixed_map_seeds[start:start + count].copy()
            search = RoutedPoolSearch(
                training,
                mcts,
                (model_a, model_b),
                num_envs=count,
                num_threads=num_threads,
                max_pending_leaves=max_pending_leaves,
                visible_action_history=visible_action_history,
                belief_retries=evaluation_belief_retries,
                seed=int(seeds[0]),
                belief_action_mode=belief_action_mode,
                search_top_actions_by_agent=(mcts.search_top_actions, model_b_top),
                search_random_actions_by_agent=(mcts.search_random_actions, model_b_random),
            )
            request = search.reset(seeds, assignments)
            done = np.zeros(count, dtype=bool)
            moves = np.zeros(count, dtype=np.int32)
            candidate_decisions = np.zeros(count, dtype=np.int32)
            candidate_particles = np.zeros(count, dtype=np.int32)
            candidate_city_captures = np.zeros(count, dtype=np.int32)
            candidate_capital_captures = np.zeros(count, dtype=np.int32)
            opponent_city_captures = np.zeros(count, dtype=np.int32)
            opponent_capital_captures = np.zeros(count, dtype=np.int32)
            records: dict[int, Dict[str, Any]] = {}
            progress_counted = np.zeros(count, dtype=bool)
            while True:
                terminal = np.asarray(request["terminated"], dtype=bool)
                done |= terminal
                horizon = (~done) & (moves >= training.max_moves)
                done |= horizon
                active = ~done
                if not active.any():
                    break
                agent_ids = np.asarray(request["agent_id"], dtype=np.int32)
                mcts_mask = active & (
                    ((agent_ids == 0) & bool(model_a_uses_mcts))
                    | ((agent_ids == 1) & bool(model_b_uses_mcts))
                )
                try:
                    root, recovery_failures = _evaluation_search_with_recovery(
                        search,
                        request,
                        active,
                        mcts_mask=mcts_mask,
                    )
                    evaluation_position_recovery_cycles += len(recovery_failures)
                except BeliefRecoveryRequired as error:
                    evaluation_position_recovery_cycles += (
                        _EVALUATION_POSITION_RECOVERY_CYCLES
                    )
                    pair_rows = _evaluation_pair_rows(error.env_ids, count)
                    if pair_rows.size == 0:
                        # A malformed diagnostic must not abort a multi-hour
                        # gate. Restart the first still-active pair instead.
                        first_active = int(np.flatnonzero(active)[0])
                        pair_rows = _evaluation_pair_rows([first_active], count)
                    pair_starts = pair_rows[::2]
                    invalid_evaluation_attempts += int(error.env_ids.size)
                    discarded_evaluation_pairs += int(pair_starts.size)
                    if (
                        not replace_failed_pairs
                        or discarded_evaluation_pairs
                        > _MAX_EVALUATION_PAIR_REPLACEMENTS
                    ):
                        return _invalid_evaluation_result(
                            totals,
                            scores,
                            reason=(
                                "belief recovery failed on the fixed paired maps"
                                if not replace_failed_pairs
                                else "belief recovery exhausted the maximum "
                                f"of {_MAX_EVALUATION_PAIR_REPLACEMENTS} "
                                "paired-map replacements"
                            ),
                            invalid_attempts=invalid_evaluation_attempts,
                            discarded_pairs=discarded_evaluation_pairs,
                            replacement_events=replacement_events,
                        )

                    already_counted = int(progress_counted[pair_rows].sum())
                    if already_counted:
                        progress.update(-already_counted)
                    totals["model_a_decisions"] -= float(
                        candidate_decisions[pair_rows].sum()
                    )
                    totals["model_a_particles"] -= float(
                        candidate_particles[pair_rows].sum()
                    )

                    reset_seeds = np.empty(pair_rows.size, dtype=np.uint32)
                    for offset, pair_start in enumerate(pair_starts):
                        old_seed = int(seeds[int(pair_start)])
                        replacement_seed, replacement_pair_cursor = (
                            _next_evaluation_replacement_seed(
                                training.seed,
                                replacement_pair_cursor,
                                used_evaluation_seeds,
                            )
                        )
                        local_rows = pair_rows[2 * offset:2 * offset + 2]
                        reset_seeds[2 * offset:2 * offset + 2] = replacement_seed
                        seeds[local_rows] = replacement_seed
                        replacement_events.append({
                            "pair_game_index": int(indices[int(pair_start)]),
                            "old_map_seed": old_seed,
                            "replacement_map_seed": replacement_seed,
                            "invalid_env_ids": error.env_ids.tolist(),
                            "reason": str(error),
                            "diagnostics": list(error.diagnostics)[:16],
                        })

                    for env in pair_rows:
                        records.pop(int(env), None)
                    done[pair_rows] = False
                    moves[pair_rows] = 0
                    progress_counted[pair_rows] = False
                    candidate_decisions[pair_rows] = 0
                    candidate_particles[pair_rows] = 0
                    candidate_city_captures[pair_rows] = 0
                    candidate_capital_captures[pair_rows] = 0
                    opponent_city_captures[pair_rows] = 0
                    opponent_capital_captures[pair_rows] = 0
                    search.counters.belief_slot_resets += int(pair_rows.size)
                    search.pool.reset_slots_into(
                        pair_rows,
                        reset_seeds,
                        np.ascontiguousarray(assignments[pair_rows], dtype=np.int32),
                        request,
                    )
                    progress.set_description("evaluate A vs model_b (belief pair replaced)")
                    continue
                active_rows = np.flatnonzero(active)
                action_rows = _sample_active_rows(
                    root,
                    request,
                    active_rows,
                    np.random.default_rng(int(seeds[0]) ^ int(moves.sum()) ^ 0xE7A1),
                    temperature=0.0,
                    exploration_rate=0.0,
                )
                actions = np.full(count, -1, dtype=np.int32)
                actions[active_rows] = np.asarray(root["action_id"], dtype=np.int32)[
                    active_rows, action_rows[active_rows]
                ]
                candidate_turns = active & (np.asarray(request["agent_id"]) == 0)
                candidate_decisions[candidate_turns] += 1
                candidate_particles[candidate_turns] += max(1, int(mcts.particles))
                totals["model_a_decisions"] += float(candidate_turns.sum())
                totals["model_a_particles"] += float(
                    candidate_turns.sum() * max(1, int(mcts.particles))
                )
                city_by_env, capital_by_env = native_capture_flags(
                    request, action_rows, active
                )
                for env in active_rows:
                    if candidate_turns[env]:
                        candidate_city_captures[env] += int(city_by_env[env])
                        candidate_capital_captures[env] += int(capital_by_env[env])
                    else:
                        opponent_city_captures[env] += int(city_by_env[env])
                        opponent_capital_captures[env] += int(capital_by_env[env])
                search.pool.step_into_checked(
                    actions,
                    np.asarray(root["state_id"], dtype=np.uint64),
                    request,
                )
                moves[active] += 1
                drained = search.drain_completed()
                for row in range(len(drained.get("env_id", []))):
                    env = int(drained["env_id"][row])
                    records[env] = {name: value[row] for name, value in drained.items()}
                    done[env] = True
                    if not progress_counted[env]:
                        progress_counted[env] = True
                        progress.update(1)
            # Drain records produced by the final native step.
            drained = search.drain_completed()
            for row in range(len(drained.get("env_id", []))):
                env = int(drained["env_id"][row])
                records[env] = {name: value[row] for name, value in drained.items()}
                done[env] = True
                if not progress_counted[env]:
                    progress_counted[env] = True
                    progress.update(1)
            for env, game_index in enumerate(indices):
                record = records.get(env)
                winner = int(record["winner_player"]) if record is not None else -1
                game_moves = int(record["moves"]) if record is not None else int(moves[env])
                candidate_player = 0 if assignments[env, 0] == 0 else 1
                score = 0.5 if winner < 0 else float(winner == candidate_player)
                scores.append(score)
                totals["moves"] += game_moves
                totals["model_a_city_captures"] += int(candidate_city_captures[env])
                totals["model_a_capital_captures"] += int(candidate_capital_captures[env])
                totals["opponent_city_captures"] += int(opponent_city_captures[env])
                totals["opponent_capital_captures"] += int(opponent_capital_captures[env])
                if score == 1.0:
                    totals["model_a_wins"] += 1.0
                elif score == 0.0:
                    totals["model_a_losses"] += 1.0
                else:
                    totals["draws"] += 1.0
                game = {
                    "game_index": int(game_index),
                    "map_seed": int(seeds[env]),
                    "model_a_player": candidate_player,
                    "winner_player": winner,
                    "score_a": score,
                    "moves": game_moves,
                    "model_a_city_captures": int(candidate_city_captures[env]),
                    "model_a_capital_captures": int(candidate_capital_captures[env]),
                    "opponent_city_captures": int(opponent_city_captures[env]),
                    "opponent_capital_captures": int(opponent_capital_captures[env]),
                    "model_a_decisions": int(candidate_decisions[env]),
                    "model_a_particles": float(candidate_particles[env]),
                    "truncated": int(record is None),
                }
                completed_game_rows.append(game)
            unfinished = int((~progress_counted).sum())
            if unfinished:
                progress.update(unfinished)
                progress_counted[:] = True
            routing.add(search.counters)
            progress.set_postfix(
                wins=int(totals["model_a_wins"]),
                draws=int(totals["draws"]),
                losses=int(totals["model_a_losses"]),
            )
    _paired_uncertainty(totals, scores)
    totals["valid"] = 1.0
    totals["completed_games"] = float(len(scores))
    totals["planned_games"] = float(games)
    totals["invalid_evaluations"] = 0.0
    totals["discarded_evaluation_pairs"] = float(discarded_evaluation_pairs)
    totals["invalid_evaluation_attempts"] = float(invalid_evaluation_attempts)
    totals["evaluation_replacement_events"] = replacement_events
    totals["evaluated_map_seeds"] = [
        int(row["map_seed"]) for row in completed_game_rows
    ]
    totals["evaluation_position_recovery_cycles"] = float(
        evaluation_position_recovery_cycles
    )
    totals["self_play_pool_leaf_rows"] = float(routing.leaf_rows)
    totals["self_play_pool_leaf_waves"] = float(routing.leaf_waves)
    totals["self_play_pool_belief_retries"] = float(routing.belief_retries)
    totals["self_play_pool_belief_fallback_rows"] = float(routing.belief_fallback_rows)
    totals["self_play_pool_belief_rejections"] = float(routing.belief_rejections)
    totals["self_play_pool_belief_rejected_particles"] = float(
        routing.belief_rejected_particles
    )
    totals["self_play_pool_belief_rejected_envs"] = float(routing.belief_rejected_envs)
    totals["self_play_pool_belief_resampled_envs"] = float(routing.belief_resampled_envs)
    totals["self_play_pool_belief_retry_successes"] = float(routing.belief_retry_successes)
    totals["self_play_pool_belief_conservative_attempts"] = float(
        routing.belief_conservative_attempts
    )
    totals["belief_rejected_particles"] = float(routing.belief_rejected_particles)
    totals["belief_rejected_envs"] = float(routing.belief_rejected_envs)
    totals["belief_resampled_envs"] = float(routing.belief_resampled_envs)
    totals["belief_retry_successes"] = float(routing.belief_retry_successes)
    totals["belief_conservative_attempts"] = float(
        routing.belief_conservative_attempts
    )
    totals["belief_slot_resets"] = float(routing.belief_slot_resets)
    totals["belief_discarded_prefixes"] = 0.0
    totals["mcts_top_actions"] = float(mcts.search_top_actions)
    totals["mcts_random_actions"] = float(mcts.search_random_actions)
    totals["model_b_mcts_top_actions"] = float(model_b_top)
    totals["model_b_mcts_random_actions"] = float(model_b_random)
    totals.update(_action_pruning_metrics(routing))
    if game_callback is not None:
        for game in completed_game_rows:
            game_callback(game)
    return totals


def train_population_pool(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts: MCTSConfig,
    *,
    games: int,
    updates_per_game: int,
    candidate: PolicyValueBeliefNet,
    opponent_models: Sequence[PolicyValueBeliefNet],
    opponent_weights: Sequence[float],
    anchor_model: PolicyValueBeliefNet | None = None,
    anchor_policy_weight: float = 0.0,
    anchor_value_weight: float = 0.25,
    critical_decision_fraction: float = 0.0,
    critical_minimum_score: float = 0.15,
    critical_replay_fraction: float = 0.0,
    critical_only: bool = False,
    selective_mcts: bool = False,
    selective_mcts_fraction: float = 0.50,
    selective_mcts_minimum_envs: int = 2,
    selective_mcts_audit_fraction: float = 0.10,
    reanalysis_path: Path | None = None,
    reanalysis_capacity: int = 4_096,
    reanalysis_max_bytes: int = 64 * 1024**2,
    reanalysis_positions: int = 0,
    reanalysis_replay_fraction: float = 0.0,
    optimisation_block_updates: int = 32,
    optimisation_screen_games: int = 0,
    optimisation_screen_max_regression: float = 0.10,
    offline_collection: bool = False,
    belief_only: bool = False,
    value_belief_heads_only: bool = False,
    raw_policy_only: bool = False,
    num_threads: int | None = None,
    max_pending_leaves: int = 4,
    visible_action_history: int = 0,
    belief_retries: int = 4,
    belief_action_mode: str = "strict",
    max_envs: int = 128,
    game_callback: GameCallback | None = None,
    reserved_seeds: Sequence[int] = (),
    replay_examples: Sequence[TrainingExample] = (),
    replay_snapshot_callback: Callable[[Sequence[TrainingExample]], None] | None = None,
    extra_replay_updates: int = 0,
    run: TrainingRun | None = None,
) -> Dict[str, Any]:
    """Train candidate agent 0 against PFSP with frozen-policy retention.

    When ``offline_collection`` is set, MCTS generates a complete fixed
    generation before any optimiser step.  This is the stable policy-iteration
    mode: every root target in the replay buffer comes from the same frozen
    apprentice, rather than a network that changes game by game.

    ``raw_policy_only`` uses the raw policy for both players and masks policy
    loss for every collected move. Combined with ``value_belief_heads_only``
    it is a safe value/belief bootstrap before MCTS is allowed to teach policy.
    """
    if games < 0 or updates_per_game < 0 or extra_replay_updates < 0:
        raise ValueError("games, updates_per_game and extra_replay_updates must not be negative")
    reserved_seed_set = {int(value) for value in reserved_seeds}
    replay_examples, replay_reserved_excluded, replay_unknown_excluded = (
        _filter_replay_examples_excluding_reserved(
            replay_examples,
            reserved_seed_set,
        )
    )
    if games == 0 and not (extra_replay_updates and replay_examples):
        return {
            "population_pool_games": 0.0,
            "population_pool_updates": 0.0,
            "population_pool_replay_reserved_excluded": float(
                replay_reserved_excluded
            ),
            "population_pool_replay_unknown_seed_excluded": float(
                replay_unknown_excluded
            ),
        }
    if not opponent_models:
        raise ValueError("population training requires at least one opponent model")
    if max_envs < 1:
        raise ValueError("max_envs must be positive")
    if belief_action_mode not in {"strict", "root_shield"}:
        raise ValueError("belief_action_mode must be 'strict' or 'root_shield'")
    if not 0.0 <= anchor_policy_weight <= 1.0:
        raise ValueError("anchor_policy_weight must be between zero and one")
    if not 0.0 <= anchor_value_weight <= 1.0:
        raise ValueError("anchor_value_weight must be between zero and one")
    if not 0.0 <= critical_decision_fraction <= 1.0:
        raise ValueError("critical_decision_fraction must be between zero and one")
    if critical_minimum_score < 0.0:
        raise ValueError("critical_minimum_score must be non-negative")
    if not 0.0 <= critical_replay_fraction <= 1.0:
        raise ValueError("critical_replay_fraction must be between zero and one")
    if not 0.0 <= training.engine_action_constraint_replay_fraction <= 1.0:
        raise ValueError(
            "engine_action_constraint_replay_fraction must be between zero and one"
        )
    if mcts.belief_particle_temperature < 0.0:
        raise ValueError("belief_particle_temperature must not be negative")
    if reanalysis_capacity < 1:
        raise ValueError("reanalysis_capacity must be positive")
    if reanalysis_max_bytes < 1:
        raise ValueError("reanalysis_max_bytes must be positive")
    if reanalysis_positions < 0:
        raise ValueError("reanalysis_positions must not be negative")
    if optimisation_block_updates < 1:
        raise ValueError("optimisation_block_updates must be positive")
    if optimisation_screen_games < 0 or optimisation_screen_games % 2:
        raise ValueError("optimisation_screen_games must be a non-negative even number")
    if optimisation_screen_max_regression < 0.0:
        raise ValueError("optimisation_screen_max_regression must be non-negative")
    if not 0.0 <= reanalysis_replay_fraction <= 1.0:
        raise ValueError("reanalysis_replay_fraction must be between zero and one")
    if reanalysis_positions and reanalysis_path is None:
        raise ValueError("reanalysis_positions requires a persistent reanalysis_path")
    if critical_only and critical_decision_fraction <= 0.0:
        raise ValueError(
            "critical_only requires a positive critical_decision_fraction"
        )
    if not 0.0 < selective_mcts_fraction <= 1.0:
        raise ValueError("selective_mcts_fraction must be in (0, 1]")
    if selective_mcts_minimum_envs < 0:
        raise ValueError("selective_mcts_minimum_envs must not be negative")
    if not 0.0 <= selective_mcts_audit_fraction <= 1.0:
        raise ValueError("selective_mcts_audit_fraction must be in [0, 1]")
    if belief_only and value_belief_heads_only:
        raise ValueError("belief_only and value_belief_heads_only are exclusive")
    if anchor_model is candidate:
        raise ValueError("anchor_model must be a frozen copy, not the trainable candidate")
    if training.max_replay_epochs <= 0.0:
        raise ValueError("max_replay_epochs must be positive")
    weights = np.asarray(opponent_weights, dtype=np.float64)
    if weights.shape != (len(opponent_models),) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("opponent_weights must be finite, non-negative and match opponents")
    if float(weights.sum()) <= 0.0:
        raise ValueError("opponent_weights must have a positive sum")
    weights /= weights.sum()
    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "population_self_play_pool")
    run.write_config(
        training=training,
        model=model_config,
        mcts=mcts,
        stage={
            "games": games,
            "updates_per_game": updates_per_game,
            "max_envs": max_envs,
            "num_threads": num_threads,
            "max_pending_leaves": max_pending_leaves,
            "visible_action_history": visible_action_history,
            "belief_retries": belief_retries,
            "opponent_weights": weights.tolist(),
            "anchor_policy_weight": anchor_policy_weight,
            "anchor_value_weight": anchor_value_weight,
            "critical_decision_fraction": critical_decision_fraction,
            "critical_minimum_score": critical_minimum_score,
            "critical_replay_fraction": critical_replay_fraction,
            "critical_only": critical_only,
            "selective_mcts": selective_mcts,
            "selective_mcts_fraction": selective_mcts_fraction,
            "selective_mcts_minimum_envs": selective_mcts_minimum_envs,
            "selective_mcts_audit_fraction": selective_mcts_audit_fraction,
            "reanalysis_path": str(reanalysis_path) if reanalysis_path else "",
            "reanalysis_capacity": reanalysis_capacity,
            "reanalysis_max_bytes": reanalysis_max_bytes,
            "reanalysis_positions": reanalysis_positions,
            "reanalysis_replay_fraction": reanalysis_replay_fraction,
            "optimisation_block_updates": optimisation_block_updates,
            "optimisation_screen_games": optimisation_screen_games,
            "optimisation_screen_max_regression": (
                optimisation_screen_max_regression
            ),
            "offline_collection": offline_collection,
            "belief_only": belief_only,
            "value_belief_heads_only": value_belief_heads_only,
            "raw_policy_only": raw_policy_only,
            "belief_action_mode": belief_action_mode,
        },
    )
    value_target_training = (
        replace(
            training,
            shaping_target_weight=0.0,
            reward_discount=1.0,
            td_lambda=1.0,
        )
        if offline_collection
        else training
    )
    candidate.to(device)
    # Calibration must not silently degrade the champion's policy/value via
    # the shared encoder. Value bootstrap likewise keeps raw policy behaviour
    # fixed while learning only outcome/belief heads from actual games.
    for name, parameter in candidate.named_parameters():
        trainable = True
        if belief_only:
            trainable = name.startswith("belief_")
        elif value_belief_heads_only:
            trainable = name.startswith(("value_head", "belief_"))
        parameter.requires_grad_(trainable)
    frozen = []
    for model in opponent_models:
        frozen.append(_freeze_reference_model(model, device))
    if anchor_model is not None:
        anchor_model = _freeze_reference_model(anchor_model, device)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    replay = ReplayBuffer(
        training.replay_capacity,
        preserve_constraint_fraction=0.40 if belief_only else 0.0,
        preserve_bridge_fraction=0.20 if belief_only else 0.0,
    )
    # A calibration run is deliberately short, but its supervised maps should
    # not be forgotten when the next CLI invocation starts.  The caller owns
    # persistence; this generic pool only seeds its bounded in-memory window.
    if replay_examples:
        replay.extend(list(replay_examples)[-training.replay_capacity:])
    rng = np.random.default_rng(training.seed ^ 0x91A6E)
    reanalysis_store = (
        PersistentReanalyseBuffer(
            reanalysis_path,
            reanalysis_capacity,
            reanalysis_max_bytes,
        )
        if reanalysis_path is not None
        else None
    )
    reanalysis_loaded = reanalysis_store.load() if reanalysis_store else 0
    reanalysis_refreshed = 0
    reanalysis_failed = 0
    reanalysis_reserved_excluded = 0
    if reanalysis_store is not None and reanalysis_positions:
        selected_reanalysis, reanalysis_reserved_excluded = (
            _sample_reanalysis_positions_excluding_reserved(
                reanalysis_store.positions,
                reanalysis_positions,
                rng,
                reserved_seed_set,
            )
        )
        reanalysed, reanalysis_failed = _reanalyse_critical_positions(
            selected_reanalysis,
            candidate=candidate,
            training=training,
            mcts=mcts,
            device=device,
            minimum_score=critical_minimum_score,
            visible_action_history=visible_action_history,
            anchor_model=anchor_model,
            anchor_policy_weight=anchor_policy_weight,
            anchor_value_weight=anchor_value_weight,
        )
        replay.extend(reanalysed)
        reanalysis_refreshed = len(reanalysed)
    losses: list[Dict[str, float]] = []
    updates = 0
    completed_count = 0
    terminal_count = 0
    truncated_count = 0
    examples_count = reanalysis_refreshed
    decisions = 0
    anchor_decisions = 0
    anchor_agreements = 0
    anchor_l1_sum = 0.0
    mcts_raw_decisions = 0
    mcts_raw_agreements = 0
    mcts_raw_l1_sum = 0.0
    critical_decisions = 0
    critical_score_sum = 0.0
    discarded_noncritical_examples = 0
    total_moves = 0
    score_sum = 0.0
    opponent_stats = [
        {"games": 0.0, "wins": 0.0, "draws": 0.0, "losses": 0.0, "score": 0.0}
        for _ in frozen
    ]
    routing = SearchCounters()
    recovery_seed_cursor = 0
    recovery_seed_history: set[int] = set()
    reanalysis_saved = 0
    reanalysis_reserved_rejected = 0
    optimisation_blocks = 0
    optimisation_rollbacks = 0
    optimisation_attempted = 0
    optimisation_screen_attempts = 0
    optimisation_screen_rollbacks = 0
    optimisation_screen_invalid = 0
    optimisation_screen_score: float | None = None
    optimisation_screen_best_score: float | None = None
    optimisation_screen_model_state: dict[str, torch.Tensor] | None = None
    optimisation_screen_optimiser_state: dict[str, Any] | None = None

    def cpu_model_state() -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in candidate.state_dict().items()
        }

    def cpu_optimizer_state() -> dict[str, Any]:
        def clone(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, Mapping):
                return {key: clone(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple(clone(item) for item in value)
            return copy.deepcopy(value)

        return clone(optimiser.state_dict())

    def restore_verified_screen_state() -> None:
        if (
            optimisation_screen_model_state is None
            or optimisation_screen_optimiser_state is None
        ):
            raise RuntimeError("optimisation screen rollback state is missing")
        candidate.load_state_dict(optimisation_screen_model_state, strict=True)
        # PyTorch casts optimiser tensors to their parameter devices while
        # loading.  Keeping the snapshot on CPU therefore avoids a second
        # GPU-sized AdamW copy between blocks.
        optimiser.load_state_dict(optimisation_screen_optimiser_state)

    def evaluate_optimisation_screen() -> dict[str, Any] | None:
        """Cheap fixed-arena raw gate used between optimisation blocks.

        This is not an MCTS promotion decision.  It catches a catastrophic
        policy regression early, before a full generation can overwrite the
        last verified state.  It intentionally uses raw policy on both sides,
        so it needs neither belief particles nor a second costly tree search.
        """
        if optimisation_screen_games <= 0 or anchor_model is None:
            return None
        screen_seed = (int(training.seed) + 700_000_019) % (2**31 - 1)
        screen_seeds = np.repeat(
            np.arange(
                screen_seed,
                screen_seed + int(optimisation_screen_games) // 2,
                dtype=np.uint32,
            ),
            2,
        )
        return evaluate_direct_matches(
            candidate,
            anchor_model,
            device,
            replace(training, seed=screen_seed),
            int(optimisation_screen_games),
            map_seeds=screen_seeds,
            model_a_label="candidate block screen",
            model_b_label="frozen anchor",
            visible_action_history=visible_action_history,
            num_threads=num_threads,
            max_actions=mcts.native_max_actions,
        )

    def memory_metrics() -> dict[str, float]:
        if device.type != "cuda":
            return {
                "cuda_memory_allocated": 0.0,
                "cuda_memory_reserved": 0.0,
            }
        return {
            "cuda_memory_allocated": float(torch.cuda.memory_allocated(device)),
            "cuda_memory_reserved": float(torch.cuda.memory_reserved(device)),
        }

    def record_event(phase: str, **details: Any) -> None:
        event = getattr(run, "event", None)
        if callable(event):
            event(phase, **details)

    def optimise_block(count: int, *, phase: str) -> int:
        """Optimise a bounded block and surface its actual lifetime clearly."""
        nonlocal updates, optimisation_blocks, optimisation_rollbacks
        nonlocal optimisation_attempted
        nonlocal optimisation_screen_attempts, optimisation_screen_rollbacks
        nonlocal optimisation_screen_invalid, optimisation_screen_score
        nonlocal optimisation_screen_best_score
        nonlocal optimisation_screen_model_state, optimisation_screen_optimiser_state
        completed = 0
        if (
            optimisation_screen_games > 0
            and anchor_model is not None
            and optimisation_screen_model_state is None
        ):
            initial_screen = evaluate_optimisation_screen()
            optimisation_screen_attempts += 1
            if initial_screen is not None and bool(initial_screen.get("valid", 0.0)):
                optimisation_screen_score = float(initial_screen["model_a_score"])
                optimisation_screen_best_score = optimisation_screen_score
            else:
                optimisation_screen_invalid += 1
            # Even when the baseline arena itself is temporarily invalid,
            # retain a coherent rollback point for a later finite/numerical
            # failure in this same generation.
            optimisation_screen_model_state = cpu_model_state()
            optimisation_screen_optimiser_state = cpu_optimizer_state()
        while completed < count and len(replay) >= training.batch_size:
            block = min(int(optimisation_block_updates), count - completed)
            record_event(
                "optimise",
                block=optimisation_blocks + 1,
                source=phase,
                updates_planned=block,
                updates_retained=updates,
                replay_examples=len(replay),
                reanalysis_bytes=(
                    reanalysis_store.bytes_used if reanalysis_store else 0
                ),
                **memory_metrics(),
            )
            print(
                "population optimise | "
                f"block={optimisation_blocks + 1} | updates={block} | "
                f"replay={len(replay)} | "
                f"reanalysis_bytes="
                f"{reanalysis_store.bytes_used if reanalysis_store else 0}",
                flush=True,
            )
            block_losses: list[Dict[str, float]] = []
            for _ in range(block):
                batch = replay.sample(
                    training.batch_size,
                    rng,
                    training.priority_replay_fraction,
                    training.bridge_replay_fraction,
                    critical_replay_fraction,
                    reanalysis_replay_fraction,
                    training.engine_action_constraint_replay_fraction,
                )
                result = train_batch(
                    candidate, optimiser, stack_examples(batch, device), training
                )
                optimisation_attempted += 1
                if not all(np.isfinite(value) for value in result.values()):
                    # train_batch normally catches numerical issues before the
                    # optimiser step.  If a future loss violates that
                    # contract, stop the block instead of treating NaN as
                    # evidence of a useful candidate.
                    optimisation_rollbacks += 1
                    record_event(
                        "rollback",
                        source=phase,
                        reason="non_finite_training_metric",
                        updates_retained=updates,
                    )
                    return completed
                losses.append(result)
                block_losses.append(result)
                updates += 1
                completed += 1
            optimisation_blocks += 1
            means = {
                f"block_{key}": float(np.mean([row[key] for row in block_losses]))
                for key in block_losses[0]
            }
            run.log(
                phase_optimise=1.0,
                optimisation_block=optimisation_blocks,
                optimisation_updates_attempted=optimisation_attempted,
                optimisation_updates_retained=updates,
                replay_examples=len(replay),
                reanalysis_bytes=(
                    reanalysis_store.bytes_used if reanalysis_store else 0
                ),
                **memory_metrics(),
                **means,
            )
            record_event(
                "optimise_complete",
                block=optimisation_blocks,
                source=phase,
                updates_attempted=optimisation_attempted,
                updates_retained=updates,
            )
            if optimisation_screen_games > 0 and anchor_model is not None:
                screen = evaluate_optimisation_screen()
                optimisation_screen_attempts += 1
                if screen is None or not bool(screen.get("valid", 0.0)):
                    optimisation_screen_invalid += 1
                    record_event(
                        "optimisation_screen_invalid",
                        block=optimisation_blocks,
                        reason=(
                            "screen unavailable"
                            if screen is None
                            else str(screen.get("invalid_reason", "invalid"))
                        ),
                    )
                else:
                    optimisation_screen_score = float(screen["model_a_score"])
                    if (
                        optimisation_screen_best_score is not None
                        and optimisation_screen_score
                        < optimisation_screen_best_score
                        - float(optimisation_screen_max_regression)
                    ):
                        restore_verified_screen_state()
                        optimisation_screen_rollbacks += 1
                        optimisation_rollbacks += 1
                        record_event(
                            "rollback",
                            source=phase,
                            reason="raw_policy_screen_regression",
                            block=optimisation_blocks,
                            score=optimisation_screen_score,
                            best_score=optimisation_screen_best_score,
                            updates_retained=updates - block,
                        )
                        print(
                            "population optimise rollback | "
                            f"block={optimisation_blocks} | "
                            f"screen={optimisation_screen_score:.4f} | "
                            f"best={optimisation_screen_best_score:.4f}",
                            flush=True,
                        )
                        updates -= block
                        completed -= block
                        del losses[-block:]
                        return completed
                    if (
                        optimisation_screen_best_score is None
                        or optimisation_screen_score > optimisation_screen_best_score
                    ):
                        optimisation_screen_best_score = optimisation_screen_score
                    optimisation_screen_model_state = cpu_model_state()
                    optimisation_screen_optimiser_state = cpu_optimizer_state()
        return completed

    def optimise_until(target: int) -> None:
        nonlocal updates
        capped = capped_update_target(
            target,
            examples_count,
            training.batch_size,
            training.max_replay_epochs,
        )
        optimise_block(max(0, capped - updates), phase="fresh_replay")

    def optimise_replay_updates(count: int) -> int:
        """Run a bounded offline optimisation block over the existing replay.

        This deliberately bypasses the fresh-example epoch cap: calibration
        replay is persistent, and a belief-only job must be able to train on
        positions collected by earlier jobs without paying for another rollout.
        """
        nonlocal updates
        return optimise_block(count, phase="persistent_replay")

    record_event(
        "collect",
        games_planned=games,
        offline_collection=offline_collection,
        max_envs=max_envs,
        reanalysis_limit_bytes=(reanalysis_store.max_bytes if reanalysis_store else 0),
        **memory_metrics(),
    )
    print(
        "population collect | "
        f"games={games} | offline={int(offline_collection)} | "
        f"reanalysis_limit_bytes="
        f"{reanalysis_store.max_bytes if reanalysis_store else 0}",
        flush=True,
    )
    with tqdm(total=games, desc=f"population SelfPlayPool ({device})", unit="game", dynamic_ncols=True) as progress:
        for start in range(0, games, int(max_envs)):
            count = min(int(max_envs), games - start)
            game_indices = np.arange(start, start + count, dtype=np.int64)
            seeds = (training.seed + game_indices).astype(np.uint32)
            opponent_indices = rng.choice(len(frozen), size=count, p=weights).astype(np.int32)
            candidate_players = (game_indices % 2).astype(np.int32)
            assignments = np.empty((count, 2), dtype=np.int32)
            for env in range(count):
                opponent_agent = 1 + int(opponent_indices[env])
                if candidate_players[env] == 0:
                    assignments[env] = (0, opponent_agent)
                else:
                    assignments[env] = (opponent_agent, 0)
            search = RoutedPoolSearch(
                training,
                mcts,
                (candidate, *frozen),
                num_envs=count,
                num_threads=num_threads,
                max_pending_leaves=max_pending_leaves,
                visible_action_history=visible_action_history,
                belief_retries=belief_retries,
                seed=int(seeds[0]),
                belief_action_mode=belief_action_mode,
                collect_engine_action_constraints=(
                    training.engine_action_constraint_weight > 0.0
                ),
            )
            request = search.reset(seeds, assignments)
            pending: list[tuple[TrainingExample | None, int, float] | None] = [None] * count
            trajectories: list[list[Transition]] = [[] for _ in range(count)]
            action_prefixes: list[list[int]] = [[] for _ in range(count)]
            critical_prefixes: list[dict[int, tuple[int, ...]]] = [
                {} for _ in range(count)
            ]
            done = np.zeros(count, dtype=bool)
            moves = np.zeros(count, dtype=np.int32)
            records: dict[int, Dict[str, Any]] = {}

            def consume_pending() -> None:
                for env, old in enumerate(pending):
                    if old is not None:
                        example, actor, reward = old
                        trajectories[env].append((actor, reward, example))
                        pending[env] = None

            vector_iteration = 0
            while True:
                consume_pending()
                terminal = np.asarray(request["terminated"], dtype=bool)
                newly_terminal = terminal & ~done
                horizon = (~terminal) & (~done) & (moves >= training.max_moves)
                drained = search.drain_completed()
                for row in range(len(drained.get("env_id", []))):
                    env = int(drained["env_id"][row])
                    records[env] = {name: value[row] for name, value in drained.items()}
                for env in np.flatnonzero(newly_terminal | horizon):
                    exact = (
                        np.asarray(request["terminal_values"][env], dtype=np.float32)
                        if newly_terminal[env]
                        else None
                    )
                    finished = _finish_trajectory(
                        trajectories[env], value_target_training, exact
                    )
                    if critical_only:
                        original_count = len(finished)
                        finished = [
                            example
                            for example in finished
                            if example.policy_loss_weight <= 0.0
                            or example.critical_priority
                        ]
                        discarded_noncritical_examples += (
                            original_count - len(finished)
                        )
                    if exact is not None:
                        for example in finished:
                            prefix = critical_prefixes[env].get(id(example))
                            if prefix is not None:
                                map_seed = int(seeds[env])
                                if map_seed in reserved_seed_set:
                                    reanalysis_reserved_rejected += 1
                                elif reanalysis_store is not None:
                                    # The normal replay and the durable
                                    # reanalysis store share immutable arrays.
                                    # Reanalysis later replaces only its
                                    # policy target on a new example.
                                    reanalysis_saved += int(reanalysis_store.add(
                                        ReanalysePosition(
                                            example=example,
                                            map_seed=map_seed,
                                            action_prefix=prefix,
                                            player_id=int(example.player_id),
                                            map_size=int(training.map_size),
                                            map_type=str(training.map_type),
                                            visible_action_history=int(
                                                visible_action_history
                                            ),
                                        )
                                    ))
                    replay.extend(finished)
                    examples_count += len(finished)
                    trajectories[env].clear()
                    done[env] = True
                    completed_count += 1
                    if exact is None:
                        truncated_count += 1
                    else:
                        terminal_count += 1
                    if not offline_collection:
                        optimise_until(completed_count * updates_per_game)
                    progress.update(1)
                active = ~done
                if not active.any():
                    break
                curriculum = max(0.0, 1.0 - completed_count / max(1, games))
                mcts_mask = active.copy()
                direct_policy = None
                direct_value = None
                if raw_policy_only:
                    mcts_mask = np.zeros(count, dtype=bool)
                elif selective_mcts:
                    (
                        mcts_mask,
                        direct_policy,
                        direct_value,
                    ) = search.selective_mcts_mask(
                        request,
                        active,
                        fraction=selective_mcts_fraction,
                        minimum_envs=selective_mcts_minimum_envs,
                        audit_fraction=selective_mcts_audit_fraction,
                        rng=rng,
                    )
                try:
                    root = search.search(
                        request,
                        active,
                        root_temperature=1.0,
                        # Root visit temperature controls which *legal move*
                        # self-play executes. It must not also randomise the
                        # hidden-map decoder: the old coupling made a default
                        # 1.0 action temperature create noisy, action-
                        # inconsistent belief particles.
                        belief_temperature=mcts.belief_particle_temperature,
                        prior_scale=curriculum,
                        root_noise=True,
                        mcts_mask=mcts_mask,
                        direct_policy=direct_policy,
                        direct_value=direct_value,
                        raw_policy_only=raw_policy_only,
                        moves_played=moves if raw_policy_only else None,
                    )
                except BeliefRecoveryRequired as error:
                    reset_rows, active_reset_rows, inactive_reset_rows = partition_recovery_envs(
                        error.env_ids, active
                    )
                    if reset_rows.size == 0:
                        raise BeliefQualityError(
                            "population belief recovery returned no valid slot ids",
                            {
                                "invalid_recovery_env_ids": error.env_ids.tolist(),
                                "belief_root_decisions": search.counters.root_env_decisions,
                                "belief_rejected_envs": search.counters.belief_rejected_envs,
                                "belief_rejected_particles": search.counters.belief_rejected_particles,
                            },
                        ) from error
                    discarded_prefixes = 0
                    for env in active_reset_rows:
                        discarded_prefixes += int(bool(trajectories[int(env)]))
                        finished = _finish_trajectory(
                            trajectories[int(env)], value_target_training, None
                        )
                        if critical_only:
                            original_count = len(finished)
                            finished = [
                                example
                                for example in finished
                                if example.policy_loss_weight <= 0.0
                                or example.critical_priority
                            ]
                            discarded_noncritical_examples += (
                                original_count - len(finished)
                            )
                        replay.extend(finished)
                        examples_count += len(finished)
                        trajectories[int(env)].clear()
                        action_prefixes[int(env)].clear()
                        critical_prefixes[int(env)].clear()
                        pending[int(env)] = None
                        records.pop(int(env), None)
                    search.counters.belief_discarded_prefixes += discarded_prefixes
                    search.counters.belief_slot_resets += int(reset_rows.size)
                    new_seeds = replacement_seeds(
                        training.seed,
                        recovery_seed_cursor,
                        int(reset_rows.size),
                        excluded=reserved_seed_set | recovery_seed_history,
                    )
                    recovery_seed_cursor += int(reset_rows.size)
                    recovery_seed_history.update(int(seed) for seed in new_seeds)
                    if active_reset_rows.size:
                        active_positions = np.flatnonzero(active[reset_rows])
                        seeds[active_reset_rows] = new_seeds[active_positions]
                        moves[active_reset_rows] = 0
                    search.pool.reset_slots_into(
                        reset_rows,
                        new_seeds,
                        np.ascontiguousarray(assignments[reset_rows], dtype=np.int32),
                        request,
                    )
                    run.log(
                        belief_recovery_event=1,
                        belief_slot_resets=search.counters.belief_slot_resets,
                        belief_active_slot_resets=int(active_reset_rows.size),
                        belief_inactive_slot_resets=int(inactive_reset_rows.size),
                        belief_discarded_prefixes=search.counters.belief_discarded_prefixes,
                        belief_root_decisions=search.counters.root_env_decisions,
                    )
                    if not offline_collection:
                        optimise_until(completed_count * updates_per_game)
                    if recovery_rate_exceeded(
                        search.counters.belief_slot_resets,
                        search.counters.root_env_decisions,
                    ):
                        run.log(
                            belief_quality_failed=1,
                            belief_slot_resets=search.counters.belief_slot_resets,
                            belief_root_decisions=search.counters.root_env_decisions,
                            replay_examples=len(replay),
                        )
                        raise BeliefQualityError(
                            "population belief slot reset rate exceeded 1%; "
                            "candidate was not safe to continue",
                            {
                                "belief_slot_resets": search.counters.belief_slot_resets,
                                "belief_root_decisions": search.counters.root_env_decisions,
                                "belief_rejected_envs": search.counters.belief_rejected_envs,
                                "belief_rejected_particles": search.counters.belief_rejected_particles,
                                "replay_examples": len(replay),
                                "optimisation_updates": updates,
                            },
                        ) from error
                    continue
                active_rows = np.flatnonzero(active)
                action_rows = _sample_active_rows(
                    root,
                    request,
                    active_rows,
                    rng,
                    temperature=np.asarray(
                        [self_play_temperature(mcts, int(move)) for move in moves],
                        dtype=np.float64,
                    ),
                    exploration_rate=training.exploration_rate * curriculum,
                )
                root_agents = np.asarray(request["agent_id"], dtype=np.int32)
                candidate_rows = active_rows[root_agents[active_rows] == 0]
                examples_by_env: dict[int, TrainingExample] = {}
                if candidate_rows.size:
                    candidate_request = _slice_rows(request, candidate_rows)
                    candidate_root = _slice_rows(root, candidate_rows)
                    state_ids = np.asarray(request["state_id"], dtype=np.uint64)
                    belief_targets = np.asarray(
                        search.pool.belief_targets_numpy(state_ids), dtype=np.int32
                    )
                    anchor_policy = None
                    anchor_value = None
                    if anchor_model is not None and (
                        anchor_policy_weight > 0.0 or anchor_value_weight > 0.0
                    ):
                        anchor_policy, anchor_value = _reference_policy_value(
                            anchor_model,
                            candidate_request,
                            device,
                            training,
                        )
                        measured = _policy_distance(
                            np.asarray(candidate_root["policy"], dtype=np.float32),
                            anchor_policy,
                            np.asarray(candidate_root["action_mask"], dtype=bool),
                        )
                        anchor_decisions += measured[0]
                        anchor_agreements += measured[1]
                        anchor_l1_sum += measured[2]
                    raw_candidate_policy = None
                    if critical_decision_fraction > 0.0 or selective_mcts:
                        # Direct rows already contain their raw policy in the
                        # hybrid root.  Run this extra reference forward only
                        # for MCTS rows, where it measures the actual search
                        # improvement instead of spending GPU time twice.
                        candidate_mcts_local = mcts_mask[candidate_rows]
                        raw_candidate_policy = np.asarray(
                            candidate_root["policy"], dtype=np.float32
                        ).copy()
                        if candidate_mcts_local.any():
                            candidate.eval()
                            raw_candidate_policy[candidate_mcts_local] = _reference_policy(
                                candidate,
                                _slice_rows(
                                    candidate_request,
                                    np.flatnonzero(candidate_mcts_local),
                                ),
                                device,
                                training,
                            )
                            measured = _policy_distance(
                                np.asarray(
                                    candidate_root["policy"], dtype=np.float32
                                )[candidate_mcts_local],
                                raw_candidate_policy[candidate_mcts_local],
                                np.asarray(
                                    candidate_root["action_mask"], dtype=bool
                                )[candidate_mcts_local],
                            )
                            mcts_raw_decisions += measured[0]
                            mcts_raw_agreements += measured[1]
                            mcts_raw_l1_sum += measured[2]
                    candidate_examples = _root_examples(
                        candidate_request,
                        candidate_root,
                        training,
                        belief_targets[candidate_rows],
                        anchor_policy=anchor_policy,
                        anchor_policy_weight=anchor_policy_weight,
                        anchor_value=anchor_value,
                        anchor_value_weight=anchor_value_weight,
                        engine_action_constraint_tiles=(
                            search.last_engine_action_constraint_tiles[candidate_rows]
                        ),
                        engine_action_constraint_scores=(
                            search.last_engine_action_constraint_scores[candidate_rows]
                        ),
                    )
                    if raw_candidate_policy is not None:
                        marked, critical_batch_score = _mark_critical_mcts_examples(
                            candidate_examples,
                            np.asarray(
                                candidate_root["policy"], dtype=np.float32
                            ),
                            raw_candidate_policy,
                            np.asarray(
                                candidate_root["action_mask"], dtype=bool
                            ),
                            fraction=critical_decision_fraction,
                            minimum_score=critical_minimum_score,
                        )
                        critical_decisions += marked
                        critical_score_sum += critical_batch_score
                    if selective_mcts:
                        # The selective scheduler has already paid for these
                        # roots.  Retain every selected candidate MCTS target
                        # for policy learning; otherwise the legacy
                        # disagreement threshold can leave an early run with
                        # virtually no policy supervision at all.
                        for local, example in enumerate(candidate_examples):
                            if mcts_mask[int(candidate_rows[local])]:
                                example.critical_priority = True
                    root_values = np.asarray(root["root_value"], dtype=np.float32)
                    for env, example in zip(candidate_rows, candidate_examples):
                        # Preserve the actual per-slot map provenance.  The
                        # calibration reservoir may survive into a run with a
                        # different held-out seed schedule.
                        example.map_seed = int(seeds[int(env)])
                        if not mcts_mask[int(env)]:
                            # A raw policy target is only an action source,
                            # never an outcome-improved policy teacher. Keep
                            # its belief/value supervision but mask policy
                            # cross-entropy, including value bootstrap games.
                            example.policy_loss_weight = 0.0
                        example.bootstrap_value = float(root_values[env])
                        examples_by_env[int(env)] = example
                        if example.critical_priority:
                            critical_prefixes[int(env)][id(example)] = tuple(
                                action_prefixes[int(env)]
                            )
                    decisions += len(candidate_examples)
                rewards = (
                    np.zeros(active_rows.size, dtype=np.float32)
                    if offline_collection
                    else _dense_shaped_rewards(request, active_rows, action_rows)
                )
                actors = np.asarray(request["to_play"], dtype=np.int32)
                for offset, env in enumerate(active_rows):
                    example = examples_by_env.get(int(env))
                    if example is not None:
                        example.shaped_reward = float(rewards[offset])
                    pending[env] = (example, int(actors[env]), float(rewards[offset]))
                actions = np.full(count, -1, dtype=np.int32)
                actions[active_rows] = np.asarray(root["action_id"], dtype=np.int32)[
                    active_rows, action_rows[active_rows]
                ]
                if raw_policy_only:
                    search.step_public_policy(actions, root["state_id"], request)
                else:
                    search.pool.step_into_checked(actions, root["state_id"], request)
                for env in active_rows:
                    action_prefixes[int(env)].append(int(actions[int(env)]))
                moves[active] += 1
                vector_iteration += 1
                if vector_iteration == 1 or vector_iteration % 25 == 0:
                    progress.set_postfix(
                        active=int(active.sum()), replay=len(replay), updates=updates
                    )

            for env, game_index in enumerate(game_indices):
                record = records.get(env)
                winner = int(record["winner_player"]) if record is not None else -1
                game_moves = int(record["moves"]) if record is not None else int(moves[env])
                candidate_player = int(candidate_players[env])
                score = 0.5 if winner < 0 else float(winner == candidate_player)
                score_sum += score
                total_moves += game_moves
                stats = opponent_stats[int(opponent_indices[env])]
                stats["games"] += 1.0
                stats["score"] += score
                stats["wins" if score == 1.0 else "losses" if score == 0.0 else "draws"] += 1.0
                if game_callback is not None:
                    game_callback({
                        "game_index": int(game_index),
                        "map_seed": int(seeds[env]),
                        "model_a_player": candidate_player,
                        "winner_player": winner,
                        "score_a": score,
                        "moves": game_moves,
                        "truncated": int(record is None),
                        "opponent_model_index": int(opponent_indices[env]),
                    })
            run.log(
                completed_games=completed_count,
                terminal_games=terminal_count,
                truncated_games=truncated_count,
                replay_examples=len(replay),
                optimisation_updates=updates,
                candidate_decisions=decisions,
                champion_anchor_weight=(
                    anchor_policy_weight if anchor_model is not None else 0.0
                ),
                champion_value_anchor_weight=(
                    anchor_value_weight if anchor_model is not None else 0.0
                ),
                champion_anchor_agreement=(
                    anchor_agreements / max(1, anchor_decisions)
                ),
                champion_anchor_mean_l1=(anchor_l1_sum / max(1, anchor_decisions)),
                mcts_raw_policy_agreement=(
                    mcts_raw_agreements / max(1, mcts_raw_decisions)
                ),
                mcts_raw_policy_mean_l1=(
                    mcts_raw_l1_sum / max(1, mcts_raw_decisions)
                ),
                mcts_raw_policy_decisions=mcts_raw_decisions,
                critical_mcts_decisions=critical_decisions,
                critical_mcts_mean_score=(
                    critical_score_sum / max(1, critical_decisions)
                ),
                critical_mcts_replay_fraction=critical_replay_fraction,
                critical_mcts_only=float(critical_only),
                discarded_noncritical_examples=discarded_noncritical_examples,
                reanalysis_loaded=reanalysis_loaded,
                reanalysis_refreshed=reanalysis_refreshed,
                reanalysis_failed=reanalysis_failed,
                reanalysis_buffer_size=(
                    len(reanalysis_store.positions) if reanalysis_store else 0
                ),
                mean_score=score_sum / max(1, completed_count),
                belief_rejected_particles=search.counters.belief_rejected_particles,
                belief_rejected_envs=search.counters.belief_rejected_envs,
                belief_action_equivalence_checked_particles=(
                    search.counters.belief_action_equivalence_checked_particles
                ),
                belief_action_equivalence_rejected_particles=(
                    search.counters.belief_action_equivalence_rejected_particles
                ),
                belief_resampled_envs=search.counters.belief_resampled_envs,
                belief_retry_successes=search.counters.belief_retry_successes,
                belief_conservative_attempts=search.counters.belief_conservative_attempts,
                belief_slot_resets=search.counters.belief_slot_resets,
                belief_discarded_prefixes=search.counters.belief_discarded_prefixes,
                belief_engine_action_constraint_particles=(
                    search.counters.engine_action_constraint_particles
                ),
                belief_engine_action_constraint_missing_actions=(
                    search.counters.engine_action_constraint_missing_actions
                ),
                belief_engine_action_constraint_extra_actions=(
                    search.counters.engine_action_constraint_extra_actions
                ),
                belief_engine_action_constraint_score=(
                    search.counters.engine_action_constraint_score_sum
                    / max(1, search.counters.engine_action_constraint_particles)
                ),
                belief_reset_rate=(
                    search.counters.belief_slot_resets
                    / max(1, search.counters.root_env_decisions)
                ),
                completion_rate=completed_count / max(1, games),
                effective_replay_epochs=effective_replay_epochs(
                    updates, training.batch_size, examples_count
                ),
                **_action_pruning_metrics(search.counters),
            )
            if losses:
                run.log(
                    completed_games=completed_count,
                    optimisation_updates=updates,
                    **losses[-1],
                )
            routing.add(search.counters)

    record_event(
        "collect_complete",
        games_completed=completed_count,
        replay_examples=len(replay),
        reanalysis_bytes=(reanalysis_store.bytes_used if reanalysis_store else 0),
        **memory_metrics(),
    )
    print(
        "population collect complete | "
        f"games={completed_count}/{games} | replay={len(replay)} | "
        f"reanalysis_bytes="
        f"{reanalysis_store.bytes_used if reanalysis_store else 0}",
        flush=True,
    )
    optimise_until(games * updates_per_game)
    offline_updates = optimise_replay_updates(extra_replay_updates)
    if reanalysis_store is not None:
        reanalysis_store.save()
    if replay_snapshot_callback is not None:
        replay_snapshot_callback(replay.snapshot())
    summary: Dict[str, Any] = {
        "population_pool_games": float(games),
        "population_pool_terminals": float(terminal_count),
        "population_pool_truncated": float(truncated_count),
        "population_pool_updates": float(updates),
        "population_pool_optimisation_blocks": float(optimisation_blocks),
        "population_pool_optimisation_attempted": float(optimisation_attempted),
        "population_pool_optimisation_rollbacks": float(optimisation_rollbacks),
        "population_pool_optimisation_screen_games": float(
            optimisation_screen_games
        ),
        "population_pool_optimisation_screen_attempts": float(
            optimisation_screen_attempts
        ),
        "population_pool_optimisation_screen_invalid": float(
            optimisation_screen_invalid
        ),
        "population_pool_optimisation_screen_rollbacks": float(
            optimisation_screen_rollbacks
        ),
        "population_pool_optimisation_screen_score": float(
            optimisation_screen_score
            if optimisation_screen_score is not None
            else -1.0
        ),
        "population_pool_optimisation_screen_best_score": float(
            optimisation_screen_best_score
            if optimisation_screen_best_score is not None
            else -1.0
        ),
        "population_pool_offline_replay_updates": float(offline_updates),
        "population_pool_examples": float(examples_count),
        "population_pool_replay_reserved_excluded": float(
            replay_reserved_excluded
        ),
        "population_pool_replay_unknown_seed_excluded": float(
            replay_unknown_excluded
        ),
        "population_pool_completion_rate": float(completed_count / max(1, games)),
        "population_pool_candidate_decisions": float(decisions),
        "population_pool_anchor_policy_weight": float(
            anchor_policy_weight if anchor_model is not None else 0.0
        ),
        "population_pool_anchor_value_weight": float(
            anchor_value_weight if anchor_model is not None else 0.0
        ),
        "population_pool_anchor_agreement": float(
            anchor_agreements / max(1, anchor_decisions)
        ),
        "population_pool_anchor_mean_l1": float(
            anchor_l1_sum / max(1, anchor_decisions)
        ),
        "population_pool_mcts_raw_policy_decisions": float(mcts_raw_decisions),
        "population_pool_mcts_raw_policy_agreement": float(
            mcts_raw_agreements / max(1, mcts_raw_decisions)
        ),
        "population_pool_mcts_raw_policy_mean_l1": float(
            mcts_raw_l1_sum / max(1, mcts_raw_decisions)
        ),
        "population_pool_critical_mcts_decisions": float(critical_decisions),
        "population_pool_critical_mcts_mean_score": float(
            critical_score_sum / max(1, critical_decisions)
        ),
        "population_pool_critical_mcts_replay_fraction": float(
            critical_replay_fraction
        ),
        "population_pool_critical_only": float(critical_only),
        "population_pool_discarded_noncritical_examples": float(
            discarded_noncritical_examples
        ),
        "population_pool_critical_mcts_selection_fraction": float(
            critical_decision_fraction
        ),
        "population_pool_selective_mcts": float(selective_mcts),
        "population_pool_selective_mcts_target_fraction": float(
            selective_mcts_fraction
        ),
        "population_pool_selective_mcts_minimum_envs": float(
            selective_mcts_minimum_envs
        ),
        "population_pool_selective_mcts_audit_fraction": float(
            selective_mcts_audit_fraction
        ),
        "population_pool_raw_policy_only": float(raw_policy_only),
        "population_pool_value_belief_heads_only": float(value_belief_heads_only),
        "population_pool_offline_collection": float(offline_collection),
        "population_pool_reanalysis_loaded": float(reanalysis_loaded),
        "population_pool_reanalysis_refreshed": float(reanalysis_refreshed),
        "population_pool_reanalysis_failed": float(reanalysis_failed),
        "population_pool_reanalysis_reserved_excluded": float(
            reanalysis_reserved_excluded
        ),
        "population_pool_reanalysis_saved": float(reanalysis_saved),
        "population_pool_reanalysis_buffer_size": float(
            len(reanalysis_store.positions) if reanalysis_store else 0
        ),
        "population_pool_reanalysis_buffer_bytes": float(
            reanalysis_store.bytes_used if reanalysis_store else 0
        ),
        "population_pool_reanalysis_oversize_rejected": float(
            reanalysis_store.rejected_oversize if reanalysis_store else 0
        ),
        "population_pool_reanalysis_invalid_rejected": float(
            reanalysis_store.rejected_invalid if reanalysis_store else 0
        ),
        "population_pool_reanalysis_duplicates_replaced": float(
            reanalysis_store.replaced_duplicates if reanalysis_store else 0
        ),
        "population_pool_reanalysis_evicted": float(
            reanalysis_store.evicted if reanalysis_store else 0
        ),
        "population_pool_reanalysis_reserved_rejected": float(
            reanalysis_reserved_rejected
        ),
        "population_pool_score": float(score_sum / max(1, games)),
        "population_pool_mean_moves": float(total_moves / max(1, games)),
        "population_pool_leaf_rows": float(routing.leaf_rows),
        "population_pool_leaf_waves": float(routing.leaf_waves),
        "population_pool_belief_retries": float(routing.belief_retries),
        "population_pool_belief_action_mode": belief_action_mode,
        "population_pool_belief_fallback_rows": float(routing.belief_fallback_rows),
        "population_pool_belief_rejections": float(routing.belief_rejections),
        "belief_rejected_particles": float(routing.belief_rejected_particles),
        "belief_rejected_envs": float(routing.belief_rejected_envs),
        "belief_action_equivalence_checked_particles": float(
            routing.belief_action_equivalence_checked_particles
        ),
        "belief_action_equivalence_rejected_particles": float(
            routing.belief_action_equivalence_rejected_particles
        ),
        "belief_action_equivalence_rejected_envs": float(
            routing.belief_action_equivalence_rejected_envs
        ),
        "belief_action_equivalence_rejection_rate": float(
            routing.belief_action_equivalence_rejected_particles
            / max(1, routing.belief_action_equivalence_checked_particles)
        ),
        "belief_resampled_envs": float(routing.belief_resampled_envs),
        "belief_retry_successes": float(routing.belief_retry_successes),
        "belief_conservative_attempts": float(routing.belief_conservative_attempts),
        "belief_slot_resets": float(routing.belief_slot_resets),
        "belief_discarded_prefixes": float(routing.belief_discarded_prefixes),
        "belief_engine_action_constraint_queries": float(
            routing.engine_action_constraint_queries
        ),
        "belief_engine_action_constraint_particles": float(
            routing.engine_action_constraint_particles
        ),
        "belief_engine_action_constraint_missing_actions": float(
            routing.engine_action_constraint_missing_actions
        ),
        "belief_engine_action_constraint_extra_actions": float(
            routing.engine_action_constraint_extra_actions
        ),
        "belief_engine_action_constraint_tiles": float(
            routing.engine_action_constraint_tiles
        ),
        "belief_engine_action_constraint_score": float(
            routing.engine_action_constraint_score_sum
            / max(1, routing.engine_action_constraint_particles)
        ),
        "belief_reset_rate": float(
            routing.belief_slot_resets / max(1, routing.root_env_decisions)
        ),
        "effective_replay_epochs": effective_replay_epochs(
            updates, training.batch_size, examples_count
        ),
        "requested_optimisation_updates": float(games * updates_per_game),
        "replay_examples": float(len(replay)),
        "mcts_top_actions": float(mcts.search_top_actions),
        "mcts_random_actions": float(mcts.search_random_actions),
    }
    summary.update(_action_pruning_metrics(routing))
    if raw_policy_only:
        counts = routing.belief_particle_diagnostics
        summary.update({f"belief_sampled_{name}": float(count) for name, count in counts.items()})
        summary["belief_sampled_unit_precision"] = counts["unit_correct"] / max(1, counts["unit_predicted"])
        summary["belief_sampled_unit_recall"] = counts["unit_correct"] / max(1, counts["unit_actual"])
        summary["belief_build_errors"] = [
            {"message": message, **example}
            for message, example in sorted(
                routing.belief_build_errors.items(),
                key=lambda item: (-item[1]["count"], item[0]),
            )
        ]
    for index, stats in enumerate(opponent_stats):
        summary[f"opponent_{index}_games"] = float(stats["games"])
        summary[f"opponent_{index}_score"] = float(
            stats["score"] / max(1.0, stats["games"])
        )
    if losses:
        for key in losses[0]:
            summary[key] = float(np.mean([loss[key] for loss in losses]))
    summary["run_dir"] = str(run.finish(summary))
    return summary
