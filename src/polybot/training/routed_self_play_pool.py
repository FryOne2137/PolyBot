"""Multi-checkpoint training and evaluation on one native ``SelfPlayPool``.

The pool owns every live game and MCTS tree.  Python only routes dense belief
and leaf rows through the model selected by PolyEnv's ``agent_id`` field, then
scatters neural outputs back to the original native order.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import native_batch_to_tensors, stack_examples
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
    ismcts_rejected_envs,
    particle_temperature,
    partition_recovery_envs,
    recovery_rate_exceeded,
    replace_rejected_particles,
    replacement_seeds,
)
from .optimisation import train_batch
from .persistent_reanalysis import PersistentReanalyseBuffer, ReanalysePosition
from .replay_buffer import ReplayBuffer, TrainingExample
from .run_artifacts import TrainingRun
from .self_play_pool_train import (
    _allocate_native_buffers,
    _belief_rejection_kind,
    _finish_trajectory,
    _native_globals,
    _root_examples,
)
from .value_targets import Transition


GameCallback = Callable[[Mapping[str, Any]], None]

_EVALUATION_MIN_BELIEF_RETRIES = 64
_EVALUATION_POSITION_RECOVERY_CYCLES = 2


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


def _reanalyse_critical_positions(
    positions: Sequence[ReanalysePosition],
    *,
    candidate: PolicyValueBeliefNet,
    training: TrainingConfig,
    mcts: MCTSConfig,
    device: torch.device,
    minimum_score: float,
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
    )
    refreshed: list[TrainingExample] = []
    failed = 0
    candidate.eval()
    for position in positions:
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
            if float(visits.sum()) <= 0.0:
                visits = np.asarray(decision.priors, dtype=np.float32)
            visits /= max(float(visits.sum()), 1e-8)
            tensors = stack_examples([previous], device)
            with torch.inference_mode():
                output = candidate(
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
            raw = torch.softmax(
                output.policy_logits[0, : visits.size].float(), dim=-1
            ).cpu().numpy()
            top = int(visits.argmax())
            score = float(visits[top] * max(0.0, visits[top] - raw[top]))
            if score < minimum_score or top == int(raw.argmax()):
                continue
            updated = copy.deepcopy(previous)
            updated.policy_target = visits.astype(np.float32, copy=False)
            updated.critical_priority = True
            updated.critical_score = score
            updated.reanalysis_priority = True
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
    root_env_decisions: int = 0
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
            if name == "search_prior_mass_retained_sum":
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
    }


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
        self.counters = SearchCounters()
        self._action_generator: torch.Generator | None = None
        if self.device.type in ("cpu", "cuda"):
            self._action_generator = torch.Generator(device=self.device)
            self._action_generator.manual_seed(int(seed) ^ 0x5EA2C11)
        self._capture_root_candidates = False
        self._root_candidate_action_ids: dict[int, np.ndarray] = {}
        for model in self.models:
            model.to(self.device).eval()
        map_type = MapType.Drylands if training.map_type == "drylands" else MapType.Lakes
        native_threads = 0 if num_threads is None else max(1, int(num_threads))
        self.pool = SelfPlayPool(
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
        # Pinned tensors own the memory exported through their NumPy views.
        self._buffer_owners = (
            request_owners,
            root_owners,
            all_player_owners,
            leaf_owners,
            completed_owners,
        )

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
                    _native_globals(
                        flat_states[rows], self.training.map_size, self.training.map_type
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
    def _expand_leaves(self, leaf_count: int, prior_scale: float) -> None:
        batch = {name: value[:leaf_count] for name, value in self.leaves.items()}
        agents = np.asarray(batch["agent_id"], dtype=np.int32)
        _validate_agents(agents, len(self.models), "leaf packet")
        max_actions = np.asarray(batch["action_mask"]).shape[1]
        logits = np.full((leaf_count, max_actions), -np.inf, dtype=np.float32)
        values = np.zeros(leaf_count, dtype=np.float32)
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
            ).to(output.policy_logits.dtype)
            routed_logits, selected, pruning = prune_search_logits(
                routed_logits,
                tensors.action_categorical,
                tensors.action_mask,
                top_actions=self.search_top_actions_by_agent[int(agent)],
                random_actions=self.search_random_actions_by_agent[int(agent)],
                generator=self._action_generator,
            )
            self.counters.add_action_pruning(pruning)
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
    ) -> Dict[str, np.ndarray]:
        """Run routed belief completion and MCTS for one vector decision."""
        for model in self.models:
            model.eval()
        state_ids = np.asarray(request["state_id"], dtype=np.uint64)
        actors = np.asarray(request["to_play"], dtype=np.int64)
        particle_count = max(1, int(self.mcts.particles))
        self.pool.all_player_belief_requests_into(self.all_players)
        all_players = self.all_players
        active = np.asarray(active, dtype=bool)
        self.counters.root_env_decisions += int(active.sum())

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
        submitted = False
        last_error: ValueError | None = None
        rejected_rows = all_rows
        retry_pending_envs = 0
        diagnostics: tuple[Mapping[str, Any], ...] = ()
        has_acceptance_mask = callable(
            getattr(self.pool, "ismcts_belief_acceptance_mask", None)
        )
        for attempt in range(self.belief_retries + 1):
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
            if has_acceptance_mask:
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

        waves = 1 + int(
            np.ceil(max(0, self.mcts.simulations - 1) / self.max_pending_leaves)
        )
        self._root_candidate_action_ids = {}
        for wave in range(waves):
            self._capture_root_candidates = wave == 0
            leaf_count = int(
                self.pool.select_leaves_into(
                    self.leaves, max_leaves=self.leaf_capacity
                )
            )
            self.counters.leaf_waves += 1
            self.counters.leaf_rows += leaf_count
            if leaf_count == 0:
                break
            self._expand_leaves(leaf_count, prior_scale)
        self._capture_root_candidates = False
        self.pool.root_policy_into(self.root, temperature=root_temperature)
        action_counts = np.asarray(self.root["action_mask"], dtype=np.uint8).sum(axis=1)
        empty_live = np.flatnonzero(active & (action_counts == 0))
        if empty_live.size:
            raise RuntimeError(
                "SelfPlayPool returned unexpanded live roots: "
                f"{empty_live[:8].tolist()}; stats={self.pool.search_stats()}"
            )
        root_agents = np.asarray(self.root["agent_id"], dtype=np.int32)
        enabled = np.zeros(self.num_envs, dtype=bool)
        for env in np.flatnonzero(active):
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
        return self.root

    def drain_completed(self) -> Dict[str, np.ndarray]:
        count = int(self.pool.drain_completed_into(self.completed))
        return {name: np.asarray(value[:count]).copy() for name, value in self.completed.items()}


def _sample_active_rows(
    root: Mapping[str, np.ndarray],
    request: Mapping[str, np.ndarray],
    active_rows: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
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
        if temperature <= 1e-5:
            row = int(weights.argmax())
        else:
            row = int(rng.choice(count, p=weights))
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
    totals["score_standard_error"] = (
        float(np.std(map_scores, ddof=1) / np.sqrt(len(map_scores)))
        if len(map_scores) > 1
        else 0.0
    )
    totals["score_lower_95"] = max(
        0.0, totals["model_a_score"] - 1.96 * totals["score_standard_error"]
    )
    totals["mean_moves"] = totals["moves"] / max(1.0, totals["games"])
    totals["mean_model_a_particles"] = (
        totals["model_a_particles"] / max(1.0, totals["model_a_decisions"])
    )


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
    max_envs: int = 128,
    game_callback: GameCallback | None = None,
    model_b_search_top_actions: int | None = None,
    model_b_search_random_actions: int | None = None,
    model_a_label: str = "model A",
    model_b_label: str = "model B",
) -> Dict[str, Any]:
    """Run deterministic, paired candidate evaluation entirely in native pools."""
    if games < 2 or games % 2:
        raise ValueError("games must be positive and even for paired evaluation")
    if max_envs < 2:
        raise ValueError("max_envs must be at least two")
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
    used_evaluation_seeds = {
        int(training.seed) + game_index // 2 for game_index in range(games)
    }
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
                try:
                    root, recovery_failures = _evaluation_search_with_recovery(
                        search,
                        request,
                        active,
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
                chosen = np.asarray(request["action_features"], dtype=np.int32)[
                    active_rows, action_rows[active_rows]
                ]
                city = (chosen[:, 0] == 8) & np.isin(chosen[:, 8], (11, 14))
                capital = (chosen[:, 0] == 8) & (chosen[:, 8] == 14)
                for offset, env in enumerate(active_rows):
                    if candidate_turns[env]:
                        candidate_city_captures[env] += int(city[offset])
                        candidate_capital_captures[env] += int(capital[offset])
                    else:
                        opponent_city_captures[env] += int(city[offset])
                        opponent_capital_captures[env] += int(capital[offset])
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
    totals["invalid_evaluations"] = 0.0
    totals["discarded_evaluation_pairs"] = float(discarded_evaluation_pairs)
    totals["invalid_evaluation_attempts"] = float(invalid_evaluation_attempts)
    totals["evaluation_replacement_events"] = replacement_events
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
    reanalysis_path: Path | None = None,
    reanalysis_capacity: int = 4_096,
    reanalysis_positions: int = 0,
    reanalysis_replay_fraction: float = 0.0,
    offline_collection: bool = False,
    num_threads: int | None = None,
    max_pending_leaves: int = 4,
    visible_action_history: int = 0,
    belief_retries: int = 4,
    max_envs: int = 128,
    game_callback: GameCallback | None = None,
    reserved_seeds: Sequence[int] = (),
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train candidate agent 0 against PFSP with frozen-policy retention.

    When ``offline_collection`` is set, MCTS generates a complete fixed
    generation before any optimiser step.  This is the stable policy-iteration
    mode: every root target in the replay buffer comes from the same frozen
    apprentice, rather than a network that changes game by game.
    """
    if games < 0 or updates_per_game < 0:
        raise ValueError("games and updates_per_game must not be negative")
    if games == 0:
        return {"population_pool_games": 0.0, "population_pool_updates": 0.0}
    if not opponent_models:
        raise ValueError("population training requires at least one opponent model")
    if max_envs < 1:
        raise ValueError("max_envs must be positive")
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
    if reanalysis_capacity < 1:
        raise ValueError("reanalysis_capacity must be positive")
    if reanalysis_positions < 0:
        raise ValueError("reanalysis_positions must not be negative")
    if not 0.0 <= reanalysis_replay_fraction <= 1.0:
        raise ValueError("reanalysis_replay_fraction must be between zero and one")
    if reanalysis_positions and reanalysis_path is None:
        raise ValueError("reanalysis_positions requires a persistent reanalysis_path")
    if critical_only and critical_decision_fraction <= 0.0:
        raise ValueError(
            "critical_only requires a positive critical_decision_fraction"
        )
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
            "reanalysis_path": str(reanalysis_path) if reanalysis_path else "",
            "reanalysis_capacity": reanalysis_capacity,
            "reanalysis_positions": reanalysis_positions,
            "reanalysis_replay_fraction": reanalysis_replay_fraction,
            "offline_collection": offline_collection,
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
    frozen = []
    for model in opponent_models:
        frozen.append(_freeze_reference_model(model, device))
    if anchor_model is not None:
        anchor_model = _freeze_reference_model(anchor_model, device)
    optimiser = torch.optim.AdamW(
        candidate.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed ^ 0x91A6E)
    reanalysis_store = (
        PersistentReanalyseBuffer(reanalysis_path, reanalysis_capacity)
        if reanalysis_path is not None
        else None
    )
    reanalysis_loaded = reanalysis_store.load() if reanalysis_store else 0
    reanalysis_refreshed = 0
    reanalysis_failed = 0
    if reanalysis_store is not None and reanalysis_positions:
        reanalysed, reanalysis_failed = _reanalyse_critical_positions(
            reanalysis_store.sample(reanalysis_positions, rng),
            candidate=candidate,
            training=training,
            mcts=mcts,
            device=device,
            minimum_score=critical_minimum_score,
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
    reserved_seed_set = {int(value) for value in reserved_seeds}
    recovery_seed_history: set[int] = set()
    new_reanalysis_positions: list[ReanalysePosition] = []

    def optimise_until(target: int) -> None:
        nonlocal updates
        capped = capped_update_target(
            target,
            examples_count,
            training.batch_size,
            training.max_replay_epochs,
        )
        while updates < capped and len(replay) >= training.batch_size:
            batch = replay.sample(
                training.batch_size,
                rng,
                training.priority_replay_fraction,
                training.bridge_replay_fraction,
                critical_replay_fraction,
                reanalysis_replay_fraction,
            )
            losses.append(
                train_batch(candidate, optimiser, stack_examples(batch, device), training)
            )
            updates += 1

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
                            if example.critical_priority
                        ]
                        discarded_noncritical_examples += (
                            original_count - len(finished)
                        )
                    if exact is not None:
                        for example in finished:
                            prefix = critical_prefixes[env].get(id(example))
                            if prefix is not None:
                                new_reanalysis_positions.append(
                                    ReanalysePosition(
                                        example=copy.deepcopy(example),
                                        map_seed=int(seeds[env]),
                                        action_prefix=prefix,
                                        player_id=int(example.player_id),
                                    )
                                )
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
                try:
                    root = search.search(
                        request,
                        active,
                        root_temperature=1.0,
                        belief_temperature=max(mcts.temperature, 0.35),
                        prior_scale=curriculum,
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
                                if example.critical_priority
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
                    temperature=mcts.temperature,
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
                    if critical_decision_fraction > 0.0:
                        # The candidate's unsearched policy is the apprentice
                        # we want to improve.  Compare it with the completed
                        # MCTS root only after search; the root policy itself
                        # stays the target stored by _root_examples.
                        candidate.eval()
                        raw_candidate_policy = _reference_policy(
                            candidate,
                            candidate_request,
                            device,
                            training,
                        )
                    candidate_examples = _root_examples(
                        candidate_request,
                        candidate_root,
                        training,
                        belief_targets[candidate_rows],
                        anchor_policy=anchor_policy,
                        anchor_policy_weight=anchor_policy_weight,
                        anchor_value=anchor_value,
                        anchor_value_weight=anchor_value_weight,
                    )
                    if raw_candidate_policy is not None:
                        marked, score_sum = _mark_critical_mcts_examples(
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
                        critical_score_sum += score_sum
                    root_values = np.asarray(root["root_value"], dtype=np.float32)
                    for env, example in zip(candidate_rows, candidate_examples):
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
                search.pool.step_into_checked(
                    actions,
                    np.asarray(root["state_id"], dtype=np.uint64),
                    request,
                )
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
                belief_resampled_envs=search.counters.belief_resampled_envs,
                belief_retry_successes=search.counters.belief_retry_successes,
                belief_conservative_attempts=search.counters.belief_conservative_attempts,
                belief_slot_resets=search.counters.belief_slot_resets,
                belief_discarded_prefixes=search.counters.belief_discarded_prefixes,
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

    optimise_until(games * updates_per_game)
    if reanalysis_store is not None:
        reanalysis_store.extend(new_reanalysis_positions)
        reanalysis_store.save()
    summary: Dict[str, float | str] = {
        "population_pool_games": float(games),
        "population_pool_terminals": float(terminal_count),
        "population_pool_truncated": float(truncated_count),
        "population_pool_updates": float(updates),
        "population_pool_examples": float(examples_count),
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
        "population_pool_offline_collection": float(offline_collection),
        "population_pool_reanalysis_loaded": float(reanalysis_loaded),
        "population_pool_reanalysis_refreshed": float(reanalysis_refreshed),
        "population_pool_reanalysis_failed": float(reanalysis_failed),
        "population_pool_reanalysis_saved": float(len(new_reanalysis_positions)),
        "population_pool_reanalysis_buffer_size": float(
            len(reanalysis_store.positions) if reanalysis_store else 0
        ),
        "population_pool_score": float(score_sum / max(1, games)),
        "population_pool_mean_moves": float(total_moves / max(1, games)),
        "population_pool_leaf_rows": float(routing.leaf_rows),
        "population_pool_leaf_waves": float(routing.leaf_waves),
        "population_pool_belief_retries": float(routing.belief_retries),
        "population_pool_belief_fallback_rows": float(routing.belief_fallback_rows),
        "population_pool_belief_rejections": float(routing.belief_rejections),
        "belief_rejected_particles": float(routing.belief_rejected_particles),
        "belief_rejected_envs": float(routing.belief_rejected_envs),
        "belief_resampled_envs": float(routing.belief_resampled_envs),
        "belief_retry_successes": float(routing.belief_retry_successes),
        "belief_conservative_attempts": float(routing.belief_conservative_attempts),
        "belief_slot_resets": float(routing.belief_slot_resets),
        "belief_discarded_prefixes": float(routing.belief_discarded_prefixes),
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
