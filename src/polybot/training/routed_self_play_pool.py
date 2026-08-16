"""Multi-checkpoint training and evaluation on one native ``SelfPlayPool``.

The pool owns every live game and MCTS tree.  Python only routes dense belief
and leaf rows through the model selected by PolyEnv's ``agent_id`` field, then
scatters neural outputs back to the original native order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import native_batch_to_tensors, stack_examples
from ..infrastructure.device import select_device
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from .global_mcts_train import _native_leaf_bias
from .belief_recovery import (
    BeliefQualityError,
    BeliefRecoveryRequired,
    capped_update_target,
    effective_replay_epochs,
    ismcts_rejected_envs,
    particle_temperature,
    recovery_rate_exceeded,
    replace_rejected_particles,
    replacement_seeds,
)
from .optimisation import train_batch
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
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not models:
            raise ValueError("at least one routed model is required")
        if not 1 <= max_pending_leaves <= 8:
            raise ValueError("max_pending_leaves must be between one and eight")
        if belief_retries < 0:
            raise ValueError("belief_retries must not be negative")
        Bardur, Imperius, MapType, SelfPlayPool = _pool_types()
        self.training = training
        self.mcts = mcts
        self.models = list(models)
        self.device = select_device(training.device)
        self.num_envs = int(num_envs)
        self.max_pending_leaves = int(max_pending_leaves)
        self.belief_retries = int(belief_retries)
        self.counters = SearchCounters()
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
                kind = _belief_rejection_kind(error)
                original = f"; learned error={last_error}" if last_error is not None else ""
                raise BeliefRecoveryRequired(
                    rejected_rows,
                    "Routed SelfPlayPool rejected both learned and conservative "
                    f"beliefs ({kind}): {error}{original}",
                    diagnostics,
                ) from error

        waves = 1 + int(
            np.ceil(max(0, self.mcts.simulations - 1) / self.max_pending_leaves)
        )
        for _ in range(waves):
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
        self.pool.root_policy_into(self.root, temperature=root_temperature)
        action_counts = np.asarray(self.root["action_mask"], dtype=np.uint8).sum(axis=1)
        empty_live = np.flatnonzero(active & (action_counts == 0))
        if empty_live.size:
            raise RuntimeError(
                "SelfPlayPool returned unexpanded live roots: "
                f"{empty_live[:8].tolist()}; stats={self.pool.search_stats()}"
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
) -> Dict[str, Any]:
    """Run deterministic, paired candidate evaluation entirely in native pools."""
    if games < 2 or games % 2:
        raise ValueError("games must be positive and even for paired evaluation")
    if max_envs < 2:
        raise ValueError("max_envs must be at least two")
    seed_everything(training.seed)
    # Keep the paired map and player slots fixed. Evaluation gets extra
    # independent stochastic belief attempts before it is declared invalid.
    evaluation_belief_retries = max(8, int(belief_retries))
    totals: Dict[str, float] = {
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
    chunk_limit = max(2, int(max_envs) - int(max_envs) % 2)
    with tqdm(total=games, desc="evaluate A vs model_b (SelfPlayPool)", unit="game", dynamic_ncols=True) as progress:
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
            for _ in range(training.max_moves):
                terminal = np.asarray(request["terminated"], dtype=bool)
                done |= terminal
                active = ~done
                if not active.any():
                    break
                try:
                    root = search.search(
                        request,
                        active,
                        root_temperature=0.0,
                        belief_temperature=0.0,
                        prior_scale=0.0,
                    )
                except BeliefRecoveryRequired as error:
                    progress.set_description("invalid belief evaluation")
                    rejected_particles = (
                        routing.belief_rejected_particles
                        + search.counters.belief_rejected_particles
                    )
                    rejected_envs = (
                        routing.belief_rejected_envs
                        + search.counters.belief_rejected_envs
                    )
                    resampled_envs = (
                        routing.belief_resampled_envs
                        + search.counters.belief_resampled_envs
                    )
                    retry_successes = (
                        routing.belief_retry_successes
                        + search.counters.belief_retry_successes
                    )
                    conservative_attempts = (
                        routing.belief_conservative_attempts
                        + search.counters.belief_conservative_attempts
                    )
                    return {
                        "valid": 0.0,
                        "invalid_evaluations": 1.0,
                        "requested_games": float(games),
                        "games": 0.0,
                        "evaluation_seed": float(training.seed),
                        "invalid_envs": float(error.env_ids.size),
                        "invalid_env_ids": error.env_ids.tolist(),
                        "invalid_reason": str(error),
                        "invalid_diagnostics": list(error.diagnostics),
                        "belief_rejected_particles": float(rejected_particles),
                        "belief_rejected_envs": float(rejected_envs),
                        "belief_resampled_envs": float(resampled_envs),
                        "belief_retry_successes": float(retry_successes),
                        "belief_conservative_attempts": float(conservative_attempts),
                        "belief_slot_resets": 0.0,
                        "belief_discarded_prefixes": 0.0,
                        "self_play_pool_belief_rejected_particles": float(
                            rejected_particles
                        ),
                        "self_play_pool_belief_rejected_envs": float(rejected_envs),
                        "self_play_pool_belief_resampled_envs": float(resampled_envs),
                        "self_play_pool_belief_retry_successes": float(retry_successes),
                        "self_play_pool_belief_conservative_attempts": float(
                            conservative_attempts
                        ),
                    }
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
                    if not done[env]:
                        done[env] = True
                        progress.update(1)
            # Drain records produced by the final native step.
            drained = search.drain_completed()
            for row in range(len(drained.get("env_id", []))):
                env = int(drained["env_id"][row])
                records[env] = {name: value[row] for name, value in drained.items()}
                if not done[env]:
                    done[env] = True
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
            unfinished = count - len(records)
            if unfinished:
                progress.update(unfinished)
            routing.leaf_rows += search.counters.leaf_rows
            routing.leaf_waves += search.counters.leaf_waves
            routing.belief_retries += search.counters.belief_retries
            routing.belief_fallback_rows += search.counters.belief_fallback_rows
            routing.belief_rejections += search.counters.belief_rejections
            routing.belief_rejected_particles += search.counters.belief_rejected_particles
            routing.belief_rejected_envs += search.counters.belief_rejected_envs
            routing.belief_resampled_envs += search.counters.belief_resampled_envs
            routing.belief_retry_successes += search.counters.belief_retry_successes
            routing.belief_conservative_attempts += search.counters.belief_conservative_attempts
            routing.root_env_decisions += search.counters.root_env_decisions
            progress.set_postfix(
                wins=int(totals["model_a_wins"]), draws=int(totals["draws"])
            )
    _paired_uncertainty(totals, scores)
    totals["valid"] = 1.0
    totals["invalid_evaluations"] = 0.0
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
    totals["belief_slot_resets"] = 0.0
    totals["belief_discarded_prefixes"] = 0.0
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
    num_threads: int | None = None,
    max_pending_leaves: int = 4,
    visible_action_history: int = 0,
    belief_retries: int = 4,
    max_envs: int = 128,
    game_callback: GameCallback | None = None,
    reserved_seeds: Sequence[int] = (),
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train candidate agent 0 against a PFSP population in native batches."""
    if games < 0 or updates_per_game < 0:
        raise ValueError("games and updates_per_game must not be negative")
    if games == 0:
        return {"population_pool_games": 0.0, "population_pool_updates": 0.0}
    if not opponent_models:
        raise ValueError("population training requires at least one opponent model")
    if max_envs < 1:
        raise ValueError("max_envs must be positive")
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
        },
    )
    candidate.to(device)
    frozen = []
    for model in opponent_models:
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        frozen.append(model)
    optimiser = torch.optim.AdamW(
        candidate.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed ^ 0x91A6E)
    losses: list[Dict[str, float]] = []
    updates = 0
    completed_count = 0
    terminal_count = 0
    truncated_count = 0
    examples_count = 0
    decisions = 0
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
                training.batch_size, rng, training.priority_replay_fraction
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
                    finished = _finish_trajectory(trajectories[env], training, exact)
                    replay.extend(finished)
                    examples_count += len(finished)
                    trajectories[env].clear()
                    done[env] = True
                    completed_count += 1
                    if exact is None:
                        truncated_count += 1
                    else:
                        terminal_count += 1
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
                    reset_rows = np.asarray(
                        [env for env in error.env_ids if active[int(env)]], dtype=np.int32
                    )
                    if reset_rows.size == 0:
                        raise
                    discarded_prefixes = 0
                    for env in reset_rows:
                        discarded_prefixes += int(bool(trajectories[int(env)]))
                        finished = _finish_trajectory(trajectories[int(env)], training, None)
                        replay.extend(finished)
                        examples_count += len(finished)
                        trajectories[int(env)].clear()
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
                    seeds[reset_rows] = new_seeds
                    moves[reset_rows] = 0
                    search.pool.reset_slots_into(
                        reset_rows,
                        new_seeds,
                        np.ascontiguousarray(assignments[reset_rows], dtype=np.int32),
                        request,
                    )
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
                    state_ids = np.asarray(request["state_id"], dtype=np.uint64)
                    belief_targets = np.asarray(
                        search.pool.belief_targets_numpy(state_ids), dtype=np.int32
                    )
                    candidate_examples = _root_examples(
                        _slice_rows(request, candidate_rows),
                        _slice_rows(root, candidate_rows),
                        training,
                        belief_targets[candidate_rows],
                    )
                    root_values = np.asarray(root["root_value"], dtype=np.float32)
                    for env, example in zip(candidate_rows, candidate_examples):
                        example.bootstrap_value = float(root_values[env])
                        examples_by_env[int(env)] = example
                    decisions += len(candidate_examples)
                rewards = _dense_shaped_rewards(request, active_rows, action_rows)
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
            )
            routing.leaf_rows += search.counters.leaf_rows
            routing.leaf_waves += search.counters.leaf_waves
            routing.belief_retries += search.counters.belief_retries
            routing.belief_fallback_rows += search.counters.belief_fallback_rows
            routing.belief_rejections += search.counters.belief_rejections
            routing.belief_rejected_particles += search.counters.belief_rejected_particles
            routing.belief_rejected_envs += search.counters.belief_rejected_envs
            routing.belief_resampled_envs += search.counters.belief_resampled_envs
            routing.belief_retry_successes += search.counters.belief_retry_successes
            routing.belief_conservative_attempts += search.counters.belief_conservative_attempts
            routing.belief_slot_resets += search.counters.belief_slot_resets
            routing.belief_discarded_prefixes += search.counters.belief_discarded_prefixes
            routing.root_env_decisions += search.counters.root_env_decisions

    optimise_until(games * updates_per_game)
    summary: Dict[str, float | str] = {
        "population_pool_games": float(games),
        "population_pool_terminals": float(terminal_count),
        "population_pool_truncated": float(truncated_count),
        "population_pool_updates": float(updates),
        "population_pool_examples": float(examples_count),
        "population_pool_completion_rate": float(completed_count / max(1, games)),
        "population_pool_candidate_decisions": float(decisions),
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
    }
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
