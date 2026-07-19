"""High-throughput belief-MCTS self-play via PolyEnv ``SelfPlayPool``.

Unlike the legacy global trainer, this path never creates Python ``GameEnv``
objects in its hot loop. PolyEnv owns the live games, detached belief roots,
MCTS forest and persistent C++ workers; PyTorch receives only dense batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import (
    NATIVE_ACTION_CATEGORICAL_COLUMNS,
    NATIVE_ACTION_CONTINUOUS_COLUMNS,
    native_batch_to_tensors,
    stack_examples,
)
from ..data.token_schema import EVENT_SUMMARY_SIZE
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .global_mcts_train import _native_leaf_bias
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer, TrainingExample


def _native_globals(state: np.ndarray, map_size: int, map_type: str) -> np.ndarray:
    """CPU counterpart of ``native_batch_to_tensors`` global features."""
    state = np.asarray(state, dtype=np.int32)
    result = np.empty((state.shape[0], 9), dtype=np.float32)
    result[:, 0] = state[:, 0] / 100.0
    result[:, 1] = state[:, 1] / 16.0
    result[:, 2] = state[:, 4] / 50.0
    result[:, 3] = state[:, 7] / 20.0
    result[:, 4] = state[:, 5] / 32.0
    result[:, 5] = state[:, 6] / 16.0
    result[:, 6] = np.maximum(state[:, 8], 0) / 1024.0
    result[:, 7] = float(map_size) / 32.0
    result[:, 8] = 1.0 if map_type == "drylands" else 0.0
    return result


def _allocate_native_buffers(pool: object, method: str, device: torch.device, *args: int) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    """Allocate reusable native packets, pinned on CUDA for fast host transfers."""
    spec = getattr(pool, method)(*args)
    torch_types = {
        np.dtype(np.int32): torch.int32,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.uint64): torch.uint64,
        np.dtype(np.float32): torch.float32,
    }
    arrays: Dict[str, np.ndarray] = {}
    owners: Dict[str, torch.Tensor] = {}
    for name, field in spec.items():
        shape, dtype = tuple(field["shape"]), np.dtype(field["dtype"])
        if device.type == "cuda" and dtype in torch_types:
            owner = torch.empty(shape, dtype=torch_types[dtype], device="cpu", pin_memory=True)
            owners[name] = owner
            arrays[name] = owner.numpy()
        else:
            arrays[name] = np.empty(shape, dtype=dtype)
    return arrays, owners


def _root_examples(
    request: Dict[str, np.ndarray],
    root: Dict[str, np.ndarray],
    training: TrainingConfig,
    belief_targets: np.ndarray,
) -> List[TrainingExample]:
    """Create records from public MCTS roots plus training-only belief labels.

    ``belief_targets`` are requested *after* MCTS from a dedicated API. They
    never enter the belief completion, legal-action check, policy or value
    inference path, so they cannot leak hidden state into a played action.
    """
    maps = np.asarray(request["map_tokens"], dtype=np.int64)
    state = np.asarray(request["state"], dtype=np.int32)
    features = np.asarray(request["action_features"], dtype=np.int64)
    action_mask = np.asarray(request["action_mask"], dtype=bool)
    player_ids = np.asarray(request["to_play"], dtype=np.int64)
    policies = np.asarray(root["policy"], dtype=np.float32)
    histories = np.asarray(request.get("visible_action_history", np.empty((maps.shape[0], 0, 12), dtype=np.int32)))
    history_masks = np.asarray(request.get("visible_action_history_mask", np.empty((maps.shape[0], 0), dtype=bool)), dtype=bool)
    examples: List[TrainingExample] = []
    globals_ = _native_globals(state, training.map_size, training.map_type)
    for index in range(maps.shape[0]):
        count = int(action_mask[index].sum())
        if count <= 0:
            raise RuntimeError("SelfPlayPool returned a live root without legal actions")
        policy = policies[index, :count].copy()
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0.0:
            policy.fill(1.0 / count)
        else:
            policy /= total
        examples.append(TrainingExample(
            map_tokens=maps[index].copy(),
            global_features=globals_[index].copy(),
            event_features=np.zeros(EVENT_SUMMARY_SIZE, dtype=np.float32),
            # Index columns after slicing actions. NumPy advanced indexing in
            # one expression moves the feature axis to the front, producing
            # [fields, actions] and corrupting variable-length replay rows.
            action_categorical=features[index, :count][:, NATIVE_ACTION_CATEGORICAL_COLUMNS].copy(),
            action_continuous=features[index, :count][:, NATIVE_ACTION_CONTINUOUS_COLUMNS].astype(np.float32, copy=True),
            policy_target=policy,
            full_map_target=belief_targets[index].copy(),
            hidden_mask=maps[index, :, 0] == 0,
            player_id=int(player_ids[index]),
            action_history=histories[index].astype(np.float32, copy=True),
            action_history_mask=history_masks[index].copy(),
        ))
    return examples


def _sample_root_rows(
    root: Dict[str, np.ndarray],
    rng: np.random.Generator,
    temperature: float,
) -> np.ndarray:
    policies = np.asarray(root["policy"], dtype=np.float64)
    mask = np.asarray(root["action_mask"], dtype=bool)
    rows = np.empty(policies.shape[0], dtype=np.int64)
    for index in range(policies.shape[0]):
        count = int(mask[index].sum())
        if count <= 0:
            raise RuntimeError("SelfPlayPool root has no legal action")
        weights = policies[index, :count]
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            weights = np.full(count, 1.0 / count, dtype=np.float64)
        else:
            weights = weights / total
        rows[index] = int(weights.argmax()) if temperature <= 1e-5 else int(rng.choice(count, p=weights))
    return rows


@torch.inference_mode()
def _all_player_particles(
    request: Dict[str, np.ndarray],
    model: PolicyValueBeliefNet,
    device: torch.device,
    training: TrainingConfig,
    temperature: float,
) -> np.ndarray:
    """Sample one fog-safe hypothesis for every player of every live game."""
    maps = np.asarray(request["map_tokens"], dtype=np.int32)
    states = np.asarray(request["state"], dtype=np.int32)
    perspectives = np.asarray(request["belief_player"], dtype=np.int64)
    history = np.asarray(request["visible_action_history"], dtype=np.int32)
    history_mask = np.asarray(request["visible_action_history_mask"], dtype=bool)
    envs, players, tiles, fields = maps.shape
    flat_maps = maps.reshape(envs * players, tiles, fields)
    flat_states = states.reshape(envs * players, states.shape[-1])
    flat_history = history.reshape(envs * players, history.shape[-2], history.shape[-1])
    flat_history_mask = history_mask.reshape(envs * players, history_mask.shape[-1])
    # Per-player packets intentionally contain no action list. A masked dummy
    # action means the belief receives only that player's map/history here;
    # the actual actor's completion is overwritten below with its legal-action
    # conditioned inference from the ordinary current-player packet.
    count = flat_maps.shape[0]
    categorical = torch.full((count, 1, 12), -1, dtype=torch.long, device=device)
    continuous = torch.zeros((count, 1, 4), dtype=torch.float32, device=device)
    action_mask = torch.zeros((count, 1), dtype=torch.bool, device=device)
    output = model(
        torch.as_tensor(flat_maps, dtype=torch.long, device=device),
        torch.as_tensor(_native_globals(flat_states, training.map_size, training.map_type), dtype=torch.float32, device=device),
        categorical, continuous, action_mask,
        action_history=torch.as_tensor(flat_history, dtype=torch.float32, device=device),
        action_history_mask=torch.as_tensor(flat_history_mask, dtype=torch.bool, device=device),
    )
    completed = model.sample_completed_map_batch_from_belief_logits(
        torch.as_tensor(flat_maps, dtype=torch.long, device=device), output.belief_logits,
        temperature=temperature,
        perspective_players=torch.as_tensor(perspectives.reshape(-1), dtype=torch.long, device=device),
        known_city_counts=torch.as_tensor(flat_states[:, 6], dtype=torch.long, device=device),
    )
    return completed.numpy().astype(np.int32, copy=False).reshape(envs, players, tiles, fields)


@torch.inference_mode()
def _conservative_all_player_particles(
    request: Dict[str, np.ndarray],
    model: PolicyValueBeliefNet,
    device: torch.device,
) -> np.ndarray:
    """Return an action-neutral, syntactically valid fallback hypothesis.

    This path never reads a full map.  It is used only after a learned particle
    failed the engine's public legal-action consistency check, allowing a new
    model to keep collecting supervised belief data instead of resetting an
    entire vector batch indefinitely.
    """
    maps = np.asarray(request["map_tokens"], dtype=np.int32)
    states = np.asarray(request["state"], dtype=np.int32)
    perspectives = np.asarray(request["belief_player"], dtype=np.int64)
    envs, players, tiles, fields = maps.shape
    flat_maps = maps.reshape(envs * players, tiles, fields)
    completed = model.conservative_completed_map_batch(
        torch.as_tensor(flat_maps, dtype=torch.long, device=device), player_count=players,
        perspective_players=torch.as_tensor(perspectives.reshape(-1), dtype=torch.long, device=device),
        known_city_counts=torch.as_tensor(states.reshape(envs * players, states.shape[-1])[:, 6], dtype=torch.long, device=device),
    )
    return completed.numpy().astype(np.int32, copy=False).reshape(envs, players, tiles, fields)


def _belief_rejection_kind(error: ValueError) -> str:
    """Keep rejection diagnostics compact and stable in training summaries."""
    message = str(error).lower()
    if "legal action" in message:
        return "legal_actions"
    if "unit" in message:
        return "unit_state"
    if "city" in message:
        return "city_state"
    if "revealed" in message or "visible" in message:
        return "visible_rows"
    return "other"


def _format_belief_action(action: Dict[str, Any]) -> str:
    """Short, stable representation for a public legal-action difference."""
    name = str(action.get("type_fullname", "unknown"))
    action_id = int(action.get("action_id", -1))
    source = int(action.get("source_index", -1))
    target = int(action.get("target_index", -1))
    return f"{name}(id={action_id}, src={source}, dst={target})"


def _log_belief_reset_diagnostics(
    pool: object,
    state_ids: np.ndarray,
    completed_maps: np.ndarray,
    fallback_error: ValueError,
    reset_reasons: Dict[str, int],
) -> None:
    """Print why each slot is reset without exposing the true hidden map.

    The engine diagnostic returns the public real action set and actions from
    the model's submitted *candidate* world only.  Thus a line such as
    ``missing_from_belief=[move(...)]`` is actionable for debugging while
    remaining safe under fog-of-war.
    """
    def emit(message: str) -> None:
        # Do not rely on tqdm.write here: on Windows a rapidly updating native
        # progress bar can overwrite it before the user sees the reset cause.
        print(message, flush=True)

    if not hasattr(pool, "belief_acceptance_diagnostics"):
        reason = _belief_rejection_kind(fallback_error)
        reset_reasons[reason] = reset_reasons.get(reason, 0) + int(state_ids.size)
        emit(
            "belief reset: engine lacks belief_acceptance_diagnostics; "
            f"reason={reason}, error={fallback_error}"
        )
        return
    if completed_maps.ndim == 5 and not hasattr(pool, "ismcts_belief_diagnostics"):
        reason = "full_ismcts_diagnostics_unavailable"
        reset_reasons[reason] = reset_reasons.get(reason, 0) + int(state_ids.size)
        emit(
            "belief reset: installed PolyEnv lacks ismcts_belief_diagnostics; "
            f"submit_error={fallback_error}"
        )
        return
    try:
        if completed_maps.ndim == 5 and hasattr(pool, "ismcts_belief_diagnostics"):
            diagnostics = pool.ismcts_belief_diagnostics(
                np.ascontiguousarray(state_ids, dtype=np.uint64),
                np.ascontiguousarray(completed_maps, dtype=np.int32),
            )
        else:
            diagnostics = pool.belief_acceptance_diagnostics(
                np.ascontiguousarray(state_ids, dtype=np.uint64),
                np.ascontiguousarray(completed_maps, dtype=np.int32),
            )
    except Exception as error:  # diagnostics must never hide the original reset cause
        reason = "diagnostic_error"
        reset_reasons[reason] = reset_reasons.get(reason, 0) + int(state_ids.size)
        emit(f"belief reset: diagnostic failed ({error}); submit error={fallback_error}")
        return
    rejected_rows = 0
    for row in diagnostics:
        if bool(row.get("accepted", False)):
            continue
        rejected_rows += 1
        reason = str(row.get("reason", "unknown"))
        reset_reasons[reason] = reset_reasons.get(reason, 0) + 1
        missing = [_format_belief_action(action) for action in row.get("missing_actions", [])[:3]]
        extra = [_format_belief_action(action) for action in row.get("extra_actions", [])[:3]]
        details = (
            f"env={int(row.get('env_id', -1))} reason={reason} "
            f"real_actions={int(row.get('real_action_count', -1))} "
            f"belief_actions={int(row.get('belief_action_count', -1))}"
        )
        if "player" in row:
            details += f" player={int(row['player'])} particle={int(row.get('particle', -1))}"
        if missing:
            details += " missing_from_belief=[" + ", ".join(missing) + "]"
        if extra:
            details += " extra_in_belief=[" + ", ".join(extra) + "]"
        if "first_difference_index" in row:
            details += f" first_order_difference={int(row['first_difference_index'])}"
            if row.get("real_action_at_difference"):
                details += " real_at_difference=" + _format_belief_action(row["real_action_at_difference"])
            if row.get("belief_action_at_difference"):
                details += " belief_at_difference=" + _format_belief_action(row["belief_action_at_difference"])
        if row.get("error"):
            details += f" error={row['error']}"
        emit("belief reset: " + details)
    if rejected_rows == 0:
        # This should be unreachable: the full diagnostic mirrors the same
        # [env, player, particle] validation as submit_beliefs. Keep the raw
        # engine error visible instead of making an unsupported claim about a
        # non-active particle.
        reason = "full_diagnostic_empty"
        reset_reasons[reason] = reset_reasons.get(reason, 0) + int(state_ids.size)
        emit(
            "belief reset: full ISMCTS diagnostic returned no rejected particle; "
            f"submit_error={fallback_error}"
        )


def train_self_play_pool(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts: MCTSConfig,
    *,
    num_envs: int,
    steps: int,
    updates: int,
    num_threads: Optional[int] = None,
    max_pending_leaves: int = 4,
    visible_action_history: int = 64,
    belief_retries: int = 4,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Run fully native belief-MCTS rollouts and optimise in RAM.

    Per-slot validation/retry/reset keeps invalid detached belief worlds from
    stalling a whole vector. Full-map labels are read only after MCTS and are
    used solely for the belief loss; terminal rows use native terminal values
    and other rows a one-step TD target.
    """
    if num_envs < 1 or steps < 1 or updates < 0:
        raise ValueError("num_envs and steps must be positive; updates must not be negative")
    if mcts.simulations < 1:
        raise ValueError("MCTS simulations must be positive")
    if not 1 <= max_pending_leaves <= 8:
        raise ValueError("max_pending_leaves must be between one and eight")
    if belief_retries < 0:
        raise ValueError("belief_retries must not be negative")
    if not 0 <= visible_action_history <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    try:
        from PolyEnv import Bardur, Imperius, MapType, SelfPlayPool
    except ImportError as error:
        raise RuntimeError("Update PolyEnv to a version exposing SelfPlayPool.") from error
    required_pool_api = ("all_player_belief_requests", "belief_targets_numpy", "reset_slots")
    missing_pool_api = [name for name in required_pool_api if not hasattr(SelfPlayPool, name)]
    if missing_pool_api:
        raise RuntimeError(
            "PolyEnv SelfPlayPool is missing required belief-MCTS APIs: "
            + ", ".join(missing_pool_api)
            + ". Rebuild/update the local PolyEnv bindings."
        )
    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    native_threads = 0 if num_threads is None else max(1, int(num_threads))
    map_type = MapType.Drylands if training.map_type == "drylands" else MapType.Lakes
    pool = SelfPlayPool(
        num_envs=num_envs,
        seed=training.seed,
        map_size=training.map_size,
        tribes=(Bardur, Imperius),
        map_type=map_type,
        num_threads=native_threads,
        max_actions=mcts.native_max_actions,
        auto_reset=True,
        c_puct=mcts.c_puct,
        visible_action_history=visible_action_history,
        require_all_actions=True,
    )
    # The persistent native scheduler can safely use a handful of in-flight
    # leaves per tree; virtual loss prevents duplicate branch work.
    pool.configure_search(max_pending_leaves, 1.0, 0, 0)
    request, request_owners = _allocate_native_buffers(pool, "belief_batch_spec", device)
    root, root_owners = _allocate_native_buffers(pool, "root_policy_spec", device)
    # Full ISMCTS has a detached tree per live env, player belief and particle.
    leaf_capacity = num_envs * int(pool.player_count) * max(1, mcts.particles) * max_pending_leaves
    leaves, leaf_owners = _allocate_native_buffers(pool, "leaf_batch_spec", device, leaf_capacity)
    # Keep pinned tensor owners alive while C++ writes into their NumPy views.
    _ = (request_owners, root_owners, leaf_owners)
    pool.reset_into(request, seed=training.seed)
    replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed ^ 0x51F0A7)
    pending: List[Tuple[TrainingExample, int, float] | None] = [None] * num_envs
    terminal_episodes = 0
    truncated_roots = 0
    completed_targets = 0
    leaf_counts: List[int] = []
    belief_retries_used = 0
    belief_reset_rows = 0
    belief_fallback_rows = 0
    belief_rejections: Dict[str, int] = {}
    belief_reset_reasons: Dict[str, int] = {}
    model.eval()

    with tqdm(total=steps, desc=f"SelfPlayPool MCTS ({device})", unit="step", dynamic_ncols=True) as progress:
        for step in range(steps):
            root_tensors = native_batch_to_tensors(request, device, training.map_size, training.map_type)
            with torch.inference_mode():
                root_output = model(
                    root_tensors.map_tokens, root_tensors.global_features,
                    root_tensors.action_categorical, root_tensors.action_continuous,
                    root_tensors.action_mask, root_tensors.event_features,
                    action_history=root_tensors.action_history,
                    action_history_mask=root_tensors.action_history_mask,
                )
            state_ids = np.asarray(request["state_id"], dtype=np.uint64)
            actors = np.asarray(request["to_play"], dtype=np.int64)
            particle_count = max(1, int(mcts.particles))

            def actor_particle_batch() -> np.ndarray:
                return np.stack([
                    model.sample_completed_map_batch_from_belief_logits(
                        root_tensors.map_tokens, root_output.belief_logits,
                        # The active root must reproduce the already public
                        # legal-action set exactly. Greedy decoding is far more
                        # action-stable than sampling an early, uncertain
                        # belief distribution; uncertainty remains in the
                        # non-active player particles below.
                        temperature=0.0,
                        perspective_players=torch.as_tensor(actors, dtype=torch.long, device=device),
                        known_city_counts=torch.as_tensor(
                            np.asarray(request["state"], dtype=np.int32)[:, 6], dtype=torch.long, device=device
                        ),
                    ).numpy().astype(np.int32, copy=False)
                    for _ in range(particle_count)
                ], axis=1)

            # Full ISMCTS: every player supplies its own fog observation and
            # particle. The active actor is additionally conditioned on its
            # actual legal-action list, which is where blocked paths reveal
            # public information about hidden occupancy.
            all_players = pool.all_player_belief_requests()

            def learned_particles() -> np.ndarray:
                particles = np.stack([
                    _all_player_particles(all_players, model, device, training, max(mcts.temperature, 0.35))
                    for _ in range(particle_count)
                ], axis=2)
                # Re-sample this tensor on *every* retry. The active player's
                # completion alone is required to preserve the public legal
                # root action set, so retaining a rejected copy made retries
                # deterministic failures.
                particles[np.arange(num_envs), actors] = actor_particle_batch()
                return particles

            particles = learned_particles()
            submitted = False
            for attempt in range(belief_retries + 1):
                try:
                    pool.submit_beliefs(state_ids, np.ascontiguousarray(particles))
                    submitted = True
                    break
                except ValueError as error:
                    kind = _belief_rejection_kind(error)
                    belief_rejections[kind] = belief_rejections.get(kind, 0) + num_envs
                    if attempt == belief_retries:
                        break
                    particles = learned_particles()
                    belief_retries_used += num_envs
            if not submitted:
                # Bootstrap safely rather than throw away the whole batch. The
                # fallback is a low-information *hypothesis*, not a copy of
                # the engine's hidden map. It also gives the belief head more
                # labelled positions until learned structured particles begin
                # satisfying the action-equivalence test on their own.
                fallback = _conservative_all_player_particles(all_players, model, device)
                fallback = np.repeat(fallback[:, :, None], particle_count, axis=2)
                try:
                    pool.submit_beliefs(state_ids, np.ascontiguousarray(fallback))
                    submitted = True
                    belief_fallback_rows += num_envs
                except ValueError as error:
                    kind = _belief_rejection_kind(error)
                    belief_rejections[kind] = belief_rejections.get(kind, 0) + num_envs
                    _log_belief_reset_diagnostics(
                        pool,
                        state_ids,
                        fallback,
                        error,
                        belief_reset_reasons,
                    )
                    rows = np.arange(num_envs, dtype=np.int32)
                    pool.reset_slots(rows)
                    pool.belief_requests_into(request)
                    belief_reset_rows += num_envs
                    continue

            # One select call returns up to B * pending leaves. The native
            # virtual-loss scheduler reserves distinct paths in that batch.
            # The first unexpanded root yields only one leaf per tree; after
            # that, each call can reserve ``max_pending_leaves`` paths.
            waves = 1 + int(np.ceil(max(0, mcts.simulations - 1) / max_pending_leaves))
            for _ in range(waves):
                leaf_count = int(pool.select_leaves_into(leaves, max_leaves=leaf_capacity))
                leaf_counts.append(leaf_count)
                if leaf_count == 0:
                    break
                leaf_batch = {name: value[:leaf_count] for name, value in leaves.items()}
                leaf_tensors = native_batch_to_tensors(leaf_batch, device, training.map_size, training.map_type)
                with torch.inference_mode():
                    leaf_output = model(
                        leaf_tensors.map_tokens, leaf_tensors.global_features,
                        leaf_tensors.action_categorical, leaf_tensors.action_continuous,
                        leaf_tensors.action_mask, leaf_tensors.event_features,
                        action_history=leaf_tensors.action_history,
                        action_history_mask=leaf_tensors.action_history_mask,
                        include_belief=False,
                    )
                    logits = leaf_output.policy_logits + _native_leaf_bias(
                        leaf_tensors.action_categorical, leaf_tensors.action_continuous,
                        leaf_tensors.action_mask, mcts.teacher_prior_weight,
                    ).to(leaf_output.policy_logits.dtype)
                pool.expand_and_backup(
                    np.asarray(leaf_batch["leaf_id"], dtype=np.uint64),
                    logits.float().cpu().numpy(),
                    leaf_output.value.float().cpu().numpy(),
                )

            pool.root_policy_into(root, temperature=1.0)
            root_action_counts = np.asarray(root["action_mask"], dtype=np.uint8).sum(axis=1)
            empty_roots = np.flatnonzero(root_action_counts == 0)
            if empty_roots.size:
                request_action_counts = np.asarray(request["action_mask"], dtype=np.uint8).sum(axis=1)
                stats = pool.search_stats()
                selected = leaf_counts[-waves:]
                raise RuntimeError(
                    "SelfPlayPool returned an unexpanded root after MCTS. "
                    f"step={step}, empty_roots={empty_roots[:8].tolist()} "
                    f"(total={empty_roots.size}), request_action_count="
                    f"{request_action_counts[empty_roots[:8]].tolist()}, "
                    f"selected_leaves={selected}, root_visits="
                    f"{np.asarray(root['root_visit_count'])[empty_roots[:8]].tolist()}, "
                    f"search_stats={stats}. This indicates a SelfPlayPool/MCTS API mismatch "
                    "or a native engine issue, not an action-selection failure."
                )
            root_values = np.asarray(root["root_value"], dtype=np.float32)
            next_players = np.asarray(request["to_play"], dtype=np.int64)
            terminal = np.asarray(request["terminated"], dtype=bool)
            terminal_values = np.asarray(request["terminal_values"], dtype=np.float32)
            # Complete the prior action only after MCTS has evaluated its
            # successor state. Auto-reset terminal rows use the exact vector
            # terminal payoff rather than the next episode's root value.
            for index, old in enumerate(pending):
                if old is None:
                    continue
                example, actor, reward = old
                if terminal[index]:
                    example.value_target = float(terminal_values[index, actor])
                    terminal_episodes += 1
                else:
                    sign = 1.0 if int(next_players[index]) == actor else -1.0
                    example.value_target = float(np.clip(
                        training.shaping_target_weight * reward
                        + training.reward_discount * sign * root_values[index],
                        -1.0,
                        1.0,
                    ))
                replay.extend([example])
                completed_targets += 1

            belief_targets = np.asarray(pool.belief_targets_numpy(state_ids), dtype=np.int32)
            examples = _root_examples(request, root, training, belief_targets)
            action_rows = _sample_root_rows(root, rng, mcts.temperature)
            action_features = np.asarray(request["action_features"], dtype=np.int64)
            chosen = action_features[np.arange(num_envs), action_rows]
            # Visible, dense curriculum signal; the policy target itself stays
            # the MCTS visit distribution rather than this heuristic.
            rewards = (
                ((chosen[:, 0] == 8) & (chosen[:, 8] == 14)).astype(np.float32) * 1.0
                + ((chosen[:, 0] == 8) & (chosen[:, 8] == 11)).astype(np.float32) * 0.45
                + (chosen[:, 0] == 1).astype(np.float32) * (0.08 + np.clip(chosen[:, 15], 0, 10) * 0.012)
                - (chosen[:, 0] == 3).astype(np.float32) * 0.05
            )
            for index, example in enumerate(examples):
                example.shaped_reward = float(rewards[index])
                pending[index] = (example, int(next_players[index]), float(rewards[index]))
            actions = np.asarray(root["action_id"], dtype=np.int32)[np.arange(num_envs), action_rows]
            truncated_roots += int(np.asarray(request["action_truncated"], dtype=bool).sum())
            pool.step_into_checked(
                actions.astype(np.int32, copy=False),
                np.asarray(root["state_id"], dtype=np.uint64), request,
            )
            progress.update(1)
            if (step + 1) % 10 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    replay=len(replay), terminals=terminal_episodes,
                    fallbacks=belief_fallback_rows, resets=belief_reset_rows,
                )

    losses: List[Dict[str, float]] = []
    for _ in tqdm(range(updates), desc=f"SelfPlayPool optimising ({device})", unit="update", dynamic_ncols=True):
        if len(replay) < training.batch_size:
            break
        batch = replay.sample(training.batch_size, rng, training.priority_replay_fraction)
        losses.append(train_batch(model, optimiser, stack_examples(batch, device), training))

    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float] = {
        "training_seed": float(training.seed),
        "self_play_pool_envs": float(num_envs),
        "self_play_pool_steps": float(steps),
        "self_play_pool_env_steps": float(num_envs * steps),
        "self_play_pool_threads": float(pool.num_threads),
        "self_play_pool_pending_leaves": float(max_pending_leaves),
        "self_play_pool_visible_action_history": float(visible_action_history),
        "self_play_pool_terminals": float(terminal_episodes),
        "self_play_pool_completed_targets": float(completed_targets),
        "self_play_pool_belief_retries": float(belief_retries_used),
        "self_play_pool_belief_fallback_rows": float(belief_fallback_rows),
        "self_play_pool_belief_reset_rows": float(belief_reset_rows),
        "self_play_pool_truncated_roots": float(truncated_roots),
        "replay_examples": float(len(replay)),
    }
    summary.update({f"self_play_pool_belief_rejection_{kind}": float(count)
                    for kind, count in sorted(belief_rejections.items())})
    summary.update({f"self_play_pool_belief_reset_reason_{kind}": float(count)
                    for kind, count in sorted(belief_reset_reasons.items())})
    if losses:
        summary.update({key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]})
    return summary
