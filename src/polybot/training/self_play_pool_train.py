"""High-throughput belief-MCTS self-play via PolyEnv ``SelfPlayPool``.

Unlike the legacy global trainer, this path never creates Python ``GameEnv``
objects in its hot loop. PolyEnv owns the live games, detached belief roots,
MCTS forest and persistent C++ workers; PyTorch receives only dense batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
) -> List[TrainingExample]:
    """Create policy/value records from public root requests only.

    ``SelfPlayPool`` never exposes an authoritative full map. Self-play rows
    therefore deliberately carry an all-false belief mask: policy/value learn
    from them, while any optional teacher/belief stage supplies supervised fog
    labels. This keeps the pool's no-leak contract intact.
    """
    maps = np.asarray(request["map_tokens"], dtype=np.int64)
    state = np.asarray(request["state"], dtype=np.int32)
    features = np.asarray(request["action_features"], dtype=np.int64)
    action_mask = np.asarray(request["action_mask"], dtype=bool)
    player_ids = np.asarray(request["to_play"], dtype=np.int64)
    policies = np.asarray(root["policy"], dtype=np.float32)
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
            action_categorical=features[index, :count, NATIVE_ACTION_CATEGORICAL_COLUMNS].copy(),
            action_continuous=features[index, :count, NATIVE_ACTION_CONTINUOUS_COLUMNS].astype(np.float32, copy=True),
            policy_target=policy,
            full_map_target=maps[index].copy(),
            hidden_mask=np.zeros(maps.shape[1], dtype=bool),
            player_id=int(player_ids[index]),
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
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Run fixed-horizon, fully native belief-MCTS rollouts and optimise in RAM.

    A vector horizon avoids imposing a Python-side game timeout on individual
    native slots. Terminal transitions are still labelled exactly from
    ``terminal_values``; non-terminal rows use the next root value as a
    one-step TD target. The bounded replay is process memory only.
    """
    if num_envs < 1 or steps < 1 or updates < 0:
        raise ValueError("num_envs and steps must be positive; updates must not be negative")
    if mcts.simulations < 1:
        raise ValueError("MCTS simulations must be positive")
    if not 1 <= max_pending_leaves <= 8:
        raise ValueError("max_pending_leaves must be between one and eight")
    try:
        from PolyEnv import Bardur, Imperius, MapType, SelfPlayPool
    except ImportError as error:
        raise RuntimeError("Update PolyEnv to a version exposing SelfPlayPool.") from error

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
    )
    # The persistent native scheduler can safely use a handful of in-flight
    # leaves per tree; virtual loss prevents duplicate branch work.
    pool.configure_search(max_pending_leaves, 1.0, 0, 0)
    request, request_owners = _allocate_native_buffers(pool, "belief_batch_spec", device)
    root, root_owners = _allocate_native_buffers(pool, "root_policy_spec", device)
    leaf_capacity = num_envs * max_pending_leaves
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
    model.eval()

    with tqdm(total=steps, desc=f"SelfPlayPool MCTS ({device})", unit="step", dynamic_ncols=True) as progress:
        for step in range(steps):
            root_tensors = native_batch_to_tensors(request, device, training.map_size, training.map_type)
            with torch.inference_mode():
                root_output = model(
                    root_tensors.map_tokens, root_tensors.global_features,
                    root_tensors.action_categorical, root_tensors.action_continuous,
                    root_tensors.action_mask, root_tensors.event_features,
                )
                completed = model.sample_completed_map_batch_from_belief_logits(
                    root_tensors.map_tokens, root_output.belief_logits,
                    temperature=max(mcts.temperature, 0.35),
                )
            pool.submit_beliefs(
                np.asarray(request["state_id"], dtype=np.uint64),
                completed.cpu().numpy().astype(np.int32, copy=False),
            )

            # One select call returns up to B * pending leaves. The native
            # virtual-loss scheduler reserves distinct paths in that batch.
            # The first unexpanded root yields only one leaf per tree; after
            # that, each call can reserve ``max_pending_leaves`` paths.
            waves = 1 + int(np.ceil(max(0, mcts.simulations - 1) / max_pending_leaves))
            for _ in range(waves):
                leaf_count = int(pool.select_leaves_into(leaves, max_leaves=leaf_capacity))
                if leaf_count == 0:
                    break
                leaf_batch = {name: value[:leaf_count] for name, value in leaves.items()}
                leaf_tensors = native_batch_to_tensors(leaf_batch, device, training.map_size, training.map_type)
                with torch.inference_mode():
                    leaf_output = model(
                        leaf_tensors.map_tokens, leaf_tensors.global_features,
                        leaf_tensors.action_categorical, leaf_tensors.action_continuous,
                        leaf_tensors.action_mask, leaf_tensors.event_features,
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

            examples = _root_examples(request, root, training)
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
        "self_play_pool_envs": float(num_envs),
        "self_play_pool_steps": float(steps),
        "self_play_pool_env_steps": float(num_envs * steps),
        "self_play_pool_threads": float(pool.num_threads),
        "self_play_pool_pending_leaves": float(max_pending_leaves),
        "self_play_pool_terminals": float(terminal_episodes),
        "self_play_pool_completed_targets": float(completed_targets),
        "self_play_pool_truncated_roots": float(truncated_roots),
        "replay_examples": float(len(replay)),
    }
    if losses:
        summary.update({key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]})
    return summary
