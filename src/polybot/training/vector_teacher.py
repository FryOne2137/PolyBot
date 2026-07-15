"""High-throughput policy curriculum using PolyEnv's native VectorGameEnv."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..data.token_schema import EVENT_SUMMARY_SIZE
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..models.policy_value_belief import PolicyValueBeliefNet


# VectorGameEnv action_features layout documented by PolyEnv.  These indexes
# adapt its dense [B, Amax, 17] tensor to this model's established 12+4 action
# encoder; no strings or per-environment Python packets are constructed.
_CATEGORICAL_COLUMNS = np.asarray((0, 1, 2, 10, 3, 4, 5, 6, 7, 8, 9, 14), dtype=np.int64)
_CONTINUOUS_COLUMNS = np.asarray((12, 13, 15, 16), dtype=np.int64)


def _global_features(state: np.ndarray, map_size: int, map_type: str) -> np.ndarray:
    """Adapt VectorGameEnv's documented compact state to the policy input."""
    result = np.empty((state.shape[0], 9), dtype=np.float32)
    result[:, 0] = state[:, 0] / 100.0       # turn
    result[:, 1] = state[:, 1] / 16.0        # current player
    result[:, 2] = state[:, 4] / 50.0        # stars
    result[:, 3] = state[:, 7] / 20.0        # income
    result[:, 4] = state[:, 5] / 32.0        # own units
    result[:, 5] = state[:, 6] / 16.0        # own cities
    result[:, 6] = np.maximum(state[:, 8], 0) / 1024.0  # pending city id
    result[:, 7] = float(map_size) / 32.0
    result[:, 8] = 1.0 if map_type == "drylands" else 0.0
    return result


def _event_summary(batch: Dict[str, np.ndarray]) -> np.ndarray:
    """Compress native visible-event windows to the current model's 12 fields."""
    events = np.asarray(batch["visible_event_features"], dtype=np.int32)
    mask = np.asarray(batch["visible_event_mask"], dtype=bool)
    result = np.zeros((events.shape[0], EVENT_SUMMARY_SIZE), dtype=np.float32)
    if events.shape[1] == 0:
        return result
    result[:, 0] = np.minimum(mask.sum(axis=1), 128) / 128.0
    types = events[:, :, 2]
    for event_type in range(8):
        result[:, 1 + event_type] = ((types == event_type) & mask).sum(axis=1) / 32.0
    result[:, 9] = np.clip(np.where(mask, np.maximum(events[:, :, 6], 0), 0).sum(axis=1), 0, 200) / 200.0
    result[:, 10] = ((events[:, :, 24] > 0) & mask).sum(axis=1) / 16.0
    result[:, 11] = ((events[:, :, 25] > 0) & mask).sum(axis=1) / 16.0
    return result


def _teacher_targets(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vectorised legal-action teacher using stable PolyEnv numeric IDs.

    It favours visible city captures/founding/attacks and avoids end-turn when
    productive legal actions exist.  Inputs are solely the current vector
    player observation and legal action rows.
    """
    action_type = features[..., 0]
    tile_action = features[..., 8]
    affordable = features[..., 14]
    damage = features[..., 15]
    score = np.zeros(mask.shape, dtype=np.float32)
    score += np.where((action_type == 8) & (tile_action == 14), 5.0, 0.0)  # capture city
    score += np.where((action_type == 8) & (tile_action == 11), 3.0, 0.0)  # found city
    score += np.where(action_type == 1, 0.8 + np.clip(damage, 0, 10) * 0.12, 0.0)
    score += np.where((action_type == 8) & np.isin(tile_action, (10, 12)), 0.8, 0.0)
    score += np.where(np.isin(action_type, (5, 6)), 0.45, 0.0)
    score += np.where(np.isin(action_type, (4, 7)), 0.25, 0.0)
    score -= np.where(action_type == 3, 2.0, 0.0)  # end turn
    score -= np.where(affordable == 0, 1.0, 0.0)
    score = np.where(mask, score, -np.inf)
    score -= np.max(score, axis=1, keepdims=True)
    weights = np.exp(np.where(mask, score, -np.inf)).astype(np.float32)
    weights *= mask
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)


def _vector_tensors(
    batch: Dict[str, np.ndarray],
    device: torch.device,
    training: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    features = np.asarray(batch["action_features"], dtype=np.int64)
    mask = np.asarray(batch["action_mask"], dtype=bool)
    return {
        "map_tokens": torch.as_tensor(batch["map_tokens"], dtype=torch.long, device=device),
        "global_features": torch.as_tensor(_global_features(batch["state"], training.map_size, training.map_type), device=device),
        "event_features": torch.as_tensor(_event_summary(batch), dtype=torch.float32, device=device),
        "action_categorical": torch.as_tensor(features[..., _CATEGORICAL_COLUMNS], dtype=torch.long, device=device),
        "action_continuous": torch.as_tensor(features[..., _CONTINUOUS_COLUMNS], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
    }


def train_vector_teacher(
    training: TrainingConfig,
    model_config: ModelConfig,
    num_envs: int,
    steps: int,
    num_threads: Optional[int] = None,
    max_actions: int = 512,
    visible_event_history: int = 0,
    updates_per_step: int = 1,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Train policy on native C++ vector environments; save only model weights.

    VectorGameEnv intentionally exposes player-view tokens only.  Therefore
    this stage trains the policy curriculum, not the belief decoder; belief
    remains trained exclusively by scalar full-map-supervised warm-up.
    """
    if num_envs < 1 or steps < 1 or max_actions < 1 or updates_per_step < 1:
        raise ValueError("num_envs, steps, max_actions and updates_per_step must be positive")
    if training.map_type != "lakes":
        raise ValueError("the documented VectorGameEnv constructor currently exposes the default lakes ruleset only")
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("Update PolyEnv to a version exposing VectorGameEnv.") from error

    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    cuda_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=cuda_amp)
    env = VectorGameEnv(
        num_envs=num_envs,
        seed=training.seed,
        map_size=training.map_size,
        players=(Bardur, Imperius),
        num_threads=num_threads,
        max_actions=max_actions,
        auto_reset=True,
        visible_event_history=visible_event_history,
    )
    batch = env.reset(seed=training.seed)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = 0
    terminal_count = 0
    invalid_actions = 0

    with tqdm(total=steps, desc=f"vector teacher ({device})", unit="step", dynamic_ncols=True) as progress:
        for step in range(steps):
            tensors = _vector_tensors(batch, device, training)
            features = np.asarray(batch["action_features"], dtype=np.int64)
            valid = np.asarray(batch["action_mask"], dtype=bool)
            target = _teacher_targets(features, valid)
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)

            model.train()
            for _ in range(updates_per_step):
                optimiser.zero_grad(set_to_none=True)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda_amp else nullcontext()
                with autocast:
                    output = model(
                        tensors["map_tokens"], tensors["global_features"], tensors["action_categorical"],
                        tensors["action_continuous"], tensors["action_mask"], tensors["event_features"],
                    )
                    log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
                    policy_loss = -(
                        target_tensor * torch.where(tensors["action_mask"], log_probabilities, 0.0)
                    ).sum(dim=1).mean()
                scaler.scale(policy_loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                loss_sum += policy_loss.detach().float()
                loss_count += 1

            rows = target.argmax(axis=1)
            action_ids = np.asarray(batch["action_id"], dtype=np.int32)[np.arange(num_envs), rows]
            batch = env.step(action_ids)
            terminal_count += int(np.asarray(batch["terminated"], dtype=np.uint8).sum())
            invalid_actions += int((np.asarray(batch["action_valid"], dtype=np.uint8) == 0).sum())
            progress.update(1)
            # Avoid a CUDA->CPU synchronisation for a cosmetic metric every
            # environment step; it creates visible GPU idle gaps on fast cards.
            if (step + 1) % 10 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    policy_loss=f"{float(policy_loss.detach().cpu()):.4f}",
                    terminals=terminal_count,
                )

    if model_out is not None:
        save_model(model_out, model, model_config)
    return {
        "vector_envs": float(num_envs),
        "vector_steps": float(steps),
        "environment_steps": float(num_envs * steps),
        "optimisation_updates": float(loss_count),
        "policy_loss": float((loss_sum / max(1, loss_count)).detach().cpu()),
        "terminal_episodes": float(terminal_count),
        "invalid_actions": float(invalid_actions),
    }
