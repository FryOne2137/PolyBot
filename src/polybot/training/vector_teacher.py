"""High-throughput policy curriculum using PolyEnv's native VectorGameEnv."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..data.polyenv_features import vector_event_action_history
from ..data.token_schema import EVENT_SUMMARY_SIZE
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from .run_artifacts import TrainingRun


# VectorGameEnv action_features layout documented by PolyEnv.  These indexes
# adapt its dense [B, Amax, 17] tensor to this model's established 12+4 action
# encoder; no strings or per-environment Python packets are constructed.
_CATEGORICAL_COLUMNS = np.asarray((0, 1, 2, 10, 3, 4, 5, 6, 7, 8, 9, 14), dtype=np.int64)
_CONTINUOUS_COLUMNS = np.asarray((12, 13, 15, 16), dtype=np.int64)


@lru_cache(maxsize=8)
def _manhattan_distances(map_size: int) -> np.ndarray:
    """Pairwise board distances used by the visible-objective curriculum."""
    tiles = int(map_size) * int(map_size)
    indices = np.arange(tiles, dtype=np.int32)
    rows, columns = indices // map_size, indices % map_size
    return (
        np.abs(rows[:, None] - rows[None, :])
        + np.abs(columns[:, None] - columns[None, :])
    ).astype(np.float32)


def _visible_goal_progress(
    features: np.ndarray,
    map_tokens: np.ndarray,
    state: np.ndarray,
    map_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score legal moves toward the next visible strategic objective.

    In the opening, an available village has priority over an enemy unit. Once
    the player owns more than one city, visible enemy cities take priority.
    Enemy units remain a fallback only when there is no expansion or conquest
    objective.  The former union of all three made a nearby enemy unit pull
    early policies away from villages, which is exactly the local optimum the
    short-horizon curriculum must avoid.
    """
    features = np.asarray(features, dtype=np.int64)
    tokens = np.asarray(map_tokens, dtype=np.int64)
    state = np.asarray(state, dtype=np.int64)
    batch, actions = features.shape[:2]
    tiles = tokens.shape[1]
    if tiles != int(map_size) * int(map_size):
        return np.zeros((batch, actions), dtype=np.float32), np.zeros(
            (batch, actions), dtype=bool
        )
    visible = tokens[..., 0] == 1
    players = state[:, 1]
    village = visible & (tokens[..., 14] == 1)
    enemy_city = (
        visible
        & (tokens[..., 14] == 2)
        & (tokens[..., 16] >= 0)
        & (tokens[..., 16] != players[:, None])
    )
    enemy_unit = (
        visible
        & (tokens[..., 4] >= 0)
        & (tokens[..., 3] >= 0)
        & (tokens[..., 3] != players[:, None])
    )
    own_cities = np.maximum(state[:, 6], 0)
    has_village = village.any(axis=1)
    has_enemy_city = enemy_city.any(axis=1)
    has_enemy_unit = enemy_unit.any(axis=1)
    prefer_village = has_village & ((own_cities <= 1) | ~has_enemy_city)
    prefer_enemy_city = ~prefer_village & has_enemy_city
    prefer_enemy_unit = ~(prefer_village | prefer_enemy_city) & has_enemy_unit
    goals = (
        (prefer_village[:, None] & village)
        | (prefer_enemy_city[:, None] & enemy_city)
        | (prefer_enemy_unit[:, None] & enemy_unit)
    )
    distances = _manhattan_distances(int(map_size))
    large = float(tiles * 2)
    distance_to_goal = np.where(
        goals[:, None, :], distances[None, :, :], large
    ).min(axis=2)
    no_goal = ~goals.any(axis=1)
    distance_to_goal[no_goal] = 0.0

    sources = np.clip(features[..., 1], 0, tiles - 1)
    targets = np.clip(features[..., 2], 0, tiles - 1)
    rows = np.arange(batch)[:, None]
    is_move = features[..., 0] == 0
    progress = distance_to_goal[rows, sources] - distance_to_goal[rows, targets]
    enters_village = is_move & village[rows, targets]
    enters_enemy_city = is_move & enemy_city[rows, targets]
    explores_hidden = is_move & ~visible[rows, targets] & no_goal[:, None]
    score = np.zeros((batch, actions), dtype=np.float32)
    score += np.where(is_move, np.clip(progress, -2.0, 3.0) * 0.65, 0.0)
    score += np.where(enters_village, 4.0, 0.0)
    score += np.where(enters_enemy_city, 4.5, 0.0)
    score += np.where(explores_hidden, 0.35, 0.0)
    return score, enters_village


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


def _teacher_targets(
    features: np.ndarray,
    mask: np.ndarray,
    map_tokens: np.ndarray | None = None,
    state: np.ndarray | None = None,
    map_size: int | None = None,
) -> np.ndarray:
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
    if map_tokens is not None and state is not None and map_size is not None:
        goal_score, _ = _visible_goal_progress(
            features, map_tokens, state, int(map_size)
        )
        score += goal_score
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
    action_history, action_history_mask = vector_event_action_history(batch)
    return {
        "map_tokens": torch.as_tensor(batch["map_tokens"], dtype=torch.long, device=device),
        "global_features": torch.as_tensor(_global_features(batch["state"], training.map_size, training.map_type), device=device),
        "event_features": torch.as_tensor(_event_summary(batch), dtype=torch.float32, device=device),
        "action_history": torch.as_tensor(
            action_history, dtype=torch.float32, device=device
        ),
        "action_history_mask": torch.as_tensor(
            action_history_mask, dtype=torch.bool, device=device
        ),
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
    max_actions: int = 1024,
    visible_event_history: int = 0,
    updates_per_step: int = 1,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
    run: TrainingRun | None = None,
    anchor_model: Optional[PolicyValueBeliefNet] = None,
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    learner_action_probability: float = 0.50,
) -> Dict[str, float | str]:
    """Train policy on native C++ vector environments and record scalar plots.

    VectorGameEnv intentionally exposes player-view tokens only.  Therefore
    this stage trains the policy curriculum, not the belief decoder; belief
    remains trained exclusively by scalar full-map-supervised warm-up.
    """
    if num_envs < 1 or steps < 1 or max_actions < 1 or updates_per_step < 1:
        raise ValueError("num_envs, steps, max_actions and updates_per_step must be positive")
    if training.map_type != "lakes":
        raise ValueError("the documented VectorGameEnv constructor currently exposes the default lakes ruleset only")
    for name, value in {
        "policy_anchor_weight": policy_anchor_weight,
        "value_anchor_weight": value_anchor_weight,
        "learner_action_probability": learner_action_probability,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("Update PolyEnv to a version exposing VectorGameEnv.") from error

    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "vector_teacher")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "num_envs": num_envs,
            "steps": steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "visible_event_history": visible_event_history,
            "updates_per_step": updates_per_step,
            "model_out": model_out,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "learner_action_probability": learner_action_probability,
        },
    )
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    if anchor_model is model:
        raise ValueError("anchor_model must be a frozen copy")
    if anchor_model is not None:
        anchor_model = anchor_model.to(device).eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_amp)
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
    policy_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    policy_anchor_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    value_anchor_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = 0
    terminal_count = 0
    invalid_actions = 0
    learner_actions = 0
    village_moves = 0
    rng = np.random.default_rng(training.seed ^ 0xDA66E2)

    with tqdm(total=steps, desc=f"vector teacher ({device})", unit="step", dynamic_ncols=True) as progress:
        for step in range(steps):
            tensors = _vector_tensors(batch, device, training)
            features = np.asarray(batch["action_features"], dtype=np.int64)
            valid = np.asarray(batch["action_mask"], dtype=bool)
            target = _teacher_targets(
                features,
                valid,
                np.asarray(batch["map_tokens"], dtype=np.int64),
                np.asarray(batch["state"], dtype=np.int64),
                training.map_size,
            )
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)

            model.train()
            step_losses = []
            step_policy_losses = []
            step_policy_anchor_losses = []
            step_value_anchor_losses = []
            for _ in range(updates_per_step):
                optimiser.zero_grad(set_to_none=True)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda_amp else nullcontext()
                with autocast:
                    output = model(
                        tensors["map_tokens"], tensors["global_features"], tensors["action_categorical"],
                        tensors["action_continuous"], tensors["action_mask"], tensors["event_features"],
                        action_history=tensors["action_history"],
                        action_history_mask=tensors["action_history_mask"],
                        include_belief=False,
                    )
                    log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
                    policy_loss = -(
                        target_tensor * torch.where(tensors["action_mask"], log_probabilities, 0.0)
                    ).sum(dim=1).mean()
                    anchor_policy_loss = policy_loss.new_zeros(())
                    anchor_value_loss = policy_loss.new_zeros(())
                    if anchor_model is not None:
                        with torch.no_grad():
                            anchor_output = anchor_model(
                                tensors["map_tokens"], tensors["global_features"],
                                tensors["action_categorical"], tensors["action_continuous"],
                                tensors["action_mask"], tensors["event_features"],
                                action_history=tensors["action_history"],
                                action_history_mask=tensors["action_history_mask"],
                                include_belief=False,
                            )
                            anchor_policy = torch.softmax(
                                anchor_output.policy_logits, dim=-1
                            )
                        anchor_policy_loss = -(
                            anchor_policy
                            * torch.where(
                                tensors["action_mask"], log_probabilities, 0.0
                            )
                        ).sum(dim=1).mean()
                        anchor_value_loss = F.smooth_l1_loss(
                            output.value, anchor_output.value
                        )
                    loss = (
                        policy_loss
                        + policy_anchor_weight * anchor_policy_loss
                        + value_anchor_weight * anchor_value_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                loss_sum += loss.detach().float()
                policy_loss_sum += policy_loss.detach().float()
                policy_anchor_loss_sum += anchor_policy_loss.detach().float()
                value_anchor_loss_sum += anchor_value_loss.detach().float()
                loss_count += 1
                step_losses.append(float(loss.detach().cpu()))
                step_policy_losses.append(float(policy_loss.detach().cpu()))
                step_policy_anchor_losses.append(
                    float(anchor_policy_loss.detach().cpu())
                )
                step_value_anchor_losses.append(
                    float(anchor_value_loss.detach().cpu())
                )

            teacher_rows = target.argmax(axis=1)
            learner_rows = output.policy_logits.detach().float().argmax(dim=1).cpu().numpy()
            use_learner = rng.random(num_envs) < learner_action_probability
            rows = np.where(use_learner, learner_rows, teacher_rows)
            learner_actions += int(use_learner.sum())
            _, enters_village = _visible_goal_progress(
                features,
                np.asarray(batch["map_tokens"], dtype=np.int64),
                np.asarray(batch["state"], dtype=np.int64),
                training.map_size,
            )
            village_moves += int(enters_village[np.arange(num_envs), rows].sum())
            action_ids = np.asarray(batch["action_id"], dtype=np.int32)[np.arange(num_envs), rows]
            batch = env.step(action_ids)
            terminal_count += int(np.asarray(batch["terminated"], dtype=np.uint8).sum())
            invalid_actions += int((np.asarray(batch["action_valid"], dtype=np.uint8) == 0).sum())
            run.log(
                step=step + 1,
                update=loss_count,
                environment_step=(step + 1) * num_envs,
                loss=float(np.mean(step_losses)),
                policy_loss=float(np.mean(step_policy_losses)),
                policy_anchor_loss=float(np.mean(step_policy_anchor_losses)),
                value_anchor_loss=float(np.mean(step_value_anchor_losses)),
                terminal_episodes=terminal_count,
                invalid_actions=invalid_actions,
                learner_actions=learner_actions,
                village_moves=village_moves,
            )
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
    summary: Dict[str, float | str] = {
        "vector_envs": float(num_envs),
        "vector_steps": float(steps),
        "environment_steps": float(num_envs * steps),
        "optimisation_updates": float(loss_count),
        "loss": float((loss_sum / max(1, loss_count)).detach().cpu()),
        "policy_loss": float(
            (policy_loss_sum / max(1, loss_count)).detach().cpu()
        ),
        "policy_anchor_loss": float(
            (policy_anchor_loss_sum / max(1, loss_count)).detach().cpu()
        ),
        "value_anchor_loss": float(
            (value_anchor_loss_sum / max(1, loss_count)).detach().cpu()
        ),
        "terminal_episodes": float(terminal_count),
        "invalid_actions": float(invalid_actions),
        "learner_actions": float(learner_actions),
        "village_moves": float(village_moves),
        "policy_anchor_weight": float(
            policy_anchor_weight if anchor_model is not None else 0.0
        ),
        "value_anchor_weight": float(
            value_anchor_weight if anchor_model is not None else 0.0
        ),
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
