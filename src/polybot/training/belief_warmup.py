"""Supervised warm-up for map belief without saving generated games."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..data.polyenv_features import (
    action_features,
    event_summary,
    global_features_from_obs,
    stack_examples,
    visible_events_or_empty,
)
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .optimisation import belief_loss
from .replay_buffer import TrainingExample


_BELIEF_ONLY_PREFIXES = (
    "belief_action_",
    "belief_settlement_context",
    "belief_unit_context",
    "belief_heads",
)


def _configure_belief_only_training(model: PolicyValueBeliefNet) -> Dict[str, bool]:
    """Freeze every parameter that can change policy/value behaviour."""
    original = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(_BELIEF_ONLY_PREFIXES))
    return original


def _restore_trainable_parameters(model: PolicyValueBeliefNet, original: Dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original[name])


def _new_environment(seed: int, config: TrainingConfig) -> object:
    try:
        from PolyEnv import Bardur, GameEnv, Imperius
    except ImportError as error:
        raise RuntimeError("PolyEnv must be installed before training") from error
    return GameEnv(
        seed=seed,
        map_size=config.map_size,
        map_type=config.map_type,
        players=(Bardur, Imperius),
    )


def _belief_example(env: object) -> TrainingExample:
    """Build an in-memory supervised example; full_map is never model input."""
    packet = env.model_request_numpy()
    map_tokens = np.asarray(packet["map_tokens"], dtype=np.int64)
    categories, continuous = action_features(packet["actions"])
    action_count = categories.shape[0]
    return TrainingExample(
        map_tokens=map_tokens.copy(),
        global_features=global_features_from_obs(packet["obs"]),
        event_features=event_summary(visible_events_or_empty(env)),
        action_categorical=categories,
        action_continuous=continuous,
        policy_target=np.full(action_count, 1.0 / action_count, dtype=np.float32),
        full_map_target=np.asarray(env.full_map_numpy(), dtype=np.int64).copy(),
        hidden_mask=map_tokens[:, 0] == 0,
        player_id=int(packet["obs"].get("current_player", env.current_player())),
    )


def _belief_update(
    model: PolicyValueBeliefNet,
    optimiser: torch.optim.Optimizer,
    examples: List[TrainingExample],
    device: torch.device,
) -> float:
    batch = stack_examples(examples, device)
    model.train()
    output = model(
        batch["map_tokens"],
        batch["global_features"],
        batch["action_categorical"],
        batch["action_continuous"],
        batch["action_mask"],
        batch["event_features"],
        action_history=batch["action_history"],
        action_history_mask=batch["action_history_mask"],
    )
    loss = belief_loss(
        output.belief_logits,
        batch["full_map_target"],
        batch["hidden_mask"],
    )
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    return float(loss.detach().cpu())


def warmup_belief(
    training: TrainingConfig,
    model_config: ModelConfig,
    positions: int,
    batch_size: int,
    min_hidden_tiles: int = 24,
    max_moves_per_world: int = 96,
    model_out: Optional[Path] = None,
    model: PolicyValueBeliefNet | None = None,
    preserve_policy_value: bool = False,
) -> Dict[str, float]:
    """Learn belief on naturally foggy, action-diverse random positions.

    Late random games reveal almost every tile and make map completion a
    trivial all-visible task.  Limiting their horizon and requiring a minimum
    hidden-tile count concentrates supervision on the exact distribution used
    by fog-of-war MCTS, without artificially hiding player memory.
    """
    if min_hidden_tiles < 0:
        raise ValueError("min_hidden_tiles must be non-negative")
    if max_moves_per_world < 1:
        raise ValueError("max_moves_per_world must be positive")
    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    original_trainable = (
        _configure_belief_only_training(model) if preserve_policy_value else None
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("belief warm-up has no trainable parameters")
    optimiser = torch.optim.AdamW(
        trainable_parameters,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    rng = np.random.default_rng(training.seed)
    pending: List[TrainingExample] = []
    losses: List[float] = []
    env = _new_environment(training.seed, training)
    environment_index = 0
    moves_in_world = 0
    accepted_hidden_tiles: List[int] = []
    progress_description = f"belief warm-up ({device})"
    try:
        with tqdm(
            total=positions,
            desc=progress_description,
            unit="position",
            dynamic_ncols=True,
        ) as progress:
            position = 0
            while position < positions:
                example = _belief_example(env)
                hidden_tiles = int(example.hidden_mask.sum())
                # Do not let nearly solved boards dominate supervised completion.
                # This is a natural curriculum: no previously seen tile is hidden
                # artificially, so the model still receives a valid player view.
                if hidden_tiles < min_hidden_tiles or moves_in_world >= max_moves_per_world:
                    environment_index += 1
                    env = _new_environment(training.seed + environment_index, training)
                    moves_in_world = 0
                    continue

                pending.append(example)
                accepted_hidden_tiles.append(hidden_tiles)
                legal_ids = np.asarray(env.model_request_numpy()["actions"]["action_id"], dtype=np.int64)
                env.step_fast(int(rng.choice(legal_ids)))
                moves_in_world += 1

                if len(pending) == batch_size:
                    latest_loss = _belief_update(model, optimiser, pending, device)
                    losses.append(latest_loss)
                    pending.clear()
                    progress.set_postfix(
                        belief_loss=f"{latest_loss:.4f}",
                        hidden_tiles=hidden_tiles,
                    )
                progress.update(1)
                if env.is_done():
                    environment_index += 1
                    env = _new_environment(training.seed + environment_index, training)
                    moves_in_world = 0
                position += 1

        if pending:
            losses.append(_belief_update(model, optimiser, pending, device))
    finally:
        if original_trainable is not None:
            _restore_trainable_parameters(model, original_trainable)
    if model_out is not None:
        save_model(model_out, model, model_config)
    return {
        "positions": float(positions),
        "belief_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_hidden_tiles": float(np.mean(accepted_hidden_tiles)) if accepted_hidden_tiles else 0.0,
        "min_hidden_tiles": float(min_hidden_tiles),
        "preserved_policy_value": float(preserve_policy_value),
    }
