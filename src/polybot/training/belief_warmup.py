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
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.heuristic_policy import choose_teacher_action
from .optimisation import belief_loss
from .replay_buffer import TrainingExample
from .run_artifacts import TrainingRun


_BELIEF_ONLY_PREFIXES = ("belief_",)


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
    training: TrainingConfig,
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
        batch["action_categorical"],
        batch["action_mask"],
        training.legal_action_belief_weight,
        presence_focal_gamma=training.belief_presence_focal_gamma,
    )
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    return float(loss.detach().cpu())


@torch.inference_mode()
def _belief_rollout_action(
    env: object,
    example: TrainingExample,
    model: PolicyValueBeliefNet | None,
    device: torch.device,
    rng: np.random.Generator,
    *,
    teacher_action_probability: float,
    policy_action_probability: float,
) -> tuple[int, str]:
    """Choose a legal rollout action without exposing hidden state to policy.

    Belief labels remain ``full_map_numpy()`` only.  The action source changes
    merely the *distribution of naturally encountered fog states*: teacher
    reaches useful public objectives, the frozen/current policy visits its own
    states and random play preserves coverage of unusual legal branches.
    """
    packet = env.model_request_numpy()
    action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
    if not action_ids.size:
        raise RuntimeError("belief warm-up environment returned no legal actions")
    draw = float(rng.random())
    if draw < teacher_action_probability:
        row = choose_teacher_action(packet, rng)
        return int(action_ids[row]), "teacher"
    if draw < teacher_action_probability + policy_action_probability:
        if model is None:
            raise RuntimeError("policy belief rollout requires a model")
        batch = stack_examples([example], device)
        model.eval()
        output = model(
            batch["map_tokens"],
            batch["global_features"],
            batch["action_categorical"],
            batch["action_continuous"],
            batch["action_mask"],
            batch["event_features"],
            action_history=batch["action_history"],
            action_history_mask=batch["action_history_mask"],
            include_belief=False,
        )
        probabilities = torch.softmax(output.policy_logits[0].float(), dim=-1)
        row = int(torch.multinomial(probabilities, 1).item())
        return int(action_ids[row]), "policy"
    return int(rng.choice(action_ids)), "random"


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
    teacher_action_probability: float = 0.0,
    policy_action_probability: float = 0.0,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Learn belief on naturally foggy states from mixed public rollouts.

    Late random games reveal almost every tile and make map completion a
    trivial all-visible task.  Limiting their horizon and requiring a minimum
    hidden-tile count concentrates supervision on the exact distribution used
    by fog-of-war MCTS, without artificially hiding player memory.
    """
    if min_hidden_tiles < 0:
        raise ValueError("min_hidden_tiles must be non-negative")
    if max_moves_per_world < 1:
        raise ValueError("max_moves_per_world must be positive")
    if not 0.0 <= teacher_action_probability <= 1.0:
        raise ValueError("teacher_action_probability must be between zero and one")
    if not 0.0 <= policy_action_probability <= 1.0:
        raise ValueError("policy_action_probability must be between zero and one")
    if teacher_action_probability + policy_action_probability > 1.0:
        raise ValueError("belief rollout probabilities must sum to at most one")
    if policy_action_probability and model is None:
        raise ValueError("policy_action_probability requires model")
    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "belief")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "positions": positions,
            "batch_size": batch_size,
            "min_hidden_tiles": min_hidden_tiles,
            "max_moves_per_world": max_moves_per_world,
            "preserve_policy_value": preserve_policy_value,
            "teacher_action_probability": teacher_action_probability,
            "policy_action_probability": policy_action_probability,
            "model_out": model_out,
        },
    )
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
    rollout_sources = {"random": 0, "teacher": 0, "policy": 0}
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
                action_id, action_source = _belief_rollout_action(
                    env,
                    example,
                    model,
                    device,
                    rng,
                    teacher_action_probability=teacher_action_probability,
                    policy_action_probability=policy_action_probability,
                )
                rollout_sources[action_source] += 1
                env.step_fast(action_id)
                moves_in_world += 1

                if len(pending) == batch_size:
                    latest_loss = _belief_update(model, optimiser, pending, device, training)
                    losses.append(latest_loss)
                    pending.clear()
                    run.log(
                        position=position + 1,
                        update=len(losses),
                        belief_loss=latest_loss,
                        hidden_tiles=hidden_tiles,
                        mean_hidden_tiles=float(np.mean(accepted_hidden_tiles)),
                        belief_rollout_random=float(rollout_sources["random"]),
                        belief_rollout_teacher=float(rollout_sources["teacher"]),
                        belief_rollout_policy=float(rollout_sources["policy"]),
                    )
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
            latest_loss = _belief_update(model, optimiser, pending, device, training)
            losses.append(latest_loss)
            run.log(
                position=positions,
                update=len(losses),
                belief_loss=latest_loss,
                hidden_tiles=float(np.mean([row.hidden_mask.sum() for row in pending])),
                mean_hidden_tiles=float(np.mean(accepted_hidden_tiles)),
            )
    finally:
        if original_trainable is not None:
            _restore_trainable_parameters(model, original_trainable)
    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float | str] = {
        "positions": float(positions),
        "belief_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_hidden_tiles": float(np.mean(accepted_hidden_tiles)) if accepted_hidden_tiles else 0.0,
        "min_hidden_tiles": float(min_hidden_tiles),
        "preserved_policy_value": float(preserve_policy_value),
        "belief_rollout_random": float(rollout_sources["random"]),
        "belief_rollout_teacher": float(rollout_sources["teacher"]),
        "belief_rollout_policy": float(rollout_sources["policy"]),
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
