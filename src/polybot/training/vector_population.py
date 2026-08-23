"""Fast raw-policy population play in VectorGameEnv, without any MCTS."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Deque, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..evaluation.matches import evaluate_direct_matches
from ..infrastructure.device import select_device
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from .direct_curriculum import model_state_on_cpu, restore_model_state
from .run_artifacts import TrainingRun
from .vector_actor_critic import _normalise_advantage, _visible_action_reward
from .vector_teacher import (
    _teacher_targets,
    _vector_tensors,
    _visible_goal_progress,
)


_OBJECTIVE_REPLAY_FIELDS = (
    "map_tokens",
    "state",
    "action_features",
    "action_mask",
    "visible_event_features",
    "visible_event_mask",
)


@dataclass(frozen=True)
class _ObjectiveReplayExample:
    """One public candidate decision preceding a real in-game success.

    The examples intentionally contain only the same visible tensors consumed
    by the raw policy.  They are behaviour-cloning targets, not fabricated
    value targets: reaching a village or a city is useful, but not proof that
    the whole game will be won.
    """

    observation: Mapping[str, np.ndarray]
    action_row: int
    weight: float


@dataclass
class _PPOExample:
    """One on-policy candidate decision with immutable public inputs."""

    observation: Mapping[str, np.ndarray]
    action_row: int
    old_log_probability: float
    old_value: float
    advantage: float = 0.0
    return_target: float = 0.0
    # For finished games this is the undiscounted game outcome from the
    # candidate's perspective (+1 / 0 / -1).  It deliberately remains
    # separate from the GAE return used to construct the policy advantage:
    # a 700-move game must still teach its opening positions who eventually
    # won.  A forced partial-trajectory flush falls back to ``return_target``.
    value_target: float | None = None
    # Policy updates can use only the upper advantage quantile while the value
    # and PPG objectives continue to see every completed position.
    policy_selected: bool = True


@dataclass
class _PPOTransition:
    """One native action in a slot, including opponent actions for GAE."""

    value: float
    reward: float
    next_value: float
    sign: float
    terminal: bool
    candidate_example: _PPOExample | None


@dataclass(frozen=True)
class _PPGExample:
    """One diverse public position retained for the auxiliary value phase."""

    observation: Mapping[str, np.ndarray]
    return_target: float


@dataclass(frozen=True)
class _PPGTargetExample:
    """A PPG position with the policy frozen at auxiliary-phase start."""

    observation: Mapping[str, np.ndarray]
    policy_log_probabilities: np.ndarray
    return_target: float


def _ppo_snapshot(
    batch: Mapping[str, np.ndarray], row: int, action_row: int
) -> tuple[Dict[str, np.ndarray], int]:
    """Copy and compact only public observations needed for PPO replay."""
    result: Dict[str, np.ndarray] = {}
    for field in _OBJECTIVE_REPLAY_FIELDS:
        if field in {"action_features", "action_mask"}:
            continue
        value = np.asarray(batch[field])[row]
        # PolyEnv's public token and action fields are documented int32 but
        # their categorical/runtime ranges fit in int16. Keeping a rollout in
        # RAM at this width avoids buffering multi-gigabyte full trajectories.
        if np.issubdtype(value.dtype, np.integer) and value.dtype != np.bool_:
            value = value.astype(np.int16, copy=False)
        result[field] = value.copy()
    legal_rows = np.flatnonzero(np.asarray(batch["action_mask"])[row])
    compact_action_row = int(np.searchsorted(legal_rows, action_row))
    if (
        compact_action_row >= legal_rows.size
        or int(legal_rows[compact_action_row]) != int(action_row)
    ):
        raise RuntimeError("PPO attempted to snapshot an illegal action")
    result["action_features"] = np.asarray(
        batch["action_features"]
    )[row, legal_rows].astype(np.int16, copy=True)
    result["action_mask"] = np.ones(legal_rows.size, dtype=bool)
    return result, compact_action_row


def _collate_public_observations(
    examples: Sequence[_PPOExample | _PPGExample | _PPGTargetExample],
) -> Dict[str, np.ndarray]:
    """Pad variable legal-action sets shared by PPO and PPG minibatches."""
    if not examples:
        raise ValueError("public observation batch must not be empty")
    batch_size = len(examples)
    maximum_actions = max(
        example.observation["action_features"].shape[0]
        for example in examples
    )
    action_features = np.zeros(
        (
            batch_size,
            maximum_actions,
            examples[0].observation["action_features"].shape[1],
        ),
        dtype=np.int16,
    )
    action_mask = np.zeros((batch_size, maximum_actions), dtype=bool)
    for row, example in enumerate(examples):
        actions = example.observation["action_features"]
        action_features[row, : actions.shape[0]] = actions
        action_mask[row, : actions.shape[0]] = True
    observation = {
        field: np.stack(
            [example.observation[field] for example in examples], axis=0
        )
        for field in _OBJECTIVE_REPLAY_FIELDS
        if field not in {"action_features", "action_mask"}
    }
    observation["action_features"] = action_features
    observation["action_mask"] = action_mask
    return observation


def _finish_ppo_trace(
    transitions: Sequence[_PPOTransition],
    *,
    gamma: float,
    gae_lambda: float,
    terminal_outcome: float | None = None,
    terminal_advantage_weight: float = 0.75,
) -> list[_PPOExample]:
    """Compute sign-correct GAE through candidate and opponent actions.

    The value head is always from the active player's viewpoint.  Therefore a
    turn switch changes the sign both in the bootstrap and in the recursive
    advantage.  This is the multi-step counterpart of the old one-step TD
    target and correctly propagates a terminal loss through the opponent turn.
    """
    if not 0.0 <= terminal_advantage_weight <= 1.0:
        raise ValueError("terminal_advantage_weight must be between zero and one")
    advantage = 0.0
    result: list[_PPOExample] = []
    for transition in reversed(transitions):
        continuation = 0.0 if transition.terminal else transition.sign
        delta = (
            transition.reward
            + gamma * continuation * transition.next_value
            - transition.value
        )
        advantage = delta + gamma * gae_lambda * continuation * advantage
        if transition.candidate_example is not None:
            example = transition.candidate_example
            # GAE retains a small, early-learning signal for visible tactical
            # events.  It cannot by itself credit an opening move in a
            # 700-turn game: lambda < 1 exponentially erases the terminal
            # result.  On genuine game completion, mix in Monte-Carlo outcome
            # advantage from this acting player's perspective so PPO receives
            # the same long-horizon objective as value and PPG.
            outcome_advantage = (
                float(terminal_outcome) - float(example.old_value)
                if terminal_outcome is not None
                else float(advantage)
            )
            example.advantage = float(
                (1.0 - terminal_advantage_weight) * advantage
                + terminal_advantage_weight * outcome_advantage
            )
            example.return_target = float(transition.value + advantage)
            example.value_target = (
                float(terminal_outcome)
                if terminal_outcome is not None
                else example.return_target
            )
            result.append(example)
    result.reverse()
    return result


def _ppo_batch(
    examples: Sequence[_PPOExample],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collate a public on-policy PPO minibatch."""
    if not examples:
        raise ValueError("PPO batch must not be empty")
    return (
        _collate_public_observations(examples),
        np.asarray([example.action_row for example in examples], dtype=np.int64),
        np.asarray(
            [example.old_log_probability for example in examples],
            dtype=np.float32,
        ),
        np.asarray([example.advantage for example in examples], dtype=np.float32),
        np.asarray([example.return_target for example in examples], dtype=np.float32),
    )


def _mark_top_advantage_examples(
    examples: Sequence[_PPOExample],
    fraction: float,
) -> int:
    """Mark the upper advantage quantile for the PPO policy loss.

    Keeping all examples for the critic avoids throwing away terminal labels.
    The actor, however, benefits from ignoring low-value/noisy transitions in
    long self-play games.  The deterministic stable ordering makes this easy
    to regression-test when advantages tie.
    """
    if not examples:
        return 0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top advantage fraction must be in (0, 1]")
    selected_count = max(1, int(np.ceil(len(examples) * fraction)))
    order = np.argsort(
        np.asarray([example.advantage for example in examples], dtype=np.float32),
        kind="stable",
    )
    for example in examples:
        example.policy_selected = False
    for index in order[-selected_count:]:
        examples[int(index)].policy_selected = True
    return selected_count


@torch.no_grad()
def _update_parameter_ema(
    ema_model: PolicyValueBeliefNet,
    model: PolicyValueBeliefNet,
    decay: float,
) -> None:
    """Update a deployment-only EMA without adding training gradients."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise ValueError("EMA model must have the same parameter structure")
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.lerp_(model_parameters[name].detach(), 1.0 - decay)
    # The current models do not have floating running-stat buffers, but copy
    # every buffer defensively so an architecture extension cannot leave EMA
    # with stale integer masks or counters.
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA model must have the same buffer structure")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name].detach())


def _ppg_batch(
    examples: Sequence[_PPGTargetExample],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Collate one auxiliary PPG minibatch with a padded frozen policy."""
    if not examples:
        raise ValueError("PPG batch must not be empty")
    maximum_actions = max(
        example.policy_log_probabilities.size for example in examples
    )
    reference_log_probabilities = np.zeros(
        (len(examples), maximum_actions), dtype=np.float32
    )
    for row, example in enumerate(examples):
        count = int(example.policy_log_probabilities.size)
        if count != int(example.observation["action_features"].shape[0]):
            raise ValueError("PPG policy target must match its legal actions")
        reference_log_probabilities[row, :count] = (
            example.policy_log_probabilities
        )
    return (
        _collate_public_observations(examples),
        reference_log_probabilities,
        np.asarray(
            [example.return_target for example in examples],
            dtype=np.float32,
        ),
    )


def _masked_policy_kl(
    reference_log_probabilities: torch.Tensor,
    current_logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(reference || current), excluding padded legal-action columns."""
    if (
        reference_log_probabilities.shape != current_logits.shape
        or action_mask.shape != current_logits.shape
    ):
        raise ValueError("PPG policy tensors must have matching shapes")
    current_log_probabilities = F.log_softmax(current_logits.float(), dim=-1)
    reference_policy = torch.where(
        action_mask,
        reference_log_probabilities.float().exp(),
        torch.zeros_like(reference_log_probabilities, dtype=torch.float32),
    )
    difference = torch.where(
        action_mask,
        reference_log_probabilities.float() - current_log_probabilities,
        torch.zeros_like(current_log_probabilities),
    )
    return (reference_policy * difference).sum(dim=1).mean()


def _objective_replay_success_weights(
    features: np.ndarray,
    selected_rows: np.ndarray,
    map_tokens: np.ndarray,
    state: np.ndarray,
    candidate_rows: np.ndarray,
    terminal: np.ndarray,
    terminal_reward: np.ndarray,
    map_size: int,
) -> np.ndarray:
    """Return public success weights for candidate decisions only.

    A successful prefix is created only after an observable event: entering a
    visible village, founding or capturing a city, or winning the game.  This
    avoids the previous arbitrary fixed-horizon objective rollouts and never
    reads PolyEnv's authoritative hidden map.
    """
    count = int(features.shape[0])
    rows = np.arange(count, dtype=np.int64)
    chosen = features[rows, selected_rows]
    candidate_mask = np.zeros(count, dtype=bool)
    candidate_mask[np.asarray(candidate_rows, dtype=np.int64)] = True
    _, enters_village = _visible_goal_progress(
        features, map_tokens, state, int(map_size)
    )
    village = enters_village[rows, selected_rows]
    capture_city = (chosen[:, 0] == 8) & (chosen[:, 8] == 14)
    found_city = (chosen[:, 0] == 8) & (chosen[:, 8] == 11)
    win = np.asarray(terminal, dtype=bool) & (
        np.asarray(terminal_reward, dtype=np.float32) > 0.0
    )
    weights = np.zeros(count, dtype=np.float32)
    weights = np.maximum(weights, np.where(village, 0.25, 0.0))
    weights = np.maximum(weights, np.where(found_city, 0.35, 0.0))
    weights = np.maximum(weights, np.where(capture_city, 0.60, 0.0))
    weights = np.maximum(weights, np.where(win, 1.00, 0.0))
    return np.where(candidate_mask, weights, 0.0).astype(np.float32)


def _objective_replay_snapshot(
    batch: Mapping[str, np.ndarray], row: int, action_row: int
) -> _ObjectiveReplayExample:
    """Copy one visible decision before VectorGameEnv advances the slot."""
    return _ObjectiveReplayExample(
        observation={
            field: np.asarray(batch[field])[row].copy()
            for field in _OBJECTIVE_REPLAY_FIELDS
        },
        action_row=int(action_row),
        weight=1.0,
    )


def _objective_replay_batch(
    examples: Sequence[_ObjectiveReplayExample],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Collate public replay examples for one extra policy-only update."""
    if not examples:
        raise ValueError("objective replay batch must not be empty")
    return (
        {
            field: np.stack(
                [example.observation[field] for example in examples], axis=0
            )
            for field in _OBJECTIVE_REPLAY_FIELDS
        },
        np.asarray(
            [example.action_row for example in examples], dtype=np.int64
        ),
        np.asarray([example.weight for example in examples], dtype=np.float32),
    )


def _model_rows(
    model: PolicyValueBeliefNet,
    tensors: Mapping[str, torch.Tensor],
    rows: np.ndarray,
) -> Any:
    index = torch.as_tensor(rows, dtype=torch.long, device=tensors["map_tokens"].device)
    return model(
        tensors["map_tokens"].index_select(0, index),
        tensors["global_features"].index_select(0, index),
        tensors["action_categorical"].index_select(0, index),
        tensors["action_continuous"].index_select(0, index),
        tensors["action_mask"].index_select(0, index),
        tensors["event_features"].index_select(0, index),
        action_history=tensors["action_history"].index_select(0, index),
        action_history_mask=tensors["action_history_mask"].index_select(
            0, index
        ),
        include_belief=False,
    )


def _index_copy_policy_rows_(
    target: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Copy policy rows without depending on the active AMP dtype."""
    target.index_copy_(
        0,
        index.to(device=target.device, dtype=torch.long),
        source.to(device=target.device, dtype=target.dtype),
    )


def _train_vector_population_legacy(
    training: TrainingConfig,
    model_config: ModelConfig,
    candidate: PolicyValueBeliefNet,
    opponent_models: Sequence[PolicyValueBeliefNet],
    opponent_weights: Sequence[float],
    *,
    num_envs: int,
    steps: int,
    num_threads: int | None,
    max_actions: int,
    anchor_model: PolicyValueBeliefNet,
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    opponent_policy_weight: float = 0.50,
    teacher_weight: float = 0.25,
    shaping_weight: float = 0.20,
    greedy_advantage_weight: float = 0.25,
    entropy_weight: float = 0.004,
    gamma: float = 0.995,
    objective_replay_horizon: int = 16,
    objective_replay_capacity: int = 512,
    objective_replay_batch_size: int = 32,
    objective_replay_interval: int = 16,
    objective_replay_weight: float = 0.35,
    objective_replay_recency: float = 0.92,
    objective_replay_max_epochs: int = 4,
    game_callback: Callable[[Mapping[str, Any]], None] | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train one raw policy against routed frozen opponents in a C++ batch.

    Candidate turns use actor-critic. Opponent turns distil the selected
    frozen population policy into the candidate, while value learning remains
    zero-sum and player-relative on every row. No belief particles, tree nodes
    or MCTS simulations are created in this phase.
    """
    if num_envs < 2 or steps < 1 or max_actions < 1:
        raise ValueError("num_envs >= 2, steps and max_actions are required")
    if training.map_type != "lakes":
        raise ValueError(
            "VectorGameEnv population play currently supports lakes only"
        )
    if not opponent_models:
        raise ValueError("direct population play requires an opponent")
    if anchor_model is candidate or any(
        opponent is candidate for opponent in opponent_models
    ):
        raise ValueError(
            "candidate, anchor and frozen opponents must be separate models"
        )
    weights = np.asarray(opponent_weights, dtype=np.float64)
    if (
        weights.shape != (len(opponent_models),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("opponent_weights must be finite and positive")
    weights /= weights.sum()
    for name, value in {
        "policy_anchor_weight": policy_anchor_weight,
        "value_anchor_weight": value_anchor_weight,
        "opponent_policy_weight": opponent_policy_weight,
        "teacher_weight": teacher_weight,
        "shaping_weight": shaping_weight,
        "greedy_advantage_weight": greedy_advantage_weight,
        "exploration_rate": training.exploration_rate,
        "objective_replay_weight": objective_replay_weight,
        "objective_replay_recency": objective_replay_recency,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if objective_replay_horizon < 1:
        raise ValueError("objective_replay_horizon must be positive")
    if objective_replay_capacity < 1:
        raise ValueError("objective_replay_capacity must be positive")
    if objective_replay_batch_size < 1:
        raise ValueError("objective_replay_batch_size must be positive")
    if objective_replay_interval < 1:
        raise ValueError("objective_replay_interval must be positive")
    if objective_replay_max_epochs < 1:
        raise ValueError("objective_replay_max_epochs must be positive")
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("PolyEnv VectorGameEnv is required") from error

    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "direct_population")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "num_envs": num_envs,
            "steps": steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "opponent_weights": weights.tolist(),
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "opponent_policy_weight": opponent_policy_weight,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
            "greedy_advantage_weight": greedy_advantage_weight,
            "exploration_rate": training.exploration_rate,
            "objective_replay_horizon": objective_replay_horizon,
            "objective_replay_capacity": objective_replay_capacity,
            "objective_replay_batch_size": objective_replay_batch_size,
            "objective_replay_interval": objective_replay_interval,
            "objective_replay_weight": objective_replay_weight,
            "objective_replay_recency": objective_replay_recency,
            "objective_replay_max_epochs": objective_replay_max_epochs,
        },
    )
    candidate.to(device)
    candidate.train()
    anchor_model.to(device).eval()
    for parameter in anchor_model.parameters():
        parameter.requires_grad_(False)
    frozen: list[PolicyValueBeliefNet] = []
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
        visible_event_history=0,
    )
    batch = env.reset(seed=training.seed)
    rng = np.random.default_rng(training.seed ^ 0xD1EC7)
    opponent_indices = rng.choice(
        len(frozen), size=num_envs, p=weights
    ).astype(np.int32)
    candidate_players = (np.arange(num_envs, dtype=np.int32) & 1).copy()
    episode_moves = np.zeros(num_envs, dtype=np.int32)
    episode_numbers = np.zeros(num_envs, dtype=np.int64)
    objective_histories: list[Deque[_ObjectiveReplayExample]] = [
        deque(maxlen=objective_replay_horizon) for _ in range(num_envs)
    ]
    objective_replay: Deque[_ObjectiveReplayExample] = deque(
        maxlen=objective_replay_capacity
    )
    completed_games = wins = losses = draws = 0
    candidate_turns = opponent_turns = 0
    candidate_exploration_turns = 0
    objective_replay_events = 0
    objective_replay_positions = 0
    objective_replay_updates = 0
    objective_replay_update_credits = 0
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "greedy_advantage_loss": 0.0,
        "opponent_policy_loss": 0.0,
        "value_loss": 0.0,
        "teacher_loss": 0.0,
        "policy_anchor_loss": 0.0,
        "value_anchor_loss": 0.0,
        "entropy": 0.0,
        "reward": 0.0,
        "objective_replay_loss": 0.0,
    }

    with tqdm(
        total=steps,
        desc=f"direct population ({device}, no MCTS)",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for step in range(steps):
            tensors = _vector_tensors(batch, device, training)
            features = np.asarray(batch["action_features"], dtype=np.int64)
            legal = np.asarray(batch["action_mask"], dtype=bool)
            state = np.asarray(batch["state"], dtype=np.int32)
            players_before = state[:, 1]
            candidate_rows = np.flatnonzero(players_before == candidate_players)
            opponent_rows = np.flatnonzero(players_before != candidate_players)
            candidate_turns += int(candidate_rows.size)
            opponent_turns += int(opponent_rows.size)
            teacher_target = torch.as_tensor(
                _teacher_targets(
                    features,
                    legal,
                    np.asarray(batch["map_tokens"], dtype=np.int64),
                    state,
                    training.map_size,
                ),
                dtype=torch.float32,
                device=device,
            )

            candidate.train()
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if cuda_amp
                else nullcontext()
            )
            with autocast:
                output = candidate(
                    tensors["map_tokens"],
                    tensors["global_features"],
                    tensors["action_categorical"],
                    tensors["action_continuous"],
                    tensors["action_mask"],
                    tensors["event_features"],
                    include_belief=False,
                )
                distribution = Categorical(logits=output.policy_logits)
                action_rows = distribution.sample()
                entropy_rows = distribution.entropy()

            # Evaluation is greedy.  Keeping most learner turns greedy makes
            # the rollout/value targets describe the policy that promotion
            # actually measures, while the small sampled subset still gives
            # REINFORCE a genuine exploration signal.  Previously every
            # candidate action was sampled even with --exploration-rate 0.10,
            # making this otherwise supervised population phase needlessly
            # noisy and unlike its own held-out gate.
            exploratory_rows = np.zeros(num_envs, dtype=bool)
            if candidate_rows.size:
                exploratory_rows[candidate_rows] = (
                    rng.random(candidate_rows.size)
                    < float(training.exploration_rate)
                )
                exploit_rows = candidate_rows[~exploratory_rows[candidate_rows]]
                if exploit_rows.size:
                    exploit_index = torch.as_tensor(
                        exploit_rows, dtype=torch.long, device=device
                    )
                    greedy_rows = output.policy_logits.argmax(dim=-1)
                    action_rows.index_copy_(
                        0,
                        exploit_index,
                        greedy_rows.index_select(0, exploit_index),
                    )
            candidate_exploration_turns += int(exploratory_rows.sum())

            # Keep distillation targets in fp32 even when the candidate
            # forward pass is autocast to fp16. Frozen opponents run outside
            # autocast and may otherwise produce an incompatible dtype.
            opponent_target = torch.zeros_like(
                output.policy_logits, dtype=torch.float32
            )
            for opponent_index, opponent in enumerate(frozen):
                rows = opponent_rows[
                    opponent_indices[opponent_rows] == opponent_index
                ]
                if not rows.size:
                    continue
                with torch.inference_mode():
                    opponent_output = _model_rows(opponent, tensors, rows)
                    opponent_policy = torch.softmax(
                        opponent_output.policy_logits, dim=-1
                    )
                    selected = opponent_output.policy_logits.argmax(dim=-1)
                row_tensor = torch.as_tensor(
                    rows, dtype=torch.long, device=device
                )
                _index_copy_policy_rows_(
                    opponent_target, row_tensor, opponent_policy
                )
                action_rows.index_copy_(0, row_tensor, selected)

            selected_rows = action_rows.detach().cpu().numpy().astype(np.int64)
            shaped = _visible_action_reward(
                features,
                selected_rows,
                np.asarray(batch["map_tokens"], dtype=np.int64),
                state,
                training.map_size,
            )
            action_ids = np.asarray(batch["action_id"], dtype=np.int32)[
                np.arange(num_envs), selected_rows
            ]
            next_batch = env.step(action_ids)
            terminal = np.asarray(next_batch["terminated"], dtype=bool)
            terminal_reward = np.asarray(next_batch["reward"], dtype=np.float32)
            reward = shaping_weight * shaped + terminal_reward

            # A prefix is retained only when the candidate just achieved an
            # observable objective.  `batch` contains the public observation
            # from before the action; keeping it is therefore valid even
            # though VectorGameEnv has already advanced to `next_batch`.
            if objective_replay_weight > 0.0 and candidate_rows.size:
                success_weights = _objective_replay_success_weights(
                    features,
                    selected_rows,
                    np.asarray(batch["map_tokens"], dtype=np.int64),
                    state,
                    candidate_rows,
                    terminal,
                    terminal_reward,
                    training.map_size,
                )
                for env_id in candidate_rows:
                    history = objective_histories[int(env_id)]
                    history.append(
                        _objective_replay_snapshot(
                            batch, int(env_id), int(selected_rows[env_id])
                        )
                    )
                    success_weight = float(success_weights[env_id])
                    if success_weight <= 0.0:
                        continue
                    objective_replay_events += 1
                    # The action that created the event is weighted most, but
                    # the preceding public decisions receive credit too. This
                    # is a bounded return-to-go, not an arbitrary rollout.
                    for age, example in enumerate(reversed(history)):
                        objective_replay.append(
                            replace(
                                example,
                                weight=(
                                    success_weight
                                    * objective_replay_recency**age
                                ),
                            )
                        )
                        objective_replay_positions += 1
                        # One stored public decision may be drawn at most a
                        # small configured number of times on average. Without
                        # this credit budget, a quiet replay of 32 examples
                        # could be optimised every 16 vector steps forever.
                        objective_replay_update_credits = min(
                            objective_replay_update_credits
                            + objective_replay_max_epochs,
                            len(objective_replay)
                            * objective_replay_max_epochs,
                        )
            # Native auto-reset affects only finished slots. Their history
            # must never leak into the following episode in that slot.
            for env_id in np.flatnonzero(terminal):
                objective_histories[int(env_id)].clear()

            candidate.eval()
            with torch.no_grad():
                next_tensors = _vector_tensors(next_batch, device, training)
                next_output = candidate(
                    next_tensors["map_tokens"],
                    next_tensors["global_features"],
                    next_tensors["action_categorical"],
                    next_tensors["action_continuous"],
                    next_tensors["action_mask"],
                    next_tensors["event_features"],
                    include_belief=False,
                )
                players_after = np.asarray(
                    next_batch["state"], dtype=np.int32
                )[:, 1]
                sign = torch.as_tensor(
                    np.where(players_after == players_before, 1.0, -1.0),
                    dtype=torch.float32,
                    device=device,
                )
                done = torch.as_tensor(
                    terminal, dtype=torch.float32, device=device
                )
                reward_tensor = torch.as_tensor(
                    reward, dtype=torch.float32, device=device
                )
                td_target = (
                    reward_tensor
                    + gamma * (1.0 - done) * sign * next_output.value
                )
                anchor_output = anchor_model(
                    tensors["map_tokens"],
                    tensors["global_features"],
                    tensors["action_categorical"],
                    tensors["action_continuous"],
                    tensors["action_mask"],
                    tensors["event_features"],
                    include_belief=False,
                )
                anchor_policy = torch.softmax(
                    anchor_output.policy_logits, dim=-1
                )
            candidate.train()

            advantage = td_target - output.value
            log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
            valid_log_probabilities = torch.where(
                tensors["action_mask"],
                log_probabilities,
                torch.zeros_like(log_probabilities),
            )
            exploratory_candidate_rows = np.flatnonzero(exploratory_rows)
            if exploratory_candidate_rows.size:
                candidate_index = torch.as_tensor(
                    exploratory_candidate_rows, dtype=torch.long, device=device
                )
                candidate_advantage = _normalise_advantage(
                    advantage.index_select(0, candidate_index)
                )
                candidate_log_probability = distribution.log_prob(
                    action_rows
                ).index_select(0, candidate_index)
                policy_loss = -(
                    candidate_log_probability
                    * candidate_advantage.detach()
                ).mean()
                entropy = entropy_rows.index_select(
                    0, candidate_index
                ).mean()
            else:
                policy_loss = output.value.sum() * 0.0
                entropy = output.value.sum() * 0.0
            greedy_candidate_rows = candidate_rows[
                ~exploratory_rows[candidate_rows]
            ]
            if greedy_candidate_rows.size:
                greedy_index = torch.as_tensor(
                    greedy_candidate_rows, dtype=torch.long, device=device
                )
                greedy_actions = action_rows.index_select(0, greedy_index)
                greedy_log_probability = valid_log_probabilities.index_select(
                    0, greedy_index
                ).gather(1, greedy_actions.unsqueeze(1)).squeeze(1)
                # Greedy rows match evaluation. Reinforce only a positive TD
                # advantage, so the stable rollout mode can improve rather
                # than merely preserve frozen-policy targets.
                positive_advantage = advantage.index_select(
                    0, greedy_index
                ).detach().clamp(min=0.0, max=1.0)
                greedy_advantage_loss = -(
                    greedy_log_probability * positive_advantage
                ).mean()
            else:
                greedy_advantage_loss = output.value.sum() * 0.0
            if opponent_rows.size:
                opponent_index_tensor = torch.as_tensor(
                    opponent_rows, dtype=torch.long, device=device
                )
                opponent_policy_loss = -(
                    opponent_target.index_select(0, opponent_index_tensor)
                    * valid_log_probabilities.index_select(
                        0, opponent_index_tensor
                    )
                ).sum(dim=1).mean()
            else:
                opponent_policy_loss = output.value.sum() * 0.0
            value_loss = F.smooth_l1_loss(output.value, td_target)
            teacher_loss = -(
                teacher_target * valid_log_probabilities
            ).sum(dim=1).mean()
            policy_anchor_loss = -(
                anchor_policy * valid_log_probabilities
            ).sum(dim=1).mean()
            value_anchor_loss = F.smooth_l1_loss(
                output.value, anchor_output.value
            )
            objective_replay_loss = output.value.sum() * 0.0
            if (
                objective_replay_weight > 0.0
                and len(objective_replay) >= objective_replay_batch_size
                and objective_replay_update_credits
                >= objective_replay_batch_size
                and (step + 1) % objective_replay_interval == 0
            ):
                replay_indices = rng.choice(
                    len(objective_replay),
                    size=objective_replay_batch_size,
                    replace=False,
                )
                replay_examples = [
                    objective_replay[int(index)] for index in replay_indices
                ]
                replay_batch, replay_actions, replay_weights = (
                    _objective_replay_batch(replay_examples)
                )
                replay_tensors = _vector_tensors(
                    replay_batch, device, training
                )
                replay_action_tensor = torch.as_tensor(
                    replay_actions, dtype=torch.long, device=device
                )
                replay_weight_tensor = torch.as_tensor(
                    replay_weights, dtype=torch.float32, device=device
                )
                replay_autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if cuda_amp
                    else nullcontext()
                )
                with replay_autocast:
                    replay_output = candidate(
                        replay_tensors["map_tokens"],
                        replay_tensors["global_features"],
                        replay_tensors["action_categorical"],
                        replay_tensors["action_continuous"],
                        replay_tensors["action_mask"],
                        replay_tensors["event_features"],
                        include_belief=False,
                    )
                    replay_log_probabilities = F.log_softmax(
                        replay_output.policy_logits, dim=-1
                    )
                    replay_log_probability = replay_log_probabilities.gather(
                        1, replay_action_tensor.unsqueeze(1)
                    ).squeeze(1)
                    objective_replay_loss = -(
                        replay_weight_tensor * replay_log_probability
                    ).sum() / replay_weight_tensor.sum().clamp_min(1e-6)
                objective_replay_updates += 1
                objective_replay_update_credits -= objective_replay_batch_size
            loss = (
                policy_loss
                + greedy_advantage_weight * greedy_advantage_loss
                + opponent_policy_weight * opponent_policy_loss
                + 0.50 * value_loss
                + teacher_weight * teacher_loss
                + policy_anchor_weight * policy_anchor_loss
                + value_anchor_weight * value_anchor_loss
                + objective_replay_weight * objective_replay_loss
                - entropy_weight * entropy
            )
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            episode_moves += 1
            for env_id in np.flatnonzero(terminal):
                completed_games += 1
                selected_opponent = int(opponent_indices[env_id])
                if terminal_reward[env_id] != 0.0:
                    winner_player = int(players_before[env_id])
                    if terminal_reward[env_id] < 0.0:
                        winner_player = 1 - winner_player
                    candidate_won = winner_player == candidate_players[env_id]
                    wins += int(candidate_won)
                    losses += int(not candidate_won)
                    score = float(candidate_won)
                else:
                    draws += 1
                    winner_player = -1
                    score = 0.5
                if game_callback is not None:
                    # Native auto-reset creates the next world internally.
                    # Preserve stable vector-slot lineage without pretending
                    # that its internal map seed is directly observable.
                    game_callback({
                        "game_index": completed_games - 1,
                        "map_seed": int(training.seed),
                        "map_seed_is_collection_seed": 1,
                        "vector_env_id": int(env_id),
                        "vector_episode": int(episode_numbers[env_id]),
                        "model_a_player": int(candidate_players[env_id]),
                        "winner_player": winner_player,
                        "score_a": score,
                        "moves": int(episode_moves[env_id]),
                        "opponent_model_index": selected_opponent,
                    })
                candidate_players[env_id] = 1 - candidate_players[env_id]
                opponent_indices[env_id] = int(
                    rng.choice(len(frozen), p=weights)
                )
                episode_moves[env_id] = 0
                episode_numbers[env_id] += 1

            metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "greedy_advantage_loss": float(
                    greedy_advantage_loss.detach().cpu()
                ),
                "opponent_policy_loss": float(
                    opponent_policy_loss.detach().cpu()
                ),
                "value_loss": float(value_loss.detach().cpu()),
                "teacher_loss": float(teacher_loss.detach().cpu()),
                "policy_anchor_loss": float(
                    policy_anchor_loss.detach().cpu()
                ),
                "value_anchor_loss": float(value_anchor_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "reward": float(reward.mean()),
                "objective_replay_loss": float(
                    objective_replay_loss.detach().cpu()
                ),
            }
            for key, value in metrics.items():
                totals[key] += value
            batch = next_batch
            run.log(
                step=step + 1,
                environment_step=(step + 1) * num_envs,
                completed_games=completed_games,
                candidate_wins=wins,
                candidate_losses=losses,
                draws=draws,
                candidate_turns=candidate_turns,
                candidate_exploration_turns=candidate_exploration_turns,
                opponent_turns=opponent_turns,
                objective_replay_events=objective_replay_events,
                objective_replay_positions=objective_replay_positions,
                objective_replay_size=len(objective_replay),
                objective_replay_updates=objective_replay_updates,
                objective_replay_effective_epochs=(
                    objective_replay_updates * objective_replay_batch_size
                    / max(1, objective_replay_positions)
                ),
                **metrics,
            )
            progress.update(1)
            if (step + 1) % 20 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    games=completed_games,
                    score=f"{(wins + 0.5 * draws) / max(1, completed_games):.3f}",
                )

    summary: Dict[str, float | str] = {
        "direct_population_envs": float(num_envs),
        "direct_population_steps": float(steps),
        "direct_population_environment_steps": float(num_envs * steps),
        "direct_population_completed_games": float(completed_games),
        "direct_population_wins": float(wins),
        "direct_population_losses": float(losses),
        "direct_population_draws": float(draws),
        "direct_population_score": float(
            (wins + 0.5 * draws) / max(1, completed_games)
        ),
        "direct_population_candidate_turns": float(candidate_turns),
        "direct_population_candidate_exploration_turns": float(
            candidate_exploration_turns
        ),
        "direct_population_effective_exploration_rate": float(
            candidate_exploration_turns / max(1, candidate_turns)
        ),
        "direct_population_opponent_turns": float(opponent_turns),
        "direct_population_objective_replay_events": float(
            objective_replay_events
        ),
        "direct_population_objective_replay_positions": float(
            objective_replay_positions
        ),
        "direct_population_objective_replay_size": float(len(objective_replay)),
        "direct_population_objective_replay_updates": float(
            objective_replay_updates
        ),
        "direct_population_objective_replay_weight": float(
            objective_replay_weight
        ),
        "direct_population_objective_replay_horizon": float(
            objective_replay_horizon
        ),
        "direct_population_objective_replay_max_epochs": float(
            objective_replay_max_epochs
        ),
        "direct_population_objective_replay_effective_epochs": float(
            objective_replay_updates * objective_replay_batch_size
            / max(1, objective_replay_positions)
        ),
        **{
            "direct_population_" + key: value / max(1, steps)
            for key, value in totals.items()
        },
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary


def train_vector_population(
    training: TrainingConfig,
    model_config: ModelConfig,
    candidate: PolicyValueBeliefNet,
    opponent_models: Sequence[PolicyValueBeliefNet],
    opponent_weights: Sequence[float],
    *,
    num_envs: int,
    steps: int,
    num_threads: int | None,
    max_actions: int,
    anchor_model: PolicyValueBeliefNet,
    visible_action_history: int = 32,
    policy_anchor_weight: float = 0.10,
    value_anchor_weight: float = 0.10,
    opponent_policy_weight: float = 0.0,
    teacher_weight: float = 0.0,
    shaping_weight: float = 0.20,
    shaping_final_weight: float | None = None,
    shaping_progress_start: float = 0.0,
    shaping_progress_end: float = 1.0,
    greedy_advantage_weight: float = 0.0,
    entropy_weight: float = 0.002,
    gamma: float = 1.0,
    ppo_rollout_steps: int = 32,
    ppo_epochs: int = 2,
    ppo_minibatch_size: int = 256,
    ppo_clip_range: float = 0.15,
    ppo_gae_lambda: float = 0.95,
    ppo_terminal_advantage_weight: float = 0.75,
    ppo_top_advantage_fraction: float = 0.50,
    ppo_target_kl: float = 0.015,
    ppo_hard_kl: float = 0.030,
    ppo_min_learning_rate: float = 1.0e-5,
    ppo_lr_backoff: float = 0.5,
    ppo_rollout_temperature: float = 0.85,
    ppo_max_pending_examples: int = 32_768,
    objective_replay_horizon: int = 16,
    objective_replay_capacity: int = 512,
    objective_replay_batch_size: int = 32,
    objective_replay_interval: int = 16,
    objective_replay_weight: float = 0.35,
    objective_replay_recency: float = 0.92,
    objective_replay_max_epochs: int = 4,
    ppg_enabled: bool = True,
    ppg_interval: int = 8,
    ppg_buffer_capacity: int = 8_192,
    ppg_samples_per_rollout: int = 512,
    ppg_phase_samples: int = 4_096,
    ppg_epochs: int = 2,
    ppg_minibatch_size: int = 256,
    ppg_policy_kl_weight: float = 1.0,
    ppg_value_weight: float = 1.0,
    ema_model: PolicyValueBeliefNet | None = None,
    ema_decay: float = 0.999,
    game_callback: Callable[[Mapping[str, Any]], None] | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train the raw policy with signed, multi-step PPO/GAE trajectories.

    Only candidate actions receive an actor loss.  The per-slot trace still
    contains every intervening opponent action, so a terminal result and all
    intermediate rewards are propagated back with the correct player sign.
    A trace normally remains in RAM until its native game terminates. A bounded
    safety cap or the end of a training block may bootstrap an unfinished
    trace from its observed next value; neither case is ever labelled as a
    draw.
    """
    if num_envs < 2 or steps < 1 or max_actions < 1:
        raise ValueError("num_envs >= 2, steps and max_actions are required")
    if not 0 <= visible_action_history <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    if training.map_type != "lakes":
        raise ValueError(
            "VectorGameEnv population play currently supports lakes only"
        )
    if not opponent_models:
        raise ValueError("direct population play requires an opponent")
    if anchor_model is candidate or any(
        opponent is candidate for opponent in opponent_models
    ):
        raise ValueError(
            "candidate, anchor and frozen opponents must be separate models"
        )
    weights = np.asarray(opponent_weights, dtype=np.float64)
    if (
        weights.shape != (len(opponent_models),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("opponent_weights must be finite and positive")
    weights /= weights.sum()
    for name, value in {
        "policy_anchor_weight": policy_anchor_weight,
        "value_anchor_weight": value_anchor_weight,
        "opponent_policy_weight": opponent_policy_weight,
        "teacher_weight": teacher_weight,
        "greedy_advantage_weight": greedy_advantage_weight,
        "shaping_weight": shaping_weight,
        "entropy_weight": entropy_weight,
        "ppo_clip_range": ppo_clip_range,
        "ppo_gae_lambda": ppo_gae_lambda,
        "ppo_target_kl": ppo_target_kl,
        "ppo_hard_kl": ppo_hard_kl,
        "objective_replay_weight": objective_replay_weight,
        "objective_replay_recency": objective_replay_recency,
        "ppg_policy_kl_weight": ppg_policy_kl_weight,
        "ppg_value_weight": ppg_value_weight,
    }.items():
        if value < 0.0 or (
            name not in {
                "ppo_hard_kl",
                "ppg_policy_kl_weight",
                "ppg_value_weight",
            }
            and value > 1.0
        ):
            raise ValueError(f"{name} has an invalid value")
    if ppo_hard_kl < ppo_target_kl:
        raise ValueError("ppo_hard_kl must be at least ppo_target_kl")
    if ppo_min_learning_rate <= 0.0:
        raise ValueError("ppo_min_learning_rate must be positive")
    if not 0.0 < ppo_lr_backoff < 1.0:
        raise ValueError("ppo_lr_backoff must be in (0, 1)")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 < ppo_top_advantage_fraction <= 1.0:
        raise ValueError("ppo_top_advantage_fraction must be in (0, 1]")
    if not 0.0 <= ppo_terminal_advantage_weight <= 1.0:
        raise ValueError(
            "ppo_terminal_advantage_weight must be between zero and one"
        )
    resolved_shaping_final_weight = (
        shaping_weight if shaping_final_weight is None else shaping_final_weight
    )
    if (
        resolved_shaping_final_weight < 0.0
        or not 0.0 <= shaping_progress_start <= 1.0
        or not 0.0 <= shaping_progress_end <= 1.0
        or shaping_progress_end < shaping_progress_start
    ):
        raise ValueError("invalid shaping anneal configuration")
    if ema_model is not None and not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if (
        ppo_rollout_steps < 1
        or ppo_epochs < 1
        or ppo_minibatch_size < 1
        or ppo_max_pending_examples < ppo_minibatch_size
    ):
        raise ValueError("PPO rollout, epoch and minibatch sizes must be positive")
    if not 0.1 <= ppo_rollout_temperature <= 2.0:
        raise ValueError("ppo_rollout_temperature must be between 0.1 and 2.0")
    for name, value in {
        "objective_replay_horizon": objective_replay_horizon,
        "objective_replay_capacity": objective_replay_capacity,
        "objective_replay_batch_size": objective_replay_batch_size,
        "objective_replay_interval": objective_replay_interval,
        "objective_replay_max_epochs": objective_replay_max_epochs,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if objective_replay_capacity < objective_replay_batch_size:
        raise ValueError(
            "objective_replay_capacity must be at least its batch size"
        )
    for name, value in {
        "ppg_interval": ppg_interval,
        "ppg_buffer_capacity": ppg_buffer_capacity,
        "ppg_samples_per_rollout": ppg_samples_per_rollout,
        "ppg_phase_samples": ppg_phase_samples,
        "ppg_epochs": ppg_epochs,
        "ppg_minibatch_size": ppg_minibatch_size,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if ppg_enabled and (
        ppg_buffer_capacity < ppg_phase_samples
        or ppg_phase_samples < ppg_minibatch_size
        or ppg_buffer_capacity < ppg_samples_per_rollout
    ):
        raise ValueError(
            "PPG requires buffer >= phase samples >= minibatch and buffer "
            ">= samples per rollout"
        )
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("PolyEnv VectorGameEnv is required") from error

    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "direct_population_ppo")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "algorithm": (
                "single_network_ppg" if ppg_enabled else "signed_ppo_gae"
            ),
            "num_envs": num_envs,
            "steps": steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "visible_action_history": visible_action_history,
            "opponent_weights": weights.tolist(),
            "shaping_weight": shaping_weight,
            "shaping_final_weight": resolved_shaping_final_weight,
            "shaping_progress_start": shaping_progress_start,
            "shaping_progress_end": shaping_progress_end,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "ppo_rollout_steps": ppo_rollout_steps,
            "ppo_epochs": ppo_epochs,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_range": ppo_clip_range,
            "ppo_gamma": gamma,
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_terminal_advantage_weight": ppo_terminal_advantage_weight,
            "ppo_top_advantage_fraction": ppo_top_advantage_fraction,
            "ppo_target_kl": ppo_target_kl,
            "ppo_hard_kl": ppo_hard_kl,
            "ppo_min_learning_rate": ppo_min_learning_rate,
            "ppo_lr_backoff": ppo_lr_backoff,
            "ppo_rollout_temperature": ppo_rollout_temperature,
            "ppo_max_pending_examples": ppo_max_pending_examples,
            "objective_replay_horizon": objective_replay_horizon,
            "objective_replay_capacity": objective_replay_capacity,
            "objective_replay_batch_size": objective_replay_batch_size,
            "objective_replay_interval": objective_replay_interval,
            "objective_replay_weight": objective_replay_weight,
            "objective_replay_recency": objective_replay_recency,
            "objective_replay_max_epochs": objective_replay_max_epochs,
            "ppg_enabled": ppg_enabled,
            "ppg_interval": ppg_interval,
            "ppg_buffer_capacity": ppg_buffer_capacity,
            "ppg_samples_per_rollout": ppg_samples_per_rollout,
            "ppg_phase_samples": ppg_phase_samples,
            "ppg_epochs": ppg_epochs,
            "ppg_minibatch_size": ppg_minibatch_size,
            "ppg_policy_kl_weight": ppg_policy_kl_weight,
            "ppg_value_weight": ppg_value_weight,
            "ema_enabled": ema_model is not None,
            "ema_decay": ema_decay if ema_model is not None else None,
            # Kept in the artifacts so obsolete command-line flags remain
            # auditable rather than silently affecting the PPO objective.
            "legacy_opponent_policy_weight": opponent_policy_weight,
            "legacy_teacher_weight": teacher_weight,
            "legacy_greedy_advantage_weight": greedy_advantage_weight,
        },
    )
    candidate.to(device).eval()
    if ema_model is not None:
        if ema_model is candidate:
            raise ValueError("ema_model must be a separate frozen copy")
        ema_model.to(device).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    anchor_model.to(device).eval()
    for parameter in anchor_model.parameters():
        parameter.requires_grad_(False)
    frozen: list[PolicyValueBeliefNet] = []
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
        visible_event_history=visible_action_history,
    )
    batch = env.reset(seed=training.seed)
    rng = np.random.default_rng(training.seed ^ 0x50504F)
    opponent_indices = rng.choice(
        len(frozen), size=num_envs, p=weights
    ).astype(np.int32)
    candidate_players = (np.arange(num_envs, dtype=np.int32) & 1).copy()
    traces: list[list[_PPOTransition]] = [[] for _ in range(num_envs)]
    objective_histories: list[Deque[_ObjectiveReplayExample]] = [
        deque(maxlen=objective_replay_horizon) for _ in range(num_envs)
    ]
    objective_replay: Deque[_ObjectiveReplayExample] = deque(
        maxlen=objective_replay_capacity
    )
    ppg_replay: Deque[_PPGExample] = deque(maxlen=ppg_buffer_capacity)
    episode_moves = np.zeros(num_envs, dtype=np.int32)
    episode_numbers = np.zeros(num_envs, dtype=np.int64)
    completed_games = wins = losses = draws = 0
    candidate_turns = opponent_turns = 0
    ppo_samples = ppo_updates = ppo_rollouts = 0
    ppo_policy_samples = 0
    ppo_behaviour_kl_limits = 0
    completed_examples: list[_PPOExample] = []
    pending_candidate_examples = 0
    ppo_forced_truncations = 0
    objective_replay_events = 0
    objective_replay_positions = 0
    objective_replay_updates = 0
    objective_replay_update_credits = 0
    ppg_samples_added = 0
    ppg_phases = 0
    ppg_updates = 0
    ema_updates = 0
    ppo_update_target = max(
        ppo_minibatch_size, num_envs * ppo_rollout_steps // 2
    )
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "champion_kl": 0.0,
        "behaviour_kl": 0.0,
        "behaviour_kl_limited": 0.0,
        "clip_fraction": 0.0,
        "entropy": 0.0,
        "objective_replay_loss": 0.0,
        "policy_selected_fraction": 0.0,
    }
    # A PPO update happens only after enough completed trajectories have been
    # collected.  Keep its most recent value separately from the metrics for
    # *this environment step*, otherwise tqdm misleadingly displays 0.000 on
    # every collection-only step.
    last_champion_kl = 0.0
    reward_total = 0.0
    shaping_weight_total = 0.0
    history_tokens_seen = 0
    history_rows_seen = 0
    ppg_totals = {
        "loss": 0.0,
        "value_loss": 0.0,
        "policy_kl": 0.0,
    }

    def optimise_rollout(examples: Sequence[_PPOExample]) -> Dict[str, float]:
        """Run PPO with a local trust region around the rollout policy."""
        nonlocal ppo_updates, ppo_policy_samples, ppo_behaviour_kl_limits
        nonlocal ema_updates
        nonlocal objective_replay_updates, objective_replay_update_credits
        if not examples:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "champion_kl": 0.0,
                "behaviour_kl": 0.0,
                "behaviour_kl_limited": 0.0,
                "clip_fraction": 0.0,
                "entropy": 0.0,
                "objective_replay_loss": 0.0,
            }
        advantages = np.asarray(
            [example.advantage for example in examples], dtype=np.float32
        )
        advantages = (advantages - advantages.mean()) / max(
            float(advantages.std()), 1e-4
        )
        for example, advantage in zip(examples, advantages):
            example.advantage = float(advantage)
        _mark_top_advantage_examples(
            examples, ppo_top_advantage_fraction
        )
        totals_local = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "champion_kl": 0.0,
            "behaviour_kl": 0.0,
            "behaviour_kl_limited": 0.0,
            "clip_fraction": 0.0,
            "entropy": 0.0,
            "objective_replay_loss": 0.0,
            "policy_selected_fraction": 0.0,
        }
        local_updates = 0
        for _ in range(ppo_epochs):
            # An epoch may reuse a rollout many times.  If its behaviour KL
            # grows too large, restore only this epoch and lower the learning
            # rate; never discard the whole 6k-step population block.
            epoch_start_state = model_state_on_cpu(candidate)
            epoch_start_ema_state = (
                model_state_on_cpu(ema_model)
                if ema_model is not None
                else None
            )
            epoch_optimiser_state = deepcopy(optimiser.state_dict())
            epoch_scaler_state = deepcopy(scaler.state_dict())
            epoch_totals = dict(totals_local)
            epoch_local_updates = local_updates
            epoch_ppo_updates = ppo_updates
            epoch_ppo_policy_samples = ppo_policy_samples
            epoch_objective_replay_updates = objective_replay_updates
            epoch_objective_replay_credits = objective_replay_update_credits
            epoch_ema_updates = ema_updates
            limited_epoch = False
            order = rng.permutation(len(examples))
            for start in range(0, len(order), ppo_minibatch_size):
                subset = [examples[int(index)] for index in order[start:start + ppo_minibatch_size]]
                replay_batch, actions, old_log_probabilities, advantages_batch, returns = _ppo_batch(
                    subset
                )
                tensors = _vector_tensors(replay_batch, device, training)
                action_tensor = torch.as_tensor(
                    actions, dtype=torch.long, device=device
                )
                old_log_tensor = torch.as_tensor(
                    old_log_probabilities, dtype=torch.float32, device=device
                )
                advantage_tensor = torch.as_tensor(
                    advantages_batch, dtype=torch.float32, device=device
                )
                return_tensor = torch.as_tensor(
                    [
                        example.return_target
                        if example.value_target is None
                        else example.value_target
                        for example in subset
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                policy_selected_tensor = torch.as_tensor(
                    [example.policy_selected for example in subset],
                    dtype=torch.bool,
                    device=device,
                )
                # PPO compares against the rollout policy. Keeping dropout
                # disabled makes that ratio deterministic while preserving
                # gradients through the model parameters.
                candidate.eval()
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if cuda_amp
                    else nullcontext()
                )
                with autocast:
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
                    policy_log_probabilities = F.log_softmax(
                        output.policy_logits / ppo_rollout_temperature, dim=-1
                    )
                    new_log_probability = policy_log_probabilities.gather(
                        1, action_tensor.unsqueeze(1)
                    ).squeeze(1).float()
                    ratio = torch.exp(new_log_probability - old_log_tensor)
                    unclipped = ratio * advantage_tensor
                    clipped = torch.clamp(
                        ratio, 1.0 - ppo_clip_range, 1.0 + ppo_clip_range
                    ) * advantage_tensor
                    policy_terms = -torch.minimum(unclipped, clipped)
                    selected_policy_count = int(policy_selected_tensor.sum().item())
                    policy_loss = (
                        policy_terms[policy_selected_tensor].mean()
                        if selected_policy_count
                        else policy_terms.sum() * 0.0
                    )
                    # In PPG's policy phase the critic head may keep tracking
                    # fresh returns, but its gradient must not rewrite the
                    # shared map/history representation. The later auxiliary
                    # phase is the only place where value learning updates the
                    # shared encoder.
                    policy_phase_value = (
                        candidate.value_head(output.state.detach()).squeeze(-1)
                        if ppg_enabled
                        else output.value
                    )
                    value_loss = F.smooth_l1_loss(
                        policy_phase_value.float(), return_tensor
                    )
                    entropy_per_example = Categorical(
                        logits=output.policy_logits / ppo_rollout_temperature
                    ).entropy()
                    entropy = (
                        entropy_per_example[policy_selected_tensor].mean()
                        if selected_policy_count
                        else entropy_per_example.sum() * 0.0
                    )
                # The champion is frozen, but its logits are part of the KL
                # and value-anchor losses.  ``no_grad`` avoids building its
                # backward graph while keeping ordinary tensors that PyTorch
                # can safely retain for the candidate's backward pass.
                with torch.no_grad():
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
                    anchor_log_probabilities = F.log_softmax(
                        anchor_output.policy_logits / ppo_rollout_temperature,
                        dim=-1,
                    )
                    anchor_policy = anchor_log_probabilities.exp()
                legal_actions = tensors["action_mask"]
                kl_difference = torch.where(
                    legal_actions,
                    anchor_log_probabilities
                    - policy_log_probabilities.float(),
                    torch.zeros_like(policy_log_probabilities.float()),
                )
                champion_kl = (anchor_policy * kl_difference).sum(dim=1).mean()
                behaviour_kl = 0.5 * (
                    new_log_probability - old_log_tensor
                ).square().mean()
                if float(behaviour_kl.detach().cpu()) > ppo_hard_kl:
                    restore_model_state(candidate, epoch_start_state)
                    if ema_model is not None and epoch_start_ema_state is not None:
                        restore_model_state(ema_model, epoch_start_ema_state)
                    optimiser.load_state_dict(epoch_optimiser_state)
                    scaler.load_state_dict(epoch_scaler_state)
                    totals_local = epoch_totals
                    local_updates = epoch_local_updates
                    ppo_updates = epoch_ppo_updates
                    ppo_policy_samples = epoch_ppo_policy_samples
                    objective_replay_updates = epoch_objective_replay_updates
                    objective_replay_update_credits = (
                        epoch_objective_replay_credits
                    )
                    ema_updates = epoch_ema_updates
                    previous_learning_rate = float(optimiser.param_groups[0]["lr"])
                    adapted_learning_rate = max(
                        ppo_min_learning_rate,
                        previous_learning_rate * ppo_lr_backoff,
                    )
                    for parameter_group in optimiser.param_groups:
                        parameter_group["lr"] = adapted_learning_rate
                    ppo_behaviour_kl_limits += 1
                    totals_local["behaviour_kl_limited"] += 1.0
                    print(
                        "direct population PPO behaviour-KL limit | "
                        f"kl={float(behaviour_kl.detach().cpu()):.5f} "
                        f"hard={ppo_hard_kl:.5f} | restored current epoch | "
                        f"lr={previous_learning_rate:.2e}->{adapted_learning_rate:.2e}",
                        flush=True,
                    )
                    limited_epoch = True
                    break
                kl_multiplier = policy_anchor_weight * max(
                    1.0,
                    float(champion_kl.detach().cpu())
                    / max(ppo_target_kl, 1e-6),
                )
                value_anchor_loss = F.smooth_l1_loss(
                    policy_phase_value.float(), anchor_output.value.float()
                )
                objective_replay_loss = output.value.sum() * 0.0
                if (
                    objective_replay_weight > 0.0
                    and len(objective_replay) >= objective_replay_batch_size
                    and objective_replay_update_credits
                    >= objective_replay_batch_size
                    and (ppo_updates + 1) % objective_replay_interval == 0
                ):
                    replay_indices = rng.choice(
                        len(objective_replay),
                        size=objective_replay_batch_size,
                        replace=False,
                    )
                    replay_examples = [
                        objective_replay[int(index)] for index in replay_indices
                    ]
                    replay_batch, replay_actions, replay_weights = (
                        _objective_replay_batch(replay_examples)
                    )
                    replay_tensors = _vector_tensors(
                        replay_batch, device, training
                    )
                    replay_action_tensor = torch.as_tensor(
                        replay_actions, dtype=torch.long, device=device
                    )
                    replay_weight_tensor = torch.as_tensor(
                        replay_weights, dtype=torch.float32, device=device
                    )
                    replay_output = candidate(
                        replay_tensors["map_tokens"],
                        replay_tensors["global_features"],
                        replay_tensors["action_categorical"],
                        replay_tensors["action_continuous"],
                        replay_tensors["action_mask"],
                        replay_tensors["event_features"],
                        action_history=replay_tensors["action_history"],
                        action_history_mask=replay_tensors[
                            "action_history_mask"
                        ],
                        include_belief=False,
                    )
                    replay_log_probabilities = F.log_softmax(
                        replay_output.policy_logits, dim=-1
                    )
                    replay_log_probability = replay_log_probabilities.gather(
                        1, replay_action_tensor.unsqueeze(1)
                    ).squeeze(1)
                    objective_replay_loss = -(
                        replay_weight_tensor * replay_log_probability.float()
                    ).sum() / replay_weight_tensor.sum().clamp_min(1e-6)
                    objective_replay_updates += 1
                    objective_replay_update_credits -= (
                        objective_replay_batch_size
                    )
                loss = (
                    policy_loss
                    + 0.50 * value_loss
                    + kl_multiplier * champion_kl
                    + 0.25 * value_anchor_weight * value_anchor_loss
                    + objective_replay_weight * objective_replay_loss
                    - entropy_weight * entropy
                )
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                candidate.eval()
                if ema_model is not None:
                    _update_parameter_ema(ema_model, candidate, ema_decay)
                    ema_updates += 1
                values = {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "champion_kl": float(champion_kl.detach().cpu()),
                    "behaviour_kl": float(behaviour_kl.detach().cpu()),
                    "clip_fraction": float(
                        ((ratio - 1.0).abs() > ppo_clip_range)
                        .float().mean().detach().cpu()
                    ),
                    "entropy": float(entropy.detach().cpu()),
                    "objective_replay_loss": float(
                        objective_replay_loss.detach().cpu()
                    ),
                    "policy_selected_fraction": float(
                        policy_selected_tensor.float().mean().detach().cpu()
                    ),
                }
                for name, value in values.items():
                    totals_local[name] += value
                local_updates += 1
                ppo_updates += 1
                ppo_policy_samples += int(policy_selected_tensor.sum().item())
            if limited_epoch:
                continue
        return {
            name: total / max(1, local_updates)
            for name, total in totals_local.items()
        }

    def retain_ppg_examples(examples: Sequence[_PPOExample]) -> None:
        """Keep a bounded, rollout-balanced sample for later auxiliary work."""
        nonlocal ppg_samples_added
        if not ppg_enabled or not examples:
            return
        kept = min(len(examples), ppg_samples_per_rollout)
        indices = (
            np.arange(len(examples), dtype=np.int64)
            if kept == len(examples)
            else rng.choice(len(examples), size=kept, replace=False)
        )
        for index in indices:
            example = examples[int(index)]
            ppg_replay.append(
                _PPGExample(
                    observation=example.observation,
                    return_target=float(
                        example.return_target
                        if example.value_target is None
                        else example.value_target
                    ),
                )
            )
        ppg_samples_added += kept

    def optimise_ppg_phase() -> Dict[str, float]:
        """Distil the current policy while value trains the shared encoder."""
        nonlocal ppg_phases, ppg_updates, ema_updates
        if not ppg_enabled or len(ppg_replay) < ppg_minibatch_size:
            return {}
        phase_size = min(ppg_phase_samples, len(ppg_replay))
        selected_indices = rng.choice(
            len(ppg_replay), size=phase_size, replace=False
        )
        replay_rows = list(ppg_replay)
        selected = [replay_rows[int(index)] for index in selected_indices]

        # Freeze the *current* post-PPO policy over the diverse replay states.
        # Targets from collection time would unnecessarily pull the candidate
        # back through several older policies instead of protecting only the
        # policy that enters this auxiliary phase.
        targets: list[_PPGTargetExample] = []
        candidate.eval()
        for start in range(0, len(selected), ppg_minibatch_size):
            subset = selected[start:start + ppg_minibatch_size]
            frozen_batch = _collate_public_observations(subset)
            frozen_tensors = _vector_tensors(
                frozen_batch, device, training
            )
            with torch.inference_mode():
                frozen_output = candidate(
                    frozen_tensors["map_tokens"],
                    frozen_tensors["global_features"],
                    frozen_tensors["action_categorical"],
                    frozen_tensors["action_continuous"],
                    frozen_tensors["action_mask"],
                    frozen_tensors["event_features"],
                    action_history=frozen_tensors["action_history"],
                    action_history_mask=frozen_tensors[
                        "action_history_mask"
                    ],
                    include_belief=False,
                )
                frozen_logs = F.log_softmax(
                    frozen_output.policy_logits.float(), dim=-1
                ).cpu().numpy()
            for row, example in enumerate(subset):
                legal_count = int(
                    example.observation["action_features"].shape[0]
                )
                targets.append(
                    _PPGTargetExample(
                        observation=example.observation,
                        policy_log_probabilities=frozen_logs[
                            row, :legal_count
                        ].astype(np.float32, copy=True),
                        return_target=example.return_target,
                    )
                )

        local_totals = {"loss": 0.0, "value_loss": 0.0, "policy_kl": 0.0}
        local_updates = 0
        auxiliary_progress = tqdm(
            total=ppg_epochs * (
                (len(targets) + ppg_minibatch_size - 1)
                // ppg_minibatch_size
            ),
            desc=f"PPG auxiliary phase {ppg_phases + 1}",
            unit="update",
            dynamic_ncols=True,
            leave=False,
        )
        for _ in range(ppg_epochs):
            order = rng.permutation(len(targets))
            for start in range(0, len(order), ppg_minibatch_size):
                subset = [
                    targets[int(index)]
                    for index in order[start:start + ppg_minibatch_size]
                ]
                replay_batch, reference_logs, returns = _ppg_batch(subset)
                tensors = _vector_tensors(replay_batch, device, training)
                reference_tensor = torch.as_tensor(
                    reference_logs, dtype=torch.float32, device=device
                )
                return_tensor = torch.as_tensor(
                    returns, dtype=torch.float32, device=device
                )
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if cuda_amp
                    else nullcontext()
                )
                with autocast:
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
                    value_loss = F.smooth_l1_loss(
                        output.value.float(), return_tensor
                    )
                policy_kl = _masked_policy_kl(
                    reference_tensor,
                    output.policy_logits,
                    tensors["action_mask"],
                )
                loss = (
                    ppg_value_weight * value_loss
                    + ppg_policy_kl_weight * policy_kl
                )
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                candidate.eval()
                if ema_model is not None:
                    _update_parameter_ema(ema_model, candidate, ema_decay)
                    ema_updates += 1
                values = {
                    "loss": float(loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "policy_kl": float(policy_kl.detach().cpu()),
                }
                for name, value in values.items():
                    local_totals[name] += value
                    ppg_totals[name] += value
                local_updates += 1
                ppg_updates += 1
                auxiliary_progress.update(1)
                if local_updates == 1 or local_updates % 8 == 0:
                    auxiliary_progress.set_postfix(
                        value=f"{values['value_loss']:.4f}",
                        policy_kl=f"{values['policy_kl']:.5f}",
                    )
        auxiliary_progress.close()
        candidate.eval()
        ppg_phases += 1
        return {
            name: total / max(1, local_updates)
            for name, total in local_totals.items()
        }

    with tqdm(
        total=steps,
        desc=(
            f"direct population PPG ({device}, no MCTS)"
            if ppg_enabled
            else f"direct population PPO ({device}, no MCTS)"
        ),
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for step in range(steps):
            candidate.eval()
            tensors = _vector_tensors(batch, device, training)
            history_tokens_seen += int(
                np.asarray(batch["visible_event_mask"], dtype=bool).sum()
            )
            history_rows_seen += num_envs
            features = np.asarray(batch["action_features"], dtype=np.int64)
            legal = np.asarray(batch["action_mask"], dtype=bool)
            state = np.asarray(batch["state"], dtype=np.int32)
            players_before = state[:, 1]
            candidate_rows = np.flatnonzero(players_before == candidate_players)
            opponent_rows = np.flatnonzero(players_before != candidate_players)
            candidate_turns += int(candidate_rows.size)
            opponent_turns += int(opponent_rows.size)
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
                rollout_distribution = Categorical(
                    logits=output.policy_logits / ppo_rollout_temperature
                )
                action_rows = rollout_distribution.sample()
                old_log_probabilities = rollout_distribution.log_prob(action_rows)
                values_before = output.value.float()
            # ``sample`` is intentionally run under ``inference_mode`` because it
            # is behaviour-policy data, not part of the PPO gradient.  Opponent
            # turns replace a subset of these rows below, however, and PyTorch
            # forbids an in-place write to an inference tensor outside that
            # context.  Make an ordinary tensor at the boundary.
            action_rows = action_rows.clone()
            for opponent_index, opponent in enumerate(frozen):
                rows = opponent_rows[
                    opponent_indices[opponent_rows] == opponent_index
                ]
                if not rows.size:
                    continue
                with torch.inference_mode():
                    opponent_output = _model_rows(opponent, tensors, rows)
                    selected = opponent_output.policy_logits.argmax(dim=-1)
                row_tensor = torch.as_tensor(
                    rows, dtype=torch.long, device=device
                )
                action_rows.index_copy_(0, row_tensor, selected)
            selected_rows = action_rows.detach().cpu().numpy().astype(np.int64)
            if not legal[np.arange(num_envs), selected_rows].all():
                raise RuntimeError("raw PPO selected an illegal VectorGameEnv action")
            shaped = _visible_action_reward(
                features,
                selected_rows,
                np.asarray(batch["map_tokens"], dtype=np.int64),
                state,
                training.map_size,
            )
            action_ids = np.asarray(batch["action_id"], dtype=np.int32)[
                np.arange(num_envs), selected_rows
            ]
            next_batch = env.step(action_ids)
            terminal = np.asarray(next_batch["terminated"], dtype=bool)
            terminal_reward = np.asarray(next_batch["reward"], dtype=np.float32)
            local_progress = (step + 1) / max(1, steps)
            shaping_progress = (
                shaping_progress_start
                + (shaping_progress_end - shaping_progress_start)
                * local_progress
            )
            current_shaping_weight = (
                shaping_weight
                + (resolved_shaping_final_weight - shaping_weight)
                * shaping_progress
            )
            reward = current_shaping_weight * shaped + terminal_reward
            reward_total += float(reward.mean())
            shaping_weight_total += current_shaping_weight
            if objective_replay_weight > 0.0 and candidate_rows.size:
                success_weights = _objective_replay_success_weights(
                    features,
                    selected_rows,
                    np.asarray(batch["map_tokens"], dtype=np.int64),
                    state,
                    candidate_rows,
                    terminal,
                    terminal_reward,
                    training.map_size,
                )
                for env_id in candidate_rows:
                    history = objective_histories[int(env_id)]
                    history.append(
                        _objective_replay_snapshot(
                            batch, int(env_id), int(selected_rows[env_id])
                        )
                    )
                    success_weight = float(success_weights[env_id])
                    if success_weight <= 0.0:
                        continue
                    objective_replay_events += 1
                    for age, example in enumerate(reversed(history)):
                        objective_replay.append(
                            replace(
                                example,
                                weight=(
                                    success_weight
                                    * objective_replay_recency**age
                                ),
                            )
                        )
                        objective_replay_positions += 1
                        objective_replay_update_credits = min(
                            objective_replay_update_credits
                            + objective_replay_max_epochs,
                            len(objective_replay)
                            * objective_replay_max_epochs,
                        )
            for env_id in np.flatnonzero(terminal):
                objective_histories[int(env_id)].clear()
            with torch.inference_mode():
                next_tensors = _vector_tensors(next_batch, device, training)
                next_output = candidate(
                    next_tensors["map_tokens"],
                    next_tensors["global_features"],
                    next_tensors["action_categorical"],
                    next_tensors["action_continuous"],
                    next_tensors["action_mask"],
                    next_tensors["event_features"],
                    action_history=next_tensors["action_history"],
                    action_history_mask=next_tensors[
                        "action_history_mask"
                    ],
                    include_belief=False,
                )
                next_values = next_output.value.float().cpu().numpy()
            players_after = np.asarray(next_batch["state"], dtype=np.int32)[:, 1]
            signs = np.where(players_after == players_before, 1.0, -1.0)
            candidate_mask = np.zeros(num_envs, dtype=bool)
            candidate_mask[candidate_rows] = True
            old_logs = old_log_probabilities.float().cpu().numpy()
            old_values = values_before.cpu().numpy()
            for env_id in range(num_envs):
                example = None
                if candidate_mask[env_id]:
                    observation, compact_action_row = _ppo_snapshot(
                        batch, env_id, int(selected_rows[env_id])
                    )
                    example = _PPOExample(
                        observation=observation,
                        action_row=compact_action_row,
                        old_log_probability=float(old_logs[env_id]),
                        old_value=float(old_values[env_id]),
                    )
                    pending_candidate_examples += 1
                traces[env_id].append(
                    _PPOTransition(
                        value=float(old_values[env_id]),
                        reward=float(reward[env_id]),
                        next_value=float(next_values[env_id]),
                        sign=float(signs[env_id]),
                        terminal=bool(terminal[env_id]),
                        candidate_example=example,
                    )
                )

            episode_moves += 1
            for env_id in np.flatnonzero(terminal):
                if terminal_reward[env_id] != 0.0:
                    winner_player = int(players_before[env_id])
                    if terminal_reward[env_id] < 0.0:
                        winner_player = 1 - winner_player
                    candidate_outcome = float(
                        1.0
                        if winner_player == candidate_players[env_id]
                        else -1.0
                    )
                else:
                    winner_player = -1
                    candidate_outcome = 0.0
                # Finish only a real game here.  The return crosses every
                # opponent action stored in this slot before reaching its own
                # candidate decisions.
                finished_examples = _finish_ppo_trace(
                    traces[int(env_id)],
                    gamma=gamma,
                    gae_lambda=ppo_gae_lambda,
                    terminal_outcome=candidate_outcome,
                    terminal_advantage_weight=(
                        ppo_terminal_advantage_weight
                    ),
                )
                completed_examples.extend(finished_examples)
                pending_candidate_examples -= len(finished_examples)
                traces[int(env_id)].clear()
                completed_games += 1
                selected_opponent = int(opponent_indices[env_id])
                if terminal_reward[env_id] != 0.0:
                    candidate_won = candidate_outcome > 0.0
                    wins += int(candidate_won)
                    losses += int(not candidate_won)
                    score = float(candidate_won)
                else:
                    draws += 1
                    winner_player = -1
                    score = 0.5
                if game_callback is not None:
                    game_callback({
                        "game_index": completed_games - 1,
                        "map_seed": int(training.seed),
                        "map_seed_is_collection_seed": 1,
                        "vector_env_id": int(env_id),
                        "vector_episode": int(episode_numbers[env_id]),
                        "model_a_player": int(candidate_players[env_id]),
                        "winner_player": winner_player,
                        "score_a": score,
                        "moves": int(episode_moves[env_id]),
                        "opponent_model_index": selected_opponent,
                    })
                candidate_players[env_id] = 1 - candidate_players[env_id]
                opponent_indices[env_id] = int(
                    rng.choice(len(frozen), p=weights)
                )
                episode_moves[env_id] = 0
                episode_numbers[env_id] += 1

            rollout_metrics: Dict[str, float] = {}
            ppg_metrics: Dict[str, float] = {}
            if pending_candidate_examples > ppo_max_pending_examples:
                for trace in traces:
                    if not trace:
                        continue
                    truncated_examples = _finish_ppo_trace(
                        trace, gamma=gamma, gae_lambda=ppo_gae_lambda
                    )
                    completed_examples.extend(truncated_examples)
                    pending_candidate_examples -= len(truncated_examples)
                    trace.clear()
                    ppo_forced_truncations += 1
            if step + 1 == steps:
                # The final block boundary is a bootstrap, never a fake draw.
                for trace in traces:
                    if not trace:
                        continue
                    truncated_examples = _finish_ppo_trace(
                        trace, gamma=gamma, gae_lambda=ppo_gae_lambda
                    )
                    completed_examples.extend(truncated_examples)
                    pending_candidate_examples -= len(truncated_examples)
                    trace.clear()
                    ppo_forced_truncations += 1
            if completed_examples and (
                len(completed_examples) >= ppo_update_target
                or step + 1 == steps
            ):
                examples = completed_examples
                completed_examples = []
                ppo_samples += len(examples)
                ppo_rollouts += 1
                rollout_metrics = optimise_rollout(examples)
                retain_ppg_examples(examples)
                if ppo_rollouts % ppg_interval == 0:
                    ppg_metrics = optimise_ppg_phase()
                last_champion_kl = float(
                    rollout_metrics.get("champion_kl", last_champion_kl)
                )
                for name, value in rollout_metrics.items():
                    totals[name] += value

            batch = next_batch
            run.log(
                step=step + 1,
                environment_step=(step + 1) * num_envs,
                completed_games=completed_games,
                candidate_wins=wins,
                candidate_losses=losses,
                draws=draws,
                candidate_turns=candidate_turns,
                opponent_turns=opponent_turns,
                ppo_samples=ppo_samples,
                ppo_policy_samples=ppo_policy_samples,
                ppo_updates=ppo_updates,
                ppo_rollouts=ppo_rollouts,
                ppo_pending_examples=pending_candidate_examples,
                ppo_forced_truncations=ppo_forced_truncations,
                ppg_enabled=float(ppg_enabled),
                ppg_buffer_size=len(ppg_replay),
                ppg_samples_added=ppg_samples_added,
                ppg_phases=ppg_phases,
                ppg_updates=ppg_updates,
                ppg_update_this_step=float(bool(ppg_metrics)),
                objective_replay_events=objective_replay_events,
                objective_replay_positions=objective_replay_positions,
                objective_replay_size=len(objective_replay),
                objective_replay_updates=objective_replay_updates,
                objective_replay_update_credits=(
                    objective_replay_update_credits
                ),
                ppo_behaviour_kl_limits=float(ppo_behaviour_kl_limits),
                ppo_learning_rate=float(optimiser.param_groups[0]["lr"]),
                ema_enabled=float(ema_model is not None),
                ema_updates=ema_updates,
                ppo_update_this_step=float(bool(rollout_metrics)),
                ppo_last_champion_kl=last_champion_kl,
                reward=float(reward.mean()),
                shaping_weight=current_shaping_weight,
                **{
                    "ppo_" + name: value
                    for name, value in rollout_metrics.items()
                },
                **{
                    "ppg_" + name: value
                    for name, value in ppg_metrics.items()
                },
            )
            progress.update(1)
            if (step + 1) % 20 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    games=completed_games,
                    score=f"{(wins + 0.5 * draws) / max(1, completed_games):.3f}",
                    ppo_updates=ppo_updates,
                    ppg_phases=ppg_phases,
                    replay_updates=objective_replay_updates,
                    kl=f"{last_champion_kl:.5f}",
                    bkl=f"{float(rollout_metrics.get('behaviour_kl', 0.0)):.5f}",
                    kl_limits=ppo_behaviour_kl_limits,
                )

    divisor = max(1, ppo_rollouts)
    summary: Dict[str, float | str] = {
        "direct_population_algorithm": (
            "single_network_ppg" if ppg_enabled else "signed_ppo_gae"
        ),
        "direct_population_envs": float(num_envs),
        "direct_population_visible_action_history": float(
            visible_action_history
        ),
        "direct_population_mean_visible_history_tokens": float(
            history_tokens_seen / max(1, history_rows_seen)
        ),
        "direct_population_history_gate_norm": float(
            candidate.history_gate[-1].weight.detach().float().norm().cpu()
        ),
        "direct_population_steps": float(step + 1),
        "direct_population_environment_steps": float((step + 1) * num_envs),
        "direct_population_completed_games": float(completed_games),
        "direct_population_wins": float(wins),
        "direct_population_losses": float(losses),
        "direct_population_draws": float(draws),
        "direct_population_score": float(
            (wins + 0.5 * draws) / max(1, completed_games)
        ),
        "direct_population_candidate_turns": float(candidate_turns),
        "direct_population_opponent_turns": float(opponent_turns),
        "direct_population_ppo_samples": float(ppo_samples),
        "direct_population_ppo_policy_samples": float(ppo_policy_samples),
        "direct_population_ppo_updates": float(ppo_updates),
        "direct_population_ppo_rollouts": float(ppo_rollouts),
        "direct_population_ppo_pending_examples": float(
            pending_candidate_examples
        ),
        "direct_population_ppo_forced_truncations": float(
            ppo_forced_truncations
        ),
        "direct_population_ppg_enabled": float(ppg_enabled),
        "direct_population_ppg_policy_phase_value_head_only": float(
            ppg_enabled
        ),
        "direct_population_ppg_interval": float(ppg_interval),
        "direct_population_ppg_buffer_capacity": float(
            ppg_buffer_capacity
        ),
        "direct_population_ppg_buffer_size": float(len(ppg_replay)),
        "direct_population_ppg_samples_added": float(ppg_samples_added),
        "direct_population_ppg_phase_samples": float(ppg_phase_samples),
        "direct_population_ppg_phases": float(ppg_phases),
        "direct_population_ppg_updates": float(ppg_updates),
        "direct_population_ppg_loss": float(
            ppg_totals["loss"] / max(1, ppg_updates)
        ),
        "direct_population_ppg_value_loss": float(
            ppg_totals["value_loss"] / max(1, ppg_updates)
        ),
        "direct_population_ppg_policy_kl": float(
            ppg_totals["policy_kl"] / max(1, ppg_updates)
        ),
        "direct_population_ppg_policy_kl_weight": float(
            ppg_policy_kl_weight
        ),
        "direct_population_ppg_value_weight": float(ppg_value_weight),
        "direct_population_objective_replay_events": float(
            objective_replay_events
        ),
        "direct_population_objective_replay_positions": float(
            objective_replay_positions
        ),
        "direct_population_objective_replay_size": float(
            len(objective_replay)
        ),
        "direct_population_objective_replay_updates": float(
            objective_replay_updates
        ),
        "direct_population_objective_replay_effective_epochs": float(
            objective_replay_updates * objective_replay_batch_size
            / max(1, objective_replay_positions)
        ),
        "direct_population_objective_replay_weight": float(
            objective_replay_weight
        ),
        "direct_population_ppo_update_target": float(ppo_update_target),
        "direct_population_ppo_max_pending_examples": float(
            ppo_max_pending_examples
        ),
        "direct_population_ppo_behaviour_kl_limits": float(
            ppo_behaviour_kl_limits
        ),
        "direct_population_ppo_learning_rate": float(
            optimiser.param_groups[0]["lr"]
        ),
        "direct_population_ppo_target_kl": float(ppo_target_kl),
        "direct_population_ppo_hard_kl": float(ppo_hard_kl),
        "direct_population_ppo_gamma": float(gamma),
        "direct_population_ppo_terminal_advantage_weight": float(
            ppo_terminal_advantage_weight
        ),
        "direct_population_ppo_top_advantage_fraction": float(
            ppo_top_advantage_fraction
        ),
        "direct_population_ppo_rollout_temperature": float(
            ppo_rollout_temperature
        ),
        "direct_population_ppo_reward": reward_total / max(1, step + 1),
        "direct_population_shaping_weight_start": float(shaping_weight),
        "direct_population_shaping_weight_final": float(
            resolved_shaping_final_weight
        ),
        "direct_population_shaping_weight_mean": float(
            shaping_weight_total / max(1, step + 1)
        ),
        "direct_population_ema_enabled": float(ema_model is not None),
        "direct_population_ema_decay": float(
            ema_decay if ema_model is not None else 0.0
        ),
        "direct_population_ema_updates": float(ema_updates),
        **{
            "direct_population_ppo_" + name: value / divisor
            for name, value in totals.items()
        },
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary


def evaluate_direct_population_matches(
    training: TrainingConfig,
    candidate: PolicyValueBeliefNet,
    opponent_models: Sequence[PolicyValueBeliefNet],
    opponent_weights: Sequence[float],
    *,
    games: int,
    gate_seed: int,
    visible_action_history: int = 32,
    game_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    candidate_label: str = "pretendent",
    opponent_labels: Sequence[str] | None = None,
    num_threads: int | None = None,
    max_actions: int = 1024,
) -> Dict[str, Any]:
    """Evaluate one raw network on the same paired arena for a population."""
    if not opponent_models:
        raise ValueError("direct population evaluation requires an opponent")
    if games < 2 or games % 2:
        raise ValueError("direct population games must be positive and even")
    weights = np.asarray(opponent_weights, dtype=np.float64)
    if (
        weights.shape != (len(opponent_models),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("opponent_weights must be finite and positive")
    weights /= weights.sum()
    resolved_opponent_labels = (
        [f"model B {index + 1}" for index in range(len(opponent_models))]
        if opponent_labels is None
        else [str(label) for label in opponent_labels]
    )
    if len(resolved_opponent_labels) != len(opponent_models):
        raise ValueError("opponent_labels must match opponent_models")
    results: list[Dict[str, Any]] = []
    device = next(candidate.parameters()).device
    for opponent_index, opponent in enumerate(opponent_models):
        match = evaluate_direct_matches(
            candidate,
            opponent,
            device,
            replace(
                training,
                seed=(
                    (gate_seed - 1 + opponent_index * 1_000_003)
                    % (2**31 - 1)
                    + 1
                ),
            ),
            games,
            game_callback=(
                None
                if game_callback is None
                else lambda game, selected=opponent_index: game_callback(
                    selected, game
                )
            ),
            model_a_label=candidate_label,
            model_b_label=resolved_opponent_labels[opponent_index],
            visible_action_history=visible_action_history,
            num_threads=num_threads,
            max_actions=max_actions,
        )
        results.append(match)
    return {
        "model_a_score": float(sum(
            weight * float(row["model_a_score"])
            for weight, row in zip(weights, results)
        )),
        "worst_score": float(min(
            float(row["model_a_score"]) for row in results
        )),
        "model_a_city_captures_per_game": float(sum(
            weight * float(row["model_a_city_captures_per_game"])
            for weight, row in zip(weights, results)
        )),
        "completion_rate": float(sum(
            weight * float(row["completion_rate"])
            for weight, row in zip(weights, results)
        )),
        "opponents": results,
    }


def _population_objective_progress(
    city_captures_per_game: float,
    completion_rate: float,
) -> float:
    """Combine auxiliary objectives without making either a hard gate."""
    return float(city_captures_per_game) + float(completion_rate)


def _population_block_acceptance(
    *,
    baseline: bool,
    weighted_score: float,
    worst_score: float,
    objective_progress: float,
    best_score: float,
    best_objective_progress: float,
    historical_score_floor: float,
    score_tolerance: float,
    objective_tolerance: float,
) -> tuple[bool, str]:
    """Select a population checkpoint using strength as the primary metric.

    A Pareto rule over score, worst opponent, captures and completion made a
    useful policy update almost impossible to accept: improving a win often
    changes the route and therefore a secondary visible-objective metric.
    The hard safety property is the historical floor.  Above it, score wins;
    equally strong policies can use the visible objectives as a tie-breaker.
    """
    if baseline:
        return True, "baseline"
    if worst_score < historical_score_floor:
        return False, "historical_floor"
    if weighted_score > best_score + 1e-8:
        return True, "score_improved"
    if weighted_score < best_score - score_tolerance:
        return False, "score_regression"
    if objective_progress >= best_objective_progress - objective_tolerance:
        return True, "score_tied_or_close"
    return False, "objective_regression"


def _population_block_selects_checkpoint(
    *,
    baseline: bool,
    weighted_score: float,
    objective_progress: float,
    best_score: float,
    best_objective_progress: float,
) -> tuple[bool, str]:
    """Return whether a safe block becomes the final selected checkpoint.

    ``_population_block_acceptance`` intentionally has a small tolerance so
    noisy, nearly-equal gate results do not terminate the curriculum early.
    That tolerance must *not* let a slightly weaker checkpoint overwrite the
    real high-water mark.  A checkpoint is therefore selected only for a
    strict score improvement, or for a score tie with better visible progress.
    """
    if baseline:
        return True, "baseline"
    if weighted_score > best_score + 1e-8:
        return True, "score_improved"
    if (
        abs(weighted_score - best_score) <= 1e-8
        and objective_progress > best_objective_progress + 1e-8
    ):
        return True, "score_tie_objective_improved"
    return False, "safe_but_not_best"


def train_direct_population_curriculum(
    training: TrainingConfig,
    model_config: ModelConfig,
    candidate: PolicyValueBeliefNet,
    champion: PolicyValueBeliefNet,
    opponent_models: Sequence[PolicyValueBeliefNet],
    opponent_weights: Sequence[float],
    *,
    num_envs: int,
    minimum_steps: int,
    maximum_steps: int,
    block_steps: int,
    num_threads: int | None,
    max_actions: int,
    visible_action_history: int = 32,
    gate_games: int,
    gate_seed: int,
    gate_score: float,
    historical_score_floor: float,
    readiness_city_captures_per_game: float,
    readiness_completion_rate: float,
    readiness_patience: int,
    policy_anchor_weight: float,
    value_anchor_weight: float,
    opponent_policy_weight: float,
    teacher_weight: float,
    shaping_weight: float,
    shaping_final_weight: float = 0.0,
    greedy_advantage_weight: float = 0.25,
    regression_patience: int = 2,
    score_tolerance: float = 0.02,
    objective_tolerance: float = 0.05,
    diagnostic_gate_games: int | None = 20,
    ppo_rollout_steps: int = 32,
    ppo_epochs: int = 2,
    ppo_minibatch_size: int = 256,
    ppo_clip_range: float = 0.15,
    ppo_gamma: float = 1.0,
    ppo_gae_lambda: float = 0.95,
    ppo_terminal_advantage_weight: float = 0.75,
    ppo_top_advantage_fraction: float = 0.50,
    ppo_target_kl: float = 0.015,
    ppo_hard_kl: float = 0.030,
    ppo_rollout_temperature: float = 0.85,
    ppo_max_pending_examples: int = 32_768,
    ppo_min_learning_rate: float = 1.0e-5,
    ppo_lr_backoff: float = 0.5,
    objective_replay_horizon: int = 16,
    objective_replay_capacity: int = 512,
    objective_replay_batch_size: int = 32,
    objective_replay_interval: int = 16,
    objective_replay_weight: float = 0.35,
    objective_replay_recency: float = 0.92,
    objective_replay_max_epochs: int = 4,
    ppg_enabled: bool = True,
    ppg_interval: int = 8,
    ppg_buffer_capacity: int = 8_192,
    ppg_samples_per_rollout: int = 512,
    ppg_phase_samples: int = 4_096,
    ppg_epochs: int = 2,
    ppg_minibatch_size: int = 256,
    ppg_policy_kl_weight: float = 1.0,
    ppg_value_weight: float = 1.0,
    ema_enabled: bool = True,
    ema_decay: float = 0.999,
    gate_game_callback: Callable[[str, int, int, Mapping[str, Any]], None]
    | None = None,
    training_game_callback: Callable[[int, Mapping[str, Any]], None]
    | None = None,
    candidate_label: str = "pretendent",
    opponent_labels: Sequence[str] | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, Any]:
    """Advance raw population play only while held-out population skill grows."""
    if minimum_steps < 0 or maximum_steps < minimum_steps:
        raise ValueError("invalid direct population step range")
    if not 0 <= visible_action_history <= 1024:
        raise ValueError("visible_action_history must be between 0 and 1024")
    if minimum_steps and block_steps < 1:
        raise ValueError("block_steps must be positive")
    if gate_games < 2 or gate_games % 2:
        raise ValueError("direct population gate games must be positive and even")
    resolved_diagnostic_gate_games = (
        gate_games if diagnostic_gate_games is None else int(diagnostic_gate_games)
    )
    if (
        resolved_diagnostic_gate_games < 2
        or resolved_diagnostic_gate_games % 2
        or resolved_diagnostic_gate_games > gate_games
    ):
        raise ValueError(
            "diagnostic gate games must be positive, even and no larger than the full gate"
        )
    if readiness_patience < 1:
        raise ValueError("readiness_patience must be positive")
    if regression_patience < 1:
        raise ValueError("regression_patience must be positive")
    if not 0.0 <= greedy_advantage_weight <= 1.0:
        raise ValueError(
            "greedy_advantage_weight must be between zero and one"
        )
    if score_tolerance < 0.0 or objective_tolerance < 0.0:
        raise ValueError("population acceptance tolerances must not be negative")
    if (
        ppo_rollout_steps < 1
        or ppo_epochs < 1
        or ppo_minibatch_size < 1
        or ppo_max_pending_examples < ppo_minibatch_size
    ):
        raise ValueError("PPO rollout, epoch and minibatch sizes must be positive")
    if not 0.0 < ppo_clip_range <= 1.0:
        raise ValueError("ppo_clip_range must be in (0, 1]")
    if not 0.0 < ppo_gamma <= 1.0:
        raise ValueError("ppo_gamma must be in (0, 1]")
    if not 0.0 <= ppo_gae_lambda <= 1.0:
        raise ValueError("ppo_gae_lambda must be between zero and one")
    if not 0.0 <= ppo_terminal_advantage_weight <= 1.0:
        raise ValueError(
            "ppo_terminal_advantage_weight must be between zero and one"
        )
    if not 0.0 < ppo_top_advantage_fraction <= 1.0:
        raise ValueError("ppo_top_advantage_fraction must be in (0, 1]")
    if not 0.0 < ppo_target_kl <= ppo_hard_kl:
        raise ValueError("PPO KL thresholds must satisfy 0 < target <= hard")
    if not 0.1 <= ppo_rollout_temperature <= 2.0:
        raise ValueError("ppo_rollout_temperature must be between 0.1 and 2.0")
    if ppo_min_learning_rate <= 0.0:
        raise ValueError("ppo_min_learning_rate must be positive")
    if not 0.0 < ppo_lr_backoff < 1.0:
        raise ValueError("ppo_lr_backoff must be between zero and one")
    for name, value in {
        "objective_replay_weight": objective_replay_weight,
        "objective_replay_recency": objective_replay_recency,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    for name, value in {
        "objective_replay_horizon": objective_replay_horizon,
        "objective_replay_capacity": objective_replay_capacity,
        "objective_replay_batch_size": objective_replay_batch_size,
        "objective_replay_interval": objective_replay_interval,
        "objective_replay_max_epochs": objective_replay_max_epochs,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    for name, value in {
        "ppg_interval": ppg_interval,
        "ppg_buffer_capacity": ppg_buffer_capacity,
        "ppg_samples_per_rollout": ppg_samples_per_rollout,
        "ppg_phase_samples": ppg_phase_samples,
        "ppg_epochs": ppg_epochs,
        "ppg_minibatch_size": ppg_minibatch_size,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if ppg_enabled and (
        ppg_buffer_capacity < ppg_phase_samples
        or ppg_phase_samples < ppg_minibatch_size
        or ppg_buffer_capacity < ppg_samples_per_rollout
    ):
        raise ValueError("invalid PPG buffer or minibatch sizes")
    if ppg_policy_kl_weight < 0.0 or ppg_value_weight < 0.0:
        raise ValueError("PPG loss weights must not be negative")
    if not 0.0 <= shaping_final_weight <= 1.0:
        raise ValueError("shaping_final_weight must be between zero and one")
    if ema_enabled and not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    weights = np.asarray(opponent_weights, dtype=np.float64)
    if (
        weights.shape != (len(opponent_models),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("opponent_weights must be finite and positive")
    weights /= weights.sum()
    run = run or TrainingRun.create("runs", "direct_population_curriculum")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "num_envs": num_envs,
            "minimum_steps": minimum_steps,
            "maximum_steps": maximum_steps,
            "block_steps": block_steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "visible_action_history": visible_action_history,
            "gate_games": gate_games,
            "diagnostic_gate_games": resolved_diagnostic_gate_games,
            "gate_seed": gate_seed,
            "gate_score": gate_score,
            "historical_score_floor": historical_score_floor,
            "readiness_city_captures_per_game": (
                readiness_city_captures_per_game
            ),
            "readiness_completion_rate": readiness_completion_rate,
            "readiness_patience": readiness_patience,
            "regression_patience": regression_patience,
            "score_tolerance": score_tolerance,
            "objective_tolerance": objective_tolerance,
            "opponent_weights": weights.tolist(),
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "opponent_policy_weight": opponent_policy_weight,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
            "shaping_final_weight": shaping_final_weight,
            "greedy_advantage_weight": greedy_advantage_weight,
            "algorithm": (
                "single_network_ppg" if ppg_enabled else "signed_ppo_gae"
            ),
            "ppo_rollout_steps": ppo_rollout_steps,
            "ppo_epochs": ppo_epochs,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_range": ppo_clip_range,
            "ppo_gamma": ppo_gamma,
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_terminal_advantage_weight": ppo_terminal_advantage_weight,
            "ppo_top_advantage_fraction": ppo_top_advantage_fraction,
            "ppo_target_kl": ppo_target_kl,
            "ppo_hard_kl": ppo_hard_kl,
            "ppo_rollout_temperature": ppo_rollout_temperature,
            "ppo_max_pending_examples": ppo_max_pending_examples,
            "ppo_min_learning_rate": ppo_min_learning_rate,
            "ppo_lr_backoff": ppo_lr_backoff,
            "objective_replay_horizon": objective_replay_horizon,
            "objective_replay_capacity": objective_replay_capacity,
            "objective_replay_batch_size": objective_replay_batch_size,
            "objective_replay_interval": objective_replay_interval,
            "objective_replay_weight": objective_replay_weight,
            "objective_replay_recency": objective_replay_recency,
            "objective_replay_max_epochs": objective_replay_max_epochs,
            "ppg_enabled": ppg_enabled,
            "ppg_interval": ppg_interval,
            "ppg_buffer_capacity": ppg_buffer_capacity,
            "ppg_samples_per_rollout": ppg_samples_per_rollout,
            "ppg_phase_samples": ppg_phase_samples,
            "ppg_epochs": ppg_epochs,
            "ppg_minibatch_size": ppg_minibatch_size,
            "ppg_policy_kl_weight": ppg_policy_kl_weight,
            "ppg_value_weight": ppg_value_weight,
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay if ema_enabled else None,
        },
    )
    parent_state = model_state_on_cpu(candidate)
    best_state = model_state_on_cpu(candidate)
    ema_model: PolicyValueBeliefNet | None = None
    best_ema_state: Dict[str, torch.Tensor] | None = None
    if ema_enabled:
        ema_model = PolicyValueBeliefNet(model_config)
        restore_model_state(ema_model, parent_state)
        best_ema_state = model_state_on_cpu(ema_model)
    best_score = 0.0
    best_worst_score = 0.0
    best_capture_rate = 0.0
    best_completion_rate = 0.0
    best_objective_progress = 0.0
    readiness_streak = 0
    ready = False
    best_meets_readiness = False
    accepted_blocks = selected_blocks = rollback_blocks = 0
    consecutive_regressions = 0
    maximum_consecutive_regressions = 0
    gate_rows: list[Dict[str, Any]] = []

    def gate(
        label: str,
        block: int,
        baseline: bool = False,
        *,
        games: int = resolved_diagnostic_gate_games,
    ) -> bool:
        nonlocal best_state, best_score, best_worst_score
        nonlocal best_ema_state
        nonlocal best_capture_rate, best_completion_rate
        nonlocal best_objective_progress
        nonlocal readiness_streak, ready, accepted_blocks, selected_blocks
        nonlocal rollback_blocks
        nonlocal best_meets_readiness
        nonlocal consecutive_regressions, maximum_consecutive_regressions
        evaluation = evaluate_direct_population_matches(
            training,
            candidate,
            opponent_models,
            weights,
            games=games,
            gate_seed=gate_seed,
            visible_action_history=visible_action_history,
            game_callback=(
                None
                if gate_game_callback is None
                else lambda opponent_index, game: gate_game_callback(
                    label, block, opponent_index, game
                )
            ),
            candidate_label=candidate_label,
            opponent_labels=opponent_labels,
            num_threads=num_threads,
            max_actions=max_actions,
        )
        results = evaluation["opponents"]
        weighted_score = float(evaluation["model_a_score"])
        worst_score = float(evaluation["worst_score"])
        capture_rate = float(
            evaluation["model_a_city_captures_per_game"]
        )
        completion_rate = float(evaluation["completion_rate"])
        objective_progress = _population_objective_progress(
            capture_rate, completion_rate
        )
        accepted, acceptance_reason = _population_block_acceptance(
            baseline=baseline,
            weighted_score=weighted_score,
            worst_score=worst_score,
            objective_progress=objective_progress,
            best_score=best_score,
            best_objective_progress=best_objective_progress,
            historical_score_floor=historical_score_floor,
            score_tolerance=score_tolerance,
            objective_tolerance=objective_tolerance,
        )
        selected, selection_reason = _population_block_selects_checkpoint(
            baseline=baseline,
            weighted_score=weighted_score,
            objective_progress=objective_progress,
            best_score=best_score,
            best_objective_progress=best_objective_progress,
        )
        if not accepted:
            selected = False
            selection_reason = "not_accepted"
        if accepted:
            if not baseline:
                accepted_blocks += 1
            if selected:
                best_state = model_state_on_cpu(candidate)
                if ema_model is not None:
                    best_ema_state = model_state_on_cpu(ema_model)
                best_score = weighted_score
                best_worst_score = worst_score
                best_capture_rate = capture_rate
                best_completion_rate = completion_rate
                best_objective_progress = objective_progress
                if not baseline:
                    selected_blocks += 1
            elif not baseline:
                # A 20-game diagnostic only resolves score in coarse steps.
                # Keep a safe near-tie as the *working* policy so several
                # individually too-small improvements can accumulate.  The
                # strict high-water snapshot remains available for the final
                # held-out selection, and an actual regression still rolls
                # back in the branch below.
                selection_reason = "safe_continue_accumulating"
            if baseline:
                # Baseline protects the input checkpoint but does not count as
                # a successful population-training block.
                best_meets_readiness = False
                readiness_streak = 0
                ready = False
            elif selected:
                meets = (
                    weighted_score >= gate_score
                    and worst_score >= historical_score_floor
                    and capture_rate >= readiness_city_captures_per_game
                    and completion_rate >= readiness_completion_rate
                )
                best_meets_readiness = meets
                readiness_streak = readiness_streak + 1 if meets else 0
                ready = readiness_streak >= readiness_patience
            consecutive_regressions = 0
        else:
            restore_model_state(candidate, best_state)
            if ema_model is not None and best_ema_state is not None:
                restore_model_state(ema_model, best_ema_state)
            rollback_blocks += 1
            consecutive_regressions += 1
            maximum_consecutive_regressions = max(
                maximum_consecutive_regressions,
                consecutive_regressions,
            )
        row = {
            "stage": label,
            "block": float(block),
            "weighted_score": weighted_score,
            "worst_score": worst_score,
            "city_captures_per_game": capture_rate,
            "completion_rate": completion_rate,
            "objective_progress": objective_progress,
            "accepted": float(accepted),
            "acceptance_reason": acceptance_reason,
            "selected": float(selected),
            "selection_reason": selection_reason,
            "ready": float(ready),
            "best_meets_readiness": float(best_meets_readiness),
            "readiness_streak": float(readiness_streak),
            "consecutive_regressions": float(consecutive_regressions),
            "games": float(games),
            "opponents": results,
        }
        gate_rows.append(row)
        run.log(
            direct_population_gate_score=weighted_score,
            direct_population_gate_worst=worst_score,
            direct_population_gate_captures=capture_rate,
            direct_population_gate_completion=completion_rate,
            direct_population_gate_objective_progress=objective_progress,
            direct_population_gate_accepted=float(accepted),
            direct_population_gate_acceptance_reason=acceptance_reason,
            direct_population_gate_selected=float(selected),
            direct_population_gate_selection_reason=selection_reason,
            direct_population_ready=float(ready),
            direct_population_best_meets_readiness=float(
                best_meets_readiness
            ),
            direct_population_consecutive_regressions=float(
                consecutive_regressions
            ),
            direct_population_gate_games=float(games),
        )
        if not baseline:
            print(
                "direct population diagnostic | "
                f"block={block} score={weighted_score:.4f} "
                f"worst={worst_score:.4f} accepted={int(accepted)} "
                f"selection={selection_reason}",
                flush=True,
            )
        return accepted

    gate("baseline", -1, baseline=True)
    completed = 0
    block = 0
    early_stopped = False
    early_stop_reason = ""
    training_rows: list[Dict[str, Any]] = []
    current_ppo_learning_rate = float(training.learning_rate)
    while completed < maximum_steps and (
        completed < minimum_steps
        or not ready
        or 0 < consecutive_regressions < regression_patience
    ):
        block += 1
        current_steps = min(block_steps, maximum_steps - completed)
        block_training = replace(
            training,
            seed=(training.seed - 1 + block * 2_000_003) % (2**31 - 1) + 1,
            learning_rate=current_ppo_learning_rate,
        )
        result = train_vector_population(
            block_training,
            model_config,
            candidate,
            opponent_models,
            opponent_weights,
            num_envs=num_envs,
            steps=current_steps,
            num_threads=num_threads,
            max_actions=max_actions,
            anchor_model=champion,
            visible_action_history=visible_action_history,
            policy_anchor_weight=policy_anchor_weight,
            value_anchor_weight=value_anchor_weight,
            opponent_policy_weight=opponent_policy_weight,
            teacher_weight=teacher_weight,
            shaping_weight=shaping_weight,
            shaping_final_weight=shaping_final_weight,
            shaping_progress_start=completed / max(1, maximum_steps),
            shaping_progress_end=(completed + current_steps)
            / max(1, maximum_steps),
            greedy_advantage_weight=greedy_advantage_weight,
            gamma=ppo_gamma,
            ppo_rollout_steps=ppo_rollout_steps,
            ppo_epochs=ppo_epochs,
            ppo_minibatch_size=ppo_minibatch_size,
            ppo_clip_range=ppo_clip_range,
            ppo_gae_lambda=ppo_gae_lambda,
            ppo_terminal_advantage_weight=ppo_terminal_advantage_weight,
            ppo_top_advantage_fraction=ppo_top_advantage_fraction,
            ppo_target_kl=ppo_target_kl,
            ppo_hard_kl=ppo_hard_kl,
            ppo_min_learning_rate=ppo_min_learning_rate,
            ppo_lr_backoff=ppo_lr_backoff,
            ppo_rollout_temperature=ppo_rollout_temperature,
            ppo_max_pending_examples=ppo_max_pending_examples,
            objective_replay_horizon=objective_replay_horizon,
            objective_replay_capacity=objective_replay_capacity,
            objective_replay_batch_size=objective_replay_batch_size,
            objective_replay_interval=objective_replay_interval,
            objective_replay_weight=objective_replay_weight,
            objective_replay_recency=objective_replay_recency,
            objective_replay_max_epochs=objective_replay_max_epochs,
            ppg_enabled=ppg_enabled,
            ppg_interval=ppg_interval,
            ppg_buffer_capacity=ppg_buffer_capacity,
            ppg_samples_per_rollout=ppg_samples_per_rollout,
            ppg_phase_samples=ppg_phase_samples,
            ppg_epochs=ppg_epochs,
            ppg_minibatch_size=ppg_minibatch_size,
            ppg_policy_kl_weight=ppg_policy_kl_weight,
            ppg_value_weight=ppg_value_weight,
            ema_model=ema_model,
            ema_decay=ema_decay,
            game_callback=(
                None
                if training_game_callback is None
                else lambda game, selected_block=block: training_game_callback(
                    selected_block, game
                )
            ),
            run=run.child(f"block_{block:03d}"),
        )
        training_rows.append(result)
        completed += current_steps
        adapted_learning_rate = float(
            result.get(
                "direct_population_ppo_learning_rate",
                current_ppo_learning_rate,
            )
        )
        if adapted_learning_rate < current_ppo_learning_rate - 1e-12:
            previous_learning_rate = current_ppo_learning_rate
            current_ppo_learning_rate = adapted_learning_rate
            run.log(
                direct_population_ppo_adaptive_kl_backoff=1.0,
                direct_population_ppo_behaviour_kl_limits=float(
                    result.get("direct_population_ppo_behaviour_kl_limits", 0.0)
                ),
                direct_population_ppo_previous_learning_rate=(
                    previous_learning_rate
                ),
                direct_population_ppo_learning_rate=current_ppo_learning_rate,
            )
        accepted = gate("population", block)
        if (
            completed >= minimum_steps
            and not accepted
            and consecutive_regressions >= regression_patience
        ):
            early_stopped = True
            early_stop_reason = "persistent_population_regression"
            break

    def states_match(
        first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> bool:
        return first.keys() == second.keys() and all(
            torch.equal(first[name], second[name]) for name in first
        )

    # The working state may hold several small, safe gains that a 20-game
    # diagnostic could not resolve.  The high-water state protects against
    # drift.  Compare both (and their EMA counterparts) only on the final
    # held-out arena, then keep the strongest safe policy.
    working_state = model_state_on_cpu(candidate)
    final_variants: list[tuple[str, Mapping[str, torch.Tensor], str]] = [
        ("working_raw", working_state, candidate_label),
    ]
    if not states_match(best_state, working_state):
        final_variants.append(
            ("best_raw", best_state, f"{candidate_label} best")
        )
    if ema_model is not None:
        working_ema_state = model_state_on_cpu(ema_model)
        final_variants.append(
            ("working_ema", working_ema_state, f"{candidate_label} EMA")
        )
        if (
            best_ema_state is not None
            and not states_match(best_ema_state, working_ema_state)
        ):
            final_variants.append(
                ("best_ema", best_ema_state, f"{candidate_label} best EMA")
            )

    final_candidates: list[Dict[str, Any]] = []
    for label, state, label_for_log in final_variants:
        restore_model_state(candidate, state)
        evaluation = evaluate_direct_population_matches(
            training,
            candidate,
            opponent_models,
            weights,
            games=gate_games,
            gate_seed=gate_seed,
            visible_action_history=visible_action_history,
            game_callback=(
                None
                if gate_game_callback is None
                else lambda opponent_index, game, stage=label: gate_game_callback(
                    f"final_{stage}", block, opponent_index, game
                )
            ),
            candidate_label=label_for_log,
            opponent_labels=opponent_labels,
            num_threads=num_threads,
            max_actions=max_actions,
        )
        score = float(evaluation["model_a_score"])
        worst_score = float(evaluation["worst_score"])
        final_candidates.append(
            {
                "label": label,
                "state": state,
                "evaluation": evaluation,
                "score": score,
                "worst_score": worst_score,
                "safe": score >= gate_score
                and worst_score >= historical_score_floor,
            }
        )

    raw_candidate = final_candidates[0]
    raw_full_evaluation = raw_candidate["evaluation"]
    raw_full_score = float(raw_candidate["score"])
    raw_full_worst_score = float(raw_candidate["worst_score"])
    ema_candidate = next(
        (
            item
            for item in final_candidates
            if item["label"] == "working_ema"
        ),
        None,
    )
    ema_full_score = (
        float(ema_candidate["score"]) if ema_candidate is not None else 0.0
    )
    ema_full_worst_score = (
        float(ema_candidate["worst_score"])
        if ema_candidate is not None
        else 0.0
    )
    select_ema = False
    safe_final_candidates = [
        item for item in final_candidates if bool(item["safe"])
    ]
    if safe_final_candidates:
        chosen_final = max(
            safe_final_candidates,
            key=lambda item: (
                float(item["score"]),
                float(item["worst_score"]),
                item["label"],
            ),
        )
        restore_model_state(candidate, chosen_final["state"])
        full_evaluation = chosen_final["evaluation"]
        full_score = float(chosen_final["score"])
        full_worst_score = float(chosen_final["worst_score"])
        full_safe = True
        full_selection = str(chosen_final["label"])
        select_ema = "ema" in full_selection
    else:
        full_evaluation = raw_full_evaluation
        full_score = raw_full_score
        full_worst_score = raw_full_worst_score
        full_safe = False
        full_selection = "rejected"
    print(
        "direct population final candidates | "
        + ", ".join(
            f"{item['label']}={float(item['score']):.4f}"
            f"/worst={float(item['worst_score']):.4f}"
            for item in final_candidates
        )
        + f" | selected={full_selection}",
        flush=True,
    )
    full_row = {
        "stage": "final",
        "block": float(block),
        "games": float(gate_games),
        "weighted_score": full_score,
        "worst_score": full_worst_score,
        "city_captures_per_game": float(
            full_evaluation["model_a_city_captures_per_game"]
        ),
        "completion_rate": float(full_evaluation["completion_rate"]),
        "selected": float(full_safe),
        "accepted": float(full_safe),
        "selection_reason": (
            f"full_gate_{full_selection}"
            if full_safe
            else "full_gate_rejected"
        ),
        "raw_score": raw_full_score,
        "raw_worst_score": raw_full_worst_score,
        "ema_score": ema_full_score if ema_model is not None else None,
        "ema_worst_score": (
            ema_full_worst_score if ema_model is not None else None
        ),
        "candidates": [
            {
                "label": str(item["label"]),
                "score": float(item["score"]),
                "worst_score": float(item["worst_score"]),
                "safe": float(bool(item["safe"])),
            }
            for item in final_candidates
        ],
        "opponents": full_evaluation["opponents"],
    }
    gate_rows.append(full_row)
    run.log(
        direct_population_full_gate_score=full_score,
        direct_population_full_gate_worst=full_worst_score,
        direct_population_full_gate_selected=float(full_safe),
        direct_population_full_gate_selected_ema=float(select_ema),
        direct_population_full_gate_raw_score=raw_full_score,
        direct_population_full_gate_ema_score=(
            ema_full_score if ema_model is not None else None
        ),
        direct_population_full_gate_candidates=float(len(final_candidates)),
        direct_population_full_gate_games=float(gate_games),
    )
    if full_safe:
        best_score = full_score
        best_worst_score = full_worst_score
        best_capture_rate = float(
            full_evaluation["model_a_city_captures_per_game"]
        )
        best_completion_rate = float(full_evaluation["completion_rate"])
    else:
        # Diagnostics decide which experiment deserves the expensive full
        # gate; only that full gate may retain a challenger.
        restore_model_state(candidate, parent_state)
        ready = False
        best_meets_readiness = False
    summary: Dict[str, Any] = {
        "direct_population_minimum_steps": float(minimum_steps),
        "direct_population_maximum_steps": float(maximum_steps),
        "direct_population_steps": float(completed),
        "direct_population_blocks": float(block),
        "direct_population_ready": float(ready),
        "direct_population_best_meets_readiness": float(
            best_meets_readiness
        ),
        "direct_population_best_score": float(best_score),
        "direct_population_best_worst_score": float(best_worst_score),
        "direct_population_best_city_captures_per_game": float(best_capture_rate),
        "direct_population_best_completion_rate": float(best_completion_rate),
        "direct_population_best_objective_progress": float(
            best_objective_progress
        ),
        "direct_population_score_tolerance": float(score_tolerance),
        "direct_population_objective_tolerance": float(objective_tolerance),
        "direct_population_greedy_advantage_weight": float(
            greedy_advantage_weight
        ),
        "direct_population_algorithm": (
            "single_network_ppg" if ppg_enabled else "signed_ppo_gae"
        ),
        "direct_population_diagnostic_gate_games": float(
            resolved_diagnostic_gate_games
        ),
        "direct_population_full_gate_games": float(gate_games),
        "direct_population_full_gate_score": full_score,
        "direct_population_full_gate_worst_score": full_worst_score,
        "direct_population_full_gate_selected": float(full_safe),
        "direct_population_full_gate_selection": full_selection,
        "direct_population_full_gate_raw_score": raw_full_score,
        "direct_population_full_gate_raw_worst_score": raw_full_worst_score,
        "direct_population_ema_enabled": float(ema_model is not None),
        "direct_population_ema_decay": float(
            ema_decay if ema_model is not None else 0.0
        ),
        "direct_population_full_gate_ema_score": float(ema_full_score),
        "direct_population_full_gate_ema_worst_score": float(
            ema_full_worst_score
        ),
        "direct_population_full_gate_selected_ema": float(select_ema),
        "direct_population_full_gate_candidates": [
            {
                "label": str(item["label"]),
                "score": float(item["score"]),
                "worst_score": float(item["worst_score"]),
                "safe": float(bool(item["safe"])),
            }
            for item in final_candidates
        ],
        "direct_population_ppo_learning_rate_final": current_ppo_learning_rate,
        "direct_population_ppo_gamma": float(ppo_gamma),
        "direct_population_ppo_terminal_advantage_weight": float(
            ppo_terminal_advantage_weight
        ),
        "direct_population_ppo_top_advantage_fraction": float(
            ppo_top_advantage_fraction
        ),
        "direct_population_shaping_weight_start": float(shaping_weight),
        "direct_population_shaping_weight_final": float(
            shaping_final_weight
        ),
        "direct_population_objective_replay_weight": float(
            objective_replay_weight
        ),
        "direct_population_objective_replay_horizon": float(
            objective_replay_horizon
        ),
        "direct_population_objective_replay_capacity": float(
            objective_replay_capacity
        ),
        "direct_population_objective_replay_max_epochs": float(
            objective_replay_max_epochs
        ),
        "direct_population_ppg_enabled": float(ppg_enabled),
        "direct_population_ppg_interval": float(ppg_interval),
        "direct_population_ppg_buffer_capacity": float(
            ppg_buffer_capacity
        ),
        "direct_population_ppg_samples_per_rollout": float(
            ppg_samples_per_rollout
        ),
        "direct_population_ppg_phase_samples": float(ppg_phase_samples),
        "direct_population_ppg_epochs": float(ppg_epochs),
        "direct_population_ppg_minibatch_size": float(
            ppg_minibatch_size
        ),
        "direct_population_ppg_policy_kl_weight": float(
            ppg_policy_kl_weight
        ),
        "direct_population_ppg_value_weight": float(ppg_value_weight),
        "direct_population_accepted_blocks": float(accepted_blocks),
        "direct_population_selected_blocks": float(selected_blocks),
        "direct_population_rollback_blocks": float(rollback_blocks),
        "direct_population_regression_patience": float(
            regression_patience
        ),
        "direct_population_consecutive_regressions": float(
            consecutive_regressions
        ),
        "direct_population_max_consecutive_regressions": float(
            maximum_consecutive_regressions
        ),
        "direct_population_early_stopped": float(early_stopped),
        "direct_population_early_stop_reason": early_stop_reason,
        "direct_population_gate_rows": gate_rows,
        "direct_population_training_rows": training_rows,
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
