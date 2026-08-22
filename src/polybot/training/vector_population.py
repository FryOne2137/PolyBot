"""Fast raw-policy population play in VectorGameEnv, without any MCTS."""

from __future__ import annotations

from collections import deque
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


@dataclass
class _PPOTransition:
    """One native action in a slot, including opponent actions for GAE."""

    value: float
    reward: float
    next_value: float
    sign: float
    terminal: bool
    candidate_example: _PPOExample | None


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


def _finish_ppo_trace(
    transitions: Sequence[_PPOTransition],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[_PPOExample]:
    """Compute sign-correct GAE through candidate and opponent actions.

    The value head is always from the active player's viewpoint.  Therefore a
    turn switch changes the sign both in the bootstrap and in the recursive
    advantage.  This is the multi-step counterpart of the old one-step TD
    target and correctly propagates a terminal loss through the opponent turn.
    """
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
            example.advantage = float(advantage)
            example.return_target = float(transition.value + advantage)
            result.append(example)
    result.reverse()
    return result


def _ppo_batch(
    examples: Sequence[_PPOExample],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collate a public on-policy PPO minibatch."""
    if not examples:
        raise ValueError("PPO batch must not be empty")
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
    return (
        observation,
        np.asarray([example.action_row for example in examples], dtype=np.int64),
        np.asarray(
            [example.old_log_probability for example in examples],
            dtype=np.float32,
        ),
        np.asarray([example.advantage for example in examples], dtype=np.float32),
        np.asarray([example.return_target for example in examples], dtype=np.float32),
    )


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
                        objective_replay_update_credits += (
                            objective_replay_max_epochs
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
    policy_anchor_weight: float = 0.10,
    value_anchor_weight: float = 0.10,
    opponent_policy_weight: float = 0.0,
    teacher_weight: float = 0.0,
    shaping_weight: float = 0.20,
    greedy_advantage_weight: float = 0.0,
    entropy_weight: float = 0.002,
    gamma: float = 0.995,
    ppo_rollout_steps: int = 32,
    ppo_epochs: int = 2,
    ppo_minibatch_size: int = 256,
    ppo_clip_range: float = 0.15,
    ppo_gae_lambda: float = 0.95,
    ppo_target_kl: float = 0.015,
    ppo_hard_kl: float = 0.030,
    ppo_rollout_temperature: float = 0.85,
    ppo_max_pending_examples: int = 32_768,
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
    }.items():
        if value < 0.0 or (name != "ppo_hard_kl" and value > 1.0):
            raise ValueError(f"{name} must be between zero and one")
    if ppo_hard_kl < ppo_target_kl:
        raise ValueError("ppo_hard_kl must be at least ppo_target_kl")
    if (
        ppo_rollout_steps < 1
        or ppo_epochs < 1
        or ppo_minibatch_size < 1
        or ppo_max_pending_examples < ppo_minibatch_size
    ):
        raise ValueError("PPO rollout, epoch and minibatch sizes must be positive")
    if not 0.1 <= ppo_rollout_temperature <= 2.0:
        raise ValueError("ppo_rollout_temperature must be between 0.1 and 2.0")
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
            "algorithm": "signed_ppo_gae",
            "num_envs": num_envs,
            "steps": steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "opponent_weights": weights.tolist(),
            "shaping_weight": shaping_weight,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "ppo_rollout_steps": ppo_rollout_steps,
            "ppo_epochs": ppo_epochs,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_range": ppo_clip_range,
            "ppo_gamma": gamma,
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_target_kl": ppo_target_kl,
            "ppo_hard_kl": ppo_hard_kl,
            "ppo_rollout_temperature": ppo_rollout_temperature,
            "ppo_max_pending_examples": ppo_max_pending_examples,
            # Kept in the artifacts so obsolete command-line flags remain
            # auditable rather than silently affecting the PPO objective.
            "legacy_opponent_policy_weight": opponent_policy_weight,
            "legacy_teacher_weight": teacher_weight,
            "legacy_greedy_advantage_weight": greedy_advantage_weight,
        },
    )
    candidate.to(device).eval()
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
    rng = np.random.default_rng(training.seed ^ 0x50504F)
    opponent_indices = rng.choice(
        len(frozen), size=num_envs, p=weights
    ).astype(np.int32)
    candidate_players = (np.arange(num_envs, dtype=np.int32) & 1).copy()
    traces: list[list[_PPOTransition]] = [[] for _ in range(num_envs)]
    episode_moves = np.zeros(num_envs, dtype=np.int32)
    episode_numbers = np.zeros(num_envs, dtype=np.int64)
    completed_games = wins = losses = draws = 0
    candidate_turns = opponent_turns = 0
    ppo_samples = ppo_updates = ppo_rollouts = 0
    ppo_kl_exceeded = False
    completed_examples: list[_PPOExample] = []
    pending_candidate_examples = 0
    ppo_forced_truncations = 0
    ppo_update_target = max(
        ppo_minibatch_size, num_envs * ppo_rollout_steps // 2
    )
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "champion_kl": 0.0,
        "behaviour_kl": 0.0,
        "clip_fraction": 0.0,
        "entropy": 0.0,
    }
    # A PPO update happens only after enough completed trajectories have been
    # collected.  Keep its most recent value separately from the metrics for
    # *this environment step*, otherwise tqdm misleadingly displays 0.000 on
    # every collection-only step.
    last_champion_kl = 0.0
    reward_total = 0.0

    def optimise_rollout(examples: Sequence[_PPOExample]) -> Dict[str, float]:
        """Run PPO epochs and stop before an unsafe champion-policy drift."""
        nonlocal ppo_updates, ppo_kl_exceeded
        if not examples:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "champion_kl": 0.0,
                "behaviour_kl": 0.0,
                "clip_fraction": 0.0,
                "entropy": 0.0,
            }
        advantages = np.asarray(
            [example.advantage for example in examples], dtype=np.float32
        )
        advantages = (advantages - advantages.mean()) / max(
            float(advantages.std()), 1e-4
        )
        for example, advantage in zip(examples, advantages):
            example.advantage = float(advantage)
        totals_local = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "champion_kl": 0.0,
            "behaviour_kl": 0.0,
            "clip_fraction": 0.0,
            "entropy": 0.0,
        }
        local_updates = 0
        for _ in range(ppo_epochs):
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
                    returns, dtype=torch.float32, device=device
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
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = F.smooth_l1_loss(
                        output.value.float(), return_tensor
                    )
                    entropy = Categorical(
                        logits=output.policy_logits / ppo_rollout_temperature
                    ).entropy().mean()
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
                if float(champion_kl.detach().cpu()) > ppo_hard_kl:
                    ppo_kl_exceeded = True
                    candidate.eval()
                    return {
                        **{
                            name: total / max(1, local_updates)
                            for name, total in totals_local.items()
                        },
                        "champion_kl": float(champion_kl.detach().cpu()),
                        "behaviour_kl": float(behaviour_kl.detach().cpu()),
                        "clip_fraction": float(
                            ((ratio - 1.0).abs() > ppo_clip_range)
                            .float().mean().detach().cpu()
                        ),
                    }
                kl_multiplier = policy_anchor_weight * max(
                    1.0,
                    float(champion_kl.detach().cpu())
                    / max(ppo_target_kl, 1e-6),
                )
                value_anchor_loss = F.smooth_l1_loss(
                    output.value.float(), anchor_output.value.float()
                )
                loss = (
                    policy_loss
                    + 0.50 * value_loss
                    + kl_multiplier * champion_kl
                    + 0.25 * value_anchor_weight * value_anchor_loss
                    - entropy_weight * entropy
                )
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                candidate.eval()
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
                }
                for name, value in values.items():
                    totals_local[name] += value
                local_updates += 1
                ppo_updates += 1
        return {
            name: total / max(1, local_updates)
            for name, total in totals_local.items()
        }

    with tqdm(
        total=steps,
        desc=f"direct population PPO ({device}, no MCTS)",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for step in range(steps):
            candidate.eval()
            tensors = _vector_tensors(batch, device, training)
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
            reward = shaping_weight * shaped + terminal_reward
            reward_total += float(reward.mean())
            with torch.inference_mode():
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
                # Finish only a real game here.  The return crosses every
                # opponent action stored in this slot before reaching its own
                # candidate decisions.
                finished_examples = _finish_ppo_trace(
                    traces[int(env_id)], gamma=gamma, gae_lambda=ppo_gae_lambda
                )
                completed_examples.extend(finished_examples)
                pending_candidate_examples -= len(finished_examples)
                traces[int(env_id)].clear()
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
                ppo_updates=ppo_updates,
                ppo_rollouts=ppo_rollouts,
                ppo_pending_examples=pending_candidate_examples,
                ppo_forced_truncations=ppo_forced_truncations,
                ppo_kl_exceeded=float(ppo_kl_exceeded),
                ppo_learning_rate=float(optimiser.param_groups[0]["lr"]),
                ppo_update_this_step=float(bool(rollout_metrics)),
                ppo_last_champion_kl=last_champion_kl,
                reward=float(reward.mean()),
                **{
                    "ppo_" + name: value
                    for name, value in rollout_metrics.items()
                },
            )
            progress.update(1)
            if (step + 1) % 20 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    games=completed_games,
                    score=f"{(wins + 0.5 * draws) / max(1, completed_games):.3f}",
                    ppo_updates=ppo_updates,
                    kl=f"{last_champion_kl:.5f}",
                )
            if ppo_kl_exceeded:
                break

    divisor = max(1, ppo_rollouts)
    summary: Dict[str, float | str] = {
        "direct_population_algorithm": "signed_ppo_gae",
        "direct_population_envs": float(num_envs),
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
        "direct_population_ppo_updates": float(ppo_updates),
        "direct_population_ppo_rollouts": float(ppo_rollouts),
        "direct_population_ppo_pending_examples": float(
            pending_candidate_examples
        ),
        "direct_population_ppo_forced_truncations": float(
            ppo_forced_truncations
        ),
        "direct_population_ppo_update_target": float(ppo_update_target),
        "direct_population_ppo_max_pending_examples": float(
            ppo_max_pending_examples
        ),
        "direct_population_ppo_kl_exceeded": float(ppo_kl_exceeded),
        "direct_population_ppo_target_kl": float(ppo_target_kl),
        "direct_population_ppo_hard_kl": float(ppo_hard_kl),
        "direct_population_ppo_rollout_temperature": float(
            ppo_rollout_temperature
        ),
        "direct_population_ppo_reward": reward_total / max(1, step + 1),
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
    game_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    candidate_label: str = "pretendent",
    opponent_labels: Sequence[str] | None = None,
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
    greedy_advantage_weight: float = 0.25,
    regression_patience: int = 2,
    score_tolerance: float = 0.02,
    objective_tolerance: float = 0.05,
    diagnostic_gate_games: int | None = 20,
    ppo_rollout_steps: int = 32,
    ppo_epochs: int = 2,
    ppo_minibatch_size: int = 256,
    ppo_clip_range: float = 0.15,
    ppo_gae_lambda: float = 0.95,
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
    if not 0.0 <= ppo_gae_lambda <= 1.0:
        raise ValueError("ppo_gae_lambda must be between zero and one")
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
            "greedy_advantage_weight": greedy_advantage_weight,
            "algorithm": "signed_ppo_gae",
            "ppo_rollout_steps": ppo_rollout_steps,
            "ppo_epochs": ppo_epochs,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_range": ppo_clip_range,
            "ppo_gae_lambda": ppo_gae_lambda,
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
        },
    )
    parent_state = model_state_on_cpu(candidate)
    best_state = model_state_on_cpu(candidate)
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
            game_callback=(
                None
                if gate_game_callback is None
                else lambda opponent_index, game: gate_game_callback(
                    label, block, opponent_index, game
                )
            ),
            candidate_label=candidate_label,
            opponent_labels=opponent_labels,
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
                best_score = weighted_score
                best_worst_score = worst_score
                best_capture_rate = capture_rate
                best_completion_rate = completion_rate
                best_objective_progress = objective_progress
                if not baseline:
                    selected_blocks += 1
            elif not baseline:
                # Safe near-ties keep the phase alive, but training resumes
                # from the actual strongest model rather than accumulating
                # several tolerated score drops.
                restore_model_state(candidate, best_state)
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
            policy_anchor_weight=policy_anchor_weight,
            value_anchor_weight=value_anchor_weight,
            opponent_policy_weight=opponent_policy_weight,
            teacher_weight=teacher_weight,
            shaping_weight=shaping_weight,
            greedy_advantage_weight=greedy_advantage_weight,
            ppo_rollout_steps=ppo_rollout_steps,
            ppo_epochs=ppo_epochs,
            ppo_minibatch_size=ppo_minibatch_size,
            ppo_clip_range=ppo_clip_range,
            ppo_gae_lambda=ppo_gae_lambda,
            ppo_target_kl=ppo_target_kl,
            ppo_hard_kl=ppo_hard_kl,
            ppo_rollout_temperature=ppo_rollout_temperature,
            ppo_max_pending_examples=ppo_max_pending_examples,
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
        if bool(result.get("direct_population_ppo_kl_exceeded", 0.0)):
            restore_model_state(candidate, best_state)
            rollback_blocks += 1
            previous_learning_rate = current_ppo_learning_rate
            current_ppo_learning_rate = max(
                ppo_min_learning_rate,
                current_ppo_learning_rate * ppo_lr_backoff,
            )
            run.log(
                direct_population_ppo_rollback=1.0,
                direct_population_ppo_kl_exceeded=1.0,
                direct_population_ppo_previous_learning_rate=(
                    previous_learning_rate
                ),
                direct_population_ppo_learning_rate=current_ppo_learning_rate,
            )
            if previous_learning_rate <= ppo_min_learning_rate + 1e-12:
                early_stopped = True
                early_stop_reason = "ppo_kl_limit_at_min_learning_rate"
                break
            continue
        accepted = gate("population", block)
        if (
            completed >= minimum_steps
            and not accepted
            and consecutive_regressions >= regression_patience
        ):
            early_stopped = True
            early_stop_reason = "persistent_population_regression"
            break

    restore_model_state(candidate, best_state)
    full_evaluation = evaluate_direct_population_matches(
        training,
        candidate,
        opponent_models,
        weights,
        games=gate_games,
        gate_seed=gate_seed,
        game_callback=(
            None
            if gate_game_callback is None
            else lambda opponent_index, game: gate_game_callback(
                "final", block, opponent_index, game
            )
        ),
        candidate_label=candidate_label,
        opponent_labels=opponent_labels,
    )
    full_score = float(full_evaluation["model_a_score"])
    full_worst_score = float(full_evaluation["worst_score"])
    full_safe = (
        full_score >= gate_score
        and full_worst_score >= historical_score_floor
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
        "selection_reason": "full_gate" if full_safe else "full_gate_rejected",
        "opponents": full_evaluation["opponents"],
    }
    gate_rows.append(full_row)
    run.log(
        direct_population_full_gate_score=full_score,
        direct_population_full_gate_worst=full_worst_score,
        direct_population_full_gate_selected=float(full_safe),
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
        "direct_population_algorithm": "signed_ppo_gae",
        "direct_population_diagnostic_gate_games": float(
            resolved_diagnostic_gate_games
        ),
        "direct_population_full_gate_games": float(gate_games),
        "direct_population_full_gate_score": full_score,
        "direct_population_full_gate_worst_score": full_worst_score,
        "direct_population_full_gate_selected": float(full_safe),
        "direct_population_ppo_learning_rate_final": current_ppo_learning_rate,
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
