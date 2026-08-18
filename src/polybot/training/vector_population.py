"""Fast raw-policy population play in VectorGameEnv, without any MCTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Sequence

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
from .vector_teacher import _teacher_targets, _vector_tensors


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
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    opponent_policy_weight: float = 0.50,
    teacher_weight: float = 0.25,
    shaping_weight: float = 0.20,
    entropy_weight: float = 0.004,
    gamma: float = 0.995,
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
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
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
    completed_games = wins = losses = draws = 0
    candidate_turns = opponent_turns = 0
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "opponent_policy_loss": 0.0,
        "value_loss": 0.0,
        "teacher_loss": 0.0,
        "policy_anchor_loss": 0.0,
        "value_anchor_loss": 0.0,
        "entropy": 0.0,
        "reward": 0.0,
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

            opponent_target = torch.zeros_like(output.policy_logits)
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
                opponent_target.index_copy_(0, row_tensor, opponent_policy)
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
            if candidate_rows.size:
                candidate_index = torch.as_tensor(
                    candidate_rows, dtype=torch.long, device=device
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
            loss = (
                policy_loss
                + opponent_policy_weight * opponent_policy_loss
                + 0.50 * value_loss
                + teacher_weight * teacher_loss
                + policy_anchor_weight * policy_anchor_loss
                + value_anchor_weight * value_anchor_loss
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
                opponent_turns=opponent_turns,
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
        "direct_population_opponent_turns": float(opponent_turns),
        **{
            "direct_population_" + key: value / max(1, steps)
            for key, value in totals.items()
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
    regression_patience: int = 2,
    gate_game_callback: Callable[[str, int, int, Mapping[str, Any]], None]
    | None = None,
    training_game_callback: Callable[[int, Mapping[str, Any]], None]
    | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, Any]:
    """Advance raw population play only while held-out population skill grows."""
    if minimum_steps < 0 or maximum_steps < minimum_steps:
        raise ValueError("invalid direct population step range")
    if minimum_steps and block_steps < 1:
        raise ValueError("block_steps must be positive")
    if gate_games < 2 or gate_games % 2:
        raise ValueError("direct population gate games must be positive and even")
    if readiness_patience < 1:
        raise ValueError("readiness_patience must be positive")
    if regression_patience < 1:
        raise ValueError("regression_patience must be positive")
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
            "gate_seed": gate_seed,
            "gate_score": gate_score,
            "historical_score_floor": historical_score_floor,
            "readiness_city_captures_per_game": (
                readiness_city_captures_per_game
            ),
            "readiness_completion_rate": readiness_completion_rate,
            "readiness_patience": readiness_patience,
            "regression_patience": regression_patience,
            "opponent_weights": weights.tolist(),
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "opponent_policy_weight": opponent_policy_weight,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
        },
    )
    best_state = model_state_on_cpu(candidate)
    best_score = 0.0
    best_worst_score = 0.0
    best_capture_rate = 0.0
    best_completion_rate = 0.0
    readiness_streak = 0
    ready = False
    best_meets_readiness = False
    accepted_blocks = rollback_blocks = 0
    consecutive_regressions = 0
    maximum_consecutive_regressions = 0
    gate_rows: list[Dict[str, Any]] = []

    def gate(label: str, block: int, baseline: bool = False) -> bool:
        nonlocal best_state, best_score, best_worst_score
        nonlocal best_capture_rate, best_completion_rate
        nonlocal readiness_streak, ready, accepted_blocks, rollback_blocks
        nonlocal best_meets_readiness
        nonlocal consecutive_regressions, maximum_consecutive_regressions
        evaluation = evaluate_direct_population_matches(
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
                    label, block, opponent_index, game
                )
            ),
        )
        results = evaluation["opponents"]
        weighted_score = float(evaluation["model_a_score"])
        worst_score = float(evaluation["worst_score"])
        capture_rate = float(
            evaluation["model_a_city_captures_per_game"]
        )
        completion_rate = float(evaluation["completion_rate"])
        accepted = baseline or (
            weighted_score >= best_score
            and worst_score >= best_worst_score
            and capture_rate >= best_capture_rate
            and completion_rate >= best_completion_rate
        )
        if accepted:
            best_state = model_state_on_cpu(candidate)
            best_score = weighted_score
            best_worst_score = worst_score
            best_capture_rate = capture_rate
            best_completion_rate = completion_rate
            if not baseline:
                accepted_blocks += 1
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
            "accepted": float(accepted),
            "ready": float(ready),
            "best_meets_readiness": float(best_meets_readiness),
            "readiness_streak": float(readiness_streak),
            "consecutive_regressions": float(consecutive_regressions),
            "opponents": results,
        }
        gate_rows.append(row)
        run.log(
            direct_population_gate_score=weighted_score,
            direct_population_gate_worst=worst_score,
            direct_population_gate_captures=capture_rate,
            direct_population_gate_completion=completion_rate,
            direct_population_gate_accepted=float(accepted),
            direct_population_ready=float(ready),
            direct_population_best_meets_readiness=float(
                best_meets_readiness
            ),
            direct_population_consecutive_regressions=float(
                consecutive_regressions
            ),
        )
        return accepted

    gate("baseline", -1, baseline=True)
    completed = 0
    block = 0
    early_stopped = False
    early_stop_reason = ""
    training_rows: list[Dict[str, Any]] = []
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
        accepted = gate("population", block)
        if (
            not accepted
            and best_meets_readiness
            and consecutive_regressions >= regression_patience
        ):
            ready = True
            early_stopped = True
            early_stop_reason = "persistent_population_regression"
            break

    restore_model_state(candidate, best_state)
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
        "direct_population_accepted_blocks": float(accepted_blocks),
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
