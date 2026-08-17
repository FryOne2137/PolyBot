"""In-memory behavior-cloning bootstrap from a network-free legal-action teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..data.polyenv_features import (
    action_features,
    capture_flags,
    event_summary,
    global_features_from_obs,
    stack_examples,
    visible_events_or_empty,
)
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.heuristic_policy import choose_teacher_action, teacher_policy
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer, TrainingExample
from .rewards import RewardShaper
from .run_artifacts import TrainingRun
from .self_play import SelfPlayRunner, _winner_to_value


def _assign_value_targets(
    examples: List[TrainingExample],
    transitions: List[Tuple[int, float, Optional[TrainingExample]]],
    winner: object,
    training: TrainingConfig,
) -> None:
    players = {example.player_id for example in examples}
    advantage = {player: 0.0 for player in players}
    for actor, reward, example in reversed(transitions):
        for player in players:
            advantage[player] = training.reward_discount * advantage[player] + (
                reward if actor == player else -reward
            )
        if example is not None:
            terminal = _winner_to_value(winner, example.player_id)
            example.value_target = float(np.clip(
                terminal + training.shaping_target_weight * advantage[example.player_id], -1.0, 1.0
            ))


def _teacher_game(seed: int, training: TrainingConfig) -> Tuple[List[TrainingExample], Dict[str, float]]:
    env = SelfPlayRunner._new_env(seed, training.map_size, training.map_type)
    rng = np.random.default_rng(seed ^ 0x7EAC4E)
    shaper = RewardShaper(training)
    examples: List[TrainingExample] = []
    transitions: List[Tuple[int, float, Optional[TrainingExample]]] = []
    metrics: Dict[str, float] = {"teacher_city_captures": 0.0, "teacher_capital_captures": 0.0}
    winner = -1
    moves = 0

    while not env.is_done() and moves < training.max_moves:
        packet = env.model_request_numpy()
        player = int(packet["obs"].get("current_player", env.current_player()))
        row = choose_teacher_action(packet, rng)
        categorical, continuous = action_features(packet["actions"])
        example = TrainingExample(
            map_tokens=np.asarray(packet["map_tokens"], dtype=np.int64).copy(),
            global_features=global_features_from_obs(packet["obs"]),
            event_features=event_summary(visible_events_or_empty(env)),
            action_categorical=categorical.copy(),
            action_continuous=continuous.copy(),
            policy_target=teacher_policy(packet),
            full_map_target=np.asarray(env.full_map_numpy(), dtype=np.int64).copy(),
            hidden_mask=np.asarray(packet["map_tokens"])[:, 0] == 0,
            player_id=player,
        )
        action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
        city, capital = capture_flags(packet, row)
        ok, _, terminal_reward, winner, _ = env.step_fast(int(action_ids[row]))
        if not ok:
            raise RuntimeError("heuristic teacher selected an illegal action")
        reward = shaper.observe(
            player,
            env.observation(player_id=player, visible_only=True),
            terminal=float(terminal_reward),
            captured_city=city,
            captured_capital=capital,
        )
        example.shaped_reward = reward.non_terminal
        examples.append(example)
        transitions.append((player, reward.non_terminal, example))
        metrics["teacher_city_captures"] += float(city)
        metrics["teacher_capital_captures"] += float(capital)
        moves += 1

    _assign_value_targets(examples, transitions, winner, training)
    if not env.is_done():
        for example in examples:
            example.value_loss_weight = 0.0
    metrics["moves"] = float(moves)
    metrics["examples"] = float(len(examples))
    metrics["winner"] = float(winner)
    return examples, metrics


def train_teacher_bootstrap(
    training: TrainingConfig,
    model_config: ModelConfig,
    games: int,
    updates_per_game: int,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
    *,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Clone productive teacher behavior and belief labels without disk replay."""
    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "teacher")
    run.write_config(
        training=training,
        model=model_config,
        stage={"games": games, "updates_per_game": updates_per_game, "model_out": model_out},
    )
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed)
    losses: List[Dict[str, float]] = []
    captures = 0.0
    capitals = 0.0

    with tqdm(total=games, desc=f"teacher bootstrap ({device})", unit="game", dynamic_ncols=True) as progress:
        for game in range(games):
            examples, metrics = _teacher_game(training.seed + game, training)
            replay.extend(examples)
            captures += metrics["teacher_city_captures"]
            capitals += metrics["teacher_capital_captures"]
            if len(replay) >= training.batch_size:
                game_losses: List[Dict[str, float]] = []
                for _ in range(updates_per_game):
                    examples_batch = replay.sample(
                        training.batch_size,
                        rng,
                        priority_fraction=training.priority_replay_fraction,
                        bridge_fraction=training.bridge_replay_fraction,
                    )
                    batch = stack_examples(examples_batch, device)
                    update_loss = train_batch(model, optimiser, batch, training)
                    losses.append(update_loss)
                    game_losses.append(update_loss)
            else:
                game_losses = []
            logged = {
                "game": game + 1,
                "update": len(losses),
                "replay_examples": len(replay),
                "city_captures": captures,
                "capital_captures": capitals,
                "game_moves": metrics["moves"],
            }
            if game_losses:
                logged.update({
                    name: float(np.mean([row[name] for row in game_losses]))
                    for name in game_losses[0]
                })
            run.log(**logged)
            progress.update(1)
            progress.set_postfix(captures=int(captures), examples=len(replay))

    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float | str] = {
        "teacher_games": float(games),
        "replay_examples": float(len(replay)),
        "teacher_city_captures": captures,
        "teacher_capital_captures": capitals,
    }
    if losses:
        for name in losses[0]:
            summary[name] = float(np.mean([loss[name] for loss in losses]))
    summary["run_dir"] = str(run.finish(summary))
    return summary
