"""Global batched self-play: one native MctsPool for all active games."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import (
    action_features, capture_flags, global_features_from_obs, packets_to_tensors,
    strategic_action_bias, visible_events_or_empty, event_summary, stack_examples,
)
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.heuristic_policy import choose_teacher_action
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer, TrainingExample
from .rewards import RewardShaper
from .self_play import _action_bucket, _winner_to_value, SelfPlayRunner
from .teacher_bootstrap import _teacher_game


@dataclass
class _Episode:
    index: int
    env: object
    train_player: Optional[int]
    opponent_agent: Optional[str]
    rng: np.random.Generator
    shaper: RewardShaper
    examples: List[TrainingExample] = field(default_factory=list)
    transitions: List[Tuple[int, float, Optional[TrainingExample]]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    moves: int = 0
    winner: int = -1


def _new_episode(index: int, training: TrainingConfig, train_player: Optional[int], opponent: Optional[str]) -> _Episode:
    return _Episode(
        index=index,
        env=SelfPlayRunner._new_env(training.seed + index, training.map_size, training.map_type),
        train_player=train_player,
        opponent_agent=opponent,
        rng=np.random.default_rng((training.seed + index) ^ 0xC0DEC0DE),
        shaper=RewardShaper(training),
    )


@torch.inference_mode()
def _plan_many(
    episodes: List[_Episode], model: PolicyValueBeliefNet, device: torch.device,
    training: TrainingConfig, config: MCTSConfig,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Plan one decision per episode in one global native MctsPool.

    The pool contains ``len(episodes) * particles`` isolated belief roots. It
    therefore emits a GPU batch of that size per PUCT wave, rather than a tiny
    batch per Python game.
    """
    from PolyEnv import MctsPool

    packets = [episode.env.model_request_numpy() for episode in episodes]
    events = [visible_events_or_empty(episode.env) for episode in episodes]
    tensors = packets_to_tensors(packets, device, events)
    output = model(
        tensors.map_tokens, tensors.global_features, tensors.action_categorical,
        tensors.action_continuous, tensors.action_mask, tensors.event_features,
    )
    logits = output.policy_logits.clone()
    root_priors: List[np.ndarray] = []
    for row, packet in enumerate(packets):
        bias = strategic_action_bias(
            packet, capture_city_bonus=config.capture_city_prior_bonus,
            capture_capital_bonus=config.capture_capital_prior_bonus,
            found_city_bonus=config.found_city_prior_bonus, attack_bonus=config.attack_prior_bonus,
            end_turn_penalty=config.end_turn_prior_penalty,
        )
        logits[row, :bias.size] += torch.as_tensor(bias, dtype=logits.dtype, device=device)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    for row, packet in enumerate(packets):
        count = len(packet["actions"]["action_id"])
        root_priors.append(probabilities[row, :count])

    belief_envs: List[object] = []
    owners: List[int] = []
    for row, (episode, packet) in enumerate(zip(episodes, packets)):
        perspective = int(packet["obs"].get("current_player", episode.env.current_player()))
        # The root Transformer was computed once for the whole cohort. Slice
        # only its existing belief logits; do not run another root forward.
        belief_logits = [head[row:row + 1] for head in output.belief_logits]
        particles = model.sample_completed_maps_from_belief_logits(
            tensors.map_tokens[row:row + 1], belief_logits, config.particles,
            max(config.temperature, 0.35),
        )
        for particle in particles:
            try:
                belief_envs.append(episode.env.make_belief_env(particle.cpu().numpy().astype(np.int32), perspective=perspective))
                owners.append(row)
            except Exception:
                # A particle can be invalid without exposing source truth.
                continue

    visits_by_episode = [np.zeros(len(packet["actions"]["action_id"]), dtype=np.float64) for packet in packets]
    if not belief_envs:
        return [(row, visits_by_episode[row], root_priors[row]) for row in range(len(episodes))]
    pool = MctsPool(
        belief_envs, num_threads=config.native_threads,
        max_actions=config.native_max_actions, c_puct=config.c_puct,
    )
    # Each wave exposes at most one leaf per tree. With B active games and P
    # particles, a wave has up to B*P leaves and is the natural GPU batch.
    completed = 0
    while completed < config.simulations:
        leaves = pool.select_leaves()
        leaf_ids = np.asarray(leaves["leaf_id"])
        if leaf_ids.size == 0:
            break
        from ..data.polyenv_features import native_batch_to_tensors
        leaf_tensors = native_batch_to_tensors(leaves, device, training.map_size, training.map_type)
        leaf_output = model(
            leaf_tensors.map_tokens, leaf_tensors.global_features,
            leaf_tensors.action_categorical, leaf_tensors.action_continuous,
            leaf_tensors.action_mask, leaf_tensors.event_features, include_belief=False,
        )
        pool.expand_and_backup(
            leaf_ids, leaf_output.policy_logits.float().cpu().numpy(),
            leaf_output.value.float().cpu().numpy(),
        )
        completed += 1

    roots = pool.root_policy(temperature=1.0)
    for tree, owner in enumerate(owners):
        legal = np.asarray(roots["action_id"][tree], dtype=np.int64)
        mask = np.asarray(roots["action_mask"][tree], dtype=bool)
        count = np.asarray(roots["visit_count"][tree], dtype=np.float64)
        source_ids = np.asarray(packets[owner]["actions"]["action_id"], dtype=np.int64)
        lookup = {int(action): index for index, action in enumerate(source_ids)}
        for action, valid, visits in zip(legal, mask, count):
            if valid and int(action) in lookup:
                visits_by_episode[owner][lookup[int(action)]] += visits
    return [(row, visits_by_episode[row], root_priors[row]) for row in range(len(episodes))]


def _apply_action(episode: _Episode, action_row: int, is_model: bool, policy: Optional[np.ndarray], training: TrainingConfig) -> None:
    packet = episode.env.model_request_numpy()
    events = visible_events_or_empty(episode.env)
    player = int(packet["obs"].get("current_player", episode.env.current_player()))
    example = None
    if is_model:
        categorical, continuous = action_features(packet["actions"])
        example = TrainingExample(
            map_tokens=np.asarray(packet["map_tokens"], dtype=np.int64).copy(),
            global_features=global_features_from_obs(packet["obs"]), event_features=event_summary(events),
            action_categorical=categorical.copy(), action_continuous=continuous.copy(),
            policy_target=policy.astype(np.float32), full_map_target=np.asarray(episode.env.full_map_numpy(), dtype=np.int64).copy(),
            hidden_mask=np.asarray(packet["map_tokens"])[:, 0] == 0, player_id=player,
        )
        episode.examples.append(example)
    city, capital = capture_flags(packet, action_row)
    action_id = int(np.asarray(packet["actions"]["action_id"], dtype=np.int64)[action_row])
    ok, _, terminal, winner, _ = episode.env.step_fast(action_id)
    if not ok:
        raise RuntimeError("global self-play selected an action rejected by PolyEnv")
    reward = episode.shaper.observe(player, episode.env.observation(player_id=player, visible_only=True), float(terminal), city, capital)
    if example is not None:
        example.shaped_reward = reward.non_terminal
    episode.transitions.append((player, reward.non_terminal, example))
    episode.winner = int(winner)
    episode.moves += 1
    for name, value in reward.as_dict().items():
        episode.metrics[name] = episode.metrics.get(name, 0.0) + value
    role = "model" if is_model else "opponent"
    name = str(packet["actions"].get("type_fullname", [""])[action_row])
    key = f"{role}_action_{_action_bucket(name)}"
    episode.metrics[key] = episode.metrics.get(key, 0.0) + 1.0
    episode.metrics[f"{role}_actions"] = episode.metrics.get(f"{role}_actions", 0.0) + 1.0


def _finish(episode: _Episode, training: TrainingConfig) -> None:
    players = {row.player_id for row in episode.examples}
    future = {player: 0.0 for player in players}
    for actor, shaped, example in reversed(episode.transitions):
        for player in players:
            future[player] = training.reward_discount * future[player] + (1.0 if player == actor else -1.0) * shaped
        if example is not None:
            example.value_target = float(np.clip(_winner_to_value(episode.winner, example.player_id) + training.shaping_target_weight * future[example.player_id], -1.0, 1.0))


def train_global_mcts_self_play(
    training: TrainingConfig, model_config: ModelConfig, mcts: MCTSConfig, games: int, updates_per_game: int,
    opponent: str, random_opponent_rate: float, mixed_opponent: str, teacher_replay_fraction: float,
    active_envs: int, model_out: Optional[Path], model: Optional[PolicyValueBeliefNet],
) -> Dict[str, float]:
    """High-throughput global-MCTS trainer; writes only final model weights."""
    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    replay, teacher = ReplayBuffer(training.replay_capacity), ReplayBuffer(training.replay_capacity)
    schedule_rng, sample_rng = np.random.default_rng(training.seed ^ 0xA17C0), np.random.default_rng(training.seed)
    next_game, done = 0, 0
    active: List[_Episode] = []
    summaries: Dict[str, float] = {}
    model.eval()
    with tqdm(total=games, desc=f"global native MCTS ({device})", unit="env", dynamic_ncols=True) as bar:
        while done < games:
            while next_game < games and len(active) < active_envs:
                anchor = opponent in {"random", "heuristic"} or (opponent == "mixed" and schedule_rng.random() < random_opponent_rate)
                active.append(_new_episode(next_game, training, next_game % 2 if anchor else None, (mixed_opponent if opponent == "mixed" else opponent) if anchor else None))
                next_game += 1
            model_turns = []
            for episode in active:
                packet = episode.env.model_request_numpy()
                player = int(packet["obs"].get("current_player", episode.env.current_player()))
                if episode.train_player is None or player == episode.train_player:
                    model_turns.append(episode)
                else:
                    ids = np.asarray(packet["actions"]["action_id"])
                    row = int(episode.rng.integers(len(ids))) if episode.opponent_agent == "random" else choose_teacher_action(packet, episode.rng)
                    _apply_action(episode, row, False, None, training)
            if model_turns:
                for index, visits, priors in _plan_many(model_turns, model, device, training, mcts):
                    episode = model_turns[index]
                    weights = visits if visits.sum() else priors
                    policy = weights / max(weights.sum(), 1e-8)
                    row = int(policy.argmax()) if mcts.temperature <= 1e-5 else int(episode.rng.choice(len(policy), p=policy))
                    if episode.rng.random() < training.exploration_rate:
                        bias = strategic_action_bias(episode.env.model_request_numpy())
                        weights = np.exp(np.clip(bias - bias.max(), -20.0, 0.0))
                        row = int(episode.rng.choice(len(weights), p=weights / weights.sum()))
                    _apply_action(episode, row, True, policy, training)
            completed = [episode for episode in active if episode.env.is_done() or episode.moves >= training.max_moves]
            for episode in completed:
                _finish(episode, training); replay.extend(episode.examples); active.remove(episode); done += 1; bar.update(1)
                for key, value in episode.metrics.items(): summaries[key] = summaries.get(key, 0.0) + value
                if teacher_replay_fraction > 0.0:
                    teacher_examples, _ = _teacher_game(training.seed + 1_000_000 + episode.index, training); teacher.extend(teacher_examples)
                bar.set_postfix(replay=len(replay), active=len(active), trees=len(model_turns) * mcts.particles)
    losses = []
    for _ in tqdm(range(games * updates_per_game), desc=f"optimising ({device})", unit="update", dynamic_ncols=True):
        if len(replay) < training.batch_size: break
        tc = min(int(round(training.batch_size * teacher_replay_fraction)), len(teacher)); sc = training.batch_size - tc
        rows = replay.sample(sc, sample_rng, training.priority_replay_fraction) if sc else []
        rows += teacher.sample(tc, sample_rng, training.priority_replay_fraction) if tc else []
        losses.append(train_batch(model, optimiser, stack_examples(rows, device), training))
    if model_out: save_model(model_out, model, model_config)
    result = {"games": float(games), "active_envs": float(active_envs), "replay_examples": float(len(replay)), "teacher_replay_examples": float(len(teacher))}
    result.update({"mean_" + k: v / max(1, len(replay)) for k, v in summaries.items() if k.startswith("reward_")})
    if losses: result.update({key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]})
    return result
