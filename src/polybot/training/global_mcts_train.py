"""Global batched self-play: one native MctsPool for all active games."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.heuristic_policy import choose_teacher_action, teacher_policy
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer, TrainingExample
from .rewards import RewardShaper
from .run_artifacts import TrainingRun
from .self_play import _action_bucket, SelfPlayRunner
from .teacher_bootstrap import _teacher_game
from .value_targets import assign_td_lambda_targets


@dataclass
class _Episode:
    index: int
    env: object
    train_player: Optional[int]
    opponent_agent: Optional[str]
    opponent_model_index: Optional[int]
    rng: np.random.Generator
    shaper: RewardShaper
    examples: List[TrainingExample] = field(default_factory=list)
    transitions: List[Tuple[int, float, Optional[TrainingExample]]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    moves: int = 0
    winner: int = -1


def _native_leaf_bias(
    categorical: torch.Tensor,
    continuous: torch.Tensor,
    mask: torch.Tensor,
    teacher_prior_weight: float,
    strategic_prior_weight: float = 1.0,
) -> torch.Tensor:
    """Visible-only priors for every native MCTS leaf, not just the root."""
    action_type, tile_action = categorical[..., 0], categorical[..., 9]
    damage = continuous[..., 2].clamp_min(0)
    curriculum = torch.zeros_like(action_type, dtype=torch.float32)
    curriculum += ((action_type == 8) & (tile_action == 14)).float() * 2.0  # capture city
    curriculum += ((action_type == 8) & (tile_action == 11)).float() * 3.0  # found city
    curriculum += (action_type == 1).float() * (0.8 + damage.clamp_max(10.0) * 0.12)
    curriculum += ((action_type == 8) & ((tile_action == 10) | (tile_action == 12))).float() * 0.8
    curriculum -= (action_type == 3).float() * 2.0
    bias = curriculum * float(np.clip(strategic_prior_weight, 0.0, 1.0))
    # The network still supplies every prior.  This decaying additive anchor
    # only stops early self-play from unlearning city capture/founding before
    # terminal games become common.  It is computed entirely from legal,
    # visible action features.
    teacher = torch.zeros_like(bias)
    teacher += ((action_type == 8) & (tile_action == 14)).float() * 5.0
    teacher += ((action_type == 8) & (tile_action == 11)).float() * 3.0
    teacher += (action_type == 1).float() * (0.8 + damage.clamp_max(10.0) * 0.12)
    teacher += ((action_type == 8) & ((tile_action == 10) | (tile_action == 12))).float() * 0.8
    teacher += ((action_type == 5) | (action_type == 6)).float() * 0.45
    teacher += ((action_type == 4) | (action_type == 7)).float() * 0.25
    teacher -= (action_type == 3).float() * 2.0
    bias += teacher * float(np.clip(teacher_prior_weight, 0.0, 1.0))
    return bias.masked_fill(~mask, 0.0)


def _transient_pool_threads(requested: int | None) -> int:
    """Use a safe thread count for a newly-created native MCTS pool.

    PolyEnv owns a C++ worker pool per ``MctsPool`` and a real move requires a
    new pool. The global batch already keeps the GPU busy with all roots;
    spawning a hardware-sized C++ pool for every move can exhaust Windows
    thread resources during long runs.
    """
    return 1 if requested is None else max(1, int(requested))


def _new_episode(
    index: int,
    training: TrainingConfig,
    train_player: Optional[int],
    opponent: Optional[str],
    opponent_model_index: Optional[int] = None,
) -> _Episode:
    return _Episode(
        index=index,
        env=SelfPlayRunner._new_env(training.seed + index, training.map_size, training.map_type),
        train_player=train_player,
        opponent_agent=opponent,
        opponent_model_index=opponent_model_index,
        rng=np.random.default_rng((training.seed + index) ^ 0xC0DEC0DE),
        shaper=RewardShaper(training),
    )


def _search_action(
    visits: np.ndarray,
    priors: np.ndarray,
    rng: np.random.Generator,
    mcts: MCTSConfig,
) -> Tuple[int, np.ndarray]:
    """Turn a legal MCTS root distribution into a selected action row."""
    weights = visits if visits.sum() else priors
    policy = weights / max(float(weights.sum()), 1e-8)
    if mcts.temperature <= 1e-5:
        return int(policy.argmax()), policy
    return int(rng.choice(len(policy), p=policy)), policy


@torch.inference_mode()
def _plan_many(
    episodes: List[_Episode], model: PolicyValueBeliefNet, device: torch.device,
    training: TrainingConfig, config: MCTSConfig,
) -> List[Tuple[int, np.ndarray, np.ndarray, float | None]]:
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
        bias *= float(np.clip(config.strategic_prior_weight, 0.0, 1.0))
        logits[row, :bias.size] += torch.as_tensor(bias, dtype=logits.dtype, device=device)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    for row, packet in enumerate(packets):
        count = len(packet["actions"]["action_id"])
        model_prior = probabilities[row, :count]
        teacher_prior = teacher_policy(packet)
        anchor = float(np.clip(config.teacher_prior_weight, 0.0, 1.0))
        root_priors.append((1.0 - anchor) * model_prior + anchor * teacher_prior)

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
        return [(row, visits_by_episode[row], root_priors[row], None) for row in range(len(episodes))]
    pool = MctsPool(
        belief_envs, num_threads=_transient_pool_threads(config.native_threads),
        max_actions=config.native_max_actions, c_puct=config.c_puct,
    )
    # Each wave exposes at most one leaf per tree. With B active games and P
    # particles, a wave has up to B*P leaves and is the natural GPU batch.
    # ``simulations`` is the total search budget *per real-game decision*, as
    # in the former scalar trainer. One global pool wave evaluates one leaf
    # per particle tree, so running it ``simulations`` times accidentally
    # multiplied the requested budget by ``particles``. Divide it here.
    waves = max(1, int(np.ceil(config.simulations / max(1, config.particles))))
    for _ in range(waves):
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
        leaf_logits = leaf_output.policy_logits + _native_leaf_bias(
            leaf_tensors.action_categorical, leaf_tensors.action_continuous, leaf_tensors.action_mask,
            config.teacher_prior_weight, config.strategic_prior_weight,
        ).to(leaf_output.policy_logits.dtype)
        pool.expand_and_backup(
            leaf_ids, leaf_logits.float().cpu().numpy(),
            leaf_output.value.float().cpu().numpy(),
        )

    roots = pool.root_policy(temperature=1.0)
    root_value_sums = np.zeros(len(episodes), dtype=np.float64)
    root_value_counts = np.zeros(len(episodes), dtype=np.int64)
    raw_root_values = (
        np.asarray(roots["root_value"], dtype=np.float64).reshape(-1)
        if "root_value" in roots else np.empty(0)
    )
    for tree, owner in enumerate(owners):
        legal = np.asarray(roots["action_id"][tree], dtype=np.int64)
        mask = np.asarray(roots["action_mask"][tree], dtype=bool)
        count = np.asarray(roots["visit_count"][tree], dtype=np.float64)
        source_ids = np.asarray(packets[owner]["actions"]["action_id"], dtype=np.int64)
        lookup = {int(action): index for index, action in enumerate(source_ids)}
        for action, valid, visits in zip(legal, mask, count):
            if valid and int(action) in lookup:
                visits_by_episode[owner][lookup[int(action)]] += visits
        if tree < raw_root_values.size and np.isfinite(raw_root_values[tree]):
            root_value_sums[owner] += raw_root_values[tree]
            root_value_counts[owner] += 1
    result = [
        (
            row,
            visits_by_episode[row],
            root_priors[row],
            float(root_value_sums[row] / root_value_counts[row]) if root_value_counts[row] else None,
        )
        for row in range(len(episodes))
    ]
    # Make the native worker-pool lifetime explicit. Its NumPy outputs do not
    # retain a reference to the MctsPool, so this joins its workers now.
    del pool
    return result


def _apply_action(
    episode: _Episode,
    action_row: int,
    is_model: bool,
    policy: Optional[np.ndarray],
    training: TrainingConfig,
    bootstrap_value: float | None = None,
) -> None:
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
            bootstrap_value=bootstrap_value,
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
    assign_td_lambda_targets(episode.examples, episode.transitions, episode.winner, training)
    if not episode.env.is_done():
        for example in episode.examples:
            example.value_loss_weight = 0.0


def train_global_mcts_self_play(
    training: TrainingConfig, model_config: ModelConfig, mcts: MCTSConfig, games: int, updates_per_game: int,
    opponent: str, random_opponent_rate: float, mixed_opponent: str, teacher_replay_fraction: float,
    active_envs: int,
    model_out: Optional[Path],
    model: Optional[PolicyValueBeliefNet],
    opponent_models: Sequence[PolicyValueBeliefNet] = (),
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """High-throughput global-MCTS trainer; writes only final model weights."""
    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "self_play")
    run.write_config(
        training=training,
        model=model_config,
        mcts=mcts,
        stage={
            "games": games,
            "updates_per_game": updates_per_game,
            "opponent": opponent,
            "random_opponent_rate": random_opponent_rate,
            "mixed_opponent": mixed_opponent,
            "teacher_replay_fraction": teacher_replay_fraction,
            "active_envs": active_envs,
            "model_out": model_out,
        },
    )
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    # These are immutable adversaries.  They never receive gradients or an
    # optimiser and are sampled independently for each anchored game.
    frozen_models = [opponent_model.to(device).eval() for opponent_model in opponent_models]
    if (opponent == "model" or mixed_opponent == "model") and not frozen_models:
        raise ValueError("opponent model mode requires at least one --opponent-model checkpoint")
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    replay, teacher = ReplayBuffer(training.replay_capacity), ReplayBuffer(training.replay_capacity)
    schedule_rng, sample_rng = np.random.default_rng(training.seed ^ 0xA17C0), np.random.default_rng(training.seed)
    next_game, done = 0, 0
    active: List[_Episode] = []
    summaries: Dict[str, float] = {}
    frozen_opponent_games = 0
    losses: List[Dict[str, float]] = []
    optimisation_updates = 0

    def optimise_until(target: int, current_teacher_fraction: float) -> None:
        nonlocal optimisation_updates
        desired_teacher = int(round(
            training.batch_size * float(np.clip(current_teacher_fraction, 0.0, 1.0))
        ))
        while optimisation_updates < target:
            teacher_count = min(desired_teacher, len(teacher))
            self_play_count = training.batch_size - teacher_count
            if self_play_count > len(replay):
                teacher_count = min(len(teacher), teacher_count + self_play_count - len(replay))
                self_play_count = training.batch_size - teacher_count
            if self_play_count > len(replay) or teacher_count > len(teacher):
                break
            rows = replay.sample(
                self_play_count, sample_rng, training.priority_replay_fraction
            ) if self_play_count else []
            rows += teacher.sample(
                teacher_count, sample_rng, training.priority_replay_fraction
            ) if teacher_count else []
            if len(rows) < training.batch_size:
                break
            update_loss = train_batch(model, optimiser, stack_examples(rows, device), training)
            losses.append(update_loss)
            optimisation_updates += 1
            run.log(
                game=done,
                update=optimisation_updates,
                replay_examples=len(replay),
                teacher_replay_examples=len(teacher),
                teacher_replay_fraction=current_teacher_fraction,
                **update_loss,
            )

    model.eval()
    # Persist only the requested model path before native MCTS begins. It is
    # overwritten by later stages/final training and never contains replay,
    # games or optimiser state, but remains a usable resume point after a
    # fatal native-extension abort.
    if model_out is not None:
        save_model(model_out, model, model_config)
    with tqdm(total=games, desc=f"global native MCTS ({device})", unit="env", dynamic_ncols=True) as bar:
        while done < games:
            model.eval()
            curriculum_scale = 0.0 if games == 1 else max(0.0, 1.0 - done / (games - 1))
            search_mcts = replace(
                mcts,
                strategic_prior_weight=mcts.strategic_prior_weight * curriculum_scale,
                teacher_prior_weight=mcts.teacher_prior_weight * curriculum_scale,
            )
            while next_game < games and len(active) < active_envs:
                anchor = opponent in {"random", "heuristic", "model"} or (
                    opponent == "mixed" and schedule_rng.random() < random_opponent_rate
                )
                opponent_agent = (mixed_opponent if opponent == "mixed" else opponent) if anchor else None
                opponent_model_index = (
                    int(schedule_rng.integers(len(frozen_models))) if opponent_agent == "model" else None
                )
                frozen_opponent_games += int(opponent_model_index is not None)
                active.append(_new_episode(
                    next_game,
                    training,
                    next_game % 2 if anchor else None,
                    opponent_agent,
                    opponent_model_index,
                ))
                next_game += 1
            model_turns = []
            frozen_turns: Dict[int, List[_Episode]] = {}
            for episode in active:
                packet = episode.env.model_request_numpy()
                player = int(packet["obs"].get("current_player", episode.env.current_player()))
                if episode.train_player is None or player == episode.train_player:
                    model_turns.append(episode)
                elif episode.opponent_agent == "model":
                    if episode.opponent_model_index is None:
                        raise RuntimeError("model opponent game has no frozen model index")
                    frozen_turns.setdefault(episode.opponent_model_index, []).append(episode)
                else:
                    ids = np.asarray(packet["actions"]["action_id"])
                    row = int(episode.rng.integers(len(ids))) if episode.opponent_agent == "random" else choose_teacher_action(packet, episode.rng)
                    _apply_action(episode, row, False, None, training)
            # Search frozen opponent turns with their own policy/value network.
            # Their positions intentionally never enter the candidate replay.
            for frozen_index, episodes in frozen_turns.items():
                for index, visits, priors, _ in _plan_many(
                    episodes, frozen_models[frozen_index], device, training, search_mcts,
                ):
                    row, _ = _search_action(visits, priors, episodes[index].rng, search_mcts)
                    _apply_action(episodes[index], row, False, None, training)
            if model_turns:
                for index, visits, priors, bootstrap_value in _plan_many(
                    model_turns, model, device, training, search_mcts
                ):
                    episode = model_turns[index]
                    _, search_policy = _search_action(visits, priors, episode.rng, search_mcts)
                    policy = search_policy
                    # Keep a small teacher floor after the exploration/search
                    # curriculum ends. Otherwise the final online updates are
                    # precisely the ones most likely to erase the bootstrap.
                    anchor_scale = 0.25 + 0.75 * curriculum_scale
                    target_anchor = float(np.clip(
                        training.teacher_policy_target_weight * anchor_scale, 0.0, 1.0
                    ))
                    if target_anchor:
                        teacher_target = teacher_policy(episode.env.model_request_numpy())
                        policy = (1.0 - target_anchor) * policy + target_anchor * teacher_target
                    row, _ = _search_action(visits, priors, episode.rng, search_mcts)
                    if episode.rng.random() < training.exploration_rate * curriculum_scale:
                        bias = strategic_action_bias(episode.env.model_request_numpy())
                        weights = np.exp(np.clip(bias - bias.max(), -20.0, 0.0))
                        row = int(episode.rng.choice(len(weights), p=weights / weights.sum()))
                    _apply_action(episode, row, True, policy, training, bootstrap_value)
            completed = [episode for episode in active if episode.env.is_done() or episode.moves >= training.max_moves]
            for episode in completed:
                _finish(episode, training)
                replay.extend(episode.examples)
                active.remove(episode)
                done += 1
                bar.update(1)
                for key, value in episode.metrics.items():
                    summaries[key] = summaries.get(key, 0.0) + value
                if teacher_replay_fraction > 0.0:
                    teacher_examples, _ = _teacher_game(training.seed + 1_000_000 + episode.index, training); teacher.extend(teacher_examples)
                run.log(
                    game=done,
                    update=optimisation_updates,
                    replay_examples=len(replay),
                    game_moves=episode.moves,
                    winner=episode.winner,
                    curriculum_scale=curriculum_scale,
                    **episode.metrics,
                )
                searched_episodes = len(model_turns) + sum(len(rows) for rows in frozen_turns.values())
                bar.set_postfix(
                    replay=len(replay), active=len(active), updates=optimisation_updates,
                    trees=searched_episodes * mcts.particles,
                )
            update_scale = 0.0 if games == 1 else max(0.0, 1.0 - done / (games - 1))
            optimise_until(
                done * updates_per_game,
                teacher_replay_fraction * (0.25 + 0.75 * update_scale),
            )

    optimise_until(games * updates_per_game, teacher_replay_fraction * 0.25)
    bootstrap_examples = sum(example.bootstrap_value is not None for example in replay._examples)
    if model_out: save_model(model_out, model, model_config)
    result: Dict[str, float | str] = {
        "games": float(games), "active_envs": float(active_envs), "replay_examples": float(len(replay)),
        "teacher_replay_examples": float(len(teacher)),
        "mcts_value_bootstrap_fraction": float(bootstrap_examples / max(1, len(replay))),
        "native_mcts_threads": float(_transient_pool_threads(mcts.native_threads)),
        "frozen_opponent_games": float(frozen_opponent_games),
        "optimisation_updates": float(optimisation_updates),
        "online_updates": 1.0,
        "final_curriculum_scale": 0.0,
        "final_teacher_replay_fraction": float(teacher_replay_fraction * 0.25),
        "final_teacher_policy_target_weight": float(training.teacher_policy_target_weight * 0.25),
    }
    result.update({"mean_" + k: v / max(1, len(replay)) for k, v in summaries.items() if k.startswith("reward_")})
    if losses:
        result.update({key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]})
    result["run_dir"] = str(run.finish(result))
    return result
