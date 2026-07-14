"""Self-play data generation and an in-memory replay buffer."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import TrainingConfig
from ..data.polyenv_features import (
    action_features,
    capture_flags,
    event_summary,
    global_features_from_obs,
    strategic_action_bias,
    visible_events_or_empty,
)
from ..planning.belief_mcts import BeliefMCTS
from ..planning.heuristic_policy import choose_teacher_action
from .rewards import RewardShaper
from .replay_buffer import TrainingExample


def _winner_to_value(winner: object, player_id: int) -> float:
    try:
        winner_id = int(winner)
        if winner_id < 0:
            return 0.0
        return 1.0 if winner_id == player_id else -1.0
    except (TypeError, ValueError):
        return 0.0


def _sample_strategic_action_row(packet: Dict[str, object], rng: np.random.Generator) -> int:
    """Sample a legal action from the early-training strategic curriculum."""
    bias = strategic_action_bias(packet)
    weights = np.exp(np.clip(bias - bias.max(), -20.0, 0.0))
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def _action_bucket(name: str) -> str:
    """Small, stable action taxonomy for terminal-only training diagnostics."""
    if name == "end_turn":
        return "end_turn"
    if name == "move":
        return "move"
    if name == "attack":
        return "attack"
    if name == "tile_action:found_city":
        return "found_city"
    if name == "tile_action:capture_city":
        return "capture_city"
    return "other"


class SelfPlayRunner:
    """Generate training records without ever feeding full-map labels to policy."""

    def __init__(self, planner: BeliefMCTS, training: TrainingConfig, logger: Optional[Any] = None) -> None:
        self.planner = planner
        self.training = training
        self.logger = logger

    @staticmethod
    def _new_env(seed: int, map_size: int, map_type: str) -> object:
        try:
            from PolyEnv import Bardur, GameEnv, Imperius
        except ImportError as error:
            raise RuntimeError("PolyEnv is required. Install the project dependencies first.") from error
        return GameEnv(seed=seed, map_size=map_size, players=(Bardur, Imperius), map_type=map_type)

    def play_game(
        self,
        seed: int,
        iteration: int,
        game_index: int,
        train_player: Optional[int] = None,
        opponent_agent: Optional[str] = None,
        on_search_progress: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> Tuple[List[TrainingExample], Dict[str, float]]:
        env = self._new_env(seed, self.training.map_size, self.training.map_type)
        shaper = RewardShaper(self.training)
        examples: List[TrainingExample] = []
        shaped_by_player: Dict[int, float] = {}
        reward_totals: Dict[str, float] = {}
        action_counts: Dict[str, float] = {}
        transitions: List[Tuple[int, float, Optional[TrainingExample]]] = []
        rng = np.random.default_rng(seed ^ 0xC0DEC0DE)
        exploration_actions = 0
        moves = 0
        winner = -1
        opponent_actions = 0
        max_moves = self.training.max_moves

        while not env.is_done() and moves < max_moves:
            packet = env.model_request_numpy()
            events = visible_events_or_empty(env)
            player_id = int(packet["obs"].get("current_player", env.current_player()))
            is_model_turn = train_player is None or player_id == train_player
            def report_search(completed: int, total: int, particles: int) -> None:
                if on_search_progress:
                    on_search_progress(moves, completed, total, particles)

            decision = None
            if is_model_turn:
                decision = self.planner.decide(env, on_progress=report_search)
                action_row = decision.action_row
                if rng.random() < self.training.exploration_rate:
                    action_row = _sample_strategic_action_row(packet, rng)
                    exploration_actions += 1
            else:
                if opponent_agent not in {"random", "heuristic"}:
                    raise RuntimeError("a non-model turn requires a configured opponent agent")
                legal_count = len(np.asarray(packet["actions"]["action_id"]))
                action_row = (
                    int(rng.integers(legal_count))
                    if opponent_agent == "random"
                    else choose_teacher_action(packet, rng)
                )
                opponent_actions += 1
            action_id = int(np.asarray(packet["actions"]["action_id"], dtype=np.int64)[action_row])
            full_names = packet["actions"].get("type_fullname", [])
            action_name = str(full_names[action_row]) if action_row < len(full_names) else ""
            action_role = "model" if is_model_turn else "opponent"
            action_key = f"{action_role}_action_{_action_bucket(action_name)}"
            action_counts[action_key] = action_counts.get(action_key, 0.0) + 1.0
            action_counts[f"{action_role}_actions"] = action_counts.get(f"{action_role}_actions", 0.0) + 1.0

            example: Optional[TrainingExample] = None
            if is_model_turn:
                categorical, continuous = action_features(packet["actions"])
                if decision.visit_counts.sum() > 0:
                    policy = decision.visit_counts / decision.visit_counts.sum()
                else:
                    policy = decision.priors / decision.priors.sum()
                # Ground truth is captured only as a supervised belief target.
                # It is never passed to planner/model inference.
                example = TrainingExample(
                    map_tokens=np.asarray(packet["map_tokens"], dtype=np.int64).copy(),
                    global_features=global_features_from_obs(packet["obs"]),
                    event_features=event_summary(events),
                    action_categorical=categorical.copy(),
                    action_continuous=continuous.copy(),
                    policy_target=policy.astype(np.float32),
                    full_map_target=np.asarray(env.full_map_numpy(), dtype=np.int64).copy(),
                    hidden_mask=np.asarray(packet["map_tokens"])[:, 0] == 0,
                    player_id=player_id,
                )
            captured_city, captured_capital = capture_flags(packet, action_row)
            ok, done, terminal_reward, winner, _ = env.step_fast(action_id)
            if not ok:
                raise RuntimeError("self-play selected an action that PolyEnv rejected")
            after_obs = env.observation(player_id=player_id, visible_only=True)
            reward = shaper.observe(
                player_id,
                after_obs,
                terminal=float(terminal_reward),
                captured_city=captured_city,
                captured_capital=captured_capital,
            )
            if example is not None:
                example.shaped_reward = reward.non_terminal
            transitions.append((player_id, reward.non_terminal, example))
            shaped_by_player[player_id] = shaped_by_player.get(player_id, 0.0) + reward.total
            for name, value in reward.as_dict().items():
                reward_totals[name] = reward_totals.get(name, 0.0) + value
            if example is not None:
                examples.append(example)
            if self.logger:
                self.logger.log_step({
                    "iteration": iteration,
                    "game": game_index,
                    "move": moves,
                    "player": player_id,
                    "action_id": action_id,
                    "legal_actions": int(len(packet["actions"]["action_id"])),
                    "mcts_particles": decision.valid_particles if decision is not None else 0,
                    **reward.as_dict(),
                })
            moves += 1

        players = {example.player_id for example in examples}
        future_advantage = {player: 0.0 for player in players}
        for actor, shaped_reward, example in reversed(transitions):
            for player in players:
                sign = 1.0 if player == actor else -1.0
                future_advantage[player] = (
                    self.training.reward_discount * future_advantage[player]
                    + sign * shaped_reward
                )
            if example is None:
                continue
            terminal_value = _winner_to_value(winner, example.player_id)
            example.value_target = float(np.clip(
                terminal_value + self.training.shaping_target_weight * future_advantage[example.player_id],
                -1.0,
                1.0,
            ))
        metrics: Dict[str, float] = {
            "moves": float(moves),
            "winner": float(winner),
            "examples": float(len(examples)),
            "exploration_actions": float(exploration_actions),
            "opponent_actions": float(opponent_actions),
        }
        for player, reward in shaped_by_player.items():
            metrics["shaped_reward_player_%d" % player] = reward
        metrics.update(reward_totals)
        metrics.update(action_counts)
        return examples, metrics
