from __future__ import annotations

from collections import deque
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from polybot.config import ModelConfig, TrainingConfig
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.run_artifacts import TrainingRun
from polybot.training.vector_population import (
    _PPOExample,
    _PolicySampleUsage,
    _PPGExample,
    _TerminalValueCalibration,
    _TerminalValueTrace,
    _clear_policy_replay_after_flat_rollback,
    train_vector_population,
)


def test_actor_usage_counts_samples_not_minibatch_averages():
    usage = _PolicySampleUsage()
    usage.add(
        [_PPOExample({}, 0, 0.0, 0.0, policy_selected=False)],
        np.asarray([5]), 1,
    )
    usage.add(
        [_PPOExample({}, 0, 0.0, 0.0) for _ in range(9)],
        np.zeros(9, dtype=int), 1,
    )
    metrics = usage.metrics()
    assert metrics["policy_selected_fraction"] == 0.9
    assert metrics["policy_stale_samples"] == 1
    assert metrics["policy_age_mean"] == 0.5
    assert metrics["policy_age_p50"] == 0
    assert metrics["policy_age_p95"] == 5
    assert metrics["policy_age_max"] == 5
    assert metrics["policy_advantage_filtered_samples"] == 0


def test_terminal_reservoir_is_bounded_and_covers_the_game():
    trace = _TerminalValueTrace()
    rng = np.random.default_rng(42)
    for turn in range(1000):
        trace.add(_PPOExample({"state": np.asarray([turn])}, 0, 0.0, 0.1), rng, 64)
    assert trace.seen == 1000
    assert len(trace.positions) == 64
    turns = [int(p.observation["state"][0]) for p in trace.positions]
    assert min(turns) < 100
    assert max(turns) > 900
    assert all(not hasattr(p, "action_row") for p in trace.positions)


def test_terminal_value_reports_bias_and_valid_variance():
    calibration = _TerminalValueCalibration()
    calibration.add([_PPOExample({}, 0, 0.0, -0.5)], -1.0)
    calibration.add([_PPOExample({}, 0, 0.0, 0.5)], 1.0)
    metrics = calibration.metrics()
    assert metrics["mae"] == 0.5
    assert metrics["bias"] == 0.0
    assert metrics["correlation"] == 1.0
    assert metrics["explained_variance"] == 0.75
    assert metrics["variance_valid"] == 1.0
    one_outcome = _TerminalValueCalibration()
    one_outcome.add([_PPOExample({}, 0, 0.0, 0.0)], 1.0)
    assert one_outcome.metrics()["variance_valid"] == 0.0
    assert one_outcome.metrics()["bias"] == -1.0


class _NineStepGame:
    """Deterministic public packet game: native actions terminate only at 9.

    Both slots act as player 0 throughout the episode. Only one is controlled
    by the candidate, so PPO must also handle intervening opponent-only slots.
    """

    def __init__(self, *, num_envs, max_actions, **kwargs):
        self.num_envs = num_envs
        self.max_actions = max_actions
        self.tick = 0

    def packet(self, terminal=False):
        features = np.full((self.num_envs, self.max_actions, 17), -1, dtype=np.int32)
        features[:, :2, 0] = 3
        features[:, :2, 12:14] = 0
        features[:, :2, 15:17] = 0
        features[:, 1, 4] = 1
        legal = np.zeros((self.num_envs, self.max_actions), dtype=np.uint8)
        legal[:, :2] = 1
        state = np.zeros((self.num_envs, 11), dtype=np.int32)
        state[:, 0] = self.tick % 9
        state[:, 3] = -1
        return {
            "map_tokens": np.zeros((self.num_envs, 121, 23), dtype=np.int32),
            "state": state,
            "action_features": features,
            "action_mask": legal,
            "action_id": np.tile(np.arange(self.max_actions), (self.num_envs, 1)),
            "visible_event_features": np.zeros((self.num_envs, 0, 26), dtype=np.int32),
            "visible_event_mask": np.zeros((self.num_envs, 0), dtype=np.uint8),
            "reward": np.full(self.num_envs, float(terminal), dtype=np.float32),
            "terminated": np.full(self.num_envs, terminal, dtype=np.uint8),
            "action_valid": np.ones(self.num_envs, dtype=np.uint8),
        }

    def reset(self, **kwargs):
        self.tick = 0
        return self.packet()

    def step(self, actions):
        assert len(actions) == self.num_envs
        self.tick += 1
        return self.packet(terminal=self.tick % 9 == 0)


@pytest.fixture(params=("factorized", "relational"))
def short_game_training(tmp_path, request):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = ModelConfig(
        d_model=16, transformer_layers=1, attention_heads=2, dropout=0.0,
        policy_architecture=request.param, action_interaction_dim=8,
    )
    candidate = PolicyValueBeliefNet(config)
    opponent = deepcopy(candidate)
    continuation = {}
    results = []

    def train(steps, synchronous=True):
        result = train_vector_population(
            TrainingConfig(device="cpu", seed=42, learning_rate=1e-5),
            config, candidate, [opponent], [1.0],
            num_envs=2, steps=steps, num_threads=1, max_actions=16,
            anchor_model=deepcopy(opponent), visible_action_history=0,
            policy_anchor_weight=0.0, value_anchor_weight=0.0,
            shaping_weight=0.0, objective_replay_weight=0.0,
            ppo_rollout_steps=4, ppo_epochs=1, ppo_minibatch_size=4,
            ppo_top_advantage_fraction=1.0, ppo_max_policy_age=1,
            ppo_synchronous_rollouts=synchronous,
            ppg_interval=1, ppg_buffer_capacity=64, ppg_samples_per_rollout=16,
            ppg_phase_samples=4, ppg_epochs=1, ppg_minibatch_size=4,
            continuation_state=continuation,
            run=TrainingRun.create(tmp_path, f"block-{len(results)}"),
        )
        results.append(result)
        return result

    with (
        patch("PolyEnv.VectorGameEnv", _NineStepGame),
        patch.object(TrainingRun, "_write_plots"),
    ):
        try:
            yield train, continuation, candidate
        finally:
            torch.set_num_threads(previous_threads)


def test_sync_ppo_updates_before_terminal_and_preserves_full_game_value(short_game_training):
    train, continuation, _ = short_game_training
    first = train(4)
    assert first["direct_population_completed_games"] == 0
    assert first["direct_population_ppo_updates"] > 0
    assert first["direct_population_ppo_samples"] == 4
    assert first["direct_population_value_calibration_count"] == 0
    assert first["direct_population_ppo_terminal_target_fraction"] == 0
    assert first["direct_population_ppo_pending_examples"] == 0
    assert continuation["value_traces"][0].seen == 4

    second = train(5)
    assert second["direct_population_ppo_environment_reused"] == 1
    assert second["direct_population_completed_games"] == 2
    assert second["direct_population_value_calibration_count"] == 9
    assert second["direct_population_ppo_policy_age_max"] == 0
    assert second["direct_population_ppo_policy_stale_samples"] == 0
    assert second["direct_population_ppo_policy_selected_fraction"] == 1
    assert second["direct_population_ppo_actor_applied_fraction"] == 1
    assert second["direct_population_ppg_terminal_target_fraction"] > 0
    terminals = [e for e in continuation["ppg_replay"] if e.has_terminal_value]
    assert len(terminals) == 9
    assert {int(e.observation["state"][0]) for e in terminals} == set(range(9))
    assert all(e.return_target == 1.0 for e in terminals)


def test_legacy_collection_still_waits_for_completed_games(short_game_training):
    train, _, _ = short_game_training
    result = train(4, synchronous=False)
    assert result["direct_population_ppo_updates"] == 0
    assert result["direct_population_completed_games"] == 0
    assert result["direct_population_ppo_pending_examples"] == 4


def test_rollback_keeps_pending_value_positions_but_clears_actor_data():
    trace = _TerminalValueTrace()
    trace.add(_PPOExample({}, 0, 0.0, 0.1), np.random.default_rng(1), 16)
    continuation = {
        "value_traces": [trace],
        "terminal_replay": deque(maxlen=8),
        "ppg_replay": deque([
            _PPGExample({}, 1.0, has_terminal_value=True),
            _PPGExample({}, 0.2),
        ], maxlen=8),
    }
    _clear_policy_replay_after_flat_rollback(
        continuation, num_envs=1, objective_replay_horizon=4,
        objective_replay_capacity=8, ppg_buffer_capacity=8,
        preserve_terminal_ppg=True,
    )
    assert continuation["value_traces"][0] is trace
    assert continuation["traces"] == [[]]
    assert len(continuation["ppg_replay"]) == 1
    assert continuation["ppg_replay"][0].return_target == 1.0
