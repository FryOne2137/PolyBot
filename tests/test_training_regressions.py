from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from polybot.config import MCTSConfig, ModelConfig, TrainingConfig
from polybot.data.polyenv_features import stack_examples
from polybot.data.token_schema import MAP_FEATURE_MAX
from polybot.evaluation.matches import (
    _paired_seed_and_slot,
    evaluate_direct_matches,
)
from polybot.infrastructure.randomness import seed_everything
from polybot.infrastructure.model_io import save_model
from polybot.models.policy_value_belief import (
    PolicyValueBeliefNet,
    apply_public_bridge_constraints,
)
from polybot.planning.action_pruning import (
    mandatory_search_actions,
    prune_search_logits,
    restrict_root_policy_to_candidates,
)
from polybot.training.belief_warmup import (
    _BELIEF_ONLY_PREFIXES,
    _configure_belief_only_training,
    _restore_trainable_parameters,
)
from polybot.training.direct_curriculum import (
    direct_readiness,
    model_state_on_cpu,
    restore_model_state,
    train_direct_policy_curriculum,
)
from polybot.training.belief_recovery import (
    BeliefRecoveryRequired,
    _invalid_unit_token_details,
    capped_update_target,
    ismcts_action_difference_rejections,
    effective_replay_epochs,
    ismcts_rejected_envs,
    particle_temperature,
    partition_recovery_envs,
    recovery_rate_exceeded,
    replace_rejected_particles,
    replacement_seeds,
)
from polybot.training.global_mcts_train import _native_leaf_bias
from polybot.training.league import (
    _challenger_continuation_action,
    _champion_focused_probabilities,
    _curriculum_controls,
    _flat_distillation_gate_decision,
    _mcts_probe_passed,
    _promotion_gate_decision,
    _round_champion_mode,
    _select_exploiter_checkpoint,
    train_promotion_league,
)
from polybot.training.league_db import LeagueDatabase
from polybot.training.optimisation import (
    BELIEF_LOSS_FEATURES,
    BELIEF_RUNTIME_ID_FEATURES,
    belief_terrain_metrics,
    belief_loss,
    bridge_constraint_loss,
    train_batch,
)
from polybot.training.replay_buffer import ReplayBuffer, TrainingExample
from polybot.training.routed_self_play_pool import (
    _evaluation_pair_rows,
    _evaluation_search_with_recovery,
    _freeze_reference_model,
    _next_evaluation_replacement_seed,
    _paired_pool_schedule,
    _policy_distance,
    _reference_policy,
)
from polybot.training.self_play_pool_train import (
    _belief_rejection_kind,
    _finish_trajectory,
    _root_examples,
    _scheduled_updates,
    _winner_from_terminal_values,
)
from polybot.training.vector_actor_critic import (
    _normalise_advantage,
    _visible_action_reward,
)
from polybot.training.vector_objectives import objective_gae
from polybot.training.vector_teacher import _teacher_targets, _visible_goal_progress
from polybot.training.vector_population import (
    _PPGTargetExample,
    _PPOExample,
    _PPOTransition,
    _TerminalValueCalibration,
    _ppo_batch,
    _ppo_snapshot,
    _finish_ppo_trace,
    _mark_top_advantage_examples,
    _population_block_acceptance,
    _population_block_selects_checkpoint,
    _objective_replay_batch,
    _objective_replay_snapshot,
    _objective_replay_success_weights,
    _index_copy_policy_rows_,
    _masked_policy_kl,
    _ppg_batch,
    _update_parameter_ema,
    evaluate_direct_population_matches,
    train_direct_population_curriculum,
    train_vector_population,
)


class _MemoryRun:
    directory = Path(".")

    def write_config(self, **kwargs):
        return None

    def child(self, name):
        return self

    def log(self, **kwargs):
        return None

    def set_metric_sink(self, sink):
        self.metric_sink = sink

    def finish(self, summary):
        return self.directory


def test_population_policy_copy_accepts_mixed_precision_source():
    target = torch.zeros((3, 2), dtype=torch.float16)
    source = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    _index_copy_policy_rows_(target, torch.tensor([1]), source)

    assert target.dtype == torch.float16
    torch.testing.assert_close(
        target[1].float(), source[0], atol=1e-3, rtol=0.0
    )


def test_ppo_action_rows_leave_inference_mode_before_opponent_replacement():
    with torch.inference_mode():
        sampled_rows = torch.tensor([2, 4, 6], dtype=torch.long)

    # PPO samples its behaviour policy without autograd, but the opponent rows
    # are inserted afterwards.  The clone must happen outside inference mode.
    action_rows = sampled_rows.clone()
    action_rows.index_copy_(
        0,
        torch.tensor([1], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    )

    assert action_rows.tolist() == [2, 9, 6]


def test_post_bootstrap_population_focuses_current_champion_safely():
    probabilities = _champion_focused_probabilities(
        [10, 11, 12],
        [0.40, 0.38, 0.22],
        10,
        champion_probability=0.60,
        probability_floor=0.05,
    )

    assert np.isclose(sum(probabilities), 1.0)
    assert np.isclose(probabilities[0], 0.60)
    assert min(probabilities[1:]) >= 0.05


def test_promoted_lineage_skips_teacher_and_actor_for_champion_population():
    controls = _curriculum_controls(
        champion_status="promoted",
        direct_teacher_steps=2_000,
        direct_actor_steps=1_000,
        direct_actor_max_steps=5_000,
        direct_population_steps=2_000,
        direct_population_max_steps=10_000,
        direct_policy_anchor_weight=0.75,
        direct_population_teacher_weight=0.25,
        direct_population_opponent_policy_weight=0.50,
        post_bootstrap_population_steps=4_000,
        post_bootstrap_population_max_steps=8_000,
        post_bootstrap_policy_anchor_weight=0.50,
        post_bootstrap_teacher_weight=0.05,
        post_bootstrap_opponent_policy_weight=0.75,
        direct_objective_steps=2_000,
        post_bootstrap_objective_steps=500,
    )

    assert controls["phase"] == "champion_population"
    assert controls["teacher_steps"] == 0
    # Established champions no longer run arbitrary short objective rollouts;
    # real-event prefix replay happens inside population training instead.
    assert controls["objective_steps"] == 0
    assert controls["actor_steps"] == 0
    assert controls["population_steps"] == 4_000
    assert controls["population_max_steps"] == 8_000
    assert controls["policy_anchor_weight"] == 0.50
    assert controls["population_teacher_weight"] == 0.05


def test_promoted_recovery_restores_dense_curriculum_with_anchor_controls():
    controls = _curriculum_controls(
        champion_status="promoted",
        direct_teacher_steps=2_000,
        direct_actor_steps=1_000,
        direct_actor_max_steps=5_000,
        direct_population_steps=9_000,
        direct_population_max_steps=18_000,
        direct_policy_anchor_weight=0.30,
        direct_population_teacher_weight=0.25,
        direct_population_opponent_policy_weight=0.0,
        post_bootstrap_population_steps=12_000,
        post_bootstrap_population_max_steps=18_000,
        post_bootstrap_policy_anchor_weight=0.20,
        post_bootstrap_teacher_weight=0.0,
        post_bootstrap_opponent_policy_weight=0.0,
        direct_objective_steps=1_000,
        refresh_promoted_curriculum=True,
    )

    assert controls["phase"] == "promoted_recovery"
    assert controls["post_bootstrap"]
    assert controls["teacher_steps"] == 2_000
    assert controls["objective_steps"] == 1_000
    assert controls["actor_steps"] == 1_000
    assert controls["population_steps"] == 9_000
    assert controls["policy_anchor_weight"] == 0.30


def test_auto_mode_uses_persisted_capability_to_switch_to_mcts():
    assert _round_champion_mode("auto", None) == "flat"
    assert _round_champion_mode("auto", {"mcts_ready": 0}) == "flat"
    assert _round_champion_mode("auto", {"mcts_ready": 1}) == "mcts"
    assert _round_champion_mode("flat", {"mcts_ready": 1}) == "flat"


def test_population_checkpoint_selection_prioritises_strength_with_safety_floor():
    accepted, reason = _population_block_acceptance(
        baseline=False,
        weighted_score=0.61,
        worst_score=0.45,
        objective_progress=0.30,
        best_score=0.58,
        best_objective_progress=0.80,
        historical_score_floor=0.45,
        score_tolerance=0.02,
        objective_tolerance=0.05,
    )
    assert accepted and reason == "score_improved"

    accepted, reason = _population_block_acceptance(
        baseline=False,
        weighted_score=0.62,
        worst_score=0.44,
        objective_progress=1.0,
        best_score=0.58,
        best_objective_progress=0.80,
        historical_score_floor=0.45,
        score_tolerance=0.02,
        objective_tolerance=0.05,
    )
    assert not accepted and reason == "historical_floor"


def test_population_near_tie_does_not_replace_the_actual_high_water_mark():
    selected, reason = _population_block_selects_checkpoint(
        baseline=False,
        weighted_score=0.59,
        objective_progress=1.00,
        best_score=0.60,
        best_objective_progress=0.20,
    )

    assert not selected
    assert reason == "safe_but_not_best"


def test_population_safe_near_tie_resumes_from_strongest_checkpoint():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    candidate = PolicyValueBeliefNet(config)
    champion = PolicyValueBeliefNet(config)
    gate_rows = [
        {
            "model_a_score": 0.50,
            "worst_score": 0.50,
            "model_a_city_captures_per_game": 0.10,
            "completion_rate": 0.50,
            "opponents": [],
        },
        {
            "model_a_score": 0.60,
            "worst_score": 0.60,
            "model_a_city_captures_per_game": 0.20,
            "completion_rate": 0.60,
            "opponents": [],
        },
        {
            "model_a_score": 0.59,
            "worst_score": 0.59,
            "model_a_city_captures_per_game": 0.40,
            "completion_rate": 0.80,
            "opponents": [],
        },
        {
            "model_a_score": 0.60,
            "worst_score": 0.60,
            "model_a_city_captures_per_game": 0.20,
            "completion_rate": 0.60,
            "opponents": [],
        },
    ]
    states = []
    population_kwargs = []

    def population_update(*args, **kwargs):
        population_kwargs.append(kwargs)
        with torch.no_grad():
            next(candidate.parameters()).add_(0.01)
        states.append(next(candidate.parameters()).detach().clone())
        return {"steps": 1.0}

    with (
        patch(
            "polybot.training.vector_population.evaluate_direct_population_matches",
            side_effect=gate_rows,
        ),
        patch(
            "polybot.training.vector_population.train_vector_population",
            side_effect=population_update,
        ),
    ):
        result = train_direct_population_curriculum(
            TrainingConfig(device="cpu"),
            config,
            candidate,
            champion,
            [champion],
            [1.0],
            num_envs=2,
            minimum_steps=2_000,
            maximum_steps=2_000,
            block_steps=1_000,
            num_threads=1,
            max_actions=16,
            gate_games=2,
            diagnostic_gate_games=2,
            gate_seed=100,
            gate_score=0.50,
            historical_score_floor=0.45,
            readiness_city_captures_per_game=0.05,
            readiness_completion_rate=0.05,
            readiness_patience=2,
            policy_anchor_weight=0.75,
            value_anchor_weight=0.50,
            opponent_policy_weight=0.50,
            teacher_weight=0.25,
            shaping_weight=0.20,
            objective_replay_horizon=7,
            objective_replay_capacity=96,
            objective_replay_batch_size=24,
            objective_replay_interval=5,
            objective_replay_weight=0.30,
            objective_replay_recency=0.80,
            objective_replay_max_epochs=3,
            ppg_interval=6,
            ppg_buffer_capacity=512,
            ppg_samples_per_rollout=64,
            ppg_phase_samples=256,
            ppg_epochs=3,
            ppg_minibatch_size=32,
            ppg_policy_kl_weight=1.5,
            ppg_value_weight=0.75,
            run=_MemoryRun(),
        )

    assert result["direct_population_accepted_blocks"] == 2.0
    assert result["direct_population_selected_blocks"] == 1.0
    assert result["direct_population_gate_rows"][2]["selected"] == 0.0
    torch.testing.assert_close(next(candidate.parameters()), states[0])
    assert population_kwargs[0]["objective_replay_horizon"] == 7
    assert population_kwargs[0]["objective_replay_capacity"] == 96
    assert population_kwargs[0]["objective_replay_batch_size"] == 24
    assert population_kwargs[0]["objective_replay_interval"] == 5
    assert population_kwargs[0]["objective_replay_weight"] == 0.30
    assert population_kwargs[0]["objective_replay_recency"] == 0.80
    assert population_kwargs[0]["objective_replay_max_epochs"] == 3
    assert population_kwargs[0]["ppg_interval"] == 6
    assert population_kwargs[0]["ppg_buffer_capacity"] == 512
    assert population_kwargs[0]["ppg_samples_per_rollout"] == 64
    assert population_kwargs[0]["ppg_phase_samples"] == 256
    assert population_kwargs[0]["ppg_epochs"] == 3
    assert population_kwargs[0]["ppg_minibatch_size"] == 32
    assert population_kwargs[0]["ppg_policy_kl_weight"] == 1.5
    assert population_kwargs[0]["ppg_value_weight"] == 0.75


def test_signed_ppo_gae_propagates_opponent_terminal_win_as_candidate_loss():
    example = _PPOExample(
        observation={}, action_row=0, old_log_probability=0.0, old_value=0.2
    )
    finished = _finish_ppo_trace(
        [
            _PPOTransition(
                value=0.2,
                reward=0.0,
                next_value=0.3,
                sign=-1.0,
                terminal=False,
                candidate_example=example,
            ),
            _PPOTransition(
                value=0.3,
                reward=1.0,
                next_value=0.0,
                sign=1.0,
                terminal=True,
                candidate_example=None,
            ),
        ],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert finished == [example]
    assert np.isclose(example.return_target, -1.0)
    assert example.advantage < 0.0


def test_finished_game_value_target_is_undiscounted_even_when_gae_is_short():
    example = _PPOExample(
        observation={}, action_row=0, old_log_probability=0.0, old_value=0.4
    )
    finished = _finish_ppo_trace(
        [
            _PPOTransition(
                value=0.4,
                reward=0.0,
                next_value=0.2,
                sign=1.0,
                terminal=False,
                candidate_example=example,
            ),
            _PPOTransition(
                value=0.2,
                reward=1.0,
                next_value=0.0,
                sign=1.0,
                terminal=True,
                candidate_example=None,
            ),
        ],
        gamma=0.5,
        gae_lambda=0.5,
        terminal_outcome=1.0,
    )

    assert finished == [example]
    assert example.value_target == 1.0
    assert example.return_target != example.value_target


def test_terminal_value_calibration_uses_only_terminal_outcomes():
    calibration = _TerminalValueCalibration()
    calibration.add(
        [_PPOExample({"state": np.asarray([10])}, 0, 0.0, -1.0)], -1.0
    )
    calibration.add(
        [_PPOExample({"state": np.asarray([75])}, 0, 0.0, 1.0)], 1.0
    )

    metrics = calibration.metrics()

    assert metrics["count"] == 2.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["correlation"] == 1.0
    assert metrics["opening_count"] == 1.0
    assert metrics["late_count"] == 1.0


def test_top_advantage_filter_keeps_only_the_highest_half_for_policy():
    examples = [
        _PPOExample({}, 0, 0.0, 0.0, advantage=value)
        for value in (-2.0, -1.0, 0.0, 3.0)
    ]

    selected = _mark_top_advantage_examples(examples, 0.50)

    assert selected == 2
    assert [example.policy_selected for example in examples] == [False, False, True, True]


def test_parameter_ema_tracks_the_policy_without_aliasing_its_weights():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    model = PolicyValueBeliefNet(config)
    ema = PolicyValueBeliefNet(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2.0)
        for parameter in ema.parameters():
            parameter.zero_()

    _update_parameter_ema(ema, model, 0.5)

    torch.testing.assert_close(
        next(ema.parameters()),
        torch.full_like(next(ema.parameters()), 1.0),
    )


def test_exploiter_only_replaces_main_after_the_same_development_floor():
    main = {
        "direct_population_full_gate_selected": 1.0,
        "direct_population_full_gate_score": 0.55,
        "direct_population_full_gate_worst_score": 0.48,
    }
    better_exploiter = {
        "direct_population_full_gate_selected": 1.0,
        "direct_population_full_gate_score": 0.56,
        "direct_population_full_gate_worst_score": 0.46,
    }
    brittle_exploiter = {
        "direct_population_full_gate_selected": 0.0,
        "direct_population_full_gate_score": 0.70,
        "direct_population_full_gate_worst_score": 0.30,
    }

    assert _select_exploiter_checkpoint(main, better_exploiter)
    assert not _select_exploiter_checkpoint(main, brittle_exploiter)


def test_ppo_snapshot_compacts_legal_actions_without_changing_target_row():
    batch = {
        "map_tokens": np.zeros((1, 9, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 4, 17), dtype=np.int32),
        "action_mask": np.asarray(((True, False, True, False),)),
        "visible_event_features": np.zeros((1, 0, 26), dtype=np.int32),
        "visible_event_mask": np.zeros((1, 0), dtype=bool),
    }
    batch["action_features"][0, 2, 8] = 14
    observation, action_row = _ppo_snapshot(batch, row=0, action_row=2)
    example = _PPOExample(observation, action_row, 0.0, 0.0)
    replay, actions, _, _, _ = _ppo_batch([example])

    assert actions.tolist() == [1]
    assert replay["action_mask"].tolist() == [[True, True]]
    assert replay["action_features"][0, 1, 8] == 14


def test_ppg_batch_and_policy_kl_ignore_padded_actions():
    batch = {
        "map_tokens": np.zeros((1, 9, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 3, 17), dtype=np.int32),
        "action_mask": np.asarray(((True, True, False),)),
        "visible_event_features": np.zeros((1, 2, 26), dtype=np.int32),
        "visible_event_mask": np.asarray(((False, True),)),
    }
    observation, _ = _ppo_snapshot(batch, row=0, action_row=0)
    reference = torch.log_softmax(torch.tensor((1.0, 2.0)), dim=0)
    example = _PPGTargetExample(
        observation=observation,
        policy_log_probabilities=reference.numpy(),
        return_target=0.75,
    )
    replay, frozen_logs, returns = _ppg_batch([example])
    mask = torch.as_tensor(replay["action_mask"])
    identical = torch.tensor(((1.0, 2.0),))
    changed = torch.tensor(((2.0, 1.0),))

    assert returns.tolist() == [0.75]
    assert replay["visible_event_mask"].tolist() == [[False, True]]
    assert torch.isclose(
        _masked_policy_kl(torch.as_tensor(frozen_logs), identical, mask),
        torch.tensor(0.0),
        atol=1e-6,
    )
    assert _masked_policy_kl(
        torch.as_tensor(frozen_logs), changed, mask
    ).item() > 0.0


def _example(player: int) -> TrainingExample:
    return TrainingExample(
        map_tokens=np.zeros((1, 23), dtype=np.int64),
        global_features=np.zeros(9, dtype=np.float32),
        event_features=np.zeros(12, dtype=np.float32),
        action_categorical=np.zeros((1, 12), dtype=np.int64),
        action_continuous=np.zeros((1, 4), dtype=np.float32),
        policy_target=np.ones(1, dtype=np.float32),
        full_map_target=np.zeros((1, 23), dtype=np.int64),
        hidden_mask=np.ones(1, dtype=bool),
        player_id=player,
    )


def test_pool_terminal_result_reaches_every_decision():
    examples = [_example(0), _example(1), _example(0), _example(1)]
    transitions = [
        (example.player_id, 0.0, example)
        for example in examples
    ]
    # A discount may weaken shaping, but must not turn an early certain win
    # into a value close to zero. Native MCTS backs up terminal +/-1 directly.
    training = TrainingConfig(reward_discount=0.5, td_lambda=1.0, shaping_target_weight=0.0)

    finished = _finish_trajectory(transitions, training, np.asarray([1.0, -1.0]))

    assert [id(example) for example in finished] == [id(example) for example in examples]
    assert [example.value_target for example in examples] == [1.0, -1.0, 1.0, -1.0]


def test_truncated_pool_game_does_not_teach_a_fake_draw_value():
    examples = [_example(0), _example(1)]
    transitions = [(example.player_id, 0.0, example) for example in examples]

    finished = _finish_trajectory(transitions, TrainingConfig(), terminal_values=None)

    assert len(finished) == 2
    assert [example.value_loss_weight for example in finished] == [0.0, 0.0]


def test_zero_weight_removes_truncated_example_from_value_loss():
    example = _example(0)
    example.value_target = 1.0
    example.value_loss_weight = 0.0
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0)

    result = train_batch(
        model,
        optimiser,
        stack_examples([example], torch.device("cpu")),
        TrainingConfig(belief_weight=0.0),
    )

    assert result["value_loss"] == 0.0


def test_pool_policy_target_keeps_a_frozen_predecessor_anchor():
    request = {
        "map_tokens": np.zeros((1, 1, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 2, 17), dtype=np.int32),
        "action_mask": np.ones((1, 2), dtype=bool),
        "to_play": np.zeros(1, dtype=np.int32),
    }
    root = {"policy": np.asarray([[1.0, 0.0]], dtype=np.float32)}
    beliefs = np.zeros((1, 1, 23), dtype=np.int32)

    example = _root_examples(
        request,
        root,
        TrainingConfig(),
        beliefs,
        anchor_policy=np.asarray([[0.0, 1.0]], dtype=np.float32),
        anchor_policy_weight=0.25,
    )[0]

    np.testing.assert_allclose(example.policy_target, [0.75, 0.25])


def test_pool_value_target_keeps_a_separate_frozen_predecessor_anchor():
    request = {
        "map_tokens": np.zeros((1, 1, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 1, 17), dtype=np.int32),
        "action_mask": np.ones((1, 1), dtype=bool),
        "to_play": np.zeros(1, dtype=np.int32),
    }
    root = {"policy": np.ones((1, 1), dtype=np.float32)}
    example = _root_examples(
        request,
        root,
        TrainingConfig(),
        np.zeros((1, 1, 23), dtype=np.int32),
        anchor_value=np.asarray([0.75], dtype=np.float32),
        anchor_value_weight=0.50,
    )[0]

    assert example.anchor_value_target == 0.75
    assert example.anchor_value_loss_weight == 0.50


def test_pool_engine_action_constraint_marks_hidden_action_endpoint():
    request = {
        "map_tokens": np.zeros((1, 2, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 1, 17), dtype=np.int32),
        "action_mask": np.ones((1, 1), dtype=bool),
        "to_play": np.zeros(1, dtype=np.int32),
    }
    request["map_tokens"][0, 0, 0] = 1
    root = {"policy": np.ones((1, 1), dtype=np.float32)}
    focus = np.asarray([[False, True]], dtype=bool)
    example = _root_examples(
        request, root, TrainingConfig(), np.zeros((1, 2, 23), dtype=np.int32),
        engine_action_constraint_tiles=focus,
        engine_action_constraint_scores=np.asarray([3.0], dtype=np.float32),
    )[0]

    assert example.engine_action_constraint_priority
    assert example.engine_action_constraint_score == 3.0
    assert example.engine_action_constraint_tiles.tolist() == [False, True]
    assert stack_examples([example], torch.device("cpu"))["engine_action_constraint_mask"].tolist() == [[False, True]]


def test_engine_action_constraint_loss_trains_only_focused_hidden_belief():
    example = _example(0)
    example.engine_action_constraint_tiles = np.ones(1, dtype=bool)
    example.engine_action_constraint_priority = True
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0)

    result = train_batch(
        model,
        optimiser,
        stack_examples([example], torch.device("cpu")),
        TrainingConfig(
            belief_weight=0.0,
            engine_action_constraint_weight=1.0,
            bridge_constraint_weight=0.0,
        ),
    )

    assert result["belief_engine_action_constraint_loss"] > 0.0


def test_value_anchor_coefficient_is_not_cancelled_by_loss_normalisation():
    example = _example(0)
    example.value_loss_weight = 0.0
    example.anchor_value_target = 1.0
    example.anchor_value_loss_weight = 0.5
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0)

    result = train_batch(
        model,
        optimiser,
        stack_examples([example], torch.device("cpu")),
        TrainingConfig(belief_weight=0.0),
    )

    assert np.isclose(result["value_anchor_loss"], 0.25)


def test_objective_replay_credits_only_visible_candidate_successes():
    features = np.zeros((4, 1, 17), dtype=np.int32)
    features[:, :, 14] = 1
    features[0, 0, :3] = (0, 0, 1)  # move onto the visible village
    features[1, 0, (0, 8)] = (8, 14)  # capture a city
    features[2, 0, (0, 8)] = (8, 11)  # found a city
    features[3, 0, :3] = (0, 0, 2)  # terminal win below
    tokens = np.zeros((4, 9, 23), dtype=np.int32)
    tokens[..., 0] = 1
    tokens[0, 1, 14] = 1
    state = np.zeros((4, 9), dtype=np.int32)

    weights = _objective_replay_success_weights(
        features,
        np.zeros(4, dtype=np.int64),
        tokens,
        state,
        candidate_rows=np.asarray((0, 2, 3), dtype=np.int64),
        terminal=np.asarray((False, False, False, True)),
        terminal_reward=np.asarray((0.0, 0.0, 0.0, 1.0)),
        map_size=3,
    )

    # Row 1 captured a city but belonged to the frozen opponent, so it is
    # not a learning target for the candidate.
    np.testing.assert_allclose(weights, (0.25, 0.0, 0.35, 1.0))


def test_objective_replay_snapshots_only_public_model_inputs():
    batch = {
        "map_tokens": np.zeros((1, 9, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 2, 17), dtype=np.int32),
        "action_mask": np.asarray(((True, False),)),
        "visible_event_features": np.zeros((1, 0, 26), dtype=np.int32),
        "visible_event_mask": np.zeros((1, 0), dtype=bool),
    }
    example = _objective_replay_snapshot(batch, row=0, action_row=0)
    batch["map_tokens"][0, 0, 0] = 99
    replay_batch, actions, weights = _objective_replay_batch([example])

    assert replay_batch["map_tokens"][0, 0, 0] == 0
    assert actions.tolist() == [0]
    np.testing.assert_allclose(weights, [1.0])


def test_visible_objective_curriculum_prefers_and_rewards_entering_a_village():
    features = np.zeros((1, 2, 17), dtype=np.int32)
    features[0, 0, :3] = (0, 0, 1)  # move directly onto visible village
    features[0, 1, :3] = (0, 0, 3)  # move away
    features[..., 14] = 1
    mask = np.ones((1, 2), dtype=bool)
    tokens = np.zeros((1, 9, 23), dtype=np.int32)
    tokens[..., 0] = 1
    tokens[0, 1, 14] = 1
    state = np.zeros((1, 9), dtype=np.int32)

    teacher = _teacher_targets(features, mask, tokens, state, map_size=3)
    reward_to_village = _visible_action_reward(
        features, np.asarray([0]), tokens, state, map_size=3
    )
    reward_away = _visible_action_reward(
        features, np.asarray([1]), tokens, state, map_size=3
    )

    assert teacher.argmax(axis=1).tolist() == [0]
    assert reward_to_village[0] > reward_away[0]


def test_visible_objective_prioritises_opening_village_over_enemy_unit():
    features = np.zeros((1, 2, 17), dtype=np.int32)
    features[0, :, 0] = 0
    features[0, :, 1] = 0
    features[0, :, 2] = (1, 3)  # toward unit / toward village
    features[..., 14] = 1
    mask = np.ones((1, 2), dtype=bool)
    tokens = np.zeros((1, 9, 23), dtype=np.int32)
    tokens[..., 0] = 1
    tokens[0, 6, 14] = 1  # visible village
    tokens[0, 2, 4] = 1
    tokens[0, 2, 3] = 1  # visible enemy unit
    state = np.zeros((1, 9), dtype=np.int32)
    state[0, 6] = 1  # opening: just the capital

    teacher = _teacher_targets(features, mask, tokens, state, map_size=3)
    goals, _ = _visible_goal_progress(features, tokens, state, map_size=3)

    assert goals[0, 1] > goals[0, 0]
    assert teacher.argmax(axis=1).tolist() == [1]


def test_objective_gae_transports_future_reward_across_turn_switches():
    advantages = objective_gae(
        torch.tensor([[0.10], [0.20]], dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
        gamma=0.90,
        gae_lambda=0.80,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[-0.044], [0.20]], dtype=torch.float32),
    )


def test_objective_gae_stops_at_terminal_auto_reset_rows():
    advantages = objective_gae(
        torch.tensor([[0.10], [0.90]], dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.10], [0.90]], dtype=torch.float32),
    )


def test_in_memory_curriculum_rollback_restores_every_parameter():
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    original = model_state_on_cpu(model)
    for parameter in model.parameters():
        parameter.data.add_(1.0)

    restore_model_state(model, original)

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), original[name])


def test_mcts_readiness_requires_score_objectives_and_completed_games():
    result = {
        "model_a_score": 0.55,
        "model_a_city_captures_per_game": 0.10,
        "completion_rate": 0.0,
    }

    assert not direct_readiness(
        result,
        score=0.50,
        city_captures_per_game=0.05,
        completion_rate=0.05,
    )
    result["completion_rate"] = 0.10
    assert direct_readiness(
        result,
        score=0.50,
        city_captures_per_game=0.05,
        completion_rate=0.05,
    )


def test_direct_population_gate_uses_weighted_and_worst_opponent_scores():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    candidate = PolicyValueBeliefNet(config)
    opponents = [PolicyValueBeliefNet(config), PolicyValueBeliefNet(config)]
    responses = [
        {
            "model_a_score": 0.75,
            "model_a_city_captures_per_game": 0.20,
            "completion_rate": 0.80,
        },
        {
            "model_a_score": 0.25,
            "model_a_city_captures_per_game": 0.10,
            "completion_rate": 0.40,
        },
    ]

    with patch(
        "polybot.training.vector_population.evaluate_direct_matches",
        side_effect=responses,
    ) as evaluator:
        result = evaluate_direct_population_matches(
            TrainingConfig(),
            candidate,
            opponents,
            [0.75, 0.25],
            games=2,
            gate_seed=100,
        )

    assert np.isclose(result["model_a_score"], 0.625)
    assert np.isclose(result["worst_score"], 0.25)
    assert np.isclose(result["model_a_city_captures_per_game"], 0.175)
    assert np.isclose(result["completion_rate"], 0.70)
    assert evaluator.call_args_list[0].args[3].seed == 100
    assert evaluator.call_args_list[1].args[3].seed == 1_000_103


def test_direct_population_never_freezes_the_candidate_as_its_opponent():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    candidate = PolicyValueBeliefNet(config)
    anchor = PolicyValueBeliefNet(config)

    try:
        train_vector_population(
            TrainingConfig(device="cpu"),
            config,
            candidate,
            [candidate],
            [1.0],
            num_envs=2,
            steps=1,
            num_threads=1,
            max_actions=16,
            anchor_model=anchor,
        )
    except ValueError as error:
        assert "separate models" in str(error)
    else:
        raise AssertionError("candidate alias must be rejected before freezing")


def test_direct_actor_advances_with_best_ready_state_after_persistent_regression():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    candidate = PolicyValueBeliefNet(config)
    champion = PolicyValueBeliefNet(config)
    gate_rows = [
        {
            "model_a_score": 0.50,
            "model_a_city_captures_per_game": 0.10,
            "completion_rate": 0.50,
            "score_lower_95": 0.40,
        },
        {
            "model_a_score": 0.70,
            "model_a_city_captures_per_game": 0.20,
            "completion_rate": 0.70,
            "score_lower_95": 0.60,
        },
        {
            "model_a_score": 0.80,
            "model_a_city_captures_per_game": 0.25,
            "completion_rate": 0.80,
            "score_lower_95": 0.65,
        },
        {
            "model_a_score": 0.25,
            "model_a_city_captures_per_game": 0.05,
            "completion_rate": 0.30,
            "score_lower_95": 0.10,
        },
        {
            "model_a_score": 0.10,
            "model_a_city_captures_per_game": 0.00,
            "completion_rate": 0.10,
            "score_lower_95": 0.00,
        },
    ]

    actor_parameters = []

    def actor_update(*args, **kwargs):
        with torch.no_grad():
            next(candidate.parameters()).add_(0.01)
        actor_parameters.append(next(candidate.parameters()).detach().clone())
        return {"steps": 1.0}

    with (
        patch(
            "polybot.training.direct_curriculum.evaluate_direct_matches",
            side_effect=gate_rows,
        ),
        patch(
            "polybot.training.direct_curriculum.train_vector_actor_critic",
            side_effect=actor_update,
        ) as trainer,
    ):
        result = train_direct_policy_curriculum(
            TrainingConfig(device="cpu"),
            config,
            candidate,
            champion,
            teacher_envs=1,
            teacher_steps=0,
            actor_envs=1,
            actor_steps=4_000,
            actor_block_steps=1_000,
            actor_max_steps=10_000,
            num_threads=1,
            max_actions=16,
            gate_games=2,
            gate_seed=100,
            readiness_patience=2,
            regression_patience=2,
            run=_MemoryRun(),
        )

    assert trainer.call_count == 4
    assert result["direct_actor_steps"] == 4_000.0
    assert result["direct_ready"] == 1.0
    assert result["direct_early_stopped"] == 1.0
    assert result["direct_early_stop_reason"] == "persistent_actor_regression"
    assert result["direct_best_score"] == 0.80
    assert result["direct_gate_rows"][0]["direct_readiness_streak"] == 0.0
    torch.testing.assert_close(
        next(candidate.parameters()), actor_parameters[1]
    )


def test_direct_population_advances_after_persistent_regression():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    candidate = PolicyValueBeliefNet(config)
    champion = PolicyValueBeliefNet(config)
    gate_rows = [
        {
            "model_a_score": 0.50,
            "worst_score": 0.50,
            "model_a_city_captures_per_game": 0.10,
            "completion_rate": 0.50,
            "opponents": [],
        },
        {
            "model_a_score": 0.60,
            "worst_score": 0.60,
            "model_a_city_captures_per_game": 0.20,
            "completion_rate": 0.60,
            "opponents": [],
        },
        {
            "model_a_score": 0.70,
            "worst_score": 0.65,
            "model_a_city_captures_per_game": 0.25,
            "completion_rate": 0.70,
            "opponents": [],
        },
        {
            "model_a_score": 0.30,
            "worst_score": 0.20,
            "model_a_city_captures_per_game": 0.05,
            "completion_rate": 0.30,
            "opponents": [],
        },
        {
            "model_a_score": 0.20,
            "worst_score": 0.10,
            "model_a_city_captures_per_game": 0.00,
            "completion_rate": 0.10,
            "opponents": [],
        },
    ]

    population_parameters = []

    def population_update(*args, **kwargs):
        with torch.no_grad():
            next(candidate.parameters()).add_(0.01)
        population_parameters.append(
            next(candidate.parameters()).detach().clone()
        )
        return {"steps": 1.0}

    with (
        patch(
            "polybot.training.vector_population.evaluate_direct_population_matches",
            side_effect=gate_rows,
        ),
        patch(
            "polybot.training.vector_population.train_vector_population",
            side_effect=population_update,
        ) as trainer,
    ):
        result = train_direct_population_curriculum(
            TrainingConfig(device="cpu"),
            config,
            candidate,
            champion,
            [champion],
            [1.0],
            num_envs=2,
            minimum_steps=4_000,
            maximum_steps=10_000,
            block_steps=1_000,
            num_threads=1,
            max_actions=16,
            gate_games=2,
            gate_seed=100,
            gate_score=0.50,
            historical_score_floor=0.45,
            readiness_city_captures_per_game=0.05,
            readiness_completion_rate=0.05,
            readiness_patience=2,
            policy_anchor_weight=0.75,
            value_anchor_weight=0.50,
            opponent_policy_weight=0.50,
            teacher_weight=0.25,
            shaping_weight=0.20,
            regression_patience=2,
            run=_MemoryRun(),
        )

    assert trainer.call_count == 4
    assert result["direct_population_steps"] == 4_000.0
    assert result["direct_population_ready"] == 1.0
    assert result["direct_population_early_stopped"] == 1.0
    assert result["direct_population_early_stop_reason"] == (
        "persistent_population_regression"
    )
    assert result["direct_population_best_score"] == 0.70
    assert result["direct_population_gate_rows"][0]["readiness_streak"] == 0.0
    torch.testing.assert_close(
        next(candidate.parameters()), population_parameters[1]
    )


def test_flat_promotion_uses_raw_population_results_only():
    result = _promotion_gate_decision(
        [
            {
                "opponent_id": 1,
                "sampling_probability": 0.6,
                "model_a_score": 0.60,
                "score_lower_95": 0.51,
            },
            {
                "opponent_id": 2,
                "sampling_probability": 0.4,
                "model_a_score": 0.50,
                "score_lower_95": 0.30,
            },
        ],
        1,
        promotion_score=0.55,
        promotion_lower_bound=0.50,
        population_score=0.55,
        historical_score_floor=0.45,
    )

    assert result["promoted"] is True
    assert np.isclose(result["population_score"], 0.56)


def test_safe_near_even_candidate_continues_below_half_score():
    assert _challenger_continuation_action(
        valid=True,
        candidate_score=0.49,
        population_score=0.5954,
        minimum_historical_score=0.49,
        historical_score_floor=0.45,
        has_previous_challenger=True,
    ) == "candidate"


def test_flat_mcts_distillation_requires_raw_policy_improvement():
    improved = _flat_distillation_gate_decision(
        0.50, 0.575, minimum_score=0.45
    )
    tied = _flat_distillation_gate_decision(
        0.50, 0.50, minimum_score=0.45
    )
    invalid = _flat_distillation_gate_decision(
        0.50, 0.60, minimum_score=0.45, valid=False
    )

    assert improved["accepted"] is True
    assert np.isclose(float(improved["improvement"]), 0.075)
    assert tied["accepted"] is False
    assert invalid["accepted"] is False


def test_direct_evaluation_progress_names_both_models():
    class _OneStepVector:
        @staticmethod
        def batch_spec():
            return {
                "map_tokens": {"shape": (1, 121, 23), "dtype": np.int32},
                "state": {"shape": (1, 11), "dtype": np.int32},
                "action_id": {"shape": (1, 1), "dtype": np.int32},
                "action_features": {
                    "shape": (1, 1, 17), "dtype": np.int32,
                },
                "action_mask": {"shape": (1, 1), "dtype": np.uint8},
                "visible_event_features": {
                    "shape": (1, 0, 26), "dtype": np.int32,
                },
                "visible_event_mask": {
                    "shape": (1, 0), "dtype": np.uint8,
                },
                "action_truncated": {"shape": (1,), "dtype": np.uint8},
                "total_legal_action_count": {
                    "shape": (1,), "dtype": np.int32,
                },
                "action_valid": {"shape": (1,), "dtype": np.uint8},
                "terminated": {"shape": (1,), "dtype": np.uint8},
            }

        @staticmethod
        def reset_into(batch, seed=None):
            for value in batch.values():
                value.fill(0)
            batch["state"][:, 1] = 0
            batch["state"][:, 3] = -1
            batch["action_id"][:, 0] = 7
            batch["action_mask"][:, 0] = 1
            batch["total_legal_action_count"][:] = 1

        @staticmethod
        def step_into(action_ids, batch):
            batch["action_valid"][:] = action_ids >= 0
            batch["terminated"][:] = action_ids >= 0
            batch["state"][:, 2] = action_ids >= 0
            batch["state"][:, 3] = np.where(action_ids >= 0, 0, -1)

    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    model_a = PolicyValueBeliefNet(config)
    model_b = PolicyValueBeliefNet(config)
    games = []
    with (
        patch(
            "polybot.evaluation.matches._new_paired_vector_envs",
            return_value=(_OneStepVector(), _OneStepVector()),
        ),
        patch("polybot.evaluation.matches.tqdm") as progress,
    ):
        progress.return_value.__enter__.return_value = progress.return_value
        result = evaluate_direct_matches(
            model_a,
            model_b,
            torch.device("cpu"),
            TrainingConfig(device="cpu"),
            2,
            game_callback=games.append,
            model_a_label="pretendent champion-challenger-017",
            model_b_label="champion-005",
        )

    assert progress.call_args.kwargs["desc"] == (
        "evaluate direct pretendent champion-challenger-017 vs champion-005"
    )
    assert result["model_a_score"] == 0.5
    assert [game["model_a_player"] for game in games] == [0, 1]
    assert [game["map_seed"] for game in games] == [2137, 2137]


def test_unsafe_candidate_falls_back_to_previous_challenger():
    assert _challenger_continuation_action(
        valid=True,
        candidate_score=0.44,
        population_score=0.59,
        minimum_historical_score=0.44,
        historical_score_floor=0.45,
        has_previous_challenger=True,
    ) == "previous"
    assert _challenger_continuation_action(
        valid=False,
        candidate_score=0.0,
        population_score=0.0,
        minimum_historical_score=0.0,
        historical_score_floor=0.45,
        has_previous_challenger=False,
    ) == "none"


def test_flat_league_keeps_training_a_safe_049_challenger_across_rounds():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    champion = PolicyValueBeliefNet(config)
    challenger = PolicyValueBeliefNet(config)
    historical = PolicyValueBeliefNet(config)
    with torch.no_grad():
        next(challenger.parameters()).add_(0.01)
        next(historical.parameters()).sub_(0.01)

    near_even = {
        "games": 100.0,
        "model_a_wins": 49.0,
        "model_a_losses": 51.0,
        "draws": 0.0,
        "model_a_score": 0.49,
        "score_lower_95": 0.42,
    }
    strong_historical = {
        "games": 20.0,
        "model_a_wins": 16.0,
        "model_a_losses": 4.0,
        "draws": 0.0,
        "model_a_score": 0.80,
        "score_lower_95": 0.60,
    }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        champion_path = root / "champion.pt"
        challenger_path = root / "challenger.pt"
        historical_path = root / "historical.pt"
        checkpoint_dir = root / "checkpoints"
        save_model(champion_path, champion, config)
        save_model(challenger_path, challenger, config)
        save_model(historical_path, historical, config)

        with patch(
            "polybot.training.league.evaluate_direct_matches",
            side_effect=[
                near_even,
                strong_historical,
                near_even,
                strong_historical,
            ],
        ):
            summary = train_promotion_league(
                TrainingConfig(device="cpu"),
                champion_in=champion_path,
                challenger_in=challenger_path,
                checkpoint_dir=checkpoint_dir,
                rounds=2,
                promotion_games=100,
                promotion_score=0.55,
                promotion_lower_bound=0.50,
                evaluation_seed=1_000,
                evaluation_seed_stride=101,
                round_seed_stride=103,
                checkpoint_prefix="champion",
                mcts=MCTSConfig(simulations=16, particles=1),
                num_envs=2,
                steps=0,
                updates=0,
                num_threads=1,
                max_pending_leaves=1,
                visible_action_history=0,
                belief_retries=1,
                database_path=root / "league.sqlite3",
                population_model_paths=(historical_path,),
                population_size=2,
                population_games=0,
                direct_teacher_steps=0,
                direct_actor_steps=0,
                direct_population_steps=0,
                champion_mode="flat",
                mcts_probe_games=0,
                historical_games=20,
                population_score=0.52,
                historical_score_floor=0.45,
                run=_MemoryRun(),
            )

        assert summary["league_promotions"] == 0.0
        assert Path(str(summary["league_challenger_path"])).name == (
            "champion-challenger-002.pt"
        )
        assert (checkpoint_dir / "champion-challenger-001.pt").is_file()
        assert (checkpoint_dir / "champion-challenger-002.pt").is_file()


def test_mcts_probe_requires_search32_to_strictly_beat_raw_policy():
    assert _mcts_probe_passed(
        0.70,
        0.71,
        minimum_search_score=0.55,
        minimum_improvement=0.0,
    )
    assert not _mcts_probe_passed(
        0.70,
        0.70,
        minimum_search_score=0.55,
        minimum_improvement=0.0,
    )
    assert not _mcts_probe_passed(
        0.50,
        0.70,
        minimum_search_score=0.55,
        minimum_improvement=0.0,
        valid=False,
    )


def test_flat_league_activates_mcts_gate_only_after_search32_is_better():
    config = ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    model = PolicyValueBeliefNet(config)
    direct_results = [
        {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 0.0,
            "draws": 1.0,
            "model_a_score": 0.75,
            "score_lower_95": 0.50,
        },
        {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 1.0,
            "draws": 0.0,
            "model_a_score": 0.50,
            "score_lower_95": 0.10,
        },
        {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 0.0,
            "draws": 1.0,
            "model_a_score": 0.75,
            "score_lower_95": 0.50,
        },
        {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 0.0,
            "draws": 1.0,
            "model_a_score": 0.75,
            "score_lower_95": 0.50,
        },
        {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 1.0,
            "draws": 0.0,
            "model_a_score": 0.50,
            "score_lower_95": 0.10,
        },
    ]
    passing_search_result = {
        "games": 2.0,
        "model_a_wins": 1.0,
        "model_a_losses": 0.0,
        "draws": 1.0,
        "model_a_score": 0.75,
        "score_lower_95": 0.50,
        "valid": 1.0,
    }
    non_improving_search_result = {
        "games": 2.0,
        "model_a_wins": 1.0,
        "model_a_losses": 1.0,
        "draws": 0.0,
        "model_a_score": 0.50,
        "score_lower_95": 0.10,
        "valid": 1.0,
    }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        champion_path = root / "teacher.pt"
        database_path = root / "league.sqlite3"
        save_model(champion_path, model, config)

        probe_calls = []

        def search_probe_after_checkpoint(*args, **kwargs):
            probe_calls.append(1)
            expected = f"champion-{len(probe_calls):03d}.pt"
            assert (root / "checkpoints" / expected).is_file()
            return (
                passing_search_result
                if len(probe_calls) == 1
                else non_improving_search_result
            )

        with (
            patch(
                "polybot.training.league.require_routed_self_play_pool"
            ),
            patch(
                "polybot.training.league.train_direct_policy_curriculum",
                return_value={"direct_ready": 0.0},
            ),
            patch(
                "polybot.training.league.train_direct_population_curriculum",
                return_value={"direct_population_ready": 0.0},
            ),
            patch(
                "polybot.training.league.evaluate_direct_matches",
                side_effect=direct_results,
            ),
            patch(
                "polybot.training.league.evaluate_matches_pool",
                side_effect=search_probe_after_checkpoint,
            ),
            patch(
                "polybot.training.league.train_population_pool"
            ) as population_trainer,
        ):
            summary = train_promotion_league(
                TrainingConfig(device="cpu"),
                champion_in=champion_path,
                checkpoint_dir=root / "checkpoints",
                rounds=2,
                promotion_games=2,
                promotion_score=0.55,
                promotion_lower_bound=0.50,
                evaluation_seed=1_000,
                evaluation_seed_stride=101,
                round_seed_stride=103,
                checkpoint_prefix="champion",
                mcts=MCTSConfig(simulations=16, particles=1),
                num_envs=2,
                steps=0,
                updates=0,
                num_threads=1,
                max_pending_leaves=1,
                visible_action_history=0,
                belief_retries=1,
                database_path=database_path,
                population_games=0,
                population_updates_per_game=0,
                population_workers=1,
                historical_games=2,
                population_score=0.50,
                historical_score_floor=0.45,
                direct_teacher_steps=1,
                direct_actor_steps=0,
                direct_population_steps=1,
                direct_population_max_steps=1,
                direct_population_block_steps=1,
                champion_mode="flat",
                mcts_probe_games=2,
                mcts_probe_simulations=32,
                mcts_probe_score=0.55,
                mcts_probe_minimum_improvement=0.0,
                mcts_probe_patience=1,
                run=_MemoryRun(),
            )

        population_trainer.assert_not_called()
        assert summary["league_promotions"] == 1.0
        assert summary["league_last_mcts_ready"] == 0.0
        assert Path(str(summary["league_champion_path"])).name == (
            "champion-001.pt"
        )
        with LeagueDatabase(database_path) as database:
            rows = database.connection.execute(
                """SELECT model_id, mcts_ready, mcts_evaluation_active
                   FROM model_capabilities ORDER BY model_id"""
            ).fetchall()
            models = database.connection.execute(
                """SELECT status, checkpoint_path FROM models
                   WHERE parent_model_id IS NOT NULL ORDER BY id"""
            ).fetchall()
        assert len(rows) == 2
        assert [int(row["mcts_ready"]) for row in rows] == [1, 0]
        assert [
            int(row["mcts_evaluation_active"]) for row in rows
        ] == [1, 1]
        assert [row["status"] for row in models] == [
            "promoted",
            "challenger",
        ]
        assert Path(str(models[1]["checkpoint_path"])).name == (
            "champion-002.pt"
        )


def test_league_reference_policy_is_frozen_and_never_mutated():
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    device = torch.device("cpu")
    frozen = _freeze_reference_model(model, device)
    request = {
        "map_tokens": np.zeros((1, 121, 23), dtype=np.int32),
        "state": np.zeros((1, 9), dtype=np.int32),
        "action_features": np.zeros((1, 3, 17), dtype=np.int32),
        "action_mask": np.asarray([[True, True, False]]),
        "to_play": np.zeros(1, dtype=np.int32),
    }

    policy = _reference_policy(frozen, request, device, TrainingConfig())

    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    np.testing.assert_allclose(policy[0, :2].sum(), 1.0, atol=1e-6)
    assert policy[0, 2] == 0.0
    for name, parameter in frozen.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_league_reports_policy_drift_from_champion():
    decisions, agreements, l1_sum = _policy_distance(
        np.asarray([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]], dtype=np.float32),
        np.asarray([[True, True, False], [True, True, False]]),
    )

    assert decisions == 2
    assert agreements == 1
    assert np.isclose(l1_sum, 2.2)


def test_paired_evaluation_reuses_map_and_swaps_candidate_slot():
    assert [_paired_seed_and_slot(100, index) for index in range(4)] == [
        (100, 0), (100, 1), (101, 0), (101, 1),
    ]


def test_routed_pool_schedule_duplicates_seeds_and_swaps_agent_ids():
    seeds, assignments = _paired_pool_schedule(100, start=0, count=4)

    np.testing.assert_array_equal(seeds, np.asarray([100, 100, 101, 101], np.uint32))
    np.testing.assert_array_equal(
        assignments,
        np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], np.int32),
    )


def test_one_invalid_evaluation_slot_replaces_its_complete_pair_only():
    np.testing.assert_array_equal(_evaluation_pair_rows([3], 128), [2, 3])
    np.testing.assert_array_equal(_evaluation_pair_rows([2, 3], 128), [2, 3])
    np.testing.assert_array_equal(_evaluation_pair_rows([127], 128), [126, 127])


def test_evaluation_replacement_seeds_are_unique_and_outside_original_schedule():
    used = {1_500_000_007 + index for index in range(50)}
    cursor = 0
    first, cursor = _next_evaluation_replacement_seed(
        1_500_000_007, cursor, used
    )
    second, cursor = _next_evaluation_replacement_seed(
        1_500_000_007, cursor, used
    )

    assert first != second
    assert first in used and second in used
    assert first not in range(1_500_000_007, 1_500_000_057)


class _RecoveringEvaluationSearch:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls: list[tuple[object, np.ndarray]] = []

    def search(self, request, active, **kwargs):
        self.calls.append((request, np.asarray(active).copy()))
        if len(self.calls) <= self.failures:
            raise BeliefRecoveryRequired(
                [0],
                "rejected",
                [{
                    "env_id": 0,
                    "player": 1,
                    "particle": 0,
                    "reason": "legal_action_mismatch",
                }],
            )
        return {"state_id": np.asarray([77], dtype=np.uint64)}


def test_evaluation_retries_same_position_without_changing_request_or_active_rows():
    search = _RecoveringEvaluationSearch(failures=2)
    request = {"state_id": np.asarray([77], dtype=np.uint64)}
    active = np.asarray([True])

    root, failures = _evaluation_search_with_recovery(
        search, request, active, recovery_cycles=4
    )

    assert root["state_id"][0] == 77
    assert len(failures) == 2
    assert len(search.calls) == 3
    assert all(call_request is request for call_request, _ in search.calls)
    assert all(np.array_equal(call_active, active) for _, call_active in search.calls)


def test_underscored_legal_action_diagnostic_is_not_classified_as_other():
    kind = _belief_rejection_kind(
        ValueError("acceptance mask rejected particle"),
        [{"reason": "legal_action_mismatch"}],
    )

    assert kind == "legal_actions"


def test_public_action_difference_rejects_only_the_mismatched_particle():
    class _Pool:
        @staticmethod
        def ismcts_belief_action_differences(state_ids, particles):
            assert state_ids.dtype == np.uint64
            assert particles.dtype == np.int32
            return [{
                "env_id": 0,
                "player": 1,
                "particle": 0,
                "missing_actions": [{"source_index": 4, "target_index": 8}],
                "extra_actions": [{"source_index": 8, "target_index": 4}],
            }]

    rejected, mask, count, details = ismcts_action_difference_rejections(
        _Pool(),
        np.asarray([77], dtype=np.uint64),
        np.zeros((1, 2, 2, 9, 23), dtype=np.int32),
    )

    np.testing.assert_array_equal(rejected, [0])
    assert mask[0, 1, 0]
    assert not mask[0, 1, 1]
    assert count == 1
    assert len(details[0]["missing_actions"]) == 1


def test_missing_public_action_difference_api_keeps_backward_compatibility():
    rejected, mask, count, details = ismcts_action_difference_rejections(
        object(),
        np.asarray([77], dtype=np.uint64),
        np.zeros((1, 2, 1, 9, 23), dtype=np.int32),
    )

    assert rejected.size == 0
    assert not mask.any()
    assert count == 0
    assert details == ()


def test_evaluation_exhaustion_keeps_diagnostics_from_every_same_state_cycle():
    search = _RecoveringEvaluationSearch(failures=10)
    request = {"state_id": np.asarray([77], dtype=np.uint64)}
    active = np.asarray([True])

    try:
        _evaluation_search_with_recovery(
            search, request, active, recovery_cycles=4
        )
    except BeliefRecoveryRequired as error:
        assert "recovery exhausted after 4 cycles" in str(error)
        assert len(error.diagnostics) == 4
    else:
        raise AssertionError("exhausted evaluation recovery must remain invalid")

    assert len(search.calls) == 4
    assert all(call_request is request for call_request, _ in search.calls)


def test_process_seed_controls_torch_particle_sampling():
    seed_everything(91337)
    first = torch.multinomial(torch.tensor([0.2, 0.3, 0.5]), 8, replacement=True)
    seed_everything(91337)
    second = torch.multinomial(torch.tensor([0.2, 0.3, 0.5]), 8, replacement=True)

    assert torch.equal(first, second)


def test_belief_loss_ignores_runtime_ids_and_decoder_owned_fields():
    target = torch.zeros((1, 1, 23), dtype=torch.long)
    target[..., 4] = 1   # make unit-only semantic fields applicable
    target[..., 14] = 2  # make city-only semantic fields applicable
    hidden = torch.ones((1, 1), dtype=torch.bool)
    logits = [
        torch.zeros((1, 1, largest + 2), requires_grad=True)
        for largest in MAP_FEATURE_MAX
    ]

    belief_loss(logits, target, hidden).backward()

    assert BELIEF_RUNTIME_ID_FEATURES.isdisjoint(BELIEF_LOSS_FEATURES)
    for feature, feature_logits in enumerate(logits):
        if feature in BELIEF_LOSS_FEATURES:
            assert feature_logits.grad is not None
        else:
            assert feature_logits.grad is None


def test_belief_loss_upweights_hidden_public_action_targets():
    target = torch.zeros((1, 2, 23), dtype=torch.long)
    hidden = torch.ones((1, 2), dtype=torch.bool)
    logits = [
        torch.zeros((1, 2, largest + 2), requires_grad=True)
        for largest in MAP_FEATURE_MAX
    ]
    with torch.no_grad():
        # Tile 0 is predicted correctly while the public action targets tile
        # 1, whose terrain class is intentionally predicted incorrectly.
        logits[19][0, 0, 1] = 4.0
        logits[19][0, 1, 0] = 4.0
    actions = torch.full((1, 1, 12), -1, dtype=torch.long)
    actions[0, 0, 2] = 1
    mask = torch.ones((1, 1), dtype=torch.bool)

    plain = belief_loss(logits, target, hidden)
    weighted = belief_loss(
        logits, target, hidden, actions, mask, legal_action_weight=2.0
    )

    assert weighted > plain


def test_belief_terrain_metrics_report_hidden_action_and_bridge_quality():
    # 3x3 board: centre is visible water; its north/south hidden neighbours
    # are land, so the bridge geometry label is true.
    observed = torch.full((1, 9, 23), -1, dtype=torch.long)
    observed[..., 0] = 0
    observed[:, 4, 0] = 1
    observed[:, 4, 19] = 1
    target = observed.clone()
    target[..., 19] = 1
    target[:, 1, 19] = 2
    target[:, 7, 19] = 2
    hidden = observed[..., 0] == 0
    logits = torch.zeros((1, 9, 6))
    logits[..., 2] = 12.0  # terrain id 1 / water
    logits[:, 1, 3] = 20.0  # terrain id 2 / land
    logits[:, 7, 3] = 20.0
    focus = torch.zeros_like(hidden)
    focus[:, 1] = True

    metrics = belief_terrain_metrics(logits, observed, target, hidden, focus, 3)

    assert metrics["belief_hidden_terrain_samples"] == 8.0
    assert metrics["belief_hidden_terrain_accuracy"] == 1.0
    assert metrics["belief_hidden_terrain_nll"] < 0.01
    assert metrics["belief_action_constraint_terrain_samples"] == 1.0
    assert metrics["belief_action_constraint_terrain_accuracy"] == 1.0
    assert metrics["belief_bridge_geometry_samples"] == 1.0
    assert metrics["belief_bridge_geometry_accuracy"] == 1.0


def test_belief_action_evidence_marks_public_move_source_and_target():
    actions = torch.full((1, 1, 12), -1, dtype=torch.long)
    actions[0, 0, 0] = 0  # move
    actions[0, 0, 1] = 0
    actions[0, 0, 2] = 3
    evidence = PolicyValueBeliefNet._belief_action_tile_evidence(
        actions, torch.ones((1, 1), dtype=torch.bool), tiles=4
    )

    # Token class zero is reserved for -1, hence public type 0 occupies 1.
    type_count = 11
    assert evidence.shape == (1, 4, 2 * type_count)
    assert evidence[0, 0, 1] == 1.0
    assert evidence[0, 3, type_count + 1] == 1.0


def test_single_environment_advantage_normalisation_stays_finite():
    advantage = torch.tensor([0.75])
    normalised = _normalise_advantage(advantage)

    assert torch.equal(normalised, advantage)
    assert torch.isfinite(normalised).all()


def test_pool_update_budget_is_spread_across_rollout():
    due = [_scheduled_updates(step, steps=10, updates=5) for step in range(1, 11)]
    assert due == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert _winner_from_terminal_values(np.asarray([0.0, 0.0])) == -1


def test_zero_prior_weights_remove_hand_written_leaf_bias():
    categorical = torch.full((1, 2, 12), -1, dtype=torch.long)
    categorical[0, :, 0] = torch.tensor([8, 3])
    categorical[0, 0, 9] = 14
    continuous = torch.zeros((1, 2, 4))
    mask = torch.ones((1, 2), dtype=torch.bool)

    pure = _native_leaf_bias(
        categorical, continuous, mask,
        teacher_prior_weight=0.0, strategic_prior_weight=0.0,
    )
    anchored = _native_leaf_bias(
        categorical, continuous, mask,
        teacher_prior_weight=0.0, strategic_prior_weight=1.0,
    )

    assert torch.equal(pure, torch.zeros_like(pure))
    assert anchored[0, 0] > anchored[0, 1]


def test_checkpoint_belief_warmup_freezes_policy_value_parameters():
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    original = _configure_belief_only_training(model)

    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(_BELIEF_ONLY_PREFIXES)

    _restore_trainable_parameters(model, original)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_belief_retry_temperature_becomes_stochastic_after_greedy_particle():
    temperatures = [
        particle_temperature(0.0, attempt, particle)
        for attempt in range(5)
        for particle in range(2)
    ]

    assert temperatures[0] == 0.0
    assert temperatures[1] >= 0.15
    assert [particle_temperature(0.0, attempt, 0) for attempt in range(5)] == [
        0.0, 0.15, 0.30, 0.60, 1.0,
    ]


class _DiagnosticPool:
    def ismcts_belief_diagnostics(self, state_ids, particles):
        return [{
            "accepted": False,
            "env_id": 103,
            "player": 0,
            "particle": 1,
            "reason": "legal_action_mismatch",
        }]


def test_bad_particle_one_of_two_marks_only_its_environment_and_particle():
    state_ids = np.arange(128, dtype=np.uint64)
    particles = np.zeros((128, 2, 2, 1, 23), dtype=np.int32)

    rows, mask, rejected_particles, diagnostics = ismcts_rejected_envs(
        _DiagnosticPool(), state_ids, particles
    )

    np.testing.assert_array_equal(rows, [103])
    assert rejected_particles == 1
    assert mask.sum() == 1
    assert mask[103, 0, 1]
    assert diagnostics[0]["reason"] == "legal_action_mismatch"


def test_resampling_one_bad_particle_preserves_other_127_envs_and_state_ids():
    state_ids = np.arange(10_000, 10_128, dtype=np.uint64)
    before_state_ids = state_ids.copy()
    particles = np.arange(128 * 2 * 2 * 1 * 23, dtype=np.int32).reshape(128, 2, 2, 1, 23)
    before = particles.copy()
    mask = np.zeros((128, 2, 2), dtype=bool)
    mask[103, 0, 1] = True
    replacement = np.full((1, 2, 2, 1, 23), -7, dtype=np.int32)

    replace_rejected_particles(particles, np.asarray([103]), mask, replacement)

    np.testing.assert_array_equal(state_ids, before_state_ids)
    np.testing.assert_array_equal(particles[:103], before[:103])
    np.testing.assert_array_equal(particles[104:], before[104:])
    np.testing.assert_array_equal(particles[103, 0, 0], before[103, 0, 0])
    np.testing.assert_array_equal(particles[103, 1], before[103, 1])
    assert np.all(particles[103, 0, 1] == -7)


def test_native_reset_of_one_slot_keeps_other_127_state_ids():
    from PolyEnv import Bardur, Imperius, MapType, SelfPlayPool

    pool = SelfPlayPool(
        num_envs=128,
        seed=123,
        map_size=11,
        tribes=(Bardur, Imperius),
        map_type=MapType.Lakes,
        num_threads=1,
        max_actions=1024,
        auto_reset=False,
        c_puct=1.5,
        visible_action_history=0,
        require_all_actions=True,
    )
    before = pool.belief_requests()
    reset_packet = pool.belief_requests()
    state_ids = np.asarray(before["state_id"], dtype=np.uint64).copy()
    pool.reset_slots_into(
        np.asarray([103], dtype=np.int32),
        np.asarray([99_999], dtype=np.uint32),
        np.asarray([[0, 1]], dtype=np.int32),
        reset_packet,
    )
    reset_state_ids = np.asarray(reset_packet["state_id"], dtype=np.uint64)

    unchanged = np.arange(128) != 103
    np.testing.assert_array_equal(reset_state_ids[unchanged], state_ids[unchanged])
    assert reset_state_ids[103] != state_ids[103]


class _MaskPool:
    def ismcts_belief_acceptance_mask(self, state_ids, particles):
        accepted = np.ones(particles.shape[:3], dtype=bool)
        accepted[7, 1, 0] = False
        return accepted


class _AcceptingMaskPool:
    def ismcts_belief_acceptance_mask(self, state_ids, particles):
        return np.ones(particles.shape[:3], dtype=bool)


class _MaskAndUnitDiagnosticsPool:
    def ismcts_belief_acceptance_mask(self, state_ids, particles):
        accepted = np.ones(particles.shape[:3], dtype=bool)
        accepted[7, 1, 0] = False
        return accepted

    def ismcts_belief_diagnostics(self, state_ids, particles):
        return [{
            "accepted": False,
            "env_id": 7,
            "player": 1,
            "particle": 0,
            "reason": "belief_build_error",
            "error": "completed_map_tokens contains an invalid unit state",
        }]


class _EnvMaskWithParticleDiagnosticsPool:
    def ismcts_belief_acceptance_mask(self, state_ids, particles):
        accepted = np.ones(particles.shape[0], dtype=bool)
        accepted[7] = False
        return accepted

    def ismcts_belief_diagnostics(self, state_ids, particles):
        return [{
            "accepted": False,
            "env_id": 7,
            "player": 1,
            "particle": 0,
            "reason": "belief_build_error",
        }]


def test_conservative_particles_are_not_trusted_without_acceptance_validation():
    particles = np.zeros((16, 2, 2, 1, 23), dtype=np.int32)
    rows, mask, rejected_particles, _ = ismcts_rejected_envs(
        _MaskPool(), np.arange(16, dtype=np.uint64), particles
    )

    np.testing.assert_array_equal(rows, [7])
    assert rejected_particles == 1
    assert mask[7, 1, 0]


def test_new_acceptance_mask_can_preflight_a_full_valid_packet():
    particles = np.zeros((8, 2, 2, 1, 23), dtype=np.int32)
    rows, mask, rejected_particles, _ = ismcts_rejected_envs(
        _AcceptingMaskPool(),
        np.arange(8, dtype=np.uint64),
        particles,
        submit_failed=False,
    )

    assert rows.size == 0
    assert not mask.any()
    assert rejected_particles == 0


def test_acceptance_mask_rejection_keeps_diagnostics_and_invalid_unit_fields():
    particles = np.zeros((8, 2, 2, 1, 23), dtype=np.int32)
    particles[..., 2] = -1
    particles[..., 3] = -1
    particles[..., 4] = -1
    particles[..., 21] = -1
    particles[..., 22] = -1
    invalid = particles[7, 1, 0, 0]
    invalid[2] = 15
    invalid[3] = 1
    invalid[4] = 1
    invalid[21] = 10
    invalid[22] = -1

    rows, mask, rejected_particles, diagnostics = ismcts_rejected_envs(
        _MaskAndUnitDiagnosticsPool(),
        np.arange(8, dtype=np.uint64),
        particles,
        submit_failed=False,
    )

    np.testing.assert_array_equal(rows, [7])
    assert rejected_particles == 1
    assert mask[7, 1, 0]
    assert diagnostics[0]["invalid_unit_tile"] == 0
    assert diagnostics[0]["unit_health"] == 15
    assert diagnostics[0]["unit_max_health"] == 10


def test_env_acceptance_mask_uses_diagnostics_to_keep_valid_particles():
    particles = np.zeros((8, 2, 2, 1, 23), dtype=np.int32)

    rows, mask, rejected_particles, _ = ismcts_rejected_envs(
        _EnvMaskWithParticleDiagnosticsPool(),
        np.arange(8, dtype=np.uint64),
        particles,
        submit_failed=False,
    )

    np.testing.assert_array_equal(rows, [7])
    assert rejected_particles == 1
    assert mask.sum() == 1
    assert mask[7, 1, 0]


def test_local_unit_diagnostic_detects_stale_fields_without_a_unit():
    completed = np.zeros((1, 23), dtype=np.int32)
    completed[0, 2] = 5
    completed[0, 3] = -1
    completed[0, 4] = -1
    completed[0, 21] = -1
    completed[0, 22] = -1

    details = _invalid_unit_token_details(completed, player_count=2)

    assert details is not None
    assert details["unit_validation_code"] == 2


def test_replay_update_budget_never_exceeds_configured_effective_epochs():
    allowed = capped_update_target(
        requested_updates=500,
        example_count=71,
        batch_size=64,
        max_replay_epochs=6.0,
    )

    assert allowed == 6
    assert effective_replay_epochs(allowed, 64, 71) <= 6.0


def test_one_population_slot_recovery_does_not_trip_circuit_breaker():
    assert not recovery_rate_exceeded(1, 64)
    assert recovery_rate_exceeded(2, 128)


def test_rejected_inactive_population_slot_is_still_recovered():
    active = np.ones(64, dtype=bool)
    active[:7] = False

    rows, active_rows, inactive_rows = partition_recovery_envs([3], active)

    np.testing.assert_array_equal(rows, [3])
    assert active_rows.size == 0
    np.testing.assert_array_equal(inactive_rows, [3])


def test_recovery_seed_skips_held_out_evaluation_seed():
    blocked = replacement_seeds(91337, 0, 1)[0]
    selected = replacement_seeds(91337, 0, 2, excluded=[int(blocked)])

    assert int(blocked) not in selected.tolist()
    assert len(set(selected.tolist())) == 2


def test_public_bridge_action_projects_hidden_neighbours_without_touching_visible_tiles():
    observed = torch.full((9, 23), -1, dtype=torch.long)
    observed[:, 0] = 0
    observed[4, 0] = 1
    observed[4, 19] = 1  # visible water bridge target
    observed[1, 0] = 1
    observed[1, 19] = 1  # visible water blocks the N/S orientation
    completed = observed.clone()
    completed[:, 19] = 1
    completed[observed[:, 0] == 1] = observed[observed[:, 0] == 1]
    actions = torch.full((1, 12), -1, dtype=torch.long)
    actions[0, 0] = 8
    actions[0, 2] = 4
    actions[0, 9] = 9

    apply_public_bridge_constraints(
        completed, observed, actions, torch.ones(1, dtype=torch.bool)
    )

    assert int(completed[1, 19]) == 1
    assert int(completed[3, 19]) in (2, 3, 4)
    assert int(completed[5, 19]) in (2, 3, 4)
    assert torch.equal(completed[observed[:, 0] == 1], observed[observed[:, 0] == 1])


def test_conservative_bridge_projection_suppresses_unlisted_extra_actions():
    observed = torch.full((9, 23), -1, dtype=torch.long)
    observed[:, 0] = 0
    observed[4, 0] = 1
    observed[4, 19] = 1
    observed[4, 7] = 0
    completed = observed.clone()
    completed[:, 19] = 2
    completed[4] = observed[4]

    apply_public_bridge_constraints(
        completed, observed, None, None, suppress_unlisted=True
    )

    landish = lambda index: int(completed[index, 19]) in (2, 3, 4)
    assert not (landish(1) and landish(7))
    assert not (landish(3) and landish(5))
    assert int(completed[4, 19]) == 1


def test_scalar_particle_sampler_also_applies_public_bridge_actions():
    model = PolicyValueBeliefNet(
        ModelConfig(d_model=32, transformer_layers=1, attention_heads=4)
    )
    observed = torch.full((1, 9, 23), -1, dtype=torch.long)
    observed[..., 0] = 0
    observed[:, 4, 0] = 1
    observed[:, 4, 19] = 1
    logits = [
        torch.zeros((1, 9, largest + 2))
        for largest in MAP_FEATURE_MAX
    ]
    logits[19][..., 2] = 10.0  # raw decoder prefers Water everywhere
    actions = torch.full((1, 1, 12), -1, dtype=torch.long)
    actions[:, 0, 0] = 8
    actions[:, 0, 2] = 4
    actions[:, 0, 9] = 9

    particle = model.sample_completed_maps_from_belief_logits(
        observed,
        logits,
        particles=1,
        temperature=0.0,
        action_categorical=actions,
        action_mask=torch.ones((1, 1), dtype=torch.bool),
    )[0]

    landish = lambda index: int(particle[index, 19]) in (2, 3, 4)
    assert (landish(1) and landish(7)) or (landish(3) and landish(5))


def test_bridge_constraint_loss_learns_positive_and_negative_geometry():
    map_tokens = torch.full((1, 9, 23), -1, dtype=torch.long)
    map_tokens[..., 0] = 0
    map_tokens[:, 4, 0] = 1
    map_tokens[:, 4, 19] = 1
    hidden = map_tokens[..., 0] == 0
    actions = torch.full((1, 1, 12), -1, dtype=torch.long)
    actions[:, 0, 0] = 8
    actions[:, 0, 2] = 4
    actions[:, 0, 9] = 9
    action_mask = torch.ones((1, 1), dtype=torch.bool)
    positive_target = map_tokens.clone()
    positive_target[:, 3, 19] = 2
    positive_target[:, 5, 19] = 2

    positive_logits = torch.zeros((1, 9, 6))
    positive_logits[:, :, 2] = 8.0
    positive_logits[:, (3, 5), 3] = 16.0
    negative_logits = torch.zeros((1, 9, 6))
    negative_logits[:, :, 2] = 8.0
    positive_good = bridge_constraint_loss(
        positive_logits, map_tokens, positive_target, hidden,
        actions, action_mask, 3,
    )
    positive_bad = bridge_constraint_loss(
        negative_logits, map_tokens, positive_target, hidden,
        actions, action_mask, 3,
    )
    assert positive_good < positive_bad

    negative_target = map_tokens.clone()
    no_actions = torch.zeros_like(action_mask)
    negative_good = bridge_constraint_loss(
        negative_logits, map_tokens, negative_target, hidden,
        actions, no_actions, 3,
    )
    negative_bad = bridge_constraint_loss(
        positive_logits, map_tokens, negative_target, hidden,
        actions, no_actions, 3,
    )
    assert negative_good < negative_bad


def test_bridge_positions_receive_their_configured_replay_reservation():
    replay = ReplayBuffer(32)
    rows = [_example(0) for _ in range(20)]
    for row in rows[:6]:
        row.bridge_priority = True
    replay.extend(rows)

    sampled = replay.sample(
        10, np.random.default_rng(7), priority_fraction=0.0, bridge_fraction=0.4
    )

    assert sum(row.bridge_priority for row in sampled) >= 4


def test_engine_action_constraint_positions_receive_replay_reservation():
    replay = ReplayBuffer(32)
    rows = [_example(0) for _ in range(20)]
    for row in rows[:6]:
        row.engine_action_constraint_priority = True
        row.engine_action_constraint_tiles = np.ones(1, dtype=bool)
    replay.extend(rows)

    sampled = replay.sample(
        10,
        np.random.default_rng(7),
        priority_fraction=0.0,
        bridge_fraction=0.0,
        engine_action_constraint_fraction=0.4,
    )

    assert sum(row.engine_action_constraint_priority for row in sampled) >= 4


def _pruning_packet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [[10.0, 9.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0, -2.0]]
    )
    categorical = torch.full((1, 12, 12), -1, dtype=torch.long)
    categorical[:, :, 0] = 0
    categorical[0, 5, 0] = 1  # attack
    categorical[0, 6, 0] = 3  # end turn
    categorical[0, 7, 0] = 8
    categorical[0, 7, 9] = 14  # capture city
    categorical[0, 8, 0] = 8
    categorical[0, 8, 9] = 9  # build bridge
    mask = torch.tensor(
        [[True, True, True, True, True, True, True, True, True, True, False, False]]
    )
    return logits, categorical, mask


def test_action_pruning_preserves_authoritative_legal_mask_and_tactics():
    logits, categorical, legal = _pruning_packet()
    before = legal.clone()

    masked, selected, stats = prune_search_logits(
        logits, categorical, legal, top_actions=2, random_actions=0
    )

    torch.testing.assert_close(legal, before)
    assert selected[0, :2].all()
    assert selected[0, 5:9].all()
    assert not selected[0, 2:5].any()
    assert stats.legal_actions == 10
    assert stats.selected_actions == 6
    assert stats.pruned_rows == 1
    assert torch.isfinite(masked[0, selected[0]]).all()
    assert (masked[0, legal[0] & ~selected[0]] == torch.finfo(masked.dtype).min).all()


def test_action_pruning_random_candidates_are_seeded_and_additive():
    logits, categorical, legal = _pruning_packet()
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)

    _, first, stats = prune_search_logits(
        logits,
        categorical,
        legal,
        top_actions=2,
        random_actions=2,
        generator=first_generator,
    )
    _, second, _ = prune_search_logits(
        logits,
        categorical,
        legal,
        top_actions=2,
        random_actions=2,
        generator=second_generator,
    )

    torch.testing.assert_close(first, second)
    assert stats.random_actions == 2
    assert stats.selected_actions == 8
    assert 0.0 < stats.prior_mass_retained_sum <= 1.0


def test_action_pruning_expands_top_k_for_required_prior_mass():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    categorical = torch.zeros((1, 4, 12), dtype=torch.long)
    legal = torch.ones((1, 4), dtype=torch.bool)

    _, selected, stats = prune_search_logits(
        logits,
        categorical,
        legal,
        top_actions=1,
        min_prior_mass=0.90,
    )

    assert selected.sum().item() == 3
    assert stats.prior_mass_retained_sum >= 0.90


def test_zero_top_actions_is_an_exact_backwards_compatible_off_switch():
    logits, categorical, legal = _pruning_packet()

    masked, selected, stats = prune_search_logits(
        logits, categorical, legal, top_actions=0, random_actions=8
    )

    assert masked is logits
    torch.testing.assert_close(selected, legal)
    assert stats.pruned_rows == 0
    assert stats.selected_actions == int(legal.sum())
    assert stats.prior_mass_retained_sum == 1.0


def test_mandatory_action_selector_ignores_illegal_padding():
    _, categorical, legal = _pruning_packet()
    categorical[0, 10, 0] = 1

    mandatory = mandatory_search_actions(categorical, legal)

    assert mandatory[0, 5:9].all()
    assert not mandatory[0, 10]


def test_root_policy_executes_only_remembered_candidates_without_hiding_legality():
    root = {
        "action_id": np.asarray([[10, 11, 12, -1]], dtype=np.int32),
        "action_mask": np.asarray([[True, True, True, False]]),
        "visit_count": np.asarray([[2, 7, 3, 0]], dtype=np.int32),
        "policy": np.asarray([[0.1, 0.6, 0.3, 0.0]], dtype=np.float32),
    }
    before = root["action_mask"].copy()

    stats = restrict_root_policy_to_candidates(
        root, {0: np.asarray([10, 12])}, np.asarray([True])
    )

    np.testing.assert_array_equal(root["action_mask"], before)
    np.testing.assert_allclose(root["policy"], [[0.25, 0.0, 0.75, 0.0]])
    assert stats.roots == 1
    assert stats.visits == 12
    assert stats.outside_candidate_visits == 7
    assert stats.unfiltered_choices_outside == 1


def test_root_policy_fails_open_when_candidate_capture_is_unavailable():
    root = {
        "action_id": np.asarray([[10, 11]], dtype=np.int32),
        "action_mask": np.asarray([[True, True]]),
        "visit_count": np.asarray([[1, 1]], dtype=np.int32),
        "policy": np.asarray([[0.4, 0.6]], dtype=np.float32),
    }

    stats = restrict_root_policy_to_candidates(root, {}, np.asarray([True]))

    np.testing.assert_allclose(root["policy"], [[0.4, 0.6]])
    assert stats.roots == 0
    assert stats.missing_roots == 1
