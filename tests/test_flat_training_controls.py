from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from polybot.config import ModelConfig, TrainingConfig
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.flat_experiments import FlatImprovements, flat_schedule, model_fingerprint
from polybot.training.flat_value import ValueValidationCollector, fit_value_head
from polybot.training.run_artifacts import TrainingRun
from polybot.training.vector_population import train_direct_population_curriculum, evaluate_direct_population_matches


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    return PolicyValueBeliefNet(ModelConfig(d_model=16, transformer_layers=1,
                                          attention_heads=2, dropout=0.0,
                                          policy_architecture="relational"))


def packet(turn=0):
    actions = np.zeros((1, 2, 17), dtype=np.int32)
    actions[..., 0] = 3
    state = np.zeros((1, 11), dtype=np.int32)
    state[:, 0] = turn
    return {"map_tokens": np.zeros((1, 121, 23), dtype=np.int32),
            "state": state, "action_features": actions,
            "action_mask": np.ones((1, 2), dtype=np.uint8),
            "visible_event_features": np.zeros((1, 0, 26), dtype=np.int32),
            "visible_event_mask": np.zeros((1, 0), dtype=np.uint8)}


def curriculum(tmp_path, candidate, champion, **changes):
    arguments = dict(
        num_envs=2, minimum_steps=1, maximum_steps=1, block_steps=1,
        num_threads=1, max_actions=16, gate_games=2, gate_seed=1237,
        visible_action_history=0, gate_score=0.5, historical_score_floor=0.4,
        readiness_city_captures_per_game=0.0, readiness_completion_rate=0.0,
        readiness_patience=1, policy_anchor_weight=0.0, value_anchor_weight=0.0,
        opponent_policy_weight=0.0, teacher_weight=0.0, shaping_weight=0.0,
        ema_enabled=False, run=TrainingRun.create(tmp_path, "controls"),
    )
    arguments.update(changes)
    return train_direct_population_curriculum(
        TrainingConfig(device="cpu"), candidate.config, candidate, champion,
        [champion], [1.0], **arguments,
    )


def arena(score, opponents=1):
    return {"model_a_score": score, "worst_score": score,
            "model_a_city_captures_per_game": 0.0, "completion_rate": 1.0,
            "opponents": [{"model_a_score": score, "score_lower_95": 0.0,
                           "score_standard_error": 1.0,
                           "model_a_city_captures_per_game": 0.0,
                           "completion_rate": 1.0} for _ in range(opponents)]}


def test_schedules_reach_endpoints_at_minimum_and_hold_them():
    settings = FlatImprovements()
    before = flat_schedule(0, 4000, minimum_steps=24000, maximum_steps=48000,
                           block_steps=4000, settings=settings)
    last = flat_schedule(20000, 4000, minimum_steps=24000, maximum_steps=48000,
                         block_steps=4000, settings=settings)
    extra = flat_schedule(40000, 4000, minimum_steps=24000, maximum_steps=48000,
                          block_steps=4000, settings=settings)
    assert before["block_progress"] == before["start"] == 0
    assert last["block_progress"] == last["end"] == 1
    assert extra["block_progress"] == extra["start"] == extra["end"] == 1
    legacy = flat_schedule(20000, 4000, minimum_steps=24000, maximum_steps=48000,
                           block_steps=4000, settings=None)
    assert legacy["end"] == .5
    explicit = flat_schedule(20000, 4000, minimum_steps=24000, maximum_steps=48000,
                             block_steps=4000, settings=FlatImprovements(schedule_steps=48000))
    assert explicit["end"] == .5


def test_curriculum_actually_passes_complete_schedules_before_early_end(tmp_path):
    candidate = model(); champion = deepcopy(candidate)
    captured = []; evaluations = []
    def train(*args, **kwargs):
        captured.append(kwargs)
        return {"direct_population_environment_steps": 32000.0}
    def evaluate(*args, **kwargs):
        evaluations.append(kwargs)
        return arena(.5 + .01 * len(evaluations))
    with patch("polybot.training.vector_population.train_vector_population", side_effect=train), \
         patch("polybot.training.vector_population.evaluate_direct_population_matches", side_effect=evaluate), \
         patch.object(TrainingRun, "_write_plots"):
        result = curriculum(
            tmp_path, candidate, champion, minimum_steps=24000, maximum_steps=48000,
            block_steps=4000, policy_anchor_weight=.2, flat_policy_anchor_final_weight=.05,
            ppo_rollout_steps=128, shaping_weight=.05,
            flat_improvements=FlatImprovements(value_updates=0, scientific_artifacts=False),
        )
    assert len(captured) == 6 and result["direct_population_steps"] == 24000
    assert captured[0]["ppo_rollout_steps"] == 128
    assert captured[-1]["ppo_rollout_steps"] == 256
    assert captured[0]["policy_anchor_weight"] == .2
    assert captured[-1]["policy_anchor_weight"] == pytest.approx(.05)
    assert captured[-1]["shaping_progress_end"] == 1
    assert result["flat_schedule_steps"] == 24000


@pytest.mark.parametrize("rollback", [False, True])
def test_value_fit_uses_evaluated_actor_labels_including_after_rollback(tmp_path, rollback):
    candidate = model(); champion = deepcopy(candidate)
    first = next(candidate.parameters())
    original = first.detach().clone()
    fits = []
    def version():
        return int(round(float((first.detach()-original).flatten()[0])))
    def train(*args, **kwargs):
        with torch.no_grad(): first.add_(1)
        return {}
    def evaluate(*args, **kwargs):
        v = version()
        callback = kwargs.get("position_callback")
        for game in range(2):
            if callback is not None:
                for _ in range(8): callback(0, game, packet(v), 0)
            if kwargs.get("game_callback"):
                kwargs["game_callback"](0, {"game_index": game, "map_seed": 1237,
                                            "model_a_player": game, "score_a": float(v > 0),
                                            "truncated": 0})
        return arena(0.0 if rollback and v else .5 if not v else 1.0)
    def fit(live, training, replay, validation, settings, **kwargs):
        fits.append((version(), [e.return_target for e in validation], kwargs["validation_source"]))
        return {"updates": 0, "kept": False, "reason": "test", **kwargs}
    with patch("polybot.training.vector_population.train_vector_population", side_effect=train), \
         patch("polybot.training.vector_population.evaluate_direct_population_matches", side_effect=evaluate), \
         patch("polybot.training.vector_population.fit_value_head", side_effect=fit), \
         patch.object(TrainingRun, "_write_plots"):
        curriculum(tmp_path, candidate, champion,
                   flat_improvements=FlatImprovements(value_validation_games=2, scientific_artifacts=False))
    assert fits[0][0] == (0 if rollback else 1)
    assert set(fits[0][1]) == ({-1.0} if rollback else {1.0})
    assert fits[0][2] == ("restored_policy_arena" if rollback else "current_policy_arena")
    assert fits[-1][2] == "selected_policy_arena"


def test_sparse_benchmarks_preserve_an_intermediate_screen_winner(tmp_path):
    candidate = model(); champion = deepcopy(candidate)
    first = next(candidate.parameters()); original = first.detach().clone()
    def train(*args, **kwargs):
        with torch.no_grad(): first.add_(1)
        return {}
    def evaluate(*args, **kwargs):
        v = int(round(float((first.detach()-original).flatten()[0])))
        return arena({0:.5, 1:.65, 2:.55, 3:.54, 4:.52}[v])
    with patch("polybot.training.vector_population.train_vector_population", side_effect=train), \
         patch("polybot.training.vector_population.evaluate_direct_population_matches", side_effect=evaluate), \
         patch.object(TrainingRun, "_write_plots"):
        result = curriculum(tmp_path, candidate, champion, minimum_steps=4, maximum_steps=4,
                            flat_champion_screen=True,
                            flat_improvements=FlatImprovements(value_updates=0, diagnostic_interval=2,
                                diagnostic_improvement=1.0, scientific_artifacts=False))
    assert result["flat_deferred_benchmarks"] == 2
    assert result["flat_full_benchmarks"] == 3  # Baseline, blocks 2 and 4.
    assert result["direct_population_gate_rows"][1]["weighted_score"] is None
    assert result["direct_population_gate_rows"][1]["selected"] == 0
    assert result["direct_population_full_gate_selection"] == "best_screen"
    assert torch.equal(first.detach(), original + 1)


def test_raw_evaluation_cache_ignores_value_head_but_not_policy_changes():
    candidate = model(); opponent = deepcopy(candidate); cache = {}
    full = model_fingerprint(candidate)
    policy = model_fingerprint(candidate, policy_only=True)
    def evaluate():
        return evaluate_direct_population_matches(
            TrainingConfig(device="cpu"), candidate, [opponent], [1.0],
            games=2, gate_seed=2137, evaluation_cache=cache)
    with patch("polybot.training.vector_population.evaluate_direct_matches", return_value={
        "model_a_score": .5, "model_a_city_captures_per_game":0.0, "completion_rate":1.0}) as call:
        evaluate()
        with torch.no_grad(): candidate.value_head[2].bias.add_(1)
        assert model_fingerprint(candidate) != full
        assert model_fingerprint(candidate, policy_only=True) == policy
        evaluate()
        assert call.call_count == 1
        with torch.no_grad(): next(candidate.relational_policy.parameters()).add_(.1)
        evaluate()
        assert call.call_count == 2


def test_position_collection_does_not_disable_other_opponent_cache_entries():
    candidate = model(); a = deepcopy(candidate); b = deepcopy(candidate)
    cache = {}; seen = []
    def evaluate(*args, **kwargs):
        if kwargs.get("position_callback"):
            kwargs["position_callback"](0, packet(), 0)
        return {"model_a_score":.5, "model_a_city_captures_per_game":0.0, "completion_rate":1.0}
    with patch("polybot.training.vector_population.evaluate_direct_matches", side_effect=evaluate) as call:
        for _ in range(2):
            evaluate_direct_population_matches(
                TrainingConfig(device="cpu"), candidate, [a,b], [.5,.5],
                games=2, gate_seed=123, evaluation_cache=cache,
                position_callback=lambda *args: seen.append(args), position_opponent_index=0)
    assert len(seen) == 2
    assert call.call_count == 3  # Only the position-collecting matchup repeats.


def test_current_policy_validation_requires_multiple_completed_games():
    live = model(); collector = ValueValidationCollector(2, 16, 13)
    for _ in range(8): collector.add(0, packet(), 0)
    collector.finish({"game_index":0, "score_a":1.0, "truncated":0})
    original = deepcopy(live.state_dict())
    result = fit_value_head(live, TrainingConfig(device="cpu"), collector.examples,
                           collector.examples, FlatImprovements(value_batch_size=8),
                           seed=7, validation_source="current_policy_arena")
    assert result["reason"] == "insufficient_terminal_games"
    assert all(torch.equal(v, original[k]) for k,v in live.state_dict().items())


def test_critic_backoff_keeps_the_best_intermediate_head_not_the_last_update():
    live = model()
    with torch.no_grad():
        for p in live.value_head.parameters(): p.zero_()
    train = ValueValidationCollector(2, 16, 13)
    valid = ValueValidationCollector(2, 16, 23)
    for game in range(2):
        for _ in range(4):
            train.add(game, packet(), 0)
            valid.add(game, packet(), 0)
        train.finish({"game_index":game, "score_a":1.0, "truncated":0})
        valid.finish({"game_index":game, "score_a":.55, "truncated":0})
    before = deepcopy(live.state_dict())
    result = fit_value_head(live, TrainingConfig(device="cpu"), train.examples,
                           valid.examples, FlatImprovements(value_batch_size=8,
                               value_updates=4, value_learning_rate=.3), seed=13)
    assert result["kept"] and result["attempts"] == 2
    assert result["selected_updates"] == 1
    assert result["learning_rate"] == pytest.approx(.15)
    assert result["after"]["rmse"] < result["before"]["rmse"]
    assert all(torch.equal(t, before[n]) for n,t in live.state_dict().items()
               if not n.startswith("value_head."))
