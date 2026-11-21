import numpy as np
import pytest

from polybot.config import MCTSConfig, TrainingConfig
from polybot.evaluation import value_disagreements as source
from polybot.training.routed_self_play_pool import _invalid_evaluation_result


def sample(**kwargs):
    return source._sample_source_positions(None, None, TrainingConfig(seed=100), MCTSConfig(), 4,
        num_envs=4, num_threads=1, history=0, samples_per_game=1, source_policy="mcts-zero", **kwargs)


@pytest.mark.parametrize("failed_seed", [100, 101, None])
def test_failed_source_batch_is_isolated_on_original_maps_and_only_valid_prefixes_survive(monkeypatch, failed_seed):
    calls = []

    def evaluate(model, opponent, training, mcts, games, **kwargs):
        schedule = kwargs["map_seeds"]
        calls.append(schedule)
        assert not kwargs["replace_failed_pairs"]
        invalid = games == 4 or training.seed == failed_seed
        # Also send partial callbacks from the invalid batch. They must never
        # be retained alongside a replay started again at the original root.
        packet = {"agent_id": np.array([0] * games), "action_mask": np.ones((games, 2), dtype=np.uint8),
            "map_seed": np.array(schedule), "to_play": np.array([i % 2 for i in range(games)])}
        kwargs["action_callback"](np.arange(games), packet, np.array([999 if invalid else 10 + training.seed] * games))
        for game in range(games):
            kwargs["game_callback"]({"game_index": game, "truncated": False})
        if invalid:
            return {"valid": False, "completed_games": 0, "invalid_reason": "belief failed",
                    "belief_failure_details": {"message": "bad particle", "moves": [58]}}
        return {"valid": True, "evaluated_map_seeds": schedule}

    monkeypatch.setattr(source, "evaluate_matches_pool", evaluate)
    monkeypatch.setattr(source, "_position_fingerprint", lambda packet, row: f"seed-{packet['map_seed'][row]}")
    probes, prefixes, report = sample(isolate_failed_pairs=True)
    assert calls == [[100, 100, 101, 101], [100, 100], [101, 101]]
    expected = [seed for seed in (100, 101) if seed != failed_seed]
    assert report["map_seeds"] == expected
    assert report["failed_map_seeds"] == ([] if failed_seed is None else [failed_seed])
    assert report["valid"] is (failed_seed is None)
    assert report["terminal_source_games"] == 2 * len(expected)
    assert {position.seed for position in probes} == set(expected)
    assert {position.game for position in probes} == {i for i in range(4) if 100 + i // 2 in expected}
    assert all(prefixes[position.game] == [10 + position.seed] for position in probes)
    assert sum(row["planned_games"] for row in report["source_evaluation_attempts"]) == 8
    assert report["source_evaluation_attempts"][0]["belief_failure_details"]["moves"] == [58]


def test_source_recovery_stops_at_deadline_and_does_not_retry_single_pairs_forever(monkeypatch):
    clock, calls = [0], []

    def evaluate(*args, **kwargs):
        calls.append(kwargs["map_seeds"])
        clock[0] = 10
        return {"valid": False, "invalid_reason": "belief failed"}

    monkeypatch.setattr(source, "evaluate_matches_pool", evaluate)
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])
    probes, prefixes, report = sample(isolate_failed_pairs=True, deadline=5)
    assert not probes and not prefixes and report["failed_map_seeds"] == [100, 101]
    assert report["deadline_reached"] and calls == [[100, 100, 101, 101]]
    calls.clear()
    probes, _, report = source._sample_source_positions(None, None, TrainingConfig(seed=100), MCTSConfig(), 2,
        num_envs=2, num_threads=1, history=0, samples_per_game=1, source_policy="mcts-zero", isolate_failed_pairs=True)
    assert not probes and len(calls) == 1 and report["failed_map_seeds"] == [100]


def test_wrong_fixed_schedule_and_programming_errors_are_still_fatal(monkeypatch):
    monkeypatch.setattr(source, "evaluate_matches_pool", lambda *args, **kwargs:
        {"valid": True, "evaluated_map_seeds": [200] * 4})
    with pytest.raises(RuntimeError, match="changed the fixed map schedule"):
        sample(isolate_failed_pairs=True)

    def broken(*args, **kwargs):
        raise ValueError("unexpected native shape error")

    monkeypatch.setattr(source, "evaluate_matches_pool", broken)
    with pytest.raises(ValueError, match="unexpected native shape"):
        sample(isolate_failed_pairs=True)
    monkeypatch.setattr(source, "evaluate_matches_pool", lambda *args, **kwargs:
        {"valid": False, "invalid_reason": "actual reason"})
    with pytest.raises(RuntimeError, match="actual reason"):
        sample()


def test_invalid_evaluation_retains_failure_details_without_turning_into_a_valid_score():
    totals = {"games": 4.0, "model_a_wins": 0.0, "model_a_losses": 0.0, "draws": 0.0,
              "moves": 0.0, "model_a_particles": 0.0, "model_a_decisions": 0.0}
    details = {"message": "bad particle", "map_seeds": [100, 100], "moves": [58, 50], "diagnostics": []}
    result = _invalid_evaluation_result(totals, [], reason="belief failed", invalid_attempts=1,
        discarded_pairs=1, replacement_events=[], failure_details=details)
    assert not result["valid"] and result["completed_games"] == 0
    assert result["belief_failure_details"] == details
