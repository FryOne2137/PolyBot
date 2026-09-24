import numpy as np

from polybot.config import MCTSConfig
from polybot.data.polyenv_features import (
    action_features,
    event_action_history,
    event_summary,
    global_features_from_obs,
    vector_event_action_history,
    vector_event_summary,
)
from polybot.training.mcts_exploration import (
    add_dirichlet_noise_to_root_logits,
    mix_dirichlet_root_prior,
    self_play_temperature,
)


def test_action_features_preserve_legal_action_row_count():
    actions = {
        "action_id": np.asarray([41, 99]),
        "type_id": np.asarray([0, 3]),
        "source_index": np.asarray([3, -1]),
        "target_index": np.asarray([4, -1]),
    }
    categorical, continuous = action_features(actions)
    assert categorical.shape[0] == 2
    assert continuous.shape[0] == 2
    assert categorical[0, 0] == 0
    assert categorical[1, 0] == 3


def test_dirichlet_root_noise_keeps_a_legal_probability_distribution():
    prior = np.asarray([0.70, 0.20, 0.10], dtype=np.float64)
    noisy = mix_dirichlet_root_prior(
        prior, np.random.default_rng(7), epsilon=0.25, concentration=10.0
    )

    assert np.isclose(noisy.sum(), 1.0)
    assert np.all(noisy > 0.0)
    assert not np.allclose(noisy, prior)


def test_dirichlet_noise_leaves_padded_logits_untouched():
    logits = np.asarray([[2.0, -4.0, 0.0, 9.0]], dtype=np.float32)
    mask = np.asarray([[True, False, True, False]])
    original = logits.copy()

    add_dirichlet_noise_to_root_logits(
        logits, mask, np.random.default_rng(21), epsilon=0.25, concentration=10.0
    )

    np.testing.assert_array_equal(logits[~mask], original[~mask])
    np.testing.assert_allclose(np.exp(logits[0, mask[0]]).sum(), 1.0, atol=1e-6)


def test_self_play_temperature_becomes_greedy_after_opening():
    config = MCTSConfig(temperature=0.8, temperature_drop_move=3, late_temperature=0.0)

    assert self_play_temperature(config, 2) == 0.8
    assert self_play_temperature(config, 3) == 0.0


def test_global_feature_vector_is_stable_when_optional_obs_keys_are_absent():
    vector = global_features_from_obs({"turn": 3, "stars": 5, "map_type": "lakes"})
    assert vector.shape == (9,)
    assert np.isfinite(vector).all()


def test_event_summary_contains_no_runtime_ids():
    summary = event_summary({"type_id": np.asarray([0, 1, 1]), "damage": np.asarray([0, 4, 2])})
    assert summary.shape == (12,)
    assert summary[2] > 0


def test_vector_events_match_compact_self_play_history_layout():
    events = np.full((1, 3, 26), -1, dtype=np.int32)
    mask = np.asarray([[0, 1, 1]], dtype=np.uint8)
    events[0, 1, [2, 9, 10, 3, 4, 5, 14, 15, 16, 17]] = (
        7, 0, 3, 3, 10, 11, 2, -1, 41, -1,
    )
    events[0, 2, [2, 9, 10, 3, 4, 5, 14, 15, 16, 17]] = (
        13, 1, 2, 2, 11, 12, -1, -1, -1, -1,
    )
    events[0, 2, 12] = 5  # building detail
    history, history_mask = vector_event_action_history({
        "visible_event_features": events,
        "visible_event_mask": mask,
    })
    assert history.shape == (1, 3, 12)
    assert history_mask.tolist() == [[False, True, True]]
    assert history[0, 1, :10].tolist() == [
        7, 0, 3, 3, 10, 11, 2, -1, 41, -1,
    ]
    assert history[0, 2, 10:].tolist() == [2, 5]
    assert not history[0, 0].any()

    summary = vector_event_summary({
        "visible_event_features": events,
        "visible_event_mask": mask,
    })
    assert summary.shape == (1, 12)
    assert np.isclose(summary[0, 0], 2 / 128.0)
    assert summary[0, 8] > 0.0  # event type 7


def test_scalar_event_history_is_bounded_and_right_aligned():
    events = {
        "type_id": np.asarray([7, 9, 13]),
        "actor_player": np.asarray([0, 1, 0]),
        "actor_tribe": np.asarray([3, 2, 3]),
        "flags": np.asarray([3, 2, 1]),
        "source_index": np.asarray([10, 11, -1]),
        "target_index": np.asarray([11, 12, 13]),
        "building_type": np.asarray([-1, -1, 5]),
    }
    history, mask = event_action_history(events, 2)
    assert mask.tolist() == [True, True]
    assert history[:, 0].tolist() == [9, 13]
    assert history[-1, 10:].tolist() == [2, 5]
