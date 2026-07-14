import numpy as np

from polybot.data.polyenv_features import action_features, event_summary, global_features_from_obs


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


def test_global_feature_vector_is_stable_when_optional_obs_keys_are_absent():
    vector = global_features_from_obs({"turn": 3, "stars": 5, "map_type": "lakes"})
    assert vector.shape == (9,)
    assert np.isfinite(vector).all()


def test_event_summary_contains_no_runtime_ids():
    summary = event_summary({"type_id": np.asarray([0, 1, 1]), "damage": np.asarray([0, 4, 2])})
    assert summary.shape == (12,)
    assert summary[2] > 0
