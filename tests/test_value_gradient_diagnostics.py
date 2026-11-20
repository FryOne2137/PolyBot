from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from polybot.training.flat_value import EncodedValuePosition
from polybot.training.value_gradient_diagnostics import inspect_cache_gradients


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Linear(1, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)


def data():
    def pairs(seed):
        return [EncodedValuePosition(np.array([sign], dtype=np.float32), 0, 0.5 * sign,
                                     (seed, 0, 1), "opening", gap_weight=0.1) for sign in (1, -1)]

    def points(seed):
        return [EncodedValuePosition(np.array([sign], dtype=np.float32), 0, -float(sign),
                                     (seed, 0), "opening") for sign in (1, -1)]

    return {"train": pairs(100), "selection": pairs(200)}, {"train": points(300), "selection": points(400)}


def inspect(model, ranking, calibration, **kwargs):
    return inspect_cache_gradients(model, ranking, calibration, seed=5, batch_size=4, learning_rate=0.01,
                                    pair_weight=1, cache_weight=1, balance_pair_maps=True,
                                    batches=4, probe_batches=2, **kwargs)


def test_opposing_cache_gradients_and_actual_adamw_damage_are_detected_without_mutation():
    model = ValueNet().train()
    before = deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    ranking, calibration = data()
    report, gradients, probes = inspect(model, ranking, calibration)
    expected = report["expected_phase_balanced_gradients"]
    assert expected["gap_norm"] == pytest.approx(0.1)
    assert expected["cache_norm"] == pytest.approx(2)
    assert expected["cache_to_gap_norm_ratio"] == pytest.approx(20)
    assert expected["gap_cache_cosine"] == pytest.approx(-1)
    assert expected["unweighted_gap_norm"] == pytest.approx(1)
    assert expected["gap_dot_combined"] < 0 and expected["combined_is_gap_ascent_direction"]
    assert report["minibatches"]["cache_conflict_fraction"] == 1
    steps = report["isolated_adamw_probes"]
    assert steps["gap_only"]["delta"]["train_gap_loss"]["mean"] < 0
    assert steps["cache_only"]["delta"]["train_gap_loss"]["mean"] > 0
    assert steps["cache_only"]["fraction_train_gap_worsened"] == 1
    assert len(gradients) == 4 and len(probes) == 8
    assert model.value_head.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters())


def test_expected_cache_gradient_matches_phase_sampling_not_uniform_positions():
    model = ValueNet()
    ranking, calibration = data()
    calibration["train"] = [EncodedValuePosition(np.array([1], dtype=np.float32), 0, 1, (300, 0), "opening")]
    calibration["train"].extend(EncodedValuePosition(np.array([2], dtype=np.float32), 0, -1, (301, 0), "mid")
                                for _ in range(4))
    report, _, _ = inspect(model, ranking, calibration)
    expected = report["expected_phase_balanced_gradients"]
    assert expected["cache_norm"] == pytest.approx(1)
    assert expected["uniform_cache_norm"] == pytest.approx(np.sqrt(2.8**2 + 1.2**2))


def test_selection_changes_do_not_change_training_gradients_or_probe_updates():
    model = ValueNet()
    ranking, calibration = data()
    original, batches, probes = inspect(model, ranking, calibration)
    for row in calibration["selection"] + ranking["selection"]:
        row.features[0] *= 10
    changed, new_batches, new_probes = inspect(model, ranking, calibration)
    assert original["expected_phase_balanced_gradients"] == changed["expected_phase_balanced_gradients"]
    assert batches == new_batches
    assert [row["after"]["train_gap_loss"] for row in probes] == [row["after"]["train_gap_loss"] for row in new_probes]
    assert original["isolated_adamw_probes"]["cache_only"]["delta"]["cache_selection_rmse"] != \
        changed["isolated_adamw_probes"]["cache_only"]["delta"]["cache_selection_rmse"]


def test_zero_gradients_are_explicit_and_maps_cannot_leak():
    model = ValueNet()
    ranking, calibration = data()
    for rows in (*ranking.values(), *calibration.values()):
        for row in rows:
            row.features[:] = 0
    # Only bias has a cache gradient; the gap gradient cancels exactly.
    report, batches, probes = inspect(model, ranking, calibration)
    assert report["expected_phase_balanced_gradients"]["gap_norm"] == 0
    assert report["expected_phase_balanced_gradients"]["gap_cache_cosine"] is None
    assert report["minibatches"]["cache_conflict_fraction"] is None
    json.dumps((report, batches, probes), allow_nan=False)
    calibration["selection"] = calibration["train"]
    with pytest.raises(ValueError, match="disjoint"):
        inspect(model, ranking, calibration)
