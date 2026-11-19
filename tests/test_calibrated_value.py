from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from polybot.config import MCTSConfig, TrainingConfig
from polybot.training.calibrated_value import (
    cached_terminal_positions, calibration_admission, fit_calibrated_heads, load_ranking_sources,
)
from polybot.training.flat_experiments import FlatImprovements
from polybot.training.flat_value import EncodedValuePosition
from polybot.training.run_artifacts import TrainingRun


def test_calibration_gate_rejects_a_better_ranking_with_bad_absolute_values():
    rank = {"valid": True, "preference_accuracy": 0.5, "gap_rmse": 0.2}
    reference_rank = {"valid": True, "preference_accuracy": 0.0, "gap_rmse": 0.4}
    calibration = {"valid": True, "rmse": 0.8, "bias": 0.1, "zero_rmse": 1.0}
    reference = {**calibration, "rmse": 0.9}
    assert not calibration_admission(rank, calibration, reference_rank, reference, rmse_regression=0, bias_regression=0.02)
    rejected = calibration_admission(rank, {**calibration, "rmse": 1.4, "bias": -0.6},
                                     reference_rank, reference, rmse_regression=0, bias_regression=0.02)
    assert set(rejected) == {"calibration_rmse_regression", "calibration_worse_than_zero", "calibration_bias_regression"}
    rejected = calibration_admission({**rank, "preference_accuracy": 0}, calibration, reference_rank, reference,
                                     rmse_regression=0, bias_regression=0.02)
    assert "ranking_accuracy_not_improved" in rejected
    assert calibration_admission(rank, {**calibration, "rmse": float("nan")}, reference_rank, reference,
                                 rmse_regression=0, bias_regression=0.02) == ["nonfinite_metrics"]


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.value_head = nn.Sequential(nn.Linear(2, 1), nn.Tanh())
        with torch.no_grad():
            self.value_head[0].weight.zero_()
            self.value_head[0].bias.zero_()
        self.requires_grad_(False)
        self.value_head.requires_grad_(True)


def test_joint_fit_calibrates_and_restores_all_input_weights(tmp_path):
    model = ValueNet()
    before = deepcopy(model.state_dict())

    def paired(seed):
        return [
            EncodedValuePosition(np.array([sign, 0], dtype=np.float32), 0, -0.5 * sign,
                                 (seed, 0, pair), "opening")
            for pair in range(2) for sign in (1, -1)
        ]

    def terminal(seed):
        return [
            EncodedValuePosition(np.array([sign * scale, 0], dtype=np.float32), 0, -float(sign),
                                 (seed + index, 0), "opening")
            for index, scale in enumerate((1, 2)) for sign in (1, -1)
        ]

    state, report = fit_calibrated_heads(
        model, {"train": paired(100), "selection": paired(200)},
        {"train": terminal(300), "selection": terminal(400)}, TrainingConfig(device="cpu"),
        FlatImprovements(value_updates=12, value_batch_size=4, value_learning_rate=0.05),
        weights=(0.01, 0.5), rmse_regression=0.0, bias_regression=0.1,
        run=TrainingRun(tmp_path), log_interval=12,
    )
    assert state is not None and report["selected"]["rank"]["preference_accuracy"] == 1.0
    assert report["selected"]["calibration"]["rmse"] < report["reference"]["calibration"]["rmse"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not torch.equal(state["0.weight"], before["value_head.0.weight"])
    for weight in (0.01, 0.5):
        assert (tmp_path / f"calibration-{weight:g}" / "metrics.csv").is_file()


def test_terminal_cache_reuses_labels_and_rejects_reserved_maps(tmp_path, monkeypatch):
    model, opponent = ValueNet(), ValueNet()
    calls = []

    def collect(*args, **kwargs):
        calls.append(True)
        assert kwargs["model_a_uses_mcts"]
        rows = [
            EncodedValuePosition(np.array([1, 0], dtype=np.float32), 0, 1, (100, 0), "opening"),
            EncodedValuePosition(np.array([-1, 0], dtype=np.float32), 1, -1, (101, 1), "late"),
        ]
        return rows, {"valid": True, "attempted_map_seeds": [100, 101], "labelled_maps": 2, "positions": 2}

    monkeypatch.setattr("polybot.training.calibrated_value.collect_mcts_terminal_value_games", collect)
    options = dict(cache_directory=tmp_path, excluded_map_seeds={200, 201}, positions_per_game=3,
                   pool={}, policy="mcts-zero")
    rows, info = cached_terminal_positions(model, opponent, TrainingConfig(device="cpu"), MCTSConfig(), 4, **options)
    restored, loaded = cached_terminal_positions(model, opponent, TrainingConfig(device="cpu"), MCTSConfig(), 4, **options)
    assert len(calls) == 1 and not info["cache_loaded"] and loaded["cache_loaded"]
    np.testing.assert_array_equal(restored[0].features, rows[0].features)
    with pytest.raises(ValueError, match="reserved"):
        cached_terminal_positions(model, opponent, TrainingConfig(device="cpu"), MCTSConfig(), 4,
                                   **{**options, "excluded_map_seeds": {100}})
    with torch.no_grad():
        model.value_head[0].bias.add_(0.1)
    _, changed = cached_terminal_positions(model, opponent, TrainingConfig(device="cpu"), MCTSConfig(), 4, **options)
    assert len(calls) == 2 and not changed["cache_loaded"]
    assert changed["cache_path"] != info["cache_path"]


def test_ranking_sources_keep_colliding_pair_ids_separate(monkeypatch):
    def load(*args, **kwargs):
        rows = [EncodedValuePosition(np.ones(2, dtype=np.float32), 0, target, (100, 0, 0), "opening")
                for target in (0, 1)]
        return rows, {"pairs": [{"pair_id": 0}], "reserved_map_seeds": [100]}

    monkeypatch.setattr("polybot.training.calibrated_value.load_saved_value_pairs", load)
    rows, source = load_ranking_sources(None, None, ["first", "second"], history=0)
    assert len(rows) == 4 and len({row.game_key for row in rows}) == 2
    assert source["reserved_map_seeds"] == [100]


def test_joint_fit_rejects_a_ranking_update_that_breaks_calibration(tmp_path):
    model = ValueNet()
    before = deepcopy(model.state_dict())

    def pairs(seed):
        return [EncodedValuePosition(np.array([sign, 0], dtype=np.float32), 0, -float(sign),
                                     (seed, 0, 0), "opening") for sign in (1, -1)]

    def points(seed):
        return [EncodedValuePosition(np.array([sign, 0], dtype=np.float32), 0, float(sign),
                                     (seed, 0), "opening") for sign in (1, -1)]

    state, report = fit_calibrated_heads(
        model, {"train": pairs(100), "selection": pairs(200)},
        {"train": points(300), "selection": points(400)}, TrainingConfig(device="cpu"),
        FlatImprovements(value_updates=8, value_batch_size=4, value_learning_rate=0.05),
        weights=(0.01,), rmse_regression=0.0, bias_regression=0.02,
        run=TrainingRun(tmp_path), log_interval=8,
    )
    assert state is None and report["selected"] is None
    assert report["variants"]["0.01"]["rejections"]["calibration_rmse_regression"] > 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
