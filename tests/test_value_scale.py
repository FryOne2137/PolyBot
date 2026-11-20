import numpy as np
import pytest

from polybot.training.value_scale import cross_validate_scale, fit_scale, rank_errors, scale_metrics


def arrays():
    targets = np.array([[0.1, -0.1], [-0.2, 0.2], [0.1, 0.1], [0.2, -0.1],
                        [-0.1, 0.1], [0.1, -0.2], [0.2, 0.2], [-0.1, 0.2]])
    pairs = {"predictions": targets * 4, "targets": targets, "signs": np.ones((8, 2)),
             "map_seeds": np.arange(8), "gap_weights": np.ones(8), "value_weights": np.ones(8),
             "pair_ids": np.arange(8), "players": np.zeros(8), "phases": np.full(8, "opening")}
    calibration = {"targets": np.array([1.0, -1.0] * 4), "predictions": np.array([0.8, -0.8] * 4),
                   "map_seeds": np.arange(100, 108)}
    return pairs, calibration


def test_scalar_fit_recovers_attenuation_and_preserves_positive_rankings():
    pairs, calibration = arrays()
    scale = fit_scale(pairs, calibration, pair_value_weight=0, calibration_weight=0)
    assert scale == pytest.approx(0.25)
    scaled, raw = scale_metrics(pairs, calibration, scale), scale_metrics(pairs, calibration, 1)
    assert scaled["gap"]["all"]["gap_rmse"] < 1e-12
    assert scaled["gap"]["informative"]["preference_accuracy"] == raw["gap"]["informative"]["preference_accuracy"] == 1


def test_cross_validation_is_grouped_and_chooses_losses_using_common_held_out_score():
    pairs, calibration = arrays()
    report = cross_validate_scale(pairs, calibration, folds=4, seed=21,
                                  calibration_weights=(0, 0.2, 1), pair_value_weight=1)
    assert report["selection"]["scale"] == pytest.approx(0.25)
    for method in report["methods"].values():
        held_pairs, held_calibration = [], []
        for fold in method["folds"]:
            assert not set(fold["held_pair_maps"]) & set(fold["train_pair_maps"])
            assert not set(fold["held_calibration_maps"]) & set(fold["train_calibration_maps"])
            held_pairs.extend(fold["held_pair_maps"])
            held_calibration.extend(fold["held_calibration_maps"])
        assert sorted(held_pairs) == list(range(8))
        assert sorted(held_calibration) == list(range(100, 108))
    assert len(report["fold_winners"]) == 4
    assert report == cross_validate_scale(pairs, calibration, folds=4, seed=21,
                                          calibration_weights=(0, 0.2, 1), pair_value_weight=1)


def test_zero_control_wins_if_positive_scales_damage_absolute_calibration():
    pairs, calibration = arrays()
    calibration["predictions"] *= -1
    report = cross_validate_scale(pairs, calibration, folds=4, seed=21,
                                  calibration_weights=(0, 0.2, 1), pair_value_weight=1)
    assert report["selection"]["scale"] == 0
    assert not report["methods"]["gap_only"]["eligible"]


def test_scale_fit_keeps_bounds_and_rejects_leaking_map_groups():
    pairs, calibration = arrays()
    pairs["predictions"] *= -1
    assert fit_scale(pairs, calibration, pair_value_weight=0, calibration_weight=0) == 0
    pairs["predictions"] *= -0.01
    assert fit_scale(pairs, calibration, pair_value_weight=0, calibration_weight=0) == 1
    calibration["map_seeds"][0] = pairs["map_seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        cross_validate_scale(pairs, calibration, folds=4, seed=21, calibration_weights=(0,), pair_value_weight=1)


def test_outliers_report_dominant_errors_and_original_head_regressions():
    pairs, _ = arrays()
    report = rank_errors(pairs, original_predictions=pairs["targets"], scale=0.25, top=3)
    errors = [row["squared_error_unscaled"] for row in report["all_pairs"]]
    assert errors == sorted(errors, reverse=True)
    assert report["largest_errors"][0]["error_increase_vs_original"] > 0
    assert all(row["squared_error_scaled"] < 1e-12 for row in report["all_pairs"])
    assert 0 < report["top1_error_fraction"] <= report["top3_error_fraction"] <= 1
