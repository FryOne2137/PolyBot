"""Train-only attenuation of a frozen value head, with map-held-out comparisons."""

from __future__ import annotations

from collections import Counter
import numpy as np


def subset(batch, mask):
    return {key: values[mask] for key, values in batch.items()}


def map_weights(seeds):
    _, inverse, counts = np.unique(seeds, return_inverse=True, return_counts=True)
    if not len(seeds):
        raise ValueError("scale fitting requires nonempty map groups")
    return len(seeds) / (len(counts) * counts[inverse])


def gaps(batch):
    return np.diff(batch["predictions"] * batch["signs"], axis=1).reshape(-1), np.diff(
        batch["targets"] * batch["signs"], axis=1).reshape(-1)


def fit_scale(pairs, calibration, *, pair_value_weight, calibration_weight, maximum=1.0):
    """Solve the bounded scalar least-squares fit without changing model weights."""
    if not 0 < maximum <= 1 or pair_value_weight < 0 or calibration_weight < 0:
        raise ValueError("scale bounds and loss weights must be nonnegative, with maximum in (0, 1]")
    x, y = gaps(pairs)
    pair_mass = map_weights(pairs["map_seeds"])
    terms = [(x / 2, y / 2, pair_mass * pairs["gap_weights"])]
    if pair_value_weight:
        terms.append((pairs["predictions"], pairs["targets"],
                      pair_value_weight * pair_mass[:, None] * pairs["value_weights"][:, None]))
    if calibration_weight:
        terms.append((calibration["predictions"], calibration["targets"],
                      calibration_weight * map_weights(calibration["map_seeds"])))
    numerator = denominator = 0.0
    for predicted, target, weight in terms:
        if not np.isfinite(predicted).all() or not np.isfinite(target).all() or not np.isfinite(weight).all():
            raise ValueError("nonfinite scale supervision")
        numerator += float(np.mean(weight * predicted * target))
        denominator += float(np.mean(weight * np.square(predicted)))
    return float(np.clip(numerator / denominator, 0, maximum)) if denominator > 1e-15 else 0.0


def scale_metrics(pairs, calibration, scale):
    predicted, target = gaps(pairs)
    error = scale * predicted - target
    informative = target != 0
    group_metrics = {}
    for name, mask in (("all", np.ones(len(error), dtype=bool)), ("tied", ~informative), ("informative", informative)):
        group_metrics[name] = {
            "pairs": int(mask.sum()), "gap_rmse": float(np.sqrt(np.square(error[mask]).mean())) if mask.any() else None,
            "preference_accuracy": float(np.mean(scale * predicted[mask] * target[mask] > 0))
            if name == "informative" and mask.any() else None,
        }
    value_error = scale * calibration["predictions"] - calibration["targets"]
    weights = map_weights(calibration["map_seeds"])
    return {
        "gap": group_metrics,
        "weighted_gap_mse": float(np.mean(map_weights(pairs["map_seeds"]) * pairs["gap_weights"] * np.square(error))),
        "calibration_rmse": float(np.sqrt(np.mean(weights * np.square(value_error)))),
        "calibration_bias": float(np.mean(weights * value_error)),
        "calibration_zero_rmse": float(np.sqrt(np.mean(weights * np.square(calibration["targets"])))),
    }


def cross_validate_scale(pairs, calibration, *, folds, seed, calibration_weights, pair_value_weight, maximum=1.0):
    """Compare scalar-fitting objectives on original training maps only."""
    pair_maps = np.unique(pairs["map_seeds"])
    calibration_maps = np.unique(calibration["map_seeds"])
    if np.intersect1d(pair_maps, calibration_maps).size:
        raise ValueError("pair and terminal calibration maps must be disjoint")
    if folds < 2 or folds > min(len(pair_maps), len(calibration_maps)):
        raise ValueError("each scale fold needs at least one pair map and one calibration map")
    rng = np.random.default_rng(seed)
    pair_folds = np.array_split(rng.permutation(pair_maps), folds)
    calibration_folds = np.array_split(rng.permutation(calibration_maps), folds)
    methods = {"gap_only": {"pair_value_weight": 0.0, "calibration_weight": 0.0}}
    methods.update({f"joint-{weight!r}": {"pair_value_weight": pair_value_weight, "calibration_weight": weight}
                    for weight in calibration_weights})
    report = {}
    for name, objective in methods.items():
        pair_predictions = np.empty_like(pairs["predictions"], dtype=np.float64)
        calibration_predictions = np.empty_like(calibration["predictions"], dtype=np.float64)
        scales, records = [], []
        for index, (held_pairs, held_calibration) in enumerate(zip(pair_folds, calibration_folds, strict=True)):
            pair_mask = np.isin(pairs["map_seeds"], held_pairs)
            calibration_mask = np.isin(calibration["map_seeds"], held_calibration)
            scale = fit_scale(subset(pairs, ~pair_mask), subset(calibration, ~calibration_mask), maximum=maximum, **objective)
            pair_predictions[pair_mask] = scale * pairs["predictions"][pair_mask]
            calibration_predictions[calibration_mask] = scale * calibration["predictions"][calibration_mask]
            scales.append(scale)
            records.append({"fold": index + 1, "scale": scale, "held_pair_maps": held_pairs.tolist(),
                            "held_calibration_maps": held_calibration.tolist(),
                            "train_pair_maps": pair_maps[~np.isin(pair_maps, held_pairs)].tolist(),
                            "train_calibration_maps": calibration_maps[~np.isin(calibration_maps, held_calibration)].tolist(),
                            "metrics": scale_metrics(subset(pairs, pair_mask), subset(calibration, calibration_mask), scale)})
        metrics = scale_metrics({**pairs, "predictions": pair_predictions},
                                {**calibration, "predictions": calibration_predictions}, 1.0)
        report[name] = {"objective": objective, "folds": records, "out_of_fold": metrics,
                        "scale_min": min(scales), "scale_median": float(np.median(scales)),
                        "scale_max": max(scales), "scale_std": float(np.std(scales)),
                        "eligible": metrics["calibration_rmse"] <= metrics["calibration_zero_rmse"] + 1e-9}
    controls = {name: scale_metrics(pairs, calibration, scale) for name, scale in (("zero", 0.0), ("unscaled", 1.0))}
    winners = []
    for index in range(folds):
        p = subset(pairs, np.isin(pairs["map_seeds"], pair_folds[index]))
        c = subset(calibration, np.isin(calibration["map_seeds"], calibration_folds[index]))
        zero = scale_metrics(p, c, 0.0)["weighted_gap_mse"]
        candidates = [(name, row["folds"][index]["metrics"]) for name, row in report.items()]
        candidates = [(name, metrics) for name, metrics in candidates
                      if metrics["calibration_rmse"] <= metrics["calibration_zero_rmse"] + 1e-9]
        winner = min(candidates, key=lambda item: item[1]["weighted_gap_mse"]) if candidates else None
        winners.append(winner[0] if winner and winner[1]["weighted_gap_mse"] < zero - 1e-12 else "zero")
    eligible = [(name, row) for name, row in report.items() if row["eligible"]]
    best = min(eligible, key=lambda item: (item[1]["out_of_fold"]["weighted_gap_mse"],
                                         item[1]["out_of_fold"]["calibration_rmse"], item[0])) if eligible else None
    # A scale that cannot improve held-out gap error over zero offers no
    # offline evidence for spending an arena on the same zero-valued search.
    if best is None or best[1]["out_of_fold"]["weighted_gap_mse"] >= controls["zero"]["weighted_gap_mse"] - 1e-12:
        selection = {"method": "zero", "scale": 0.0, "reason": "no_cross_validated_gap_gain_over_zero"}
    else:
        name, row = best
        scale = fit_scale(pairs, calibration, maximum=maximum, **row["objective"])
        selection = {"method": name, "scale": scale, "objective": row["objective"],
                     "reason": "train_map_cross_validation", "weighted_gap_mse": row["out_of_fold"]["weighted_gap_mse"]}
    return {"selection": selection, "methods": report, "controls": controls, "fold_count": folds, "seed": seed,
            "fold_winners": winners, "fold_winner_counts": dict(Counter(winners)),
            "selection_rule": "minimize out-of-fold map-balanced uncertainty-weighted gap MSE; calibration must beat or match zero",
            "scope": "scalar calibration of a previously trained fixed head, not cross-validation of the head's original fit"}


def rank_errors(pairs, *, original_predictions, scale, top):
    predicted, target = gaps(pairs)
    old_gap = np.diff(original_predictions * pairs["signs"], axis=1).reshape(-1)
    rows = [{
        "map_seed": int(pairs["map_seeds"][i]), "pair_id": int(pairs["pair_ids"][i]),
        "root_player": int(pairs["players"][i]), "phase": str(pairs["phases"][i]),
        "group": "tied" if target[i] == 0 else "informative", "target_gap": float(target[i]),
        "original_gap": float(old_gap[i]), "unscaled_gap": float(predicted[i]), "scaled_gap": float(scale * predicted[i]),
        "squared_error_unscaled": float((predicted[i] - target[i]) ** 2),
        "squared_error_original": float((old_gap[i] - target[i]) ** 2),
        "squared_error_scaled": float((scale * predicted[i] - target[i]) ** 2),
        "error_increase_vs_original": float((predicted[i] - target[i]) ** 2 - (old_gap[i] - target[i]) ** 2),
        "gap_weight": float(pairs["gap_weights"][i]),
    } for i in range(len(target))]
    rows.sort(key=lambda row: (-row["squared_error_unscaled"], row["pair_id"]))
    total = sum(row["squared_error_unscaled"] for row in rows)
    return {"largest_errors": rows[:top],
            "largest_regressions": sorted(rows, key=lambda row: -row["error_increase_vs_original"])[:top],
            "top_error_fraction": sum(row["squared_error_unscaled"] for row in rows[:top]) / total if total else 0.0,
            "top1_error_fraction": rows[0]["squared_error_unscaled"] / total if total else 0.0,
            "top3_error_fraction": sum(row["squared_error_unscaled"] for row in rows[:3]) / total if total else 0.0,
            "all_pairs": rows}
