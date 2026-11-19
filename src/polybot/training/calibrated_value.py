"""Fit branch rankings while guarding value calibration on other complete games."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np
import torch
from torch.nn import functional as F

from .flat_experiments import isolated_rng, model_fingerprint
from .flat_value import EncodedValuePosition, VALUE_PHASES, _metrics_by_phase, _paired_metrics, _paired_value_tensors, _value_gap
from .mcts_value import collect_mcts_terminal_value_games
from .saved_value_pairs import load_saved_value_pairs


def load_ranking_sources(model, training, directories, *, history):
    examples, sources, reserved = [], [], set()
    offset = 0
    for directory in directories:
        rows, source = load_saved_value_pairs(model, training, directory, history=history)
        examples.extend(replace(row, game_key=(*row.game_key[:2], row.game_key[2] + offset)) for row in rows)
        source["global_pair_id_offset"] = offset
        offset += max(int(row["pair_id"]) for row in source["pairs"]) + 1
        sources.append(source)
        reserved.update(source["reserved_map_seeds"])
    return examples, {"sources": sources, "reserved_map_seeds": sorted(reserved)}


def informative_pairs(examples):
    pairs = {}
    for example in examples:
        pairs.setdefault(example.game_key, []).append(example)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("ranking supervision requires two successors per position")
    return [
        example for pair in pairs.values()
        if abs(pair[1].return_target * pair[1].to_root_sign - pair[0].return_target * pair[0].to_root_sign) > 1e-8
        for example in pair
    ]


def cached_terminal_positions(model, opponent, training, mcts, games, *, cache_directory,
                              excluded_map_seeds, positions_per_game, pool, policy, require_existing=False):
    """Cache only labels from complete seat-swapped games with matching weights."""
    identity = {
        "schema": 1, "model": model_fingerprint(model), "opponent": model_fingerprint(opponent),
        "device": str(next(model.parameters()).device),
        "seed": training.seed, "games": games, "positions_per_game": positions_per_game,
        "map_size": training.map_size, "map_type": training.map_type, "player_tribes": training.player_tribes,
        "max_moves": training.max_moves, "mcts": asdict(mcts), "pool": pool, "policy": policy,
    }
    signature = json.dumps(identity, sort_keys=True)
    path = cache_directory / f"terminal-{sha256(signature.encode()).hexdigest()[:20]}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata["signature"] != signature:
                raise ValueError("terminal cache signature differs from the requested collection")
            summary = metadata["summary"]
            if set(summary["attempted_map_seeds"]) & set(excluded_map_seeds):
                raise ValueError("cached calibration maps overlap currently reserved maps; use a new calibration cache")
            rows = zip(data["features"], data["turns"], data["targets"], data["seeds"], data["players"], data["phases"], strict=True)
            examples = [
                EncodedValuePosition(feature.copy(), int(turn), float(target), (int(seed), int(player)), VALUE_PHASES[int(phase)])
                for feature, turn, target, seed, player, phase in rows
            ]
        if not examples or any(not np.isfinite(row.features).all() or row.return_target not in (-1, 0, 1) for row in examples):
            raise ValueError("terminal cache contains invalid features or outcomes")
        return examples, {**summary, "cache_path": str(path), "cache_loaded": True}

    if require_existing:
        raise FileNotFoundError(f"required calibration cache is missing: {path}; keep the original checkpoint and collection settings")

    examples, summary = collect_mcts_terminal_value_games(
        model, opponent, training, mcts, games, positions_per_game=positions_per_game,
        excluded_map_seeds=excluded_map_seeds, label="value calibration",
        model_a_uses_mcts=policy == "mcts-zero", **pool,
    )
    if summary["valid"] and examples:
        cache_directory.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(dir=cache_directory, suffix=".npz.tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream, metadata=np.array(json.dumps({"signature": signature, "summary": summary})),
                    features=np.stack([row.features for row in examples]),
                    turns=np.array([row.turn for row in examples], dtype=np.int32),
                    targets=np.array([row.return_target for row in examples], dtype=np.float32),
                    seeds=np.array([row.game_key[0] for row in examples], dtype=np.int64),
                    players=np.array([row.game_key[1] for row in examples], dtype=np.int32),
                    phases=np.array([VALUE_PHASES.index(row.phase) for row in examples], dtype=np.int8),
                )
            os.replace(filename, path)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    return examples, {**summary, "cache_path": str(path), "cache_loaded": False}


def calibration_admission(rank, calibration, reference_rank, reference_calibration, *, rmse_regression, bias_regression):
    failures = []
    accuracy, baseline = rank["preference_accuracy"], reference_rank["preference_accuracy"]
    numeric = (accuracy, baseline, rank["gap_rmse"], reference_rank["gap_rmse"],
               calibration["rmse"], calibration["zero_rmse"], calibration["bias"],
               reference_calibration["rmse"], reference_calibration["bias"])
    if (not all(row["valid"] for row in (rank, calibration, reference_rank, reference_calibration))
            or any(value is None or not np.isfinite(value) for value in numeric)):
        return ["nonfinite_metrics"]
    if accuracy is None or baseline is None or not (
        accuracy > baseline or accuracy == baseline == 1.0
    ):
        failures.append("ranking_accuracy_not_improved")
    if rank["gap_rmse"] >= reference_rank["gap_rmse"] - 1e-8:
        failures.append("ranking_gap_not_improved")
    if calibration["rmse"] > reference_calibration["rmse"] + rmse_regression:
        failures.append("calibration_rmse_regression")
    if calibration["rmse"] > calibration["zero_rmse"]:
        failures.append("calibration_worse_than_zero")
    if abs(calibration["bias"]) > abs(reference_calibration["bias"]) + bias_regression:
        failures.append("calibration_bias_regression")
    return failures


def paired_training_losses(predictions, targets, signs, gap_weights, value_weights):
    """Keep confidence outside the denominator so uncertain targets exert less force."""
    gap_error = (_value_gap(predictions, signs) - _value_gap(targets, signs)) / 2
    return (torch.mean(gap_weights * gap_error.square()),
            torch.mean(value_weights[:, None] * (predictions - targets).square()))


def fit_calibrated_heads(model, ranking, calibration, training, settings, *, weights,
                         rmse_regression, bias_regression, run, log_interval,
                         pair_value_weight=0.0, balance_pair_maps=False, update_callback=None):
    device = next(model.parameters()).device
    rank_states, rank_targets, rank_signs, ordered = _paired_value_tensors(ranking["train"], device)
    held_states, held_targets, held_signs, _ = _paired_value_tensors(ranking["selection"], device)
    point_states = torch.as_tensor(np.stack([row.features for row in calibration["train"]]), device=device)
    point_targets = torch.tensor([row.return_target for row in calibration["train"]], device=device)
    held_point_states = torch.as_tensor(np.stack([row.features for row in calibration["selection"]]), device=device)
    held_point_targets = np.array([row.return_target for row in calibration["selection"]], dtype=np.float32)
    phase_rows = [[i for i, row in enumerate(calibration["train"]) if row.phase == phase] for phase in VALUE_PHASES]
    phase_rows = [rows for rows in phase_rows if rows]
    rng = np.random.default_rng(training.seed ^ 0x43414C49)
    batches = [
        torch.tensor([int(rng.choice(phase_rows[int(rng.integers(len(phase_rows)))]))
                      for _ in range(settings.value_batch_size)], device=device)
        for _ in range(settings.value_updates)
    ]
    original = deepcopy(model.value_head.state_dict())
    map_counts = Counter(pair[0].game_key[0] for pair in ordered)
    balance = [len(ordered) / (len(map_counts) * map_counts[pair[0].game_key[0]])
               if balance_pair_maps else 1.0 for pair in ordered]
    gap_weights = torch.tensor([getattr(pair[0], "gap_weight", 1.0) * factor for pair, factor in zip(ordered, balance)], device=device)
    value_weights = torch.tensor([getattr(pair[0], "supervision_weight", 1.0) * factor
                                  for pair, factor in zip(ordered, balance)], device=device)
    best_state = best_key = None
    selected = None
    reports = {}
    model.eval()

    def measure():
        with torch.no_grad():
            rank = _paired_metrics(model.value_head(held_states).squeeze(-1).float(), held_targets,
                                   held_signs, ranking["selection"], 1.0)
            values = model.value_head(held_point_states).reshape(-1).float().cpu().numpy()
            calibrated = _metrics_by_phase(values, held_point_targets, calibration["selection"])
            train_predictions = model.value_head(rank_states).squeeze(-1).float()
            train_rank = _paired_metrics(train_predictions, rank_targets, rank_signs, ranking["train"], 1.0)
            train_gap, train_pair = paired_training_losses(train_predictions, rank_targets, rank_signs,
                                                           gap_weights, value_weights)
            point_mse = F.mse_loss(model.value_head(point_states).reshape(-1).float(), point_targets).item()
        return {"rank": rank, "calibration": calibrated, "train_rank": train_rank, "train_calibration_mse": point_mse,
                "train_weighted_gap_loss": train_gap.item(), "train_weighted_pair_mse": train_pair.item()}

    try:
        with isolated_rng():
            reference = measure()
            for weight in weights:
                model.value_head.load_state_dict(original)
                child = run.child(f"calibration-{weight!r}")
                optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=settings.value_learning_rate, weight_decay=0.0)
                rejections = Counter()
                accepted = 0
                last = reference
                for step in range(settings.value_updates + 1):
                    if step:
                        rows = batches[step - 1]
                        predictions = model.value_head(rank_states).squeeze(-1).float()
                        gap_loss, pair_loss = paired_training_losses(predictions, rank_targets, rank_signs,
                                                                     gap_weights, value_weights)
                        point_loss = F.mse_loss(model.value_head(point_states[rows]).reshape(-1).float(), point_targets[rows])
                        loss = gap_loss + pair_value_weight * pair_loss + weight * point_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
                        if not torch.isfinite(norm):
                            rejections["nonfinite_gradient"] += 1
                            break
                        optimizer.step()
                        last = measure()
                    if update_callback is not None:
                        update_callback(weight, step, last, model.value_head.state_dict())
                    failures = calibration_admission(
                        last["rank"], last["calibration"], reference["rank"], reference["calibration"],
                        rmse_regression=rmse_regression, bias_regression=bias_regression,
                    )
                    if step:
                        rejections.update(failures)
                    if step and not failures:
                        accepted += 1
                        key = (-last["rank"]["preference_accuracy"], last["rank"]["gap_rmse"],
                               last["calibration"]["rmse"], abs(last["calibration"]["bias"]))
                        if best_key is None or key < best_key:
                            best_key, best_state = key, deepcopy(model.value_head.state_dict())
                            selected = {"calibration_weight": weight, "update": step, **last}
                    child.log(
                        update=step, calibration_weight=weight, admitted=not failures,
                        train_gap_rmse=last["train_rank"]["gap_rmse"],
                        train_preference_accuracy=last["train_rank"]["preference_accuracy"],
                        train_calibration_mse=last["train_calibration_mse"],
                        train_weighted_gap_loss=last["train_weighted_gap_loss"],
                        train_weighted_pair_mse=last["train_weighted_pair_mse"], pair_value_loss_weight=pair_value_weight,
                        selection_gap_rmse=last["rank"]["gap_rmse"],
                        selection_preference_accuracy=last["rank"]["preference_accuracy"],
                        selection_value_rmse=last["calibration"]["rmse"], selection_value_bias=last["calibration"]["bias"],
                    )
                    if step == 0 or step % log_interval == 0 or step == settings.value_updates:
                        print(f"calibrated value | weight={weight:g} | update={step}/{settings.value_updates} | "
                              f"rank={last['rank']['preference_accuracy']} | "
                              f"gap_rmse={last['rank']['gap_rmse']:.4f} | "
                              f"value_rmse={last['calibration']['rmse']:.4f} | "
                              f"bias={last['calibration']['bias']:+.4f} | "
                              f"admitted={int(not failures)} | reasons={','.join(failures) or 'passed'}", flush=True)
                reports[str(weight)] = {"admitted_updates": accepted, "rejections": dict(rejections), "last": last}
                child.finish(reports[str(weight)], plots=False)
    finally:
        model.value_head.load_state_dict(original)
    return best_state, {"reference": reference, "selected": selected, "variants": reports}
