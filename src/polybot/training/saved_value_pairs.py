"""Load real successor labels and keep whole maps in one value split."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np

from ..data.polyenv_features import VISIBLE_ACTION_HISTORY_FEATURES
from .flat_value import EncodedValuePosition, VALUE_PHASES, paired_outcome_weights
from .mcts_value import _encode_value_rows


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_successor_packet(training, path, *, actor, history):
    """Read the same public successor packet used by training, plus optional routing metadata."""
    with np.load(path, allow_pickle=False) as archive:
        packet = {name: archive[name] for name in (
            "map_tokens", "state", "own_techs", "own_tribe", "initial_player_count",
            "player_scores", "player_scores_mask", "visible_action_history",
            "visible_action_history_mask", "action_id", "action_features", "to_play",
        )}
        saved_features = archive["features"].copy()
        agent = int(archive["agent_id"].item()) if "agent_id" in archive else None
    if (packet["map_tokens"].shape != (training.map_size ** 2, 24)
            or packet["state"].ndim != 1 or len(packet["state"]) < 4
            or int(packet["state"][1]) != actor or int(packet["to_play"]) != actor
            or int(packet["initial_player_count"]) != 2):
        raise ValueError(f"saved successor has incompatible map or player metadata in {path.name}")
    if (packet["visible_action_history"].shape != (history, VISIBLE_ACTION_HISTORY_FEATURES)
            or packet["visible_action_history_mask"].shape != (history,)):
        raise ValueError(f"saved action history differs from --visible-action-history in {path.name}")
    count = len(packet["action_id"])
    if count == 0 or packet["action_features"].shape != (count, 17):
        raise ValueError(f"saved successor has no complete legal-action packet in {path.name}")
    batch = {name: np.expand_dims(value, 0) for name, value in packet.items()}
    batch["action_mask"] = np.ones((1, count), dtype=np.uint8)
    return batch, saved_features, agent


def _successor(model, training, path, *, actor, history):
    batch, saved_features, _ = load_successor_packet(training, path, actor=actor, history=history)
    encoded = _encode_value_rows(
        model, training, next(model.parameters()).device, batch, np.array([0], dtype=np.int32),
    )[0]
    if (not np.isfinite(encoded).all() or saved_features.shape != encoded.shape
            or not np.allclose(saved_features, encoded, atol=2e-4, rtol=2e-4)):
        raise ValueError(f"saved features differ from the input checkpoint's frozen encoder in {path.name}")
    return encoded, int(batch["state"][0, 0])


def load_saved_value_pairs(model, training, directory: Path, *, history: int):
    """Validate labels against paired terminal rollouts before using gradients."""
    directory = directory.resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    source = config["training"]
    if (config.get("schema_version") != 1 or source["map_size"] != training.map_size
            or source["map_type"] != training.map_type
            or tuple(source["player_tribes"]) != tuple(training.player_tribes)
            or config["visible_action_history"] != history):
        raise ValueError("saved pair configuration differs from the current map, tribes or history")
    weighted = config.get("label_mode") == "weighted_soft"
    labels_file = config.get("labels_file", "labelled_pairs.jsonl")
    if labels_file not in ("labelled_pairs.jsonl", "weighted_pairs.jsonl") or weighted != (labels_file == "weighted_pairs.jsonl"):
        raise ValueError("invalid saved pair label file or weighting mode")
    paths = ("config.json", "pairs.jsonl", labels_file, "rollouts.jsonl")
    metadata = {}
    for row in _records(directory / "pairs.jsonl"):
        pair_id = int(row["pair_id"])
        if pair_id in metadata:
            raise ValueError(f"duplicate saved pair ID {pair_id}")
        metadata[pair_id] = row

    rollouts = defaultdict(list)
    seen_rollouts = set()
    for row in _records(directory / "rollouts.jsonl"):
        key = (int(row["pair_id"]), int(row["repeat"]))
        if key in seen_rollouts:
            raise ValueError(f"duplicate paired rollout {key}")
        seen_rollouts.add(key)
        outcomes = row.get("outcomes", {})
        if not outcomes or any(
            outcomes[arm].get("score_a") is None or outcomes[arm].get("truncated", False)
            for arm in ("zero", "soft_z")
        ):
            continue
        values = [float(outcomes[arm]["score_a"]) for arm in ("zero", "soft_z")]
        if not np.isfinite(values).all() or any(value not in (0.0, 0.5, 1.0) for value in values):
            raise ValueError(f"invalid terminal result in paired rollout {key}")
        rollouts[key[0]].append(values)

    model.eval()
    examples = []
    manifest = []
    seen_pairs = set()
    for label in _records(directory / labels_file):
        pair_id = int(label["pair_id"])
        if pair_id in seen_pairs or pair_id not in metadata:
            raise ValueError(f"duplicate or unknown labelled pair ID {pair_id}")
        seen_pairs.add(pair_id)
        root = metadata[pair_id]
        seed, player = int(root["map_seed"]), int(root["player"])
        if seed != label["map_seed"] or player != label["player"] or player not in (0, 1):
            raise ValueError(f"root identity differs from the label in pair {pair_id}")
        phase = root["phase"]
        if phase not in VALUE_PHASES or not rollouts[pair_id]:
            raise ValueError(f"missing phase or paired terminal outcomes in pair {pair_id}")
        scores = np.mean(rollouts[pair_id], axis=0)
        if len(rollouts[pair_id]) != label["completed_paired_repeats"]:
            raise ValueError(f"terminal rollout count differs from the label in pair {pair_id}")
        reliability = {"supervision_weight": 1.0, "gap_weight": 1.0}
        if weighted:
            if config.get("pair_weighting_version") != 1:
                raise ValueError("unsupported saved pair weighting version")
            reliability = paired_outcome_weights(rollouts[pair_id])
            if any(not np.isclose(label.get(key, np.nan), value, atol=1e-8, rtol=1e-6)
                   for key, value in reliability.items()):
                raise ValueError(f"saved uncertainty differs from paired rollouts in pair {pair_id}")
        targets = []
        for index, arm in enumerate(("zero", "soft_z")):
            branch = label["branches"][arm]
            actor = int(branch["to_play"])
            sign = 1 if actor == player else -1
            target = float(branch["return_target"])
            if (actor not in (0, 1) or branch["to_root_sign"] != sign
                    or branch["action_id"] != root["choices"][arm]["action_id"]
                    or not np.isfinite(target)
                    or not np.isclose(target, sign * (2.0 * scores[index] - 1.0), atol=1e-7, rtol=0)):
                raise ValueError(f"outcome, action or perspective differs in pair {pair_id}, branch {arm}")
            path = directory / branch["successor_file"]
            if path.parent != directory or path.suffix != ".npz":
                raise ValueError(f"invalid saved successor path in pair {pair_id}")
            features, turn = _successor(model, training, path, actor=actor, history=history)
            examples.append(EncodedValuePosition(features, turn, target, (seed, player, pair_id), phase, sign,
                                                 reliability["supervision_weight"], reliability["gap_weight"]))
            targets.append(target * sign)
        manifest.append({
            "pair_id": pair_id, "map_seed": seed, "player": player, "phase": phase,
            "completed_paired_repeats": len(rollouts[pair_id]), "root_targets": targets,
            "root_gap": targets[1] - targets[0],
            **reliability,
        })
    if not examples:
        raise ValueError("the saved dataset contains no labelled successor pairs")
    reserved = set(range(int(source["seed"]), int(source["seed"]) + int(config["source_games"]) // 2))
    reserved.update(int(seed) for seed in config.get("reserved_map_seeds", ()))
    reserved.update(int(row["map_seed"]) for row in metadata.values())
    return examples, {
        "directory": str(directory), "source_config": config, "pairs": manifest,
        "reserved_map_seeds": sorted(reserved), "labelled_pairs": len(manifest),
        "source_sha256": {name: sha256((directory / name).read_bytes()).hexdigest() for name in paths},
        "feature_source": "current_frozen_encoder_verified_against_saved_features",
    }


def split_value_pairs(examples, *, seed: int, selection_fraction: float):
    """Assign each map, both seats and all its pairs to a single split."""
    pairs = defaultdict(list)
    for row in examples:
        pairs[row.game_key].append(row)
    if any(len(pair) != 2 or pair[0].phase != pair[1].phase for pair in pairs.values()):
        raise ValueError("saved value split requires two branches in the same phase")
    maps = sorted({int(example.game_key[0]) for example in examples})
    if len(maps) < 2 or not 0 < selection_fraction < 1:
        raise ValueError("saved pair fitting needs at least two maps and a selection fraction in (0, 1)")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(maps)
    selection_count = min(len(maps) - 1, max(1, math.ceil(len(maps) * selection_fraction)))
    counts = np.zeros((len(maps), len(VALUE_PHASES) + 2), dtype=np.int32)
    indexes = {int(value): index for index, value in enumerate(shuffled)}
    for pair in pairs.values():
        index = indexes[int(pair[0].game_key[0])]
        counts[index, VALUE_PHASES.index(pair[0].phase)] += 1
        gap = pair[1].return_target * pair[1].to_root_sign - pair[0].return_target * pair[0].to_root_sign
        counts[index, -2] += int(abs(gap) > 1e-8)
        counts[index, -1] += 1
    total = counts.sum(axis=0)
    active = total > 0
    candidates = (
        combinations(range(len(maps)), selection_count)
        if math.comb(len(maps), selection_count) <= 4096
        else (rng.choice(len(maps), size=selection_count, replace=False) for _ in range(4096))
    )
    best_score, best_rows = None, None
    for rows in candidates:
        selected = counts[list(rows)].sum(axis=0)
        remaining = total - selected
        # Coverage uses phase and informative-pair counts. Model predictions
        # never enter the split decision.
        missing = np.count_nonzero(active[:-1] & ((selected[:-1] == 0) | (remaining[:-1] == 0)))
        deviation = float(np.abs(selected[active] / total[active] - selection_count / len(maps)).sum())
        score = (int(missing), deviation)
        if best_score is None or score < best_score:
            best_score, best_rows = score, list(rows)
    selection_maps = set(shuffled[best_rows].tolist())
    datasets = {
        "train": [row for row in examples if row.game_key[0] not in selection_maps],
        "selection": [row for row in examples if row.game_key[0] in selection_maps],
    }
    report = {
        "seed": seed, "unit": "whole_map_seed", "selection_fraction": selection_fraction,
        "method": "phase_and_informative_pair_coverage", "missing_categories": best_score[0],
    }
    for name, rows in datasets.items():
        pairs = defaultdict(list)
        for row in rows:
            pairs[row.game_key].append(row)
        if any(len(pair) != 2 for pair in pairs.values()):
            raise ValueError("saved value split lost a branch of a paired position")
        phases = {phase: sum(pair[0].phase == phase for pair in pairs.values()) for phase in VALUE_PHASES}
        informative = sum(abs(
            pair[1].return_target * pair[1].to_root_sign - pair[0].return_target * pair[0].to_root_sign
        ) > 1e-8 for pair in pairs.values())
        report[name] = {
            "map_seeds": sorted({int(row.game_key[0]) for row in rows}), "pairs": len(pairs),
            "pair_ids": [int(key[-1]) for key in pairs], "positions": len(rows),
            "informative_pairs": informative, "phase_pairs": phases,
            "coverage_gaps": [phase for phase, count in phases.items() if count == 0]
            + (["informative_pairs"] if informative == 0 else []),
        }
    return datasets, report
