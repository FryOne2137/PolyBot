"""Audit saved successor roles and value gradients using the MCTS model router."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .calibrated_value import calibration_batch_indices, paired_training_losses, paired_training_weights
from .flat_value import VALUE_PHASES, _paired_value_tensors, paired_gap_breakdown
from .mcts_value import _encode_value_rows
from .saved_value_pairs import load_successor_packet
from .value_gradient_diagnostics import _geometry, _statistics


def load_routing_inputs(model, opponent, training, ranking, sources, *, history, require_agent_id=False):
    """Read saved public packets, never a GameEnv, and encode opponent states with its own network."""
    wanted = {}
    for rows in ranking.values():
        for row in rows:
            wanted.setdefault(row.game_key, []).append(row)
    lookup = {}
    for source in sources:
        directory = Path(source["directory"]).resolve()
        config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
        labels = (json.loads(line) for line in (directory / config.get("labels_file", "labelled_pairs.jsonl"))
                  .read_text(encoding="utf-8-sig").splitlines() if line.strip())
        offset = source["global_pair_id_offset"]
        for label in labels:
            key = (int(label["map_seed"]), int(label["player"]), int(label["pair_id"]) + offset)
            if key not in wanted:
                continue
            if key in lookup or len(wanted[key]) != 2:
                raise ValueError("routing audit requires unique two-branch pair identities")
            branches = []
            for index, arm in enumerate(("zero", "soft_z")):
                branch, example = label["branches"][arm], wanted[key][index]
                actor = int(branch["to_play"])
                path = directory / branch["successor_file"]
                if path.parent != directory or path.suffix != ".npz":
                    raise ValueError("invalid saved successor path")
                packet, stored, agent = load_successor_packet(training, path, actor=actor, history=history)
                if stored.shape != example.features.shape or not np.allclose(stored, example.features, atol=2e-4, rtol=2e-4):
                    raise ValueError("successor identity or frozen apprentice features differ from the pinned dataset")
                inferred = int(actor != key[1])
                has_agent = agent is not None
                if agent is None:
                    if require_agent_id:
                        raise ValueError(f"successor has no saved agent_id: {path}")
                    agent = inferred
                if agent not in (0, 1):
                    raise ValueError("two-model routing requires saved agent_id zero or one")
                opponent_value = None
                feature_distance = None
                if agent == 1:
                    features = _encode_value_rows(opponent, training, next(opponent.parameters()).device,
                                                 packet, np.array([0], dtype=np.int32))[0]
                    with torch.no_grad():
                        state = torch.as_tensor(features.copy(), device=next(opponent.parameters()).device)
                        opponent_value = float(opponent.value_head(state.unsqueeze(0)).reshape(()))
                    if features.shape == example.features.shape:
                        feature_distance = float(np.linalg.norm(features - example.features))
                if opponent_value is not None and not np.isfinite(opponent_value):
                    raise ValueError("nonfinite frozen opponent value")
                branches.append({"arm": arm, "to_play": actor, "agent_id": agent,
                    "agent_source": "saved" if has_agent else "inferred_from_root_player",
                    "root_role_disagrees_with_agent": agent != inferred, "turn_changed": actor != key[1],
                    "opponent_value": opponent_value, "opponent_feature_l2_difference": feature_distance,
                    "successor_file": str(path), "successor_sha256": sha256(path.read_bytes()).hexdigest()})
            lookup[key] = branches
    if set(lookup) != set(wanted):
        raise ValueError("routing audit could not map every pinned pair to its original successor files")
    return lookup


def routing_counts(ordered, lookup):
    branches = [lookup[pair[0].game_key] for pair in ordered]
    agents = np.array([[row["agent_id"] for row in pair] for pair in branches])
    changed = np.array([[row["turn_changed"] for row in pair] for pair in branches])
    return {
        "pairs": len(ordered), "branches": 2 * len(ordered), "turn_changed_branches": int(changed.sum()),
        "neither_branch_changes_player": int((changed.sum(axis=1) == 0).sum()),
        "one_branch_changes_player": int((changed.sum(axis=1) == 1).sum()),
        "both_branches_change_player": int((changed.sum(axis=1) == 2).sum()),
        "both_apprentice": int((agents.sum(axis=1) == 0).sum()),
        "mixed_models": int((agents.sum(axis=1) == 1).sum()),
        "both_opponent": int((agents.sum(axis=1) == 2).sum()),
        "saved_agent_branches": sum(row["agent_source"] == "saved" for pair in branches for row in pair),
        "inferred_agent_branches": sum(row["agent_source"] != "saved" for pair in branches for row in pair),
        "role_metadata_disagreements": sum(row["root_role_disagrees_with_agent"] for pair in branches for row in pair),
    }


def audit_routing(model, ranking, calibration, lookup, *, seed, batch_size, batches, pair_weight, cache_weight,
                  balance_pair_maps, apprentice_scale=1.0, opponent_scale=1.0):
    """Compare legacy and routed objectives on exactly the same groups and cache draws."""
    if batches < 1 or not 0 <= apprentice_scale <= 1 or not 0 <= opponent_scale <= 1:
        raise ValueError("invalid routing audit batch count or value scales")
    parameters = tuple(parameter for parameter in model.value_head.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("routing audit needs differentiable apprentice value-head parameters")
    device = parameters[0].device
    prepared, report, rows = {}, {}, []
    for split, examples in ranking.items():
        states, targets, signs, ordered = _paired_value_tensors(examples, device)
        metadata = [lookup[pair[0].game_key] for pair in ordered]
        own = torch.tensor([[branch["agent_id"] == 0 for branch in pair] for pair in metadata], device=device)
        frozen = targets.new_tensor([[branch["opponent_value"] if branch["opponent_value"] is not None else 0.0
                                      for branch in pair] for pair in metadata])
        with torch.set_grad_enabled(split == "train"):
            raw = model.value_head(states).squeeze(-1).float()
            routed = torch.where(own, apprentice_scale * raw, opponent_scale * frozen)
            zero = torch.where(own, torch.zeros_like(raw), opponent_scale * frozen)
        groups = {"all": torch.ones(len(ordered), dtype=torch.bool, device=device),
                  "both_apprentice": own.sum(dim=1) == 2, "mixed_models": own.sum(dim=1) == 1,
                  "both_opponent": own.sum(dim=1) == 0}
        report[split] = {"counts": routing_counts(ordered, lookup), "groups": {}}
        for name, mask in groups.items():
            report[split]["groups"][name] = {
                "pairs": int(mask.sum()),
                "legacy": paired_gap_breakdown(raw[mask], targets[mask], signs[mask]),
                "routed": paired_gap_breakdown(routed[mask], targets[mask], signs[mask]),
                "routed_zero_apprentice": paired_gap_breakdown(zero[mask], targets[mask], signs[mask]),
            }
        raw_root, routed_root, target_root = [value.detach().cpu().numpy() * signs.cpu().numpy()
                                             for value in (raw, routed, targets)]
        for index, pair in enumerate(ordered):
            gap = lambda values: float(values[index, 1] - values[index, 0])
            rows.append({"split": split, "map_seed": pair[0].game_key[0], "root_player": pair[0].game_key[1],
                "pair_id": pair[0].game_key[-1], "phase": pair[0].phase, "branches": metadata[index],
                "legacy_root_values": raw_root[index].tolist(), "routed_root_values": routed_root[index].tolist(),
                "root_targets": target_root[index].tolist(), "legacy_gap": gap(raw_root),
                "routed_gap": gap(routed_root), "target_gap": gap(target_root),
                "preference_changed_by_routing": bool(np.sign(gap(raw_root)) != np.sign(gap(routed_root))),
                "has_apprentice_gradient_route": bool(own[index].any()) and apprentice_scale > 0})
        prepared[split] = (raw, routed, targets, signs, ordered, groups, own)
    raw, routed, targets, signs, ordered, groups, own = prepared["train"]
    gap_weights, value_weights = paired_training_weights(ordered, device, balance_pair_maps=balance_pair_maps)
    points = torch.as_tensor(np.stack([row.features for row in calibration["train"]]), device=device)
    point_targets = torch.tensor([row.return_target for row in calibration["train"]], device=device, dtype=torch.float32)
    phases = [row.phase for row in calibration["train"]]
    present = [phase for phase in VALUE_PHASES if phase in phases]
    phase_weights = point_targets.new_tensor([1 / (len(present) * phases.count(phase)) for phase in phases])
    errors = (model.value_head(points).reshape(-1).float() - point_targets).square()

    def vector(loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
        result = torch.cat([gradient.reshape(-1).detach().cpu().double() for gradient in gradients])
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite routing audit gradient")
        return result

    expected_cache = vector((phase_weights * errors).sum())
    draws = calibration_batch_indices(calibration["train"], seed=seed, batch_size=batch_size, updates=batches, device=device)
    sampled_cache = [vector((model.value_head(points[indexes]).reshape(-1).float() - point_targets[indexes]).square().mean())
                     for indexes in draws]
    gradients = {}
    for name, mask in groups.items():
        gradients[name] = {}
        gap_vectors = {}
        for variant, predictions in (("legacy", raw), ("routed", routed)):
            gap, branch = paired_training_losses(predictions, targets, signs,
                                                 gap_weights * mask, value_weights * mask)
            g, b = vector(gap), vector(branch)
            gap_vectors[variant] = g
            expected = _geometry(g, b, expected_cache, pair_weight=pair_weight, cache_weight=cache_weight)
            batches_geometry = [_geometry(g, b, cache, pair_weight=pair_weight, cache_weight=cache_weight)
                                for cache in sampled_cache]
            cosines = [row["gap_cache_cosine"] for row in batches_geometry if row["gap_cache_cosine"] is not None]
            has_route = variant == "legacy" or (bool(own[mask].any()) and apprentice_scale > 0)
            gradients[name][variant] = {
                "pairs": int(mask.sum()), "status": "empty" if not mask.any() else
                "no_apprentice_route" if not has_route else
                "zero_aggregate_gradient" if expected["gap_norm"] == 0 else "measured",
                "weighted_gap_loss_contribution": float(gap.detach()),
                "expected_phase_balanced": expected,
                "sampled_cache_conflict_fraction": sum(value < 0 for value in cosines) / len(cosines) if cosines else None,
                "sampled_gap_cache_cosine": _statistics(cosines),
                "sampled_cache_to_gap_norm_ratio": _statistics([row["cache_to_gap_norm_ratio"] for row in batches_geometry]),
            }
        gradients[name]["contribution_scope"] = "original full training-pair denominator and map weights, without group renormalization"
        old, new = gap_vectors["legacy"], gap_vectors["routed"]
        denominator = float(old.norm() * new.norm())
        gradients[name]["gap_gradient_change_norm"] = float((new - old).norm())
        gradients[name]["gap_gradient_cosine_legacy_vs_routed"] = float(torch.dot(old, new)) / denominator if denominator else None
    return {"splits": report, "training_gradients": gradients,
            "gradient_scope": "training pairs and training cache only; selection supplies prediction metrics only",
            "scope": "neural values at saved real successors, not complete MCTS action evaluations or simulated-leaf labels"}, rows
