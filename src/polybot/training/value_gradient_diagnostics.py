"""Read-only comparisons of ranking and terminal-cache gradients on a frozen encoder."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .calibrated_value import calibration_batch_indices, paired_training_losses, paired_training_weights
from .flat_value import VALUE_PHASES, _paired_metrics, _paired_value_tensors


def _statistics(values):
    values = np.array([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def _geometry(gap, branch, cache, *, pair_weight, cache_weight):
    def norm(vector):
        return float(torch.linalg.vector_norm(vector))

    def cosine(a, b):
        denominator = norm(a) * norm(b)
        return float(torch.clamp(torch.dot(a, b) / denominator, -1, 1)) if denominator > 0 else None

    combined = gap + pair_weight * branch + cache_weight * cache
    gap_norm = norm(gap)
    return {
        "gap_norm": gap_norm, "branch_norm": norm(branch), "cache_norm": norm(cache),
        "weighted_branch_norm": pair_weight * norm(branch), "weighted_cache_norm": cache_weight * norm(cache),
        "cache_to_gap_norm_ratio": cache_weight * norm(cache) / gap_norm if gap_norm else None,
        "gap_cache_cosine": cosine(gap, cache), "gap_branch_cosine": cosine(gap, branch),
        "branch_cache_cosine": cosine(branch, cache), "gap_combined_cosine": cosine(gap, combined),
        "combined_norm": norm(combined),
        # Positive projection means that -combined is a local SGD descent
        # direction for the actual weighted training gap, not necessarily AdamW.
        "gap_dot_combined": float(torch.dot(gap, combined)),
        "gap_dot_weighted_cache": float(cache_weight * torch.dot(gap, cache)),
        "combined_is_gap_ascent_direction": bool(torch.dot(gap, combined) < 0),
    }


def inspect_cache_gradients(model, ranking, calibration, *, seed, batch_size, learning_rate,
                            pair_weight, cache_weight, balance_pair_maps, batches=32, probe_batches=8):
    """Measure training gradients and restore every isolated one-step AdamW probe.

    Selection data supply measurements only. Expected cache gradients use
    the exact phase-balanced sampling distribution used by the trainer.
    """
    if (not 1 <= probe_batches <= batches or batch_size < 1 or learning_rate <= 0
            or pair_weight < 0 or cache_weight < 0
            or not np.isfinite((learning_rate, pair_weight, cache_weight)).all()):
        raise ValueError("invalid gradient probe settings")
    domains = [{row.game_key[0] for row in rows}
               for rows in (ranking["train"], ranking["selection"], calibration["train"], calibration["selection"])]
    if any(not domain for domain in domains) or any(a & b for i, a in enumerate(domains) for b in domains[i + 1:]):
        raise ValueError("gradient training and selection maps must be nonempty and disjoint")
    head = model.value_head
    parameters = tuple(head.parameters())
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("gradient diagnostics require trainable value-head parameters")
    device = parameters[0].device
    states, targets, signs, pairs = _paired_value_tensors(ranking["train"], device)
    held_states, held_targets, held_signs, _ = _paired_value_tensors(ranking["selection"], device)
    weights, branch_weights = paired_training_weights(pairs, device, balance_pair_maps=balance_pair_maps)
    point_states = torch.as_tensor(np.stack([row.features for row in calibration["train"]]), device=device)
    point_targets = torch.tensor([row.return_target for row in calibration["train"]], device=device, dtype=torch.float32)
    held_points = torch.as_tensor(np.stack([row.features for row in calibration["selection"]]), device=device)
    held_point_targets = torch.tensor([row.return_target for row in calibration["selection"]], device=device, dtype=torch.float32)
    phases = [row.phase for row in calibration["train"]]
    present = [phase for phase in VALUE_PHASES if phase in phases]
    phase_weights = point_targets.new_tensor([1 / (len(present) * phases.count(phase)) for phase in phases])
    draws = calibration_batch_indices(calibration["train"], seed=seed, batch_size=batch_size,
                                     updates=batches, device=device)
    initial = deepcopy(head.state_dict())
    previous_gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    was_training = head.training
    head.eval()

    def losses(rows=None):
        predictions = head(states).squeeze(-1).float()
        gap, branch = paired_training_losses(predictions, targets, signs, weights, branch_weights)
        unweighted_gap, _ = paired_training_losses(predictions, targets, signs,
                                                   torch.ones_like(weights), torch.ones_like(branch_weights))
        if rows is None:
            errors = (head(point_states).reshape(-1).float() - point_targets).square()
            cache, uniform = (phase_weights * errors).sum(), errors.mean()
        else:
            cache = F.mse_loss(head(point_states[rows]).reshape(-1).float(), point_targets[rows])
            uniform = cache
        return {"gap": gap, "branch": branch, "cache": cache, "unweighted_gap": unweighted_gap,
                "uniform_cache": uniform}

    def vector(loss):
        result = torch.cat([gradient.reshape(-1) for gradient in torch.autograd.grad(loss, parameters, retain_graph=True)])
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite value-head gradient")
        return result.detach().cpu().double()

    @torch.no_grad()
    def measure():
        predictions = head(states).squeeze(-1).float()
        gap, _ = paired_training_losses(predictions, targets, signs, weights, branch_weights)
        train = _paired_metrics(predictions, targets, signs, ranking["train"], 1)
        held = _paired_metrics(head(held_states).squeeze(-1).float(), held_targets, held_signs, ranking["selection"], 1)
        cache_errors = (head(point_states).reshape(-1).float() - point_targets).square()
        held_errors = head(held_points).reshape(-1).float() - held_point_targets
        return {"train_gap_loss": float(gap), "train_gap_rmse": train["gap_rmse"],
                "train_preference_accuracy": train["preference_accuracy"],
                "selection_gap_rmse": held["gap_rmse"], "selection_preference_accuracy": held["preference_accuracy"],
                "cache_train_mse": float(cache_errors.mean()),
                "cache_train_phase_mse": float((phase_weights * cache_errors).sum()),
                "cache_selection_rmse": float(held_errors.square().mean().sqrt()),
                "cache_selection_bias": float(held_errors.mean())}

    gradient_rows, probe_rows = [], []
    try:
        full = losses()
        gradients = {name: vector(loss) for name, loss in full.items()}
        expected = _geometry(gradients["gap"], gradients["branch"], gradients["cache"],
                             pair_weight=pair_weight, cache_weight=cache_weight)
        expected.update(gap_loss=float(full["gap"].detach()), branch_mse=float(full["branch"].detach()),
                        cache_phase_mse=float(full["cache"].detach()), uniform_cache_mse=float(full["uniform_cache"].detach()),
                        unweighted_gap_norm=float(gradients["unweighted_gap"].norm()),
                        uniform_cache_norm=float(gradients["uniform_cache"].norm()))
        del full
        before = measure()
        modes = {"gap_only": (1, 0, 0), "cache_only": (0, 0, cache_weight),
                 "gap_plus_cache": (1, 0, cache_weight), "full": (1, pair_weight, cache_weight)}
        for index, rows in enumerate(draws):
            batch_losses = losses(rows)
            cache = vector(batch_losses["cache"])
            geometry = _geometry(gradients["gap"], gradients["branch"], cache,
                                 pair_weight=pair_weight, cache_weight=cache_weight)
            gradient_rows.append({"batch": index + 1, "cache_indices": rows.cpu().tolist(), **geometry})
            del batch_losses
            if index >= probe_batches:
                continue
            for mode, (gap_factor, branch_factor, cache_factor) in modes.items():
                head.load_state_dict(initial)
                optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
                optimizer.zero_grad(set_to_none=True)
                components = losses(rows)
                objective = gap_factor * components["gap"] + branch_factor * components["branch"] + cache_factor * components["cache"]
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite isolated AdamW probe")
                optimizer.step()
                after = measure()
                delta = {key: after[key] - value for key, value in before.items()
                         if value is not None and after[key] is not None}
                probe_rows.append({"batch": index + 1, "mode": mode, "after": after, "delta": delta,
                                   "gradient_norm_before_clip": float(norm)})
                head.load_state_dict(initial)
                del components, objective, optimizer
        cosines = [row["gap_cache_cosine"] for row in gradient_rows if row["gap_cache_cosine"] is not None]
        probes = {}
        tolerance = 1e-10 + 1e-6 * abs(before["train_gap_loss"])
        for mode in modes:
            records = [row for row in probe_rows if row["mode"] == mode]
            probes[mode] = {
                "count": len(records),
                "delta": {key: _statistics([row["delta"].get(key) for row in records]) for key in before},
                "fraction_train_gap_worsened": sum(row["delta"]["train_gap_loss"] > tolerance for row in records) / len(records),
            }
        return {
            "before": before, "expected_phase_balanced_gradients": expected,
            "minibatches": {"count": batches, "gap_cache_cosine": _statistics(cosines),
                            "cache_to_gap_norm_ratio": _statistics([row["cache_to_gap_norm_ratio"] for row in gradient_rows]),
                            "cache_conflict_fraction": sum(value < 0 for value in cosines) / len(cosines) if cosines else None,
                            "combined_gap_ascent_fraction": sum(row["combined_is_gap_ascent_direction"] for row in gradient_rows) / batches},
            "isolated_adamw_probes": probes,
            "training_pair_weights": {"min": float(weights.min()), "mean": float(weights.mean()), "max": float(weights.max())},
            "pair_maps": {name: sorted({row.game_key[0] for row in examples}) for name, examples in ranking.items()},
            "cache_maps": {name: sorted({row.game_key[0] for row in examples}) for name, examples in calibration.items()},
            "head_parameter_count": sum(parameter.numel() for parameter in parameters),
            "scope": "all gradients use training maps only; selection maps supply measurements only",
            "probe_optimizer": "fresh AdamW, weight_decay=0, clip_norm=1; original optimizer moments are not available",
        }, gradient_rows, probe_rows
    finally:
        head.load_state_dict(initial)
        head.train(was_training)
        for parameter, gradient in zip(parameters, previous_gradients, strict=True):
            parameter.grad = gradient
