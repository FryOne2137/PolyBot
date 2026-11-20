"""Cache-gradient projection and transactional AdamW updates for value ranking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GapGuardConfig:
    backoffs: int = 6
    backoff_factor: float = 0.5
    minimum_learning_rate: float = 1e-7
    relative_tolerance: float = 1e-6
    rejection_patience: int = 16
    fallback_gap_only: bool = False
    fallback_calibration_relative_regression: float = 0.0

    def __post_init__(self):
        if (self.backoffs < 0 or self.rejection_patience < 1 or not 0 < self.backoff_factor < 1
                or not self.minimum_learning_rate > 0 or self.relative_tolerance < 0
                or self.fallback_calibration_relative_regression < 0
                or not all(math.isfinite(value) for value in
                           (self.backoff_factor, self.minimum_learning_rate, self.relative_tolerance,
                            self.fallback_calibration_relative_regression))):
            raise ValueError("invalid AdamW gap guard configuration")


def projected_value_gradients(gap_loss, branch_loss, cache_loss, parameters, *, pair_weight, cache_weight):
    """Remove only the cache component opposing the weighted training gap."""
    gap = torch.autograd.grad(gap_loss, parameters, retain_graph=True)
    branch = torch.autograd.grad(branch_loss, parameters, retain_graph=True)
    cache = tuple(cache_weight * value for value in torch.autograd.grad(cache_loss, parameters))
    vectors = [torch.cat([value.detach().reshape(-1).cpu().double() for value in values]) for values in (gap, branch, cache)]
    g, b, c = vectors
    if not all(torch.isfinite(vector).all() for vector in vectors):
        raise ValueError("nonfinite value-head gradient")
    dot = torch.dot(g, c)
    squared = torch.dot(g, g)
    projected = bool(dot < 0 and squared > 0)
    adjusted = c - (dot / squared) * g if projected else c
    combined = g + pair_weight * b + adjusted
    gradients, offset = [], 0
    for parameter in parameters:
        count = parameter.numel()
        gradients.append(combined[offset:offset + count].to(device=parameter.device, dtype=parameter.dtype).reshape_as(parameter).detach())
        offset += count
    denominator = float(g.norm() * c.norm())
    return gradients, {
        "cache_projected": projected, "gap_norm": float(g.norm()), "weighted_cache_norm": float(c.norm()),
        "cache_gap_cosine": float(dot) / denominator if denominator else None,
        "cache_gap_dot_before": float(dot), "cache_gap_dot_after": float(torch.dot(g, adjusted)),
        "projected_cache_norm": float(adjusted.norm()), "combined_gap_dot": float(torch.dot(g, combined)),
    }


def guarded_adamw_step(head, optimizer, gradients, gap_loss, settings, *, calibration_loss=None):
    """Retry the same gradient from identical parameters and optimizer moments.

    Acceptance reads the complete weighted training-pair loss. The optional
    fresh-moment fallback also checks full training-cache calibration. Every
    rejected trial restores parameters, buffers, moments, step counters and
    parameter groups. If all trials fail, even the original LR is restored.
    """
    parameters = tuple(head.parameters())
    if len(parameters) != len(gradients):
        raise ValueError("gradient layout differs from the value head")
    if any(group["lr"] < settings.minimum_learning_rate for group in optimizer.param_groups):
        raise ValueError("guard minimum learning rate exceeds the optimizer learning rate")
    initial, optimizer_initial = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())
    old_gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    original_parameters = [parameter.detach().clone() for parameter in parameters]
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    accepted, attempts = False, []

    def restore():
        head.load_state_dict(initial)
        optimizer.load_state_dict(deepcopy(optimizer_initial))
        for parameter, gradient in zip(parameters, old_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.clone()

    try:
        with torch.enable_grad():
            initial_gap = gap_loss()
            before = float(initial_gap.detach())
            gap_gradients = (torch.autograd.grad(initial_gap, parameters) if initial_gap.requires_grad else None)
        gap_vector = (torch.cat([value.detach().reshape(-1).cpu().double() for value in gap_gradients])
                      if gap_gradients is not None else None)
        if gap_vector is not None and not torch.isfinite(gap_vector).all():
            raise ValueError("nonfinite training gap gradient")
        if settings.fallback_gap_only and (gap_gradients is None or calibration_loss is None):
            raise ValueError("gap-only fallback requires a differentiable gap and training calibration loss")
        with torch.no_grad():
            calibration_before = float(calibration_loss()) if calibration_loss is not None else None
        if not math.isfinite(before):
            raise ValueError("nonfinite training gap before guarded update")
        if calibration_before is not None and not math.isfinite(calibration_before):
            raise ValueError("nonfinite training calibration before guarded update")
        tolerance = 1e-10 + settings.relative_tolerance * abs(before)
        calibration_limit = (calibration_before * (1 + settings.fallback_calibration_relative_regression)
                             + 1e-10 + settings.relative_tolerance * abs(calibration_before)
                             if calibration_before is not None else None)
        phases = [("projected", gradients)]
        if settings.fallback_gap_only:
            phases.append(("gap_fallback", gap_gradients))
        fallback_attempted = False
        for phase, phase_gradients in phases:
            fallback = phase == "gap_fallback"
            fallback_attempted = fallback_attempted or fallback
            previous_lrs = None
            for backoff in range(settings.backoffs + 1):
                lrs = [max(settings.minimum_learning_rate, lr * settings.backoff_factor**backoff) for lr in original_lrs]
                if lrs == previous_lrs:
                    break
                previous_lrs = lrs
                restore()
                if fallback:
                    optimizer.state.clear()
                for group, lr in zip(optimizer.param_groups, lrs, strict=True):
                    group["lr"] = lr
                for parameter, gradient in zip(parameters, phase_gradients, strict=True):
                    parameter.grad = gradient.detach().clone()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                finite = bool(torch.isfinite(norm))
                after = calibration_after = None
                direction = {"parameter_delta_norm": None, "gap_dot_parameter_delta": None,
                             "gap_step_cosine": None, "direction_is_gap_ascent": None}
                if finite:
                    optimizer.step()
                    finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
                    finite = finite and all(bool(torch.isfinite(value).all()) for state in optimizer.state.values()
                                            for value in state.values() if isinstance(value, torch.Tensor))
                    if finite:
                        delta = torch.cat([parameter.detach().reshape(-1).cpu().double() - original.reshape(-1).cpu().double()
                                           for parameter, original in zip(parameters, original_parameters, strict=True)])
                        delta_norm = float(delta.norm())
                        direction["parameter_delta_norm"] = delta_norm
                        if gap_vector is not None:
                            dot = float(torch.dot(gap_vector, delta))
                            denominator = float(gap_vector.norm()) * delta_norm
                            direction.update(gap_dot_parameter_delta=dot, direction_is_gap_ascent=dot > 0,
                                gap_step_cosine=max(-1.0, min(1.0, dot / denominator)) if denominator else None)
                        with torch.no_grad():
                            after = float(gap_loss())
                            calibration_after = float(calibration_loss()) if calibration_loss is not None else None
                        finite = math.isfinite(after) and (calibration_after is None or math.isfinite(calibration_after))
                gap_ok = finite and (after < before - tolerance if fallback else after <= before + tolerance)
                calibration_ok = not fallback or (finite and calibration_after <= calibration_limit)
                accepted = gap_ok and calibration_ok
                reasons = (["nonfinite_update"] if not finite else
                           (["fallback_no_gap_improvement" if fallback else "training_gap_regression"] if not gap_ok else [])
                           + (["fallback_calibration_regression"] if not calibration_ok else []))
                attempts.append({"attempt": backoff + 1, "optimizer_mode": phase, "learning_rates": lrs,
                    "gap_before": before, "gap_after": after if finite else None,
                    "calibration_before": calibration_before, "calibration_after": calibration_after if finite else None,
                    "fallback_calibration_limit": calibration_limit if fallback else None,
                    **direction, "accepted": accepted, "reason": "accepted" if accepted else reasons[0],
                    "rejection_reasons": reasons})
                if accepted:
                    return {"applied": True, "attempts": attempts, "rejected_trials": len(attempts) - 1,
                        "gap_before": before, "gap_after": after, "gap_tolerance": tolerance,
                        "calibration_before": calibration_before, "calibration_after": calibration_after,
                        "fallback_attempted": fallback_attempted, "fallback_applied": fallback,
                        "optimizer_moments_reset": fallback, "learning_rate": lrs[0],
                        "gradient_norm": float(norm), "reason": "fallback_accepted" if fallback else "accepted"}
        return {"applied": False, "attempts": attempts, "rejected_trials": len(attempts),
                "gap_before": before, "gap_after": before, "gap_tolerance": tolerance,
                "calibration_before": calibration_before, "calibration_after": calibration_before,
                "fallback_attempted": fallback_attempted, "fallback_applied": False, "optimizer_moments_reset": False,
                "learning_rate": original_lrs[0], "gradient_norm": None, "reason": "guard_rejected"}
    finally:
        if not accepted:
            restore()
