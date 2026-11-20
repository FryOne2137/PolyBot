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

    def __post_init__(self):
        if (self.backoffs < 0 or self.rejection_patience < 1 or not 0 < self.backoff_factor < 1
                or not self.minimum_learning_rate > 0 or self.relative_tolerance < 0
                or not all(math.isfinite(value) for value in
                           (self.backoff_factor, self.minimum_learning_rate, self.relative_tolerance))):
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


def guarded_adamw_step(head, optimizer, gradients, gap_loss, settings):
    """Retry the same gradient from identical parameters and optimizer moments.

    Acceptance reads only the complete weighted training-pair loss. Every
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
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    accepted, attempts = False, []

    def restore():
        head.load_state_dict(initial)
        optimizer.load_state_dict(deepcopy(optimizer_initial))
        for parameter, gradient in zip(parameters, old_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.clone()

    try:
        with torch.no_grad():
            before = float(gap_loss())
        if not math.isfinite(before):
            raise ValueError("nonfinite training gap before guarded update")
        tolerance = 1e-10 + settings.relative_tolerance * abs(before)
        previous_lrs = None
        for backoff in range(settings.backoffs + 1):
            lrs = [max(settings.minimum_learning_rate, lr * settings.backoff_factor**backoff) for lr in original_lrs]
            if lrs == previous_lrs:
                break
            previous_lrs = lrs
            restore()
            for group, lr in zip(optimizer.param_groups, lrs, strict=True):
                group["lr"] = lr
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient.detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            finite = bool(torch.isfinite(norm))
            after = None
            if finite:
                optimizer.step()
                finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
                finite = finite and all(bool(torch.isfinite(value).all()) for state in optimizer.state.values()
                                        for value in state.values() if isinstance(value, torch.Tensor))
                if finite:
                    with torch.no_grad():
                        after = float(gap_loss())
                    finite = math.isfinite(after)
            accepted = finite and after <= before + tolerance
            attempts.append({"attempt": backoff + 1, "learning_rates": lrs,
                             "gap_before": before, "gap_after": after if finite else None,
                             "accepted": accepted, "reason": "accepted" if accepted else
                             "training_gap_regression" if finite else "nonfinite_update"})
            if accepted:
                return {"applied": True, "attempts": attempts, "rejected_trials": len(attempts) - 1,
                        "gap_before": before, "gap_after": after, "gap_tolerance": tolerance,
                        "learning_rate": lrs[0], "gradient_norm": float(norm), "reason": "accepted"}
        return {"applied": False, "attempts": attempts, "rejected_trials": len(attempts),
                "gap_before": before, "gap_after": before, "gap_tolerance": tolerance,
                "learning_rate": original_lrs[0], "gradient_norm": None, "reason": "guard_rejected"}
    finally:
        if not accepted:
            restore()
