"""Fixed opponent evaluations for fitting the value actually consumed by search."""

from __future__ import annotations

import math

import numpy as np
import torch


def pair_routing_tensors(lookup, ordered, device):
    metadata = [lookup[pair[0].game_key] for pair in ordered]
    if any(len(pair) != 2 or any(branch["agent_id"] not in (0, 1) for branch in pair) for pair in metadata):
        raise ValueError("value fitting needs two branches with verified two-model routing")
    own = torch.tensor([[branch["agent_id"] == 0 for branch in pair] for pair in metadata], device=device)
    frozen = torch.tensor([[0.0 if branch["agent_id"] == 0 else branch["opponent_value"]
                            for branch in pair] for pair in metadata], device=device, dtype=torch.float32)
    if not torch.isfinite(frozen).all():
        raise ValueError("opponent value must be finite")
    return own, frozen.detach()


def routed_pair_predictions(head, states, routing):
    raw = head(states).squeeze(-1).float()
    if routing is None:
        return raw
    own, frozen = routing
    if own.shape != raw.shape or frozen.shape != raw.shape:
        raise ValueError("pair routing differs from the fitted successor shapes")
    return torch.where(own, raw, frozen.detach())


def paired_value_admission(benchmark, *, planned_games, minimum_maps=16):
    """Require complete fresh maps and positive paired evidence against both controls."""
    rows = benchmark.get("paired_map_scores", [])
    if (not benchmark.get("valid") or len(rows) * 2 != planned_games
            or len(rows) < minimum_maps or len({row["map_seed"] for row in rows}) != len(rows)):
        return {"accepted": False, "reason": "incomplete_or_insufficient_arena_maps", "maps": len(rows)}
    comparisons = {}
    for control in ("zero", "original"):
        differences = []
        for row in rows:
            scores = row["scores_by_player"]
            if any(name not in scores or len(scores[name]) != 2 or
                   any(value not in (0.0, 0.5, 1.0) for value in scores[name])
                   for name in ("calibrated", control)):
                raise ValueError("value arena needs terminal outcomes for both seats and both controls")
            differences.append(float(np.mean(scores["calibrated"]) - np.mean(scores[control])))
        positive, negative = sum(value > 0 for value in differences), sum(value < 0 for value in differences)
        count = positive + negative
        pvalue = sum(math.comb(count, k) for k in range(positive, count + 1)) / 2**count if count else 1.0
        comparisons[control] = {"mean_score_delta": float(np.mean(differences)), "positive_maps": positive,
            "negative_maps": negative, "tied_maps": len(rows) - count, "one_sided_sign_pvalue": pvalue,
            "alpha": 0.025, "passed": bool(np.mean(differences) > 0 and pvalue <= 0.025)}
    accepted = all(row["passed"] for row in comparisons.values())
    return {"accepted": accepted, "reason": "paired_arena_passed" if accepted else "no_confirmed_gameplay_improvement",
            "maps": len(rows), "unit": "map_mean_over_swapped_seats", "comparisons": comparisons}
