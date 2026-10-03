"""PolyBot-only candidate filtering for native MCTS leaf policies.

This module deliberately does not alter the authoritative legal-action mask.
Belief validation and PolyEnv still see every legal action.  The experiment
only gives actions outside the selected candidate set a numerically negligible
prior before the policy logits are submitted to the native MCTS pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ActionPruningStats:
    """Additive counters produced by one leaf-inference batch."""

    rows: int = 0
    pruned_rows: int = 0
    legal_actions: int = 0
    selected_actions: int = 0
    mandatory_actions: int = 0
    random_actions: int = 0
    prior_mass_retained_sum: float = 0.0


@dataclass(frozen=True)
class RootRestrictionStats:
    """Diagnostics for enforcing the candidate set on played root actions."""

    roots: int = 0
    missing_roots: int = 0
    visits: int = 0
    outside_candidate_visits: int = 0
    unfiltered_choices_outside: int = 0


def mandatory_search_actions(
    action_categorical: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Return tactical actions that must survive policy top-K filtering.

    The categorical schema is the same one used by the policy network:
    column 0 is the broad action type and column 9 is the tile action.  We keep
    attacks, end-turn (so a root is always finishable), bridge construction,
    ruins/exploration, founding a city and capturing a city.
    """
    if action_categorical.ndim != 3 or action_categorical.shape[-1] < 10:
        raise ValueError("action_categorical must have shape [B,A,C>=10]")
    if action_mask.shape != action_categorical.shape[:2]:
        raise ValueError("action_mask must match the first two action dimensions")
    action_type = action_categorical[..., 0]
    tile_action = action_categorical[..., 9]
    strategic_tile_action = (
        (action_type == 8)
        & (
            (tile_action == 9)   # build bridge
            | (tile_action == 10)  # ruin
            | (tile_action == 11)  # found city
            | (tile_action == 12)  # explorer
            | (tile_action == 14)  # capture city/capital
        )
    )
    return action_mask.bool() & (
        (action_type == 1)  # attack
        | (action_type == 3)  # end turn
        | strategic_tile_action
    )


def prune_search_logits(
    logits: torch.Tensor,
    action_categorical: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    top_actions: int,
    random_actions: int = 0,
    min_prior_mass: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ActionPruningStats]:
    """Mask leaf priors to top-K plus mandatory and random legal candidates.

    The returned selection mask is diagnostic only.  ``action_mask`` is never
    modified, because changing it in Python would make belief validation and
    the native state disagree about which actions are legal.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B,A]")
    if action_categorical.shape[:2] != logits.shape:
        raise ValueError("action_categorical must match logits [B,A]")
    if action_mask.shape != logits.shape:
        raise ValueError("action_mask must match logits [B,A]")
    if top_actions < 0 or random_actions < 0:
        raise ValueError("top_actions and random_actions must not be negative")
    if not 0.0 <= min_prior_mass <= 1.0:
        raise ValueError("min_prior_mass must be between zero and one")

    legal = action_mask.bool()
    legal_counts = legal.sum(dim=-1)
    rows = int(logits.shape[0])
    legal_total = int(legal_counts.sum().item())
    if rows == 0:
        empty = ActionPruningStats()
        return logits, legal.clone(), empty

    # Zero is the backwards-compatible off switch.  In this mode the exact
    # input tensor is returned and no stochastic samples are consumed.
    if top_actions == 0:
        stats = ActionPruningStats(
            rows=rows,
            legal_actions=legal_total,
            selected_actions=legal_total,
            mandatory_actions=0,
            prior_mass_retained_sum=float((legal_counts > 0).sum().item()),
        )
        return logits, legal.clone(), stats

    safe_logits = logits.float().masked_fill(~legal, float("-inf"))
    width = logits.shape[1]
    top_count = min(int(top_actions), width)
    selected = torch.zeros_like(legal)
    if top_count:
        top_indices = safe_logits.topk(top_count, dim=-1).indices
        selected.scatter_(1, top_indices, True)
        selected &= legal

    mandatory = mandatory_search_actions(action_categorical, legal)
    selected |= mandatory

    # Adaptive top-K: preserve the fast candidate filter for confident
    # policies but automatically expose more actions when the prior is flat.
    # This happens before random exploration so the retained-mass guarantee is
    # deterministic and directly attributable to the neural distribution.
    if min_prior_mass > 0.0:
        probabilities = torch.softmax(safe_logits, dim=-1)
        ordered = safe_logits.argsort(dim=-1, descending=True)
        ordered_probabilities = probabilities.gather(1, ordered)
        cumulative = ordered_probabilities.cumsum(dim=-1)
        required = (cumulative < min_prior_mass).sum(dim=-1).add_(1)
        required = torch.minimum(required, legal_counts)
        ranks = torch.arange(width, device=logits.device).unsqueeze(0)
        adaptive = ranks < required.unsqueeze(1)
        # Build a separate mask and merge it.  Masking rejected ranks to zero
        # and scattering the boolean ``adaptive`` mask directly into
        # ``selected`` used to write ``False`` repeatedly to action zero.
        # Consequently a retained/mandatory action at index zero could be
        # removed merely because another rank was outside the adaptive range.
        # ``ordered`` is a permutation for each row, so the temporary mask is
        # safe; OR keeps the top-K and mandatory guarantees intact.
        adaptive_selected = torch.zeros_like(legal)
        adaptive_selected.scatter_(1, ordered, adaptive)
        selected |= adaptive_selected
        selected &= legal

    random_added = torch.zeros_like(legal)
    if random_actions:
        available = legal & ~selected
        random_count = min(int(random_actions), width)
        random_scores = torch.rand(
            logits.shape,
            dtype=torch.float32,
            device=logits.device,
            generator=generator,
        ).masked_fill(~available, float("-inf"))
        random_indices = random_scores.topk(random_count, dim=-1).indices
        random_added.scatter_(1, random_indices, True)
        random_added &= available
        selected |= random_added

    # A malformed row without legal actions is left empty and remains the
    # native pool's responsibility.  Every normal legal row has top-K >= 1.
    excluded_legal = legal & ~selected
    masked_logits = logits.clone()
    masked_logits.masked_fill_(excluded_legal, torch.finfo(logits.dtype).min)

    with torch.no_grad():
        probabilities = torch.softmax(safe_logits, dim=-1)
        retained = (probabilities * selected).sum(dim=-1)
        retained = torch.where(legal_counts > 0, retained, torch.zeros_like(retained))
        stats = ActionPruningStats(
            rows=rows,
            pruned_rows=int((selected.sum(dim=-1) < legal_counts).sum().item()),
            legal_actions=legal_total,
            selected_actions=int(selected.sum().item()),
            mandatory_actions=int(mandatory.sum().item()),
            random_actions=int(random_added.sum().item()),
            prior_mass_retained_sum=float(retained.sum().item()),
        )
    return masked_logits, selected, stats


def restrict_root_policy_to_candidates(
    root: Mapping[str, np.ndarray],
    candidate_action_ids: Mapping[int, Sequence[int] | np.ndarray],
    enabled_rows: np.ndarray,
) -> RootRestrictionStats:
    """Zero root-policy mass outside candidates without changing legality.

    PolyEnv still creates the full root and supplies its full action mask. This
    final PolyBot-side projection guarantees that the executed action and MCTS
    training target come from the experimental candidate set.
    """
    policy = np.asarray(root["policy"])
    action_ids = np.asarray(root["action_id"])
    action_mask = np.asarray(root["action_mask"], dtype=bool)
    visit_count = np.asarray(
        root.get("visit_count", np.zeros_like(action_ids)), dtype=np.int64
    )
    enabled = np.asarray(enabled_rows, dtype=bool)
    if (
        policy.ndim != 2
        or action_ids.shape != policy.shape
        or action_mask.shape != policy.shape
    ):
        raise ValueError("root action_id, action_mask and policy must share shape [B,A]")
    if enabled.shape != (policy.shape[0],):
        raise ValueError("enabled_rows must have shape [B]")

    roots = 0
    missing = 0
    visits = 0
    outside_visits = 0
    outside_choices = 0
    for env in np.flatnonzero(enabled):
        legal_positions = np.flatnonzero(action_mask[env])
        candidates = np.asarray(candidate_action_ids.get(int(env), ()), dtype=np.int64)
        if legal_positions.size == 0 or candidates.size == 0:
            missing += 1
            continue
        allowed = np.isin(action_ids[env, legal_positions], candidates)
        if not allowed.any():
            missing += 1
            continue
        roots += 1
        row_visits = visit_count[env, legal_positions]
        visits += int(row_visits.sum())
        outside_visits += int(row_visits[~allowed].sum())
        original = policy[env, legal_positions].copy()
        outside_choices += int(not bool(allowed[int(original.argmax())]))
        policy[env, legal_positions[~allowed]] = 0.0
        total = float(policy[env, legal_positions].sum())
        if not np.isfinite(total) or total <= 0.0:
            policy[env, legal_positions] = (
                allowed.astype(policy.dtype) / int(allowed.sum())
            )
        else:
            policy[env, legal_positions] /= total
    return RootRestrictionStats(
        roots=roots,
        missing_roots=missing,
        visits=visits,
        outside_candidate_visits=outside_visits,
        unfiltered_choices_outside=outside_choices,
    )
