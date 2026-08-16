"""Loss functions and one explicit PyTorch optimisation step."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ..config import TrainingConfig
from ..data.token_schema import MAP_FEATURE_MAX
from ..models.policy_value_belief import (
    CITY_PARTICLE_FEATURES,
    SAFE_PARTICLE_FEATURES,
    UNIT_PARTICLE_FEATURES,
    PolicyValueBeliefNet,
)


# Train exactly the semantic fields consumed by the constrained belief-world
# decoder.  Columns 6, 15 and 22 are runtime city IDs whose numeric values are
# arbitrary within one game; treating them as categorical classes teaches no
# transferable map structure.  Visibility/cloak columns 0 and 1 are omitted as
# well because completion preserves/derives them instead of sampling them.
BELIEF_LOSS_FEATURES = frozenset(
    (*SAFE_PARTICLE_FEATURES, 14, *CITY_PARTICLE_FEATURES, *UNIT_PARTICLE_FEATURES)
)
BELIEF_RUNTIME_ID_FEATURES = frozenset((6, 15, 22))


def belief_loss(
    belief_logits: Iterable[torch.Tensor],
    full_map_target: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> torch.Tensor:
    """Field-aware cross-entropy for semantic values on hidden tiles.

    City/unit attributes are undefined outside their owning entity.  Training
    them there would teach the decoder that ``-1`` is the only useful answer.
    Their losses are therefore evaluated only on hidden city/unit rows; the
    presence heads get a moderate rare-class weight so a belief world can learn
    to place actual cities and units instead of only terrain. Runtime IDs and
    fields not sampled by the belief-world decoder contribute no gradient.
    """
    logits_by_feature = tuple(belief_logits)
    if len(logits_by_feature) != len(MAP_FEATURE_MAX):
        raise ValueError(
            f"belief_loss expected {len(MAP_FEATURE_MAX)} feature heads, "
            f"got {len(logits_by_feature)}"
        )
    feature_losses = []
    for column in sorted(BELIEF_LOSS_FEATURES):
        logits = logits_by_feature[column]
        largest_value = MAP_FEATURE_MAX[column]
        labels = (full_map_target[..., column] + 1).clamp(0, largest_value + 1)
        per_tile = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        loss_mask = hidden_mask.bool()
        if column in (9, 10, 11, 12, 13, 16, 17):
            loss_mask = loss_mask & (full_map_target[..., 14] == 2)
        elif column in (2, 3, 5, 21, 22):
            loss_mask = loss_mask & (full_map_target[..., 4] >= 1)

        weights = torch.ones_like(per_tile)
        if column == 14:
            weights = torch.where(full_map_target[..., 14] != 0, weights * 4.0, weights)
        elif column == 4:
            weights = torch.where(full_map_target[..., 4] >= 1, weights * 6.0, weights)
        denominator = (weights * loss_mask).sum().clamp_min(1.0)
        feature_losses.append((per_tile * weights * loss_mask).sum() / denominator)
    return torch.stack(feature_losses).mean()


def train_batch(
    model: PolicyValueBeliefNet,
    optimiser: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    config: TrainingConfig,
) -> Dict[str, float]:
    """Optimise policy, value and belief objectives on one padded batch."""
    model.train()
    output = model(
        batch["map_tokens"],
        batch["global_features"],
        batch["action_categorical"],
        batch["action_continuous"],
        batch["action_mask"],
        batch["event_features"],
        action_history=batch["action_history"],
        action_history_mask=batch["action_history_mask"],
    )
    policy_log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
    # Padded actions intentionally have a -inf logit.  Replace their log
    # probability with zero *before* multiplication: 0 * -inf is NaN.
    valid_log_probabilities = torch.where(
        batch["action_mask"],
        policy_log_probabilities,
        torch.zeros_like(policy_log_probabilities),
    )
    policy = -(
        batch["policy_target"] * valid_log_probabilities
    ).sum(dim=-1).mean()
    per_example_value = F.smooth_l1_loss(
        output.value, batch["value_target"], reduction="none"
    )
    value_weights = batch.get("value_loss_weight")
    if value_weights is None:
        value = per_example_value.mean()
    else:
        value_weights = value_weights.to(per_example_value.dtype)
        value = (per_example_value * value_weights).sum() / value_weights.sum().clamp_min(1.0)
    belief = belief_loss(output.belief_logits, batch["full_map_target"], batch["hidden_mask"])
    total = (
        config.policy_weight * policy
        + config.value_weight * value
        + config.belief_weight * belief
    )

    optimiser.zero_grad(set_to_none=True)
    total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

    return {
        "loss": float(total.detach().cpu()),
        "policy_loss": float(policy.detach().cpu()),
        "value_loss": float(value.detach().cpu()),
        "belief_loss": float(belief.detach().cpu()),
    }
