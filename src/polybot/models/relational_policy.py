"""Public spatial readout and ID-to-tile lookup for the shared action encoder."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class SpatialStateReadout(nn.Module):
    """Expose distributed public evidence directly to policy and value.

    All pooling uses revealed tiles. Empty visibility produces zero pooled
    features, rather than infinities or a fabricated revealed tile.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.combine = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = sqrt(rank)

    def forward(self, state, tiles, revealed):
        valid = revealed.bool()
        present = valid.any(dim=1, keepdim=True)
        mask = valid.unsqueeze(-1).to(tiles.dtype)
        mean = (tiles * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        maximum = tiles.masked_fill(~valid.unsqueeze(-1), torch.finfo(tiles.dtype).min).amax(dim=1)
        maximum = torch.where(present, maximum, torch.zeros_like(maximum))
        attention = (self.key(tiles) * self.query(state).unsqueeze(1)).sum(dim=-1) / self.scale
        # Softmax is finite even for a completely hidden board. Its result is
        # subsequently zeroed for that row, including the backward path.
        safe_valid = valid | ~present
        weights = torch.softmax(attention.masked_fill(~safe_valid, float("-inf")), dim=-1)
        attended = (weights.unsqueeze(-1).to(tiles.dtype) * tiles).sum(dim=1)
        attended = torch.where(present, attended, torch.zeros_like(attended))
        return self.norm(state + self.combine(torch.cat([state, mean, maximum, attended], dim=-1)))


def _pointer_indices(ids, objects_present, requested_ids):
    keys = ids.masked_fill(~objects_present, -1)
    sorted_ids, order = keys.sort(dim=1, stable=True)
    locations = torch.searchsorted(sorted_ids.contiguous(), requested_ids.contiguous())
    safe = locations.clamp_max(keys.shape[1] - 1)
    present = (requested_ids >= 0) & (locations < keys.shape[1]) & (sorted_ids.gather(1, safe) == requested_ids)
    return order.gather(1, safe), present


def city_tile_indices(map_tokens: torch.Tensor, city_ids: torch.Tensor):
    """Resolve public city pointers without a [batch, actions, tiles] tensor.

    Runtime identifiers are lookup keys only. No ID-dependent learned
    embedding, magnitude or ordering is exposed to the action encoder.
    """
    ids = map_tokens[..., 15]
    centres = (map_tokens[..., 0] == 1) & (map_tokens[..., 14] == 2) & (ids >= 0)
    return _pointer_indices(ids, centres, city_ids)


def unit_tile_indices(map_tokens: torch.Tensor, unit_ids: torch.Tensor):
    """Resolve unit-only actions to the public unit location."""
    ids = map_tokens[..., 6]
    units = (map_tokens[..., 0] == 1) & (map_tokens[..., 4] >= 1) & (ids >= 0)
    return _pointer_indices(ids, units, unit_ids)
