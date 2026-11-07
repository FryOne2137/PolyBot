"""Contextual legal-action scoring with cost linear in the action count."""

from __future__ import annotations

from math import isqrt, sqrt
from typing import Sequence

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
    embedding, magnitude or ordering is exposed to the scorer.
    """
    ids = map_tokens[..., 15]
    centres = (map_tokens[..., 0] == 1) & (map_tokens[..., 14] == 2) & (ids >= 0)
    return _pointer_indices(ids, centres, city_ids)


def unit_tile_indices(map_tokens: torch.Tensor, unit_ids: torch.Tensor):
    """Resolve unit-only actions to the public unit location."""
    ids = map_tokens[..., 6]
    units = (map_tokens[..., 0] == 1) & (map_tokens[..., 4] >= 1) & (ids >= 0)
    return _pointer_indices(ids, units, unit_ids)


class RelationalActionPolicy(nn.Module):
    """One shared scorer with explicit type/context and endpoint interactions.

    Tile projections are computed once. Every legal action then gathers its
    endpoints in the narrow rank dimension; no attention over the map or over
    other candidate actions is performed.
    """

    def __init__(self, d_model: int, rank: int, feature_max: Sequence[int], option_fields: Sequence[int]):
        super().__init__()
        self.option_fields = tuple(option_fields)
        self.feature_max = tuple(feature_max)
        self.type_embedding = nn.Embedding(feature_max[0] + 2, rank)
        self.options = nn.ModuleList([
            nn.Embedding(feature_max[index] + 2, rank) for index in self.option_fields
        ])
        self.continuous = nn.Sequential(nn.Linear(4, rank), nn.GELU(), nn.Linear(rank, rank))
        self.global_projection = nn.Linear(d_model, rank)
        self.source_projection = nn.Linear(d_model, rank)
        self.target_projection = nn.Linear(d_model, rank)
        self.city_projection = nn.Linear(d_model, rank)
        self.geometry = nn.Linear(8, rank)
        self.pair_projection = nn.Linear(rank, rank, bias=False)
        self.query_norm = nn.LayerNorm(rank)
        self.score = nn.Sequential(
            nn.LayerNorm(rank), nn.Linear(rank, 2 * rank), nn.GELU(), nn.Linear(2 * rank, 1),
        )

    def forward(self, state, tile_states, map_tokens, categorical, continuous):
        tiles = tile_states.shape[1]
        width = isqrt(tiles)
        if width * width != tiles:
            raise ValueError("relational policy requires a square board")
        def indices(values, largest):
            return (values + 1).clamp(0, largest + 1)
        query = self.type_embedding(indices(categorical[..., 0], self.feature_max[0]))
        query = query + self.continuous(continuous)
        for field, embedding in zip(self.option_fields, self.options):
            query = query + embedding(indices(categorical[..., field], self.feature_max[field]))
        query = self.query_norm(query + self.global_projection(state).unsqueeze(1))

        unit_index, unit_present = unit_tile_indices(map_tokens, categorical[..., 3])
        def endpoint(column, projection):
            raw = categorical[..., column]
            present = (raw >= 0) & (raw < tiles)
            safe = raw.clamp(0, tiles - 1)
            if column == 1:
                safe = torch.where(present, safe, unit_index)
                present = present | unit_present
            projected = projection(tile_states)
            selected = projected.gather(1, safe.unsqueeze(-1).expand(-1, -1, projected.shape[-1]))
            return selected * present.unsqueeze(-1).to(selected.dtype), safe, present
        source, src_index, src_present = endpoint(1, self.source_projection)
        target, dst_index, dst_present = endpoint(2, self.target_projection)
        city_index, city_present = city_tile_indices(map_tokens, categorical[..., 4])
        city_features = self.city_projection(tile_states)
        city = city_features.gather(1, city_index.unsqueeze(-1).expand(-1, -1, city_features.shape[-1]))
        city = city * city_present.unsqueeze(-1).to(city.dtype)

        both = src_present & dst_present
        dx = ((dst_index % width - src_index % width).float() / max(1, width - 1)) * both
        dy = ((dst_index // width - src_index // width).float() / max(1, width - 1)) * both
        geometry = torch.stack([
            src_present, dst_present, city_present, dx, dy, dx.abs(), dy.abs(), dx.square() + dy.square(),
        ], dim=-1).to(query.dtype)
        endpoints = source + target + city
        joint = (
            query + endpoints + query * endpoints
            + self.pair_projection(source * target) + self.geometry(geometry)
        )
        return self.score(joint).squeeze(-1)
