"""Shared permutation-equivariant legal-action encoder with linear set cost."""
from __future__ import annotations

import torch
from torch import nn

from .relational_policy import city_tile_indices, unit_tile_indices


class ActionEncoder(nn.Module):
    """Every legal action reads a small latent set, never an A-by-A matrix.

    Induced set attention reads the *complete* masked action list into M latent
    tokens and writes context back to every action. Positions in the candidate
    list and runtime IDs have no learned embeddings. Endpoint IDs are pointers.
    """

    def __init__(self, state_width, width, heads, latents, feature_max, option_fields, dropout):
        super().__init__()
        self.width = width
        self.option_fields = tuple(option_fields)
        self.feature_max = tuple(feature_max)
        self.type_embedding = nn.Embedding(feature_max[0]+2, width)
        self.options = nn.ModuleList([nn.Embedding(feature_max[i]+2, width) for i in self.option_fields])
        self.continuous = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.state_projection = nn.Linear(state_width, width)
        self.source_projection = nn.Linear(state_width, width)
        self.target_projection = nn.Linear(state_width, width)
        self.city_projection = nn.Linear(state_width, width)
        self.geometry = nn.Linear(8, width)
        self.token_norm = nn.LayerNorm(width)
        self.latent_seeds = nn.Parameter(torch.randn(latents, width) * .02)
        self.read_actions = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.read_norm = nn.LayerNorm(width)
        self.read_ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.read_ffn_norm = nn.LayerNorm(width)
        self.write_actions = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.write_norm = nn.LayerNorm(width)
        self.write_ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.write_ffn_norm = nn.LayerNorm(width)
        self.position_readout = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.to_position = nn.Linear(width, state_width, bias=False)
        self.position_norm = nn.LayerNorm(state_width)

    def forward(self, state, tiles, map_tokens, categorical, continuous, mask):
        if categorical.shape[:2] != mask.shape or continuous.shape[:2] != mask.shape:
            raise ValueError("action encoder needs aligned action features and legal mask")
        batch, count = mask.shape
        if count == 0:
            return state.new_zeros((batch, 0, self.width)), state
        def indices(values, largest):
            return (values+1).clamp(0, largest+1)
        tokens = self.type_embedding(indices(categorical[..., 0], self.feature_max[0]))
        for field, embedding in zip(self.option_fields, self.options):
            tokens = tokens + embedding(indices(categorical[..., field], self.feature_max[field]))
        tokens = tokens + self.continuous(torch.asinh(continuous.float()/20.0))
        tokens = tokens + self.state_projection(state).unsqueeze(1)
        tile_count = tiles.shape[1]
        unit_index, unit_present = unit_tile_indices(map_tokens, categorical[..., 3])
        def endpoint(column, projection):
            raw = categorical[..., column]
            present = (raw >= 0) & (raw < tile_count)
            safe = raw.clamp(0, tile_count-1)
            if column == 1:
                safe = torch.where(present, safe, unit_index)
                present = present | unit_present
            projected = projection(tiles)
            gathered = projected.gather(1, safe.unsqueeze(-1).expand(-1, -1, self.width))
            return gathered * present.unsqueeze(-1), safe, present
        source, src, src_present = endpoint(1, self.source_projection)
        target, dst, dst_present = endpoint(2, self.target_projection)
        city_index, city_present = city_tile_indices(map_tokens, categorical[..., 4])
        cities = self.city_projection(tiles).gather(1, city_index.unsqueeze(-1).expand(-1, -1, self.width))
        cities = cities * city_present.unsqueeze(-1)
        side = max(1, int(round(tile_count**.5)))
        both = src_present & dst_present
        dx = ((dst % side - src % side).float()/max(1, side-1)) * both
        dy = ((dst // side - src // side).float()/max(1, side-1)) * both
        geometry = torch.stack([src_present, dst_present, city_present, dx, dy, dx.abs(), dy.abs(), dx.square()+dy.square()], -1).to(tokens.dtype)
        tokens = self.token_norm(tokens + source + target + cities + self.geometry(geometry))
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        present = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~present, 0] = True  # neutral dummy for completely empty sets
        seeds = self.latent_seeds.unsqueeze(0).expand(batch, -1, -1)
        read, _ = self.read_actions(seeds, tokens, tokens, key_padding_mask=~safe_mask, need_weights=False)
        latent = self.read_norm(seeds + read)
        latent = self.read_ffn_norm(latent + self.read_ffn(latent))
        written, _ = self.write_actions(tokens, latent, latent, need_weights=False)
        actions = self.write_norm(tokens + written)
        actions = self.write_ffn_norm(actions + self.write_ffn(actions))
        actions = actions.masked_fill(~mask.unsqueeze(-1), 0)
        summary, _ = self.position_readout(self.state_projection(state).unsqueeze(1), actions, actions,
                                           key_padding_mask=~safe_mask, need_weights=False)
        conditioned = self.position_norm(state + self.to_position(summary.squeeze(1)))
        conditioned = torch.where(present.unsqueeze(-1), conditioned, state)
        return actions, conditioned
